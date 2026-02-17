from collections import defaultdict
from pathlib import Path
import math
import sys
import time

import torch
import numpy as np
import wandb
import torch.nn.functional as F
from omegaconf import OmegaConf

from config.config import ProjectConfig
from tridi.utils.training import (
    get_optimizer, get_scheduler, TrainState,
    resume_from_checkpoint, compute_grad_norm
)
from tridi.data import get_train_dataloader
from tridi.data.hh_batch_data import HHBatchData
from tridi.model.wrappers.mesh import MeshModel
from tridi.utils.metrics.reconstruction import get_mpjpe, get_mpjpe_pa
from tridi.model.base import TriDiModelOutput

from logging import getLogger
logger = getLogger(__name__)


class Trainer:
    def __init__(self, cfg: ProjectConfig, model):
        self.cfg = cfg
        self.device = torch.device("cuda")

        model = model.to(self.device)

        optimizer = get_optimizer(cfg, model)
        scheduler = get_scheduler(cfg, optimizer)

        # Resume from checkpoint and create the initial training state
        self.train_state: TrainState = resume_from_checkpoint(cfg, model, optimizer, scheduler)

        # Get dataloaders
        dataloader_train, dataloader_val = get_train_dataloader(cfg)

        self.model = model
        self.mesh_model: MeshModel = MeshModel(
            model_path=cfg.env.smpl_folder,
            batch_size=cfg.dataloader.batch_size,
            device=self.model.device
        )
        self.model.set_mesh_model(self.mesh_model)

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val

        # -------------------------
        # Early stopping config (robust for missing keys)
        # -------------------------
        def _sel(path: str, default):
            v = OmegaConf.select(self.cfg, path, default=default)
            return default if v is None else v

        self.es_enabled = bool(_sel("train.early_stop", False))
        self.es_metric = str(_sel("train.early_stop_metric", "VAL_loss/total"))
        self.es_mode = str(_sel("train.early_stop_mode", "min"))  # "min" or "max"
        self.es_patience = int(_sel("train.early_stop_patience", 10))
        self.es_min_delta = float(_sel("train.early_stop_min_delta", 0.0))
        self.es_warmup_epochs = int(_sel("train.early_stop_warmup_epochs", 0))
        self.es_save_best = bool(_sel("train.early_stop_save_best", True))

        # state
        self.es_best = float("inf") if self.es_mode == "min" else -float("inf")
        self.es_bad_epochs = 0
        self.es_best_step = None  # type: Optional[int]

        # try restore early-stop state from resume checkpoint (if exists)
        self._restore_early_stop_state_from_checkpoint()

    # -------------------------
    # Early stop helpers
    # -------------------------
    def _best_ptr_path(self) -> Path:
        ckpt_dir = Path(self.cfg.run.path) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        return ckpt_dir / "best_checkpoint.txt"

    def _write_best_pointer(self, ckpt_path: Path):
        # does NOT change ckpt naming; only writes a pointer file
        p = self._best_ptr_path()
        p.write_text(str(ckpt_path.name) + "\n", encoding="utf-8")
        logger.info(f"[EarlyStop] Updated best pointer: {p} -> {ckpt_path.name}")

    def _restore_early_stop_state_from_checkpoint(self):
        if not self.es_enabled:
            return
        ckpt_path = OmegaConf.select(self.cfg, "resume.checkpoint", default=None)
        if ckpt_path is None:
            return
        ckpt_path = Path(str(ckpt_path))
        if not ckpt_path.exists():
            return

        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            es = ckpt.get("early_stop", None)
            if es is None:
                return
            self.es_best = float(es.get("best", self.es_best))
            self.es_bad_epochs = int(es.get("bad_epochs", self.es_bad_epochs))
            self.es_best_step = es.get("best_step", self.es_best_step)
            logger.info(
                f"[EarlyStop] Restored state from {ckpt_path.name}: "
                f"best={self.es_best}, bad_epochs={self.es_bad_epochs}, best_step={self.es_best_step}"
            )
        except Exception as e:
            logger.warning(f"[EarlyStop] Failed to restore early-stop state: {e}")

    def _is_improved(self, current: float) -> bool:
        if self.es_mode == "min":
            return current < (self.es_best - self.es_min_delta)
        if self.es_mode == "max":
            return current > (self.es_best + self.es_min_delta)
        raise ValueError(f"Unknown early_stop_mode: {self.es_mode}")

    def _maybe_early_stop(self, val_log: dict) -> bool:
        """
        Returns True if training should stop now.
        Called once per epoch after validation.
        """
        if not self.es_enabled:
            return False
        if self.train_state.epoch < self.es_warmup_epochs:
            return False

        if self.es_metric not in val_log:
            logger.warning(
                f"[EarlyStop] metric '{self.es_metric}' not found in val_log. "
                f"Available keys: {list(val_log.keys())[:40]} ..."
            )
            return False

        current = float(val_log[self.es_metric])
        improved = self._is_improved(current)

        if improved:
            self.es_best = current
            self.es_bad_epochs = 0
            self.es_best_step = int(self.train_state.step)
            logger.info(f"[EarlyStop] Improved {self.es_metric}={current:.6f} (best). step={self.train_state.step}")

            if self.es_save_best:
                # Save a normal step checkpoint (name stays checkpoint-step-XXXXXXX.pth)
                ckpt_path = self.save_checkpoint()
                self._write_best_pointer(ckpt_path)

        else:
            self.es_bad_epochs += 1
            logger.info(
                f"[EarlyStop] No improve {self.es_metric}={current:.6f}, best={self.es_best:.6f}. "
                f"bad_epochs={self.es_bad_epochs}/{self.es_patience}"
            )
            if self.es_bad_epochs >= self.es_patience:
                logger.info(
                    f"[EarlyStop] Patience exhausted -> stop. "
                    f"epoch={self.train_state.epoch}, step={self.train_state.step}"
                )
                # Always save final checkpoint at stopping point
                self.save_checkpoint()
                if self.cfg.logging.wandb:
                    wandb.finish()
                time.sleep(2)
                return True

        return False

    # -------------------------
    # Core training logic
    # -------------------------
    def get_outputs(self, batch):
        # get gt sbj vertices and joints
        with torch.no_grad():
            gt_sbj_vertices, gt_sbj_joints, gt_second_sbj_vertices, gt_second_sbj_joints = self.mesh_model.get_smpl_th(batch)
            batch.sbj_vertices = gt_sbj_vertices
            batch.sbj_joints = gt_sbj_joints
            batch.second_sbj_vertices = gt_second_sbj_vertices
            batch.second_sbj_joints = gt_second_sbj_joints

        # aux_output is (x_0, x_t, noise, x_0_pred, timestep_sbj, timestep_second_sbj)
        denoise_loss, aux_output = self.model(batch, 'train', return_intermediate_steps=True)
        output = self.model.split_output(aux_output[3], aux_output)

        sbj_vertices, sbj_joints, second_sbj_vertices, second_sbj_joints = self.mesh_model.get_meshes_wkpts_th(
            output,
            scale=batch.scale,
            sbj_gender=batch.sbj_gender,
            second_sbj_gender=batch.second_sbj_gender,
            return_joints=True
        )
        output.sbj_vertices = sbj_vertices
        output.sbj_joints = sbj_joints
        output.second_sbj_vertices = second_sbj_vertices
        output.second_sbj_joints = second_sbj_joints

        return denoise_loss, output

    def compute_loss(self, batch: HHBatchData, output: TriDiModelOutput, denoise_loss):
        wandb_log = dict()
        loss = 0.0

        for key, weight in self.cfg.train.losses.items():
            if key == "smpl_v2v":
                gt_sbj_vertices = batch.sbj_vertices.to(output.sbj_vertices.device)
                pred_sbj_vertices = output.sbj_vertices
                loss_i = F.mse_loss(pred_sbj_vertices, gt_sbj_vertices, reduction='none')
            elif key == "second_smpl_v2v":
                gt_second_sbj_vertices = batch.second_sbj_vertices.to(output.second_sbj_vertices.device)
                pred_second_sbj_vertices = output.second_sbj_vertices
                loss_i = F.mse_loss(pred_second_sbj_vertices, gt_second_sbj_vertices, reduction='none')
            elif key.startswith("denoise"):
                loss_i = denoise_loss[key]
            else:
                raise NotImplementedError(f"No implementation for {key} loss.")

            loss_i = loss_i.mean()
            loss = loss + float(weight) * loss_i
            wandb_log[f'loss/{key}'] = loss_i.detach().cpu().item()

        wandb_log["loss/total"] = float(loss.detach().cpu().item())
        return loss, wandb_log

    def train_step(self, batch):
        self.model.train()
        denoise_loss, output = self.get_outputs(batch)
        loss, wandb_log = self.compute_loss(batch, output, denoise_loss)

        # backward
        loss.backward()
        if self.cfg.optimizer.clip_grad_norm is not None:
            grad_norm_unclipped = compute_grad_norm(self.model.parameters())
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optimizer.clip_grad_norm)
            wandb_log['grad_norm'] = grad_norm_unclipped
            wandb_log['grad_norm_clipped'] = compute_grad_norm(self.model.parameters())

        # optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        self.train_state.step += 1

        return wandb_log

    def save_checkpoint(self) -> Path:
        checkpoint_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': self.train_state.epoch,
            'step': self.train_state.step,
            'relative_step': self.train_state.step - self.train_state.initial_step,
            'cfg': self.cfg,
            'early_stop': {
                'best': self.es_best,
                'bad_epochs': self.es_bad_epochs,
                'best_step': self.es_best_step,
                'metric': self.es_metric,
                'mode': self.es_mode,
                'patience': self.es_patience,
                'min_delta': self.es_min_delta,
                'warmup_epochs': self.es_warmup_epochs,
            }
        }

        checkpoint_name = f'checkpoint-step-{self.train_state.step:07d}.pth'
        checkpoint_dir = Path(self.cfg.run.path) / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / checkpoint_name

        torch.save(checkpoint_dict, checkpoint_path)
        logger.info(f'Saved checkpoint to {checkpoint_path}')
        return checkpoint_path

    def train(self):
        total_batch_size = self.cfg.dataloader.batch_size

        logger.info(
            f'***** Starting training *****\n'
            f'    Dataset train size: {len(self.dataloader_train.dataset):_}\n'
            f'    Dataset val size: {len(self.dataloader_val.dataset):_}\n'
            f'    Dataloader train size: {len(self.dataloader_train):_}\n'
            f'    Dataloader val size: {len(self.dataloader_val):_}\n'
            f'    Batch size per device = {self.cfg.dataloader.batch_size}\n'
            f'    Total train batch size (w. parallel, dist & accum) = {total_batch_size}\n'
            f'    Gradient Accumulation steps = {self.cfg.optimizer.gradient_accumulation_steps}\n'
            f'    Max training steps = {self.cfg.train.max_steps}\n'
            f'    Training state = {self.train_state}\n'
            f'    EarlyStop: enabled={self.es_enabled}, metric={self.es_metric}, mode={self.es_mode}, '
            f'patience={self.es_patience}, min_delta={self.es_min_delta}, warmup_epochs={self.es_warmup_epochs}'
        )

        while True:
            self.model.train()
            for i, batch in enumerate(self.dataloader_train):
                if (self.cfg.train.limit_train_batches is not None) and (i >= self.cfg.train.limit_train_batches):
                    break

                wandb_log = self.train_step(batch)

                if not math.isfinite(wandb_log["loss/total"]):
                    logger.error(f"Loss is {wandb_log['loss/total']}, stopping training")
                    sys.exit(1)

                if self.cfg.logging.wandb and self.train_state.step % self.cfg.train.log_step_freq == 0:
                    wandb_log['lr'] = self.optimizer.param_groups[0]['lr']
                    wandb_log['step'] = self.train_state.step
                    wandb_log['relative_step'] = self.train_state.step - self.train_state.initial_step
                    wandb.log(wandb_log, step=self.train_state.step)

                if (self.train_state.step % self.cfg.train.checkpoint_freq == 0):
                    self.save_checkpoint()

                if self.train_state.step >= self.cfg.train.max_steps:
                    logger.info(f'Ending training at with state: {self.train_state}')
                    if self.cfg.logging.wandb:
                        wandb.finish()
                    time.sleep(2)
                    return

            # -------- Validation (end of epoch) --------
            self.model.eval()
            val_log_sum, val_metrics_sum, val_counters = defaultdict(float), defaultdict(float), defaultdict(int)

            for i, batch in enumerate(self.dataloader_val):
                val_losses, tmp_metrics, tmp_counters = self.val_step(batch)

                for k, v in val_losses.items():
                    val_log_sum[k] += float(v)
                for k in tmp_metrics.keys():
                    val_metrics_sum[k] += float(tmp_metrics[k])
                    val_counters[k] += int(tmp_counters[k])

            # average losses
            val_log = {f"VAL_{k}": v / max(1, len(self.dataloader_val)) for k, v in val_log_sum.items()}
            val_log["epoch"] = self.train_state.epoch

            # average metrics
            for k in val_metrics_sum.keys():
                val_log[f"VAL_{k}"] = val_metrics_sum[k] / (val_counters[k] + 1e-4)

            if self.cfg.logging.wandb:
                wandb.log(val_log, step=self.train_state.step)

            # ---- Early stopping decision ----
            if self._maybe_early_stop(val_log):
                return

            # Epoch complete
            self.train_state.epoch += 1

    @torch.no_grad()
    def compute_val_metrics(self, batch, output):
        METRICS = ["MPJPE", "MPJPE_PA"]
        tmp_metrics = {metric: 0.0 for metric in METRICS}
        tmp_counters = {metric: 0 for metric in METRICS}

        for i in range(batch.batch_size()):
            mpjpe = get_mpjpe(
                output.sbj_joints[i].detach().cpu().numpy(),
                batch.sbj_joints[i].detach().cpu().numpy()
            )
            tmp_metrics["MPJPE"] += mpjpe
            tmp_counters["MPJPE"] += 1

            mpjpe_pa = get_mpjpe_pa(
                output.sbj_joints[i].detach().cpu().numpy(),
                batch.sbj_joints[i].detach().cpu().numpy()
            )
            tmp_metrics["MPJPE_PA"] += mpjpe_pa
            tmp_counters["MPJPE_PA"] += 1

        return tmp_metrics, tmp_counters

    @torch.no_grad()
    def val_step(self, batch):
        denoise_loss, output = self.get_outputs(batch)
        _, wandb_log = self.compute_loss(batch, output, denoise_loss)
        val_metrics, val_counters = self.compute_val_metrics(batch, output)

        return wandb_log, val_metrics, val_counters
