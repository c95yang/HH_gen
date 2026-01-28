from logging import getLogger
from pathlib import Path
import csv
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from config.config import ProjectConfig
from tridi.utils.metrics import generation
from tridi.utils.metrics import reconstruction

logger = getLogger(__name__)


class Evaluator:
    def __init__(self, cfg: ProjectConfig):
        self.cfg = cfg

    def _detect_method_name(self) -> str:
        # 1) explicit override from CLI / yaml
        v = getattr(self.cfg.eval, "method_name", None)
        if v is not None and str(v).strip() != "":
            return str(v)

        # 2) fallback: checkpoint sniff (optional)
        ckpt_path = getattr(self.cfg.resume, "checkpoint", None)
        if ckpt_path:
            try:
                ckpt = torch.load(str(ckpt_path), map_location="cpu")
                if str(ckpt.get("model_type", "")).lower() == "nn_baseline":
                    return "NNBaseline"
            except Exception:
                pass

        return "HHgen"


    def _get_cfg_eval_samples_folder(self) -> Optional[Path]:
        try:
            v = getattr(self.cfg.eval, "samples_folder", None)
        except Exception:
            v = None
        if v is None or str(v).strip() == "":
            return None
        return Path(v)

    def _resolve_base_samples_folder(self) -> Path:
        override = self._get_cfg_eval_samples_folder()
        if override is not None:
            return override
        return Path(self.cfg.run.path) / "artifacts" / f"step_{self.cfg.resume.step}_samples"

    def evaluate(self):
        base_samples_folder = self._resolve_base_samples_folder()

        rows: List[Dict[str, Any]] = []
        exp_name = str(self.cfg.run.name)
        step_str = str(self.cfg.resume.step)

        logger.info(f"Experiment: {self.cfg.run.name} step: {self.cfg.resume.step}")

        # =========================
        # Generation metrics
        # =========================
        if self.cfg.eval.use_gen_metrics:
            logger.info("Evaluating generation")
            for dataset in self.cfg.run.datasets:
                logger.info(f"\ton {dataset}")
                samples_folder = base_samples_folder / dataset

                for sample_target in self.cfg.eval.sampling_target:
                    logger.info(f"\t  sampling target: {sample_target}")
                    metrics = {"1-NNA": [], "COV": [], "MMD": [], "SD": []}
                    samples_files = list(samples_folder.glob(f"{sample_target}/samples_rep_*.hdf5"))

                    for samples_file in samples_files:
                        metrics["1-NNA"].append(generation.nearest_neighbor_accuracy(
                            self.cfg, samples_file, dataset, "test", sample_target,
                        ))
                        metrics["COV"].append(generation.coverage(
                            self.cfg, samples_file, dataset, "test", sample_target,
                        ))
                        metrics["MMD"].append(generation.minimum_matching_distance(
                            self.cfg, samples_file, dataset, "test", sample_target,
                        ))
                        metrics["SD"].append(generation.sample_distance(
                            self.cfg, samples_file, dataset, "train", sample_target,
                        ))

                    for k, v in metrics.items():
                        if len(v) == 0:
                            continue
                        if k in ["1-NNA", "COV"]:
                            mean = float(np.mean(v)) * 100
                            std = float(np.std(v)) * 100
                            logger.info(f"\t\t{k:<6s} - {sample_target}: {mean:.2f} ± {std:.2f}")
                            rows.append({
                                "exp": exp_name, "step": step_str, "phase": "gen",
                                "dataset": dataset, "target": sample_target,
                                "metric": k, "mean_std": f"{mean:.2f} ± {std:.2f}"
                            })
                        else:
                            mean = float(np.mean(v))
                            std = float(np.std(v))
                            logger.info(f"\t\t{k:<6s} - {sample_target}: {mean:.2f} ± {std:.2f}")
                            rows.append({
                                "exp": exp_name, "step": step_str, "phase": "gen",
                                "dataset": dataset, "target": sample_target,
                                "metric": k, "mean_std": f"{mean:.2f} ± {std:.2f}"
                            })

        # =========================
        # Reconstruction metrics
        # =========================
        if self.cfg.eval.use_rec_metrics:
            logger.info("Evaluating reconstruction")
            for dataset in self.cfg.run.datasets:
                logger.info(f"\ton {dataset}")
                samples_folder = base_samples_folder / dataset

                for sample_target in self.cfg.eval.sampling_target:
                    if sample_target not in ["sbj", "second_sbj"]:
                        continue

                    samples_files = list(samples_folder.glob(f"{sample_target}/samples_rep_*.hdf5"))
                    metrics = {f"{sample_target}_MPJPE": [], f"{sample_target}_MPJPE_PA": []}

                    for samples_file in samples_files:
                        mpjpe, mpjpe_pa = reconstruction.get_sbj_metrics(
                            self.cfg, samples_file, dataset, sample_target
                        )
                        metrics[f"{sample_target}_MPJPE"].append(mpjpe)
                        metrics[f"{sample_target}_MPJPE_PA"].append(mpjpe_pa)

                    for k, v in metrics.items():
                        if len(v) == 0:
                            continue
                        val = float(np.mean(np.min(np.stack(v, 1), axis=1)) * 100)
                        logger.info(f"\t\t{k:<20s}: {val:.1f}")
                        rows.append({
                            "exp": exp_name, "step": step_str, "phase": "rec",
                            "dataset": dataset, "target": sample_target,
                            "metric": k, "value": f"{val:.1f}"
                        })

        out_dir = base_samples_folder / "_eval"
        self._save_tables(rows, out_dir)

    def _save_tables(self, rows: List[Dict[str, Any]], out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "evaluation_results.csv"
        method = self._detect_method_name()

        with open(csv_path, "w", newline="") as csvfile:
            fieldnames = ["method", "target", "metric", "value", "mean_std"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    "method": method,
                    "target": row.get("target", ""),
                    "metric": row.get("metric", ""),
                    "value": row.get("value", ""),
                    "mean_std": row.get("mean_std", ""),
                })

        logger.info(f"Saved evaluation results to {csv_path}")
