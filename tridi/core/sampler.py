from logging import getLogger
from pathlib import Path
from typing import List, Tuple

import clip
import h5py
import numpy as np
import torch
import trimesh
import wandb
from trimesh import visual as tr_visual

from tridi.data.hh_batch_data import HHBatchData

from config.config import ProjectConfig
from tridi.data import get_eval_dataloader
from tridi.model.base import TriDiModelOutput, apply_pose_only_like_baseline
from tridi.model.wrappers.contact import ContactModel
from tridi.model.wrappers.text_condition import TextConditionModel
from tridi.model.wrappers.mesh import MeshModel
from tridi.utils.geometry import rotation_6d_to_matrix
from tridi.utils.training import TrainState, resume_from_checkpoint

logger = getLogger(__name__)


class Sampler:
    def __init__(self, cfg: ProjectConfig, model):
        self.cfg = cfg

        self.device = torch.device("cuda")
        self.model = model.to(self.device)

        # Resume from checkpoint and create the initial training state
        self.train_state: TrainState = resume_from_checkpoint(cfg, model, None, None)

        # Get dataloaders
        self.dataloaders = get_eval_dataloader(cfg)

        # Model prediction -> meshes
        self.mesh_model = MeshModel(
            model_path=cfg.env.smpl_folder,
            batch_size=2 * cfg.dataloader.batch_size,
            device=self.device
        )
        self.model.set_mesh_model(self.mesh_model)

        # Create folder for artifacts
        self.base_samples_folder = (Path(self.cfg.run.path) / "artifacts"
                               / f"step_{self.cfg.resume.step}_samples")
        self.base_samples_folder.mkdir(parents=True, exist_ok=True)

        #self.text_condition_model = TextConditionModel("clip", device=self.device)
        self.text_condition_model = None
# 
    
    @torch.no_grad()
    def sample_step(self, batch):
        # make sure it's HHBatchData so we can override fields
        if isinstance(batch, dict):
            batch = HHBatchData(**batch)

        B = batch.batch_size()
        dev = batch.sbj_gender.device if hasattr(batch, "sbj_gender") else torch.device("cpu")

        # 1) resolve gender from cfg (even "random" -> explicit tensor, so mesh matches conditioning)
        g1 = self._gender_to_tensor(self.cfg.sample.sbj_gender, B, dev)          # (B,) long
        g2 = self._gender_to_tensor(self.cfg.sample.second_sbj_gender, B, dev)   # (B,) long

        # 2) override batch genders for mesh generation consistency
        # dataset uses bool: True=female, False=male
        batch.sbj_gender = (g1 == 1)
        batch.second_sbj_gender = (g2 == 1)

        # 3) pass gender to the model (conditioning)
        result = self.model(
            batch,
            "sample",
            sample_type=self.cfg.sample.mode,
            sbj_gender=g1,                 # long OK
            second_sbj_gender=g2,          # long OK
        )

        retrieved_output = None
        if isinstance(result, tuple):
            output, retrieved_output = result
        else:
            output = result

        output = self.model.split_output(output)
        if retrieved_output is not None:
            retrieved_output = self.model.split_output(retrieved_output)

        if bool(getattr(self.cfg.sample, "pose_only_like_baseline", False)):
            output = apply_pose_only_like_baseline(output, enabled=True)
            if retrieved_output is not None:
                retrieved_output = apply_pose_only_like_baseline(retrieved_output, enabled=True)

        return output, retrieved_output

    
    #new added
    def _gender_to_tensor(self, spec: str, B: int, device) -> torch.Tensor:
        """
        returns LongTensor (B,) with {0: male, 1: female}
        spec: "male" | "female" | "random"
        """
        spec = (spec or "random").lower()
        if spec == "random":
            return torch.randint(0, 2, (B,), device=device, dtype=torch.long)
        if spec in ("male", "m", "0"):
            return torch.zeros((B,), device=device, dtype=torch.long)
        if spec in ("female", "f", "1"):
            return torch.ones((B,), device=device, dtype=torch.long)
        raise ValueError(f"Unknown gender spec: {spec}")

    @staticmethod
    def sample_mode_to_str(sample_mode, contacts_mode):
        sample_str = []
        if sample_mode[0] == "1":
            sample_str.append("sbj")
        if sample_mode[1] == "1":
            sample_str.append("second_sbj")
        sample_str = "_".join(sample_str)

        return sample_str

    @torch.no_grad()
    def sample(self):
        # Log general info
        logger.info(
            f'***** Starting sampling *****\n'
            f'    Model: {self.cfg.model_denoising.name}\n'
            f'    Checkpoint step number: {self.cfg.resume.step}\n'
            f'    Number of repetitions: {self.cfg.sample.repetitions}\n'
        )

        for dataloader in self.dataloaders:
            # Log info
            logger.info(
                f'    Sampling mode {self.cfg.sample.mode} for: {dataloader.dataset.name}\n'
                f'    Number of samples: {len(dataloader.dataset)}\n'
            )
            # create folder for the samples
            sample_mode = self.sample_mode_to_str(self.cfg.sample.mode, self.cfg.sample.contacts_mode)
            samples_folder = self.base_samples_folder / f"{dataloader.dataset.name}" / f"{sample_mode}"
            samples_folder.mkdir(parents=True, exist_ok=True)

            for batch_i, batch in enumerate(dataloader):
                for repetition_id in range(self.cfg.sample.repetitions):
                    output, retrieved_output = self.sample_step(batch)

                    sbj_meshes, second_sbj_meshes = self.mesh_model.get_meshes(
                        output, batch.scale, batch.sbj_gender, batch.second_sbj_gender
                    )

                    raw_sbj_meshes = raw_second_sbj_meshes = None
                    if retrieved_output is not None:
                        raw_sbj_meshes, raw_second_sbj_meshes = self.mesh_model.get_meshes(
                            retrieved_output, batch.scale, batch.sbj_gender, batch.second_sbj_gender
                        )

                    sample_mode = self.cfg.sample.mode
                    is_baseline = getattr(self.cfg.run, "job", None) == "baseline"
                    for sample_idx in range(len(sbj_meshes)):
                        sbj = batch.sbj[sample_idx]
                        t_stamp = batch.t_stamp[sample_idx]
                        target_folder = samples_folder / sbj
                        target_folder.mkdir(parents=True, exist_ok=True)

                        prefix = f"{t_stamp:04d}_{repetition_id:02d}"
                        if is_baseline and raw_sbj_meshes is not None:
                            raw_sbj_meshes[sample_idx].export(target_folder / f"{prefix}_{sample_mode}_retrieved_raw_subject.ply")
                            raw_second_sbj_meshes[sample_idx].export(target_folder / f"{prefix}_{sample_mode}_retrieved_raw_second_subject.ply")

                        if sample_mode[0] == "1":
                            target_sbj = f"{t_stamp:04d}_{repetition_id:02d}_subject_sample.ply"
                            save_sbj = True
                        else:
                            target_sbj = f"{t_stamp:04d}_subject_GT.ply"
                            save_sbj = repetition_id == 0
                        if sample_mode[1] == "1":
                            target_second_sbj = f"{t_stamp:04d}_{repetition_id:02d}_second_subject_sample.ply"
                            save_second_sbj = True
                        else:
                            target_second_sbj = f"{t_stamp:04d}_second_subject_GT.ply"
                            save_second_sbj = repetition_id == 0
                        if save_sbj:
                            sbj_meshes[sample_idx].export(target_folder / target_sbj)
                        if save_second_sbj:
                            second_sbj_meshes[sample_idx].export(target_folder / target_second_sbj)

    @torch.no_grad()
    def sample_to_hdf5(self, target_name="samples.hdf5"):
        # Log general info
        logger.info(
            f'***** Starting sampling *****\n'
            f'    Model: {self.cfg.model_denoising.name}\n'
            f'    Checkpoint step number: {self.cfg.resume.step}\n'
            f'    Number of repetitions: {self.cfg.sample.repetitions}\n'
        )

        for dataloader in self.dataloaders:
            # Log info
            logger.info(
                f'    Sampling mode {self.cfg.sample.mode} for: {dataloader.dataset.name}\n'
                f'    Number of samples: {len(dataloader.dataset)}\n'
            )
            # create folder for the samples
            logger.info(
                f'    Sampling mode {self.cfg.sample.mode} for: {dataloader.dataset.name}\n'
                f'    Number of samples: {len(dataloader.dataset)}\n'
            )
            # create folder for the samples
            sample_mode = self.sample_mode_to_str(self.cfg.sample.mode, self.cfg.sample.contacts_mode)
            samples_folder = self.base_samples_folder / f"{dataloader.dataset.name}" / f"{sample_mode}"
            samples_folder.mkdir(parents=True, exist_ok=True)

            # Collect all unique subjects and their max timestamps
            sbj2max_t = {}
            for sample in dataloader.dataset.data:
                sbj = getattr(sample, "subject", None)
                if sbj is None:
                    sbj = getattr(sample, "sequence", None)  
                if sbj is None:
                    sbj = getattr(sample, "name", None)      
                t_stamp = sample.t_stamp

                if sbj not in sbj2max_t:
                    sbj2max_t[sbj] = t_stamp
                else:
                    sbj2max_t[sbj] = max(sbj2max_t[sbj], t_stamp)


            base_name = Path(target_name).stem
            S = self.cfg.sample.repetitions
            h5py_files = [h5py.File(str(samples_folder / f"{base_name}_rep_{s:02d}.hdf5"), "w") for s in range(S)]

            # Get faces for both subjects
            # Get faces
            sbj_f = self.mesh_model.get_faces_np().astype(np.int32)

            # ---- probe:  V and J
            probe_batch = next(iter(dataloader))
            if isinstance(probe_batch, dict):
                probe_batch = HHBatchData(**probe_batch)

            gt_v1, gt_j1, gt_v2, gt_j2 = self.mesh_model.get_smpl_th(probe_batch)  # GT meshes/joints
            V = int(gt_v1.shape[1])
            J = int(gt_j1.shape[1])

            
            V_from_faces = int(sbj_f.max()) + 1
            if V_from_faces != V:
                logger.warning(f"[HDF5] V mismatch: faces give {V_from_faces}, SMPL gives {V}. Using V={V}.")


            # Initialize hdf5 structure
            for h5py_file in h5py_files:
                for sbj, max_t in sbj2max_t.items():
                    sbj_group = h5py_file.create_group(sbj)
                    T = max_t + 1  # t_stamp is 0-indexed
                    
                    # First subject datasets
                    sbj_group.create_dataset("sbj_v", shape=(T, V, 3), dtype="f4")
                    sbj_group.create_dataset("sbj_f", data=sbj_f, dtype="i4")
                    sbj_group.create_dataset("sbj_smpl_global", shape=(T, 1, 9), dtype="f4")
                    sbj_group.create_dataset("sbj_smpl_body", shape=(T, 21, 9), dtype="f4")
                    sbj_group.create_dataset("sbj_smpl_lh", shape=(T, 15, 9), dtype="f4")
                    sbj_group.create_dataset("sbj_smpl_rh", shape=(T, 15, 9), dtype="f4")
                    sbj_group.create_dataset("sbj_smpl_transl", shape=(T, 3), dtype="f4")
                    sbj_group.create_dataset("sbj_smpl_betas", shape=(T, 10), dtype="f4")
                    sbj_group.create_dataset("sbj_j", shape=(T, J, 3), dtype="f4")
                    
                    # Second subject datasets
                    sbj_group.create_dataset("second_sbj_v", shape=(T, V, 3), dtype="f4")
                    sbj_group.create_dataset("second_sbj_f", data=sbj_f, dtype="i4")
                    sbj_group.create_dataset("second_sbj_smpl_global", shape=(T, 1, 9), dtype="f4")
                    sbj_group.create_dataset("second_sbj_smpl_body", shape=(T, 21, 9), dtype="f4")
                    sbj_group.create_dataset("second_sbj_smpl_lh", shape=(T, 15, 9), dtype="f4")
                    sbj_group.create_dataset("second_sbj_smpl_rh", shape=(T, 15, 9), dtype="f4")
                    sbj_group.create_dataset("second_sbj_smpl_transl", shape=(T, 3), dtype="f4")
                    sbj_group.create_dataset("second_sbj_smpl_betas", shape=(T, 10), dtype="f4")
                    sbj_group.create_dataset("second_sbj_j", shape=(T, J, 3), dtype="f4")

                    
                    # Attributes
                    sbj_group.attrs['T'] = T


            for batch_idx, batch in enumerate(dataloader):
                for repetition_id in range(self.cfg.sample.repetitions):
                    # Get outputs
                    output, _ = self.sample_step(batch)

                    # Convert output to meshes
                    sbj_meshes, second_sbj_meshes = self.mesh_model.get_meshes(
                        output, batch.scale, batch.sbj_gender, batch.second_sbj_gender
                    )

                    # Get joints
                    _, sbj_joints, _, second_sbj_joints = self.mesh_model.get_meshes_wkpts_th(
                        output, scale=batch.scale, sbj_gender=batch.sbj_gender, second_sbj_gender=batch.second_sbj_gender, return_joints=True
                    )
                    sbj_joints = sbj_joints.cpu().numpy()
                    second_sbj_joints = second_sbj_joints.cpu().numpy()

                    # Convert rotation from 6d to matrix for first subject
                    B = len(output.sbj_pose)
                    sbj_global = output.sbj_global.reshape(B, 1, 6)
                    sbj_global = rotation_6d_to_matrix(sbj_global).reshape(B, 1, 9).cpu().numpy()
                    
                    # sbj_pose contains all 51 joints (21 body + 15 left hand + 15 right hand)
                    sbj_pose = output.sbj_pose.reshape(B, -1, 6)  # (B, 51, 6)
                    sbj_pose = rotation_6d_to_matrix(sbj_pose).reshape(B, -1, 9)  # (B, 51, 9)
                    sbj_body = sbj_pose[:, :21].cpu().numpy()  # (B, 21, 9)
                    sbj_lh = sbj_pose[:, 21:21+15].cpu().numpy()  # (B, 15, 9)
                    sbj_rh = sbj_pose[:, 21+15:].cpu().numpy()  # (B, 15, 9)

                    # Convert rotation from 6d to matrix for second subject
                    second_sbj_global = output.second_sbj_global.reshape(B, 1, 6)
                    second_sbj_global = rotation_6d_to_matrix(second_sbj_global).reshape(B, 1, 9).cpu().numpy()
                    
                    # second_sbj_pose contains all 51 joints (21 body + 15 left hand + 15 right hand)
                    second_sbj_pose = output.second_sbj_pose.reshape(B, -1, 6)  # (B, 51, 6)
                    second_sbj_pose = rotation_6d_to_matrix(second_sbj_pose).reshape(B, -1, 9)  # (B, 51, 9)
                    second_sbj_body = second_sbj_pose[:, :21].cpu().numpy()  # (B, 21, 9)
                    second_sbj_lh = second_sbj_pose[:, 21:21+15].cpu().numpy()  # (B, 15, 9)
                    second_sbj_rh = second_sbj_pose[:, 21+15:].cpu().numpy()  # (B, 15, 9)

                    # Save to hdf5
                    for sample_idx in range(len(sbj_meshes)):
                        sbj = batch.sbj[sample_idx]
                        t_stamp = batch.t_stamp[sample_idx].item()

                        sbj_group = h5py_files[repetition_id][sbj]

                        # First subject
                        sbj_group['sbj_v'][t_stamp] = sbj_meshes[sample_idx].vertices.astype(np.float32)
                        sbj_group['sbj_smpl_global'][t_stamp] = sbj_global[sample_idx]
                        sbj_group['sbj_smpl_body'][t_stamp] = sbj_body[sample_idx]
                        sbj_group['sbj_smpl_lh'][t_stamp] = sbj_lh[sample_idx]
                        sbj_group['sbj_smpl_rh'][t_stamp] = sbj_rh[sample_idx]
                        sbj_group['sbj_smpl_transl'][t_stamp] = output.sbj_c[sample_idx].cpu().numpy()
                        sbj_group['sbj_smpl_betas'][t_stamp] = output.sbj_shape[sample_idx].cpu().numpy()
                        sbj_group['sbj_j'][t_stamp] = sbj_joints[sample_idx].astype(np.float32)

                        # Second subject
                        sbj_group['second_sbj_v'][t_stamp] = second_sbj_meshes[sample_idx].vertices.astype(np.float32)
                        sbj_group['second_sbj_smpl_global'][t_stamp] = second_sbj_global[sample_idx]
                        sbj_group['second_sbj_smpl_body'][t_stamp] = second_sbj_body[sample_idx]
                        sbj_group['second_sbj_smpl_lh'][t_stamp] = second_sbj_lh[sample_idx]
                        sbj_group['second_sbj_smpl_rh'][t_stamp] = second_sbj_rh[sample_idx]
                        sbj_group['second_sbj_smpl_transl'][t_stamp] = output.second_sbj_c[sample_idx].cpu().numpy()
                        sbj_group['second_sbj_smpl_betas'][t_stamp] = output.second_sbj_shape[sample_idx].cpu().numpy()
                        sbj_group['second_sbj_j'][t_stamp] = second_sbj_joints[sample_idx].astype(np.float32)


            # Close hdf5 files
            for h5py_file in h5py_files:
                h5py_file.close()