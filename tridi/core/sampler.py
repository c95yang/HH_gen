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

from config.config import ProjectConfig
from tridi.data import get_eval_dataloader
from tridi.model.base import TriDiModelOutput
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
    def sample_step(self, batch) -> Tuple[TriDiModelOutput, List[str]]:
        # self.model.eval()
        #print batch info
        # print("Sampling batch:")
        # print(f"  batch.sbj_contact_indexes: {batch.sbj_contact_indexes.shape}")

        output = self.model(batch, "sample", sample_type=self.cfg.sample.mode)
        # print("Output obtained from model")
        # print("Output shape:", output.shape if hasattr(output, 'shape') else "N/A")

        if isinstance(output, tuple):
            output, intermediate_outputs = output

        return self.model.split_output(output)

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
            print(f"Created samples folder: {samples_folder}")

            for batch_i, batch in enumerate(dataloader):
                for repetition_id in range(self.cfg.sample.repetitions):
                    # Get outputs
                    output = self.sample_step(batch)

                    # Convert output to meshes
                    sbj_meshes, second_sbj_meshes = self.mesh_model.get_meshes(
                        output, batch.scale, batch.sbj_gender
                    )

                    # Export meshes
                    # For conditional sampling add GT to export
                    for sample_idx in range(len(sbj_meshes)):
                        # save meshes
                        sbj = batch.sbj[sample_idx]
                        t_stamp = batch.t_stamp[sample_idx]
                        target_folder = samples_folder / sbj  
                        target_folder.mkdir(parents=True, exist_ok=True)

                        if self.cfg.sample.mode[0] == "1":
                            target_sbj = f"{t_stamp:04d}_{repetition_id:02d}_subject_sample.ply"
                            save_sbj = True
                        else:
                            target_sbj = f"{t_stamp:04d}_subject_GT.ply"
                            save_sbj = repetition_id == 0

                        if self.cfg.sample.mode[1] == "1":
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
            sample_mode = self.sample_mode_to_str(self.cfg.sample.mode, self.cfg.sample.contacts_mode)
            samples_folder = self.base_samples_folder / f"{dataloader.dataset.name}" / f"{sample_mode}"
            samples_folder.mkdir(parents=True, exist_ok=True)

            # Collect all unique subjects and their max timestamps
            sbj2max_t = {}
            for sample in dataloader.dataset.data:
                sbj = sample.subject
                t_stamp = sample.t_stamp
                if sbj not in sbj2max_t:
                    sbj2max_t[sbj] = t_stamp
                else:
                    sbj2max_t[sbj] = max(sbj2max_t[sbj], t_stamp)

            base_name = Path(target_name).stem
            S = self.cfg.sample.repetitions
            h5py_files = [h5py.File(str(samples_folder / f"{base_name}_rep_{s:02d}.hdf5"), "w") for s in range(S)]

            # Get faces for both subjects
            sbj_f = self.mesh_model.get_faces_np()
            
            # Initialize hdf5 structure
            for h5py_file in h5py_files:
                for sbj, max_t in sbj2max_t.items():
                    sbj_group = h5py_file.create_group(sbj)
                    T = max_t + 1  # t_stamp is 0-indexed
                    
                    # First subject datasets
                    sbj_group.create_dataset("sbj_v", shape=(T, 6890, 3))
                    sbj_group.create_dataset("sbj_f", shape=(sbj_f.shape[0], 3), data=sbj_f)
                    sbj_group.create_dataset("sbj_smpl_global", shape=(T, 1, 9))
                    sbj_group.create_dataset("sbj_smpl_body", shape=(T, 21, 9))
                    sbj_group.create_dataset("sbj_smpl_lh", shape=(T, 15, 9))
                    sbj_group.create_dataset("sbj_smpl_rh", shape=(T, 15, 9))
                    sbj_group.create_dataset("sbj_smpl_transl", shape=(T, 3))
                    sbj_group.create_dataset("sbj_smpl_betas", shape=(T, 10))
                    sbj_group.create_dataset("sbj_j", shape=(T, 73, 3))
                    
                    # Second subject datasets
                    sbj_group.create_dataset("second_sbj_v", shape=(T, 6890, 3))
                    sbj_group.create_dataset("second_sbj_f", shape=(sbj_f.shape[0], 3), data=sbj_f)
                    sbj_group.create_dataset("second_sbj_smpl_global", shape=(T, 1, 9))
                    sbj_group.create_dataset("second_sbj_smpl_body", shape=(T, 21, 9))
                    sbj_group.create_dataset("second_sbj_smpl_lh", shape=(T, 15, 9))
                    sbj_group.create_dataset("second_sbj_smpl_rh", shape=(T, 15, 9))
                    sbj_group.create_dataset("second_sbj_smpl_transl", shape=(T, 3))
                    sbj_group.create_dataset("second_sbj_smpl_betas", shape=(T, 10))
                    sbj_group.create_dataset("second_sbj_j", shape=(T, 73, 3))
                    
                    # Attributes
                    sbj_group.attrs['T'] = T

            # prediction loop
            for batch_idx, batch in enumerate(dataloader):
                for repetition_id in range(self.cfg.sample.repetitions):
                    # Get outputs
                    output = self.sample_step(batch)

                    # Convert output to meshes
                    sbj_meshes, second_sbj_meshes = self.mesh_model.get_meshes(
                        output, batch.scale, batch.sbj_gender
                    )

                    # Get joints
                    _, sbj_joints, _, second_sbj_joints = self.mesh_model.get_meshes_wkpts_th(
                        output, scale=batch.scale, sbj_gender=batch.sbj_gender, return_joints=True
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
                        sbj_group['sbj_v'][t_stamp] = sbj_meshes[sample_idx].vertices
                        sbj_group['sbj_smpl_global'][t_stamp] = sbj_global[sample_idx]
                        sbj_group['sbj_smpl_body'][t_stamp] = sbj_body[sample_idx]
                        sbj_group['sbj_smpl_lh'][t_stamp] = sbj_lh[sample_idx]
                        sbj_group['sbj_smpl_rh'][t_stamp] = sbj_rh[sample_idx]
                        sbj_group['sbj_smpl_transl'][t_stamp] = output.sbj_c[sample_idx].cpu().numpy()
                        sbj_group['sbj_smpl_betas'][t_stamp] = output.sbj_shape[sample_idx].cpu().numpy()
                        sbj_group['sbj_j'][t_stamp] = sbj_joints[sample_idx]

                        # Second subject
                        sbj_group['second_sbj_v'][t_stamp] = second_sbj_meshes[sample_idx].vertices
                        sbj_group['second_sbj_smpl_global'][t_stamp] = second_sbj_global[sample_idx]
                        sbj_group['second_sbj_smpl_body'][t_stamp] = second_sbj_body[sample_idx]
                        sbj_group['second_sbj_smpl_lh'][t_stamp] = second_sbj_lh[sample_idx]
                        sbj_group['second_sbj_smpl_rh'][t_stamp] = second_sbj_rh[sample_idx]
                        sbj_group['second_sbj_smpl_transl'][t_stamp] = output.second_sbj_c[sample_idx].cpu().numpy()
                        sbj_group['second_sbj_smpl_betas'][t_stamp] = output.second_sbj_shape[sample_idx].cpu().numpy()
                        sbj_group['second_sbj_j'][t_stamp] = second_sbj_joints[sample_idx]

            # Close hdf5 files
            for h5py_file in h5py_files:
                h5py_file.close()
