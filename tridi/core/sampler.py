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

            # get sequences with lengths from the dataset
            sbj2sct = dataloader.dataset.get_sbj2sct()  # sct = sequence, class_id, T

            base_name = Path(target_name).stem
            S = self.cfg.sample.repetitions
            h5py_files = [h5py.File(str(samples_folder / f"{base_name}_rep_{s:02d}.hdf5"), "w") for s in range(S)]

            for h5py_file in h5py_files:
                for sbj, sequences in sbj2sct.items():
                    sbj_group = h5py_file.create_group(sbj)
                    for seq, class_id, T in sequences:
                        seq_group = sbj_group.create_group(f"{seq}")
                        n_obj_v = self.canonical_obj_meshes[class_id].vertices.shape[0]
                        obj_f = self.canonical_obj_meshes[class_id].faces
                        sbj_f = self.mesh_model.get_faces_np()
                        # old: sbj_vertices, sbj_faces
                        seq_group.create_dataset("sbj_v", shape=(T, 6890, 3))
                        seq_group.create_dataset("sbj_f", shape=(sbj_f.shape[0], 3), data=sbj_f)
                        if self.cfg.model_conditioning.use_contacts != "NONE":
                            seq_group.create_dataset("sbj_contact_z", shape=(T, self.cfg.model.data_contact_channels))
                            seq_group.create_dataset("sbj_contact", shape=(T, 6890))
                        # old: obj_vertices, obj_faces
                        seq_group.create_dataset("obj_v", shape=(T, n_obj_v, 3))
                        seq_group.create_dataset("obj_f", shape=(obj_f.shape[0], 3), data=obj_f)
                        # sbj params, old: sbj_pose, sbj_c, sbj_shape
                        seq_group.create_dataset("sbj_smpl_pose", shape=(T, 1+21+15+15, 9))
                        seq_group.create_dataset("sbj_smpl_transl", shape=(T, 3))
                        seq_group.create_dataset("sbj_smpl_betas", shape=(T, 10))
                        seq_group.create_dataset("sbj_j", shape=(T, 73, 3))
                        # obj params
                        seq_group.create_dataset("obj_c", shape=(T, 3))
                        seq_group.create_dataset("obj_R", shape=(T, 9))
                        # attributes
                        seq_group.attrs['T'] = T


            # prediction loop
            for batch_idx, batch in enumerate(dataloader):
                for repetition_id in range(self.cfg.sample.repetitions):
                    # Get outputs
                    output, captions = self.sample_step(batch)
                    sbj_vertices, obj_vertices, sbj_joints = self.mesh_model.get_meshes_th(
                        output, batch.obj_class, batch.scale, sbj_gender=batch.sbj_gender, return_joints=True
                    )

                    # convert rotation from 6d to matrix
                    # output = output
                    sbj_pose = torch.cat([
                        output.sbj_global,
                        output.sbj_pose,
                    ], dim=1).reshape(-1, 52, 6)
                    sbj_pose = rotation_6d_to_matrix(sbj_pose).reshape(-1, 52, 9).cpu().numpy()
                    obj_R = output.obj_R.reshape(-1, 1, 6)
                    obj_R = rotation_6d_to_matrix(obj_R).reshape(-1, 9).cpu().numpy()

                    # Decode contacts
                    if self.cfg.model_conditioning.use_contacts != "NONE":
                        is_sampling_contacts = self.cfg.sample.mode[2] == "1"
                        contacts_mask = self.contact_model.decode_contacts_np(
                            batch.sbj_contacts_full, output.contacts,
                            batch.sbj_contact_indexes, is_sampling_contacts
                        )
                        if is_sampling_contacts:
                            contacts_z = output.contacts.cpu().numpy()
                        else:
                            contacts_z = batch.sbj_contacts.cpu().numpy()

                    # save to hdf5
                    sbj_vertices, obj_vertices = sbj_vertices.cpu().numpy(), obj_vertices.cpu().numpy()
                    for sample_idx, class_id in enumerate(batch.obj_class.cpu().numpy()):
                        sbj = batch.sbj[sample_idx]
                        obj = batch.obj[sample_idx]
                        act = batch.act[sample_idx]
                        t_stamp = batch.t_stamp[sample_idx].item()

                        sbj_group = h5py_files[repetition_id][sbj]
                        seq_group = sbj_group[f"{obj}_{act}"]
                        n_obj_v = self.canonical_obj_meshes[class_id].vertices.shape[0]

                        seq_group['sbj_v'][t_stamp] = sbj_vertices[sample_idx]
                        seq_group['obj_v'][t_stamp] = obj_vertices[sample_idx][:n_obj_v]
                        if self.cfg.model_conditioning.use_contacts != "NONE":
                            seq_group['sbj_contact_z'][t_stamp] = contacts_z[sample_idx]
                            seq_group['sbj_contact'][t_stamp] = contacts_mask[sample_idx]
                        seq_group['sbj_smpl_pose'][t_stamp] = sbj_pose[sample_idx]
                        seq_group['sbj_smpl_transl'][t_stamp] = output.sbj_c[sample_idx].cpu().numpy()
                        seq_group['sbj_smpl_betas'][t_stamp] = output.sbj_shape[sample_idx].cpu().numpy()
                        seq_group['sbj_j'][t_stamp] = sbj_joints[sample_idx].cpu().numpy()
                        seq_group['obj_c'][t_stamp] = output.obj_c[sample_idx].cpu().numpy()
                        seq_group['obj_R'][t_stamp] = obj_R[sample_idx]
