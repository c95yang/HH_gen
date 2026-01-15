import json
import pickle as pkl
from collections import defaultdict
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from typing import List, Optional, NamedTuple, Dict

import h5py
import numpy as np
import torch

from tridi.preprocessing.common import DatasetSample

from .hh_batch_data import HHBatchData
from ..utils.geometry import matrix_to_rotation_6d

logger = getLogger(__name__)


class H5DataSample(NamedTuple):
    sequence: str
    name: str
    t_stamp: int


@dataclass
class HHDataset:
    name: str # 'behave', 'embody3d', 'interhuman', 'chi3d'
    root: Path
    split: str # 'train', 'test'
    augment_symmetry: bool
    augment_rotation: bool
    downsample_factor: int = 1
    h5dataset_path: Path = None
    preload_data: bool = True
    subjects: Optional[List[str]] = field(default_factory=list)
    split_file: Optional[str] = None
    behave_repeat_fix: bool = False  # repeating the data for classes with only 1 fps annotations"
    include_pointnext: bool = False
    assets_folder: Optional[Path] = None
    fps: Optional[int] = 30
    max_timestamps: Optional[int] = None  # 限制每个序列的最大timestamp数量
    filter_subjects: Optional[List[str]] = None  # only load the specified subjects


    def __post_init__(self) -> None:
        # Open h5 dataset
        self.h5dataset_path = self.root / f"dataset_{self.split}_{self.fps}fps.hdf5"
        # if self.preload_data:
        #     self.h5dataset = self._preload_h5_dataset(self.h5dataset_path)
        #     logger.info("Preloaded H5 dataset into memory.")
        # else:
        self.h5dataset = h5py.File(self.h5dataset_path, "r")

        if self.split_file is not None:
            self.sequences = self._get_sequences_from_split()  
        else:
            raise ValueError("Must provide split file for HH dataset.")

        self.data = self._load_data()
        self._sort_data()
        logger.info(self.__str__())
    
    def __str__(self) -> str:
        return f"HHDataset {self.name}: split={self.split} #frames={len(self.data)}"

    def __len__(self) -> int:
        return len(self.data)
    
    def _get_sequences_from_split(self) -> List[str]:
        with open(self.split_file, "r") as fp:
            seqs = json.load(fp)

        return seqs

    @staticmethod
    def _preload_h5_dataset(h5dataset_path: Path):
        data_dict = dict()
        with h5py.File(h5dataset_path, "r") as h5_dataset:
            for seq in h5_dataset.keys():
                data_dict[seq] = dict()
                for key in h5_dataset[seq].keys():
                    data_dict[seq][key] = h5_dataset[seq][key][:]
                    # copy attributes
                data_dict[seq]["_attrs"] = dict(h5_dataset[seq].attrs)
        return data_dict

    @staticmethod
    def _apply_z_rotation_augmentation(sbj_transl, second_sbj_transl, sbj_global, second_sbj_global):
        angle = np.random.choice(np.arange(-np.pi, np.pi, np.pi / 36))

        cos, sin = np.cos(angle), np.sin(angle)
        R = np.array([
            [cos, -sin, 0.0],
            [sin,  cos, 0.0],
            [0.0,  0.0, 1.0]
        ], dtype=np.float32)

        def rot(x):
            return x @ R.T
        
        def rotate_axis_angle(axis_angle, Rz):
            R_new = Rz @ axis_angle.reshape(3, 3)
            R_new = R_new.reshape(9)
            return R_new

        # --- first subject ---
        sbj_transl = rot(sbj_transl)
        sbj_global = rotate_axis_angle(
            sbj_global, R
        )

        # --- second subject ---
        second_sbj_transl = rot(second_sbj_transl)
        second_sbj_global = rotate_axis_angle(
            second_sbj_global, R
        )

        return sbj_transl, second_sbj_transl, sbj_global, second_sbj_global
    


    @staticmethod
    def _apply_symmetry_augmentation(sbj_smpl, second_sbj_smpl):
        body_sym_map = np.array(
            [1, 0, 2, 4, 3, 5, 7, 6, 8, 10, 9, 11, 13, 12, 14, 16, 15, 18, 17, 20, 19],
            dtype=np.int64
        )

        # (x,y,z)->(-x,y,z)
        S = np.diag([-1.0, 1.0, 1.0]).astype(np.float32)
        s_vec = np.array([-1.0, 1.0, 1.0], dtype=np.float32) # for axis-angle flipping

        def flip_points(x):
            x = np.asarray(x)
            return x @ S.T

        def flip_rot(x):
            x = np.asarray(x, dtype=np.float32)

            # axis-angle: (...,3)
            if x.ndim >= 1 and x.shape[-1] == 3:
                return x * s_vec

            # rotmat: (...,3,3)
            if x.shape[-2:] == (3, 3):
                return S @ x @ S

            # flatten rotmat: (...,9)
            if x.shape[-1] == 9:
                R = x.reshape(*x.shape[:-1], 3, 3)
                Rf = S @ R @ S
                return Rf.reshape(*x.shape[:-1], 9)

            raise ValueError(f"Unsupported rotation shape: {x.shape}")

        def flip_global_orient_1x9(go):
            go = np.asarray(go, dtype=np.float32)
            assert go.shape == (1, 9)
            R = go.reshape(1, 3, 3)
            Rf = S[None, :, :] @ R @ S[None, :, :]
            return Rf.reshape(1, 9)

        def flip_params(params):
            # transl / geometry space
            if "transl" in params:
                params["transl"] = flip_points(params["transl"])

            # global orient
            if "global_orient" in params:
                go = params["global_orient"]
                params["global_orient"] = flip_global_orient_1x9(go)

            # body pose (swap L/R joints then flip)
            if "body_pose" in params:
                bp = np.asarray(params["body_pose"])
                if bp.shape[0] == body_sym_map.shape[0]:
                    bp = bp[body_sym_map]
                params["body_pose"] = flip_rot(bp)

            # hands (swap + flip)
            if "left_hand_pose" in params and "right_hand_pose" in params:
                lh = flip_rot(params["left_hand_pose"])
                rh = flip_rot(params["right_hand_pose"])
                params["left_hand_pose"], params["right_hand_pose"] = rh, lh

            # betas not changed
            return params

        # --- flip SMPL params ---
        sbj_smpl = flip_params(sbj_smpl)
        second_sbj_smpl = flip_params(second_sbj_smpl)

        return sbj_smpl, second_sbj_smpl

    def __getitem__(self, idx: int) -> HHBatchData:
        sample = self.data[idx] #H5DataSample(sequence='s02_Push_1', name='s02_Push_1', t_stamp=0)
        sequence = self.h5dataset[sample.sequence]

        sbj_gender = sequence.attrs['gender']
        second_sbj_gender = sequence.attrs['gender']
        
        # ==> augmentations
        sbj_smpl = {
            "betas": sequence['sbj_smpl_betas'][sample.t_stamp],
            "transl": sequence['sbj_smpl_transl'][sample.t_stamp],
            "global_orient": sequence['sbj_smpl_global'][sample.t_stamp],
            "body_pose": sequence['sbj_smpl_body'][sample.t_stamp],
            "left_hand_pose": sequence['sbj_smpl_lh'][sample.t_stamp],
            "right_hand_pose": sequence['sbj_smpl_rh'][sample.t_stamp],
        }

        second_sbj_smpl = {
            "betas": sequence['second_sbj_smpl_betas'][sample.t_stamp],
            "transl": sequence['second_sbj_smpl_transl'][sample.t_stamp],
            "global_orient": sequence['second_sbj_smpl_global'][sample.t_stamp],
            "body_pose": sequence['second_sbj_smpl_body'][sample.t_stamp],
            "left_hand_pose": sequence['second_sbj_smpl_lh'][sample.t_stamp],
            "right_hand_pose": sequence['second_sbj_smpl_rh'][sample.t_stamp],
        }



        if self.augment_symmetry and np.random.rand() > 0.5:
            sbj_smpl, second_sbj_smpl = self._apply_symmetry_augmentation(sbj_smpl, second_sbj_smpl)
        if self.augment_rotation and np.random.rand() > 0.25:
            sbj_smpl['transl'], second_sbj_smpl['transl'], sbj_smpl['global_orient'], second_sbj_smpl['global_orient'] = self._apply_z_rotation_augmentation(
                                                                            sbj_transl=sbj_smpl['transl'],
                                                                            second_sbj_transl=second_sbj_smpl['transl'],
                                                                            sbj_global=sbj_smpl['global_orient'],
                                                                            second_sbj_global=second_sbj_smpl['global_orient'])
                


        # <== save parameters
        sbj_pose = np.concatenate([
            sequence['sbj_smpl_body'][sample.t_stamp],
            sequence['sbj_smpl_lh'][sample.t_stamp],
            sequence['sbj_smpl_rh'][sample.t_stamp],
        ], axis=0).reshape((51, 3, 3))

        second_sbj_pose = np.concatenate([
            sequence['second_sbj_smpl_body'][sample.t_stamp],
            sequence['second_sbj_smpl_lh'][sample.t_stamp],
            sequence['second_sbj_smpl_rh'][sample.t_stamp],
        ], axis=0).reshape((51, 3, 3))

        # convert to 6d representation
        sbj_global = matrix_to_rotation_6d(sbj_smpl['global_orient'].reshape(3, 3)).reshape(-1)
        sbj_pose = matrix_to_rotation_6d(sbj_pose).reshape(-1)
        # convert to 6d representation
        second_sbj_global = matrix_to_rotation_6d(second_sbj_smpl['global_orient'].reshape(3, 3)).reshape(-1)
        second_sbj_pose = matrix_to_rotation_6d(second_sbj_pose).reshape(-1)    


        # Fill BatchData isntance
        batch_data = HHBatchData(
            # metadata
            meta={
                "name": sample.name,
                "t_stamp": sample.t_stamp,
            },
            sbj=sample.sequence,
            second_sbj=sample.sequence,
            t_stamp=sample.t_stamp,
            # subject
            sbj_shape=torch.tensor(sbj_smpl['betas'], dtype=torch.float),
            sbj_global=sbj_global,
            sbj_pose=sbj_pose,
            sbj_c=torch.tensor(sbj_smpl['transl'], dtype=torch.float),
            sbj_gender=torch.tensor(sbj_gender == 'male', dtype=torch.bool),
            # second subject
            second_sbj_shape=torch.tensor(second_sbj_smpl['betas'], dtype=torch.float),
            second_sbj_global=second_sbj_global,
            second_sbj_pose=second_sbj_pose,
            second_sbj_c=torch.tensor(second_sbj_smpl['transl'], dtype=torch.float),
            second_sbj_gender=torch.tensor(second_sbj_gender == 'male', dtype=torch.bool),

            scale=torch.tensor(sequence['prep_s'][sample.t_stamp], dtype=torch.float)
        )

        # print("batch_data: ", batch_data.to_string())

        return batch_data
    
    def _load_data(self) -> List[H5DataSample]:
        logger.info(f"HHDataset {self.name}: loading from {self.h5dataset_path}.")

        data: List[H5DataSample] = []
        total_frames = 0
        skipped_missing = 0

        # choose which sequences to load
        sequences_to_load = list(self.sequences)

       
        if self.filter_subjects is not None and len(self.filter_subjects) > 0:
            before = len(sequences_to_load)
            sequences_to_load = [s for s in sequences_to_load if s in set(self.filter_subjects)]
            logger.info(f"Filtering subjects: {before} -> {len(sequences_to_load)}")

        # 
        available = set(self.h5dataset.keys())
        missing = [s for s in sequences_to_load if s not in available]
        if len(missing) > 0:
            logger.warning(
                f"[HHDataset {self.name}] {len(missing)} sequences in split but NOT in hdf5. "
                f"Examples: {missing[:10]}"
            )
        sequences_to_load = [s for s in sequences_to_load if s in available]
        skipped_missing = len(missing)

        if len(sequences_to_load) == 0:
            raise RuntimeError(
                f"[HHDataset {self.name}] After filtering, no valid sequences left. "
                f"hdf5 keys={len(available)}, split size={len(self.sequences)}"
            )

        
        for seq_name in sequences_to_load:
            seq = self.h5dataset[seq_name]   

            # timestamps
            T_seq = int(seq.attrs.get("T", 0))
            if T_seq <= 0:
                logger.warning(f"[HHDataset {self.name}] Sequence {seq_name} has T={T_seq}, skip.")
                continue

            t_stamps = list(range(T_seq))

            # Limit the number of timestamps per sequence
            if self.max_timestamps is not None:
                t_stamps = t_stamps[: self.max_timestamps]

            # Downsample
            if self.downsample_factor > 1:
                t_stamps = t_stamps[:: self.downsample_factor]

            # build samples
            seq_data = [
                H5DataSample(sequence=seq_name, name=f"{seq_name}", t_stamp=int(t))
                for t in t_stamps
            ]
            data.extend(seq_data)
            total_frames += len(seq_data)

        logger.info(
            f"HH dataset {self.name} {self.split} has {total_frames} frames "
            f"(skipped_missing_seq={skipped_missing})."
        )
        return data


    def _sort_data(self) -> None:
        self.data = sorted(
            self.data,
            key=lambda f: (
                f.name,
                f.t_stamp or 0,
            ),
        )
