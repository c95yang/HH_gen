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

    def __getitem__(self, idx: int) -> HHBatchData:
        sample = self.data[idx]
        sequence = self.h5dataset[sample.sequence]

        sbj_gender = sequence.attrs['gender']
        sbj_pose = np.concatenate([
            sequence['sbj_smpl_body'][sample.t_stamp],
            sequence['sbj_smpl_lh'][sample.t_stamp],
            sequence['sbj_smpl_rh'][sample.t_stamp],
        ], axis=0).reshape((51, 3, 3))
        sbj_global = sequence['sbj_smpl_global'][sample.t_stamp]
        # print("sbj_global: ", sequence['sbj_smpl_global'].shape)
        # convert to 6d representation
        sbj_global = matrix_to_rotation_6d(sbj_global.reshape(3, 3)).reshape(-1)
        sbj_pose = matrix_to_rotation_6d(sbj_pose).reshape(-1)
        
        second_sbj_gender = sbj_gender
        second_sbj_pose = np.concatenate([
            sequence['second_sbj_smpl_body'][sample.t_stamp],
            sequence['second_sbj_smpl_lh'][sample.t_stamp],
            sequence['second_sbj_smpl_rh'][sample.t_stamp],
        ], axis=0).reshape((51, 3, 3))
        second_sbj_global = sequence['second_sbj_smpl_global'][sample.t_stamp]
        # convert to 6d representation
        second_sbj_global = matrix_to_rotation_6d(second_sbj_global.reshape(3, 3)).reshape(-1)
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
            sbj_shape=torch.tensor(sequence['sbj_smpl_betas'][sample.t_stamp], dtype=torch.float),
            sbj_global=sbj_global,
            sbj_pose=sbj_pose,
            sbj_c=torch.tensor(sequence['sbj_smpl_transl'][sample.t_stamp], dtype=torch.float),
            sbj_gender=torch.tensor(sbj_gender == 'male', dtype=torch.bool),
            #second subject
            second_sbj_shape=torch.tensor(sequence['sbj_smpl_betas'][sample.t_stamp], dtype=torch.float),
            second_sbj_global=second_sbj_global,
            second_sbj_pose=second_sbj_pose,
            second_sbj_c=torch.tensor(sequence['second_sbj_smpl_transl'][sample.t_stamp], dtype=torch.float),
            second_sbj_gender=torch.tensor(second_sbj_gender == 'male', dtype=torch.bool),

            scale=torch.tensor(sequence['prep_s'][sample.t_stamp], dtype=torch.float)
        )

        # print("batch_data: ", batch_data.to_string())

        return batch_data
    
    def _load_data(self) -> List[H5DataSample]:
        logger.info(f"HHDataset {self.name}: loading from {self.h5dataset_path}.")

        data = []
        T = 0
        skipped = 0
        
        # choose which sequences to load
        sequences_to_load = self.sequences
        # if self.filter_subjects is not None:
        #     sequences_to_load = [s for s in sequences_to_load if s in self.filter_subjects]
        #     logger.info(f"Filtering subjects to: {sequences_to_load}")
        
        for seq_name in sequences_to_load:
            print("sequence: ", seq_name)
            print("self.h5dataset: ", self.h5dataset.keys())
            seq = self.h5dataset[seq_name]
            # print content of seq
            # for key in seq.keys():
            #     print(f"  {key}: {seq[key].shape}")
# 
            # Try to get sequence, skip if missing
            if seq is None:
                # print(f"Skipping missing sequence: {seq_name}")
                skipped += 1
                continue

            # create timestamp
            t_stamps = list(range(seq.attrs["T"]))
            
            # Limit the number of timestamps per sequence
            if self.max_timestamps is not None:
                t_stamps = t_stamps[:self.max_timestamps]

            T += len(t_stamps)
            seq_data = [
                H5DataSample(
                    sequence=seq_name,
                    name=f"{seq_name}",
                    t_stamp=t_stamp
                ) for t_stamp in t_stamps
            ]

            if self.downsample_factor > 1:
                seq_data = seq_data[::self.downsample_factor]
            data.extend(seq_data)
        logger.info(f"HH dataset {self.name} {self.split} has {T} frames.")
        return data

    def _sort_data(self) -> None:
        self.data = sorted(
            self.data,
            key=lambda f: (
                f.name,
                f.t_stamp or 0,
            ),
        )

#python main.py -c config/env.yaml scenarios/embody3d.yaml -- run.name=embody3d run.job=train