"""
Code to preprocess 30fps annotations for the Embody3D dataset.
"""
import argparse
import json
from copy import deepcopy
from multiprocessing import set_start_method
from pathlib import Path
import os
import re

import h5py
import numpy as np
import smplx
import torch
import tqdm
import trimesh
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from .common import (tensor_to_cpu, DatasetSample, preprocess_worker, \
    get_sequences_list, init_preprocessing, \
    add_sequence_datasets_to_hdf5, add_meatada_to_hdf5)

def axis_angle_to_matrix(aa: np.array) -> np.array:
    """aa: (B, 3) -> R: (B, 3, 3)"""
    if aa.ndim > 2:
        F, N, _ = aa.shape
        aa_flat = aa.reshape(-1, 3)
        rot_mats_flat = Rotation.from_rotvec(aa_flat).as_matrix() 
        rot_mats = rot_mats_flat.reshape(F, N, 3, 3)
    else:
        rot_mats = Rotation.from_rotvec(aa).as_matrix()
    return rot_mats

def split(cfg):

    # convert to Path
    embody3d_path = Path(cfg.embody3d.root)
    assets_path = Path(cfg.embody3d.assets)
    #print(f"split():{embody3d_path}")
    # list dataset sequences
    train = []
    test = []
    sequences = get_sequences_list(
        "embody3d", embody3d_path, subjects=cfg.embody3d.categories
    )
    for sequence in sequences:
        if "Building_a_paper_tower" in sequence.name:
            test.append(sequence.name)
        elif "Going_out" in sequence.name:
            test.append(sequence.name)
        elif "Home_Renovations" in sequence.name:
            test.append(sequence.name)
        elif "Theater_reenactment_competition" in sequence.name:
            test.append(sequence.name)
        else:
            train.append(sequence.name)
                
    split_dict = {
        'train': train,
        'test' : test
    }
    # print(f"train samples length:{len(train)}")
    # print(f"test samples length:{len(test)}")
    with open(embody3d_path/"split.json", "w") as f:
        json.dump(split_dict, f, indent=4)
    with open (assets_path/"embody3d_train.json", "w") as f:
        json.dump(train, f, indent=4)
    with open (assets_path/"embody3d_test.json", "w") as f:
        json.dump(test, f, indent=4)

def preprocess(cfg):
    set_start_method('spawn')
    # convert to Path
    target_folder = Path(cfg.embody3d.target)
    embody3d_path = Path(cfg.embody3d.root)

    # list dataset sequences
    _sequences = get_sequences_list(
        "embody3d", embody3d_path, subjects=cfg.embody3d.categories
    )
    #print(f"preprocess_embody3d.py. Loaded sequences: {_sequences}")

    # filter sequences based on split
    if cfg.embody3d.split in ["train", "test"]:
        with open(cfg.embody3d.split_file, "r") as fp:
            split = json.load(fp)
            # print(f"Loaded split file {split}")
        split_sequences = split[cfg.embody3d.split]
        #print(f"preprocess_embody3d.py Loaded split sequences: {split_sequences}")
        # for seq in _sequences:
        #     print(f"{seq.name}")
        #     if seq.name in split_sequences:
        #         print(f"preprocess_embody3d.py Found matching sequence: {seq.name}")
        sequences = [seq for seq in _sequences if seq.name in split_sequences]

        #print(f"preprocess_embody3d.pyFiltered sequences: {sequences}")
        hdf5_name = f"dataset_{cfg.embody3d.split}"
        #print(f"hdf5name: {hdf5_name}")
    else:
        sequences = _sequences
        hdf5_name = "dataset"
    if cfg.embody3d.downsample == "10fps":
        hdf5_name += "_10fps"
    elif cfg.embody3d.downsample == "1fps":
        hdf5_name += "_1fps"
    else:
        hdf5_name += "_30fps"

    # init hdf5 file
    target_folder.mkdir(exist_ok=True, parents=True)
    if (target_folder / f"{hdf5_name}.hdf5").is_file():
        mode = "a"
    else:
        mode = "w"
    h5py_file = h5py.File(str(target_folder / f"{hdf5_name}.hdf5"), mode)

    # preprocess each sequence
    for sequence in tqdm.tqdm(sequences, total=len(sequences), ncols=80):
        seq_name = sequence.name

        # pattern for subject folders
        pattern = re.compile(r"^[A-Z]{3}\d{3}$")

        # get subject names
        subjects = []
        for subfolder in sorted(os.listdir(sequence)):
            if Path(sequence/subfolder).is_dir() and pattern.match(subfolder):
                subjects.append(subfolder)

        #print(f"subjects:{subjects}")
        # ============ 1 Load subject SMPL param
        # sequences : category / c--<date>--<time>--<subjects>--<rest> / ABC123/ smplx_mesh_* / .npy

        smplx_params1 = dict() 
        smplx_params2 = dict() 
        for param in ["betas", "body_pose", "left_hand_pose" , "right_hand_pose", "global_orient", "transl"]:
            # subject1
            smplx_folder1 = sequence / subjects[0] / f"smplx_mesh_{param}"
            npy_path1 = list(smplx_folder1.glob("*.npy"))[0]
            smplx_params1[param] = np.load(npy_path1)

            # subject2
            smplx_folder2 = sequence / subjects[1] / f"smplx_mesh_{param}"
            npy_path2 = list(smplx_folder2.glob("*.npy"))[0]
            smplx_params2[param] = np.load(npy_path2)

        T_original = smplx_params1["transl"].shape[0]
        #print("load subject smpl param done")
        #print(f"T_original:{T_original}")
        
        # downsample parameters if needed
        downsample_factor = 1
        if cfg.embody3d.downsample != "None":
            if cfg.embody3d.downsample == "10fps":
                downsample_factor = 3
            elif cfg.embody3d.downsample == "1fps":
                downsample_factor = 30
        #print(f"downsample_factor:{downsample_factor}")
        #print(f"T_original:{T_original} vs. T_downsampled:{T}")

        if downsample_factor != 1:
            for key in smplx_params1:
                smplx_params1[key] = smplx_params1[key][::downsample_factor]
                smplx_params2[key] = smplx_params2[key][::downsample_factor]
            T = smplx_params1["transl"].shape[0]
        else:
            T = T_original

        # ============ 2 extract vertices for subject 1
        preprocess_transforms = []

        # create smplx model
        sbj_model = smplx.build_layer(
            model_path=str(cfg.env.smpl_folder), model_type="smplx", gender="male",
            use_pca=False, num_betas=10, batch_size=T
        )
        
        # convert parameters sbj1 
        # torch.Tensor
        body_model_params1 = {
            "betas": torch.tensor(smplx_params1['betas'][:,:10],dtype=torch.float),
            "transl": torch.tensor(smplx_params1["transl"],dtype=torch.float),
            "global_orient": torch.tensor(axis_angle_to_matrix(smplx_params1['global_orient']),dtype=torch.float).reshape(T, -1, 9),
            "body_pose": torch.tensor(axis_angle_to_matrix(smplx_params1['body_pose'].reshape(-1, 21, 3)),dtype=torch.float).reshape(T, -1, 9),
            "left_hand_pose": torch.tensor(axis_angle_to_matrix(smplx_params1['left_hand_pose'].reshape(-1, 15, 3)),dtype=torch.float).reshape(T, -1, 9),
            "right_hand_pose": torch.tensor(axis_angle_to_matrix(smplx_params1['right_hand_pose'].reshape(-1, 15, 3)),dtype=torch.float).reshape(T, -1, 9),
        }
        if cfg.input_type == "smpl":
            body_model_params1["left_hand_pose"] = None
            body_model_params1["right_hand_pose"] = None

        # get smpl(-h) vertices
        sbj_output = sbj_model(pose2rot=False, get_skin=True, return_full_pose=True, **body_model_params1)
        sbj_verts = tensor_to_cpu(sbj_output.vertices)
        sbj_joints = tensor_to_cpu(sbj_output.joints)
        sbj_transl = body_model_params1["transl"].numpy()
        sbj_orient = body_model_params1["global_orient"].numpy().reshape(T, 3, 3)

        # save smpl parameters np.array
        sbj_smpl = {
            "betas": body_model_params1["betas"].numpy(),
            "transl": sbj_transl,
            "global_orient": sbj_orient.reshape(T, 1, 9),
            "body_pose": body_model_params1["body_pose"].numpy(),
            "left_hand_pose": body_model_params1["left_hand_pose"].numpy(),
            "right_hand_pose": body_model_params1["right_hand_pose"].numpy()
        }

        for i in range(T):
            # create mesh
            sbj_faces = sbj_model.faces
            sbj_mesh = trimesh.Trimesh(vertices=sbj_verts[i], faces=sbj_faces)
            # save sbj mesh
            #sbj_mesh.export(target_folder / f"{seq_name}_sbj_{i}_before.ply")

        #print("subject1 extracted")
        # ============ 3 extract vertices for subject 2
  
        # create smplx model
        second_sbj_model = smplx.build_layer(
            model_path=str(cfg.env.smpl_folder), model_type="smplx", gender="male",
            use_pca=False, num_betas=10, batch_size=T
        )

        # convert parameters sbj1 
        # torch.Tensor
        body_model_params2 = {
            "betas": torch.tensor(smplx_params2['betas'][:,:10],dtype=torch.float),
            "transl": torch.tensor(smplx_params2["transl"],dtype=torch.float),
            "global_orient": torch.tensor(axis_angle_to_matrix(smplx_params2['global_orient']),dtype=torch.float).reshape(T, -1, 9),
            "body_pose": torch.tensor(axis_angle_to_matrix(smplx_params2['body_pose'].reshape(-1, 21, 3)),dtype=torch.float).reshape(T, -1, 9),
            "left_hand_pose": torch.tensor(axis_angle_to_matrix(smplx_params2['left_hand_pose'].reshape(-1, 15, 3)),dtype=torch.float).reshape(T, -1, 9),
            "right_hand_pose": torch.tensor(axis_angle_to_matrix(smplx_params2['right_hand_pose'].reshape(-1, 15, 3)),dtype=torch.float).reshape(T, -1, 9),
        }
        if cfg.input_type == "smpl":
            body_model_params1["left_hand_pose"] = None
            body_model_params1["right_hand_pose"] = None

        # get smpl(-h) vertices
        second_sbj_output = second_sbj_model(pose2rot=False, get_skin=True, return_full_pose=True, **body_model_params2)
        second_sbj_verts = tensor_to_cpu(second_sbj_output.vertices)
        second_sbj_joints = tensor_to_cpu(second_sbj_output.joints)
        second_sbj_transl = body_model_params2["transl"].numpy()
        second_sbj_smpl = body_model_params2["global_orient"].numpy().reshape(T, 3, 3)

        # save smpl parameters np.array
        second_sbj_smpl = {
            "betas": body_model_params2["betas"].numpy(),
            "transl": second_sbj_transl,
            "global_orient": second_sbj_smpl.reshape(T, 1, 9),
            "body_pose": body_model_params2["body_pose"].numpy(),
            "left_hand_pose": body_model_params2["left_hand_pose"].numpy(),
            "right_hand_pose": body_model_params2["right_hand_pose"].numpy()
        }

        for i in range(T):
            # create mesh
            second_sbj_faces = second_sbj_model.faces
            second_sbj_mesh = trimesh.Trimesh(vertices=second_sbj_verts[i], faces=second_sbj_faces)
            # save sbj mesh
            #second_sbj_mesh.export(target_folder / f"{seq_name}_second_sbj_{i}_before.ply")
        # ===========================================
        # print("subject2 extracted")
        # ============ 7 preprocess each time stamp in parallel  
        preprocess_results = []
        # print("preprocess_transforms: ", preprocess_transforms)
        if len(preprocess_transforms) != T:
            #dummy append
            for _ in range(T - len(preprocess_transforms)):
                preprocess_transforms.append({
                    "R": np.eye(3, dtype=np.float32),
                    "t": np.zeros(3, dtype=np.float32),
                    "rot_center": np.zeros(3, dtype=np.float32)
                })
        for t in tqdm.tqdm(range(T), leave=False, total=T, ncols=80):
            sample= DatasetSample(
                subject=seq_name,
                t_stamp=t,
                sbj_mesh=deepcopy(sbj_mesh),
                sbj_pc=sbj_verts[t],
                sbj_joints=sbj_joints[t],
                sbj_smpl={
                    "betas": sbj_smpl["betas"][t],
                    "transl": sbj_smpl["transl"][t],
                    "global_orient": sbj_smpl["global_orient"][t],
                    "body_pose": sbj_smpl["body_pose"][t],
                    "left_hand_pose": sbj_smpl["left_hand_pose"][t],
                    "right_hand_pose": sbj_smpl["right_hand_pose"][t]
                },
                #second subject
                second_sbj_mesh=deepcopy(second_sbj_mesh),
                second_sbj_pc=second_sbj_verts[t],
                second_sbj_joints=second_sbj_joints[t],
                second_sbj_smpl={
                    "betas": second_sbj_smpl["betas"][t],
                    "transl": second_sbj_smpl["transl"][t],
                    "global_orient": second_sbj_smpl["global_orient"][t],
                    "body_pose": second_sbj_smpl["body_pose"][t],
                    "left_hand_pose": second_sbj_smpl["left_hand_pose"][t],
                    "right_hand_pose": second_sbj_smpl["right_hand_pose"][t]
                },
                preprocess_transforms=preprocess_transforms[t]
            )
            result = preprocess_worker(sample)
            preprocess_results.append(result)
        # ===========================================

        # print(preprocess_results[0])
        # print("preprocess done")

        # ============ 8 Save subject-specific data
        if not seq_name in h5py_file:
            h5py_file.create_group(seq_name)
        seq_group = h5py_file[seq_name]
        add_sequence_datasets_to_hdf5(seq_group, preprocess_results[0], T)
        add_meatada_to_hdf5(seq_group, seq_name, T, "male")
        for sample in preprocess_results:
            sample.dump_hdf5(seq_group)
        #print("saved to h5")

    # ============ 9 Save global info
    suffix = f"{cfg.embody3d.split}_{cfg.embody3d.downsample}"
    OmegaConf.save(config=cfg, f=str(target_folder / f"preprocess_config_{suffix}.yaml"))
    # ===========================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess Embody3D data with 30 fps annotations')

    parser.add_argument('--config', "-c", type=str, nargs="*", help='Path to YAML config(-s) file.')
    parser.add_argument("overrides", type=str, nargs="*", help="Overrides for the config.")
    arguments = parser.parse_args()

    config = init_preprocessing(arguments)
    print("Preprocessing with config:")
    print(OmegaConf.to_yaml(config))

    # create split_file
    split(config)

    # preprocess data
    preprocess(config)

#python -m tridi.preprocessing.preprocess_embody3d -c ./config/env.yaml -- embody3d.split="train" embody3d.downsample="10fps"
#python -m tridi.preprocessing.preprocess_embody3d -c ./config/env.yaml -- embody3d.split="test" embody3d.downsample="1fps"
