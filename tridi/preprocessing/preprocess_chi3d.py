"""
Code to preprocess 30fps annotations for the InerHuman dataset.
"""
import argparse
import json
import pickle as pkl
from copy import deepcopy
from multiprocessing import set_start_method
from pathlib import Path

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

def split(cfg):

    # convert to Path
    chi3d_path = Path(cfg.chi3d.root)
    assets_path = Path(cfg.chi3d.assets)
    #print(f"split():{chi3d_path}")

    # merge split files to a single json
    train = []
    test = []

    sequences = get_sequences_list("chi3d", chi3d_path)
    sequences.sort()
    # Split Scenario 1: specific motions for test the rest for train

    for sequence in sequences:
        if "Hug" in sequence.name:
            test.append(str(sequence))
        elif "Handshake" in sequence.name:
            test.append(str(sequence))
        else:
            train.append(str(sequence))

    # Split scenario 2: 80% of each sequence for train, 20% for test

    # for sequence in sequences:      
    #     test.append(str(sequence))
    #     train.append(str(sequence))
    
    # Split scenario 3: 80% of each motion for train, 20% of each motion for test
    # motion_dict={}
    # for sequence in sequences:
    #     motion, _ = sequence.stem.split()
    #     sbj_name = sequence.parents[1].name
    #     key = f"{sbj_name}_{motion}"
    #     motion_dict.setdefault(key, []).append(str(sequence))

    # for seq_list in motion_dict.values():
    #     split_idx = int(0.8 * len(seq_list))
    #     train.extend(seq_list[:split_idx])
    #     test.extend(seq_list[split_idx:])

    split_dict = {
        'train': train,
        'test' : test
    }
    # print(f"train samples length:{len(train)}")
    # print(f"test samples length:{len(test)}")
    with open(chi3d_path/"split.json", "w") as f:
        json.dump(split_dict, f, indent=4)
    
    # create split files for training
    split_train = []
    split_test = []

    for seq in train:
        seq = Path(seq)
        motion, index= seq.stem.split()
        sbj_name = seq.parents[1].name
        seq_name = f"{sbj_name}_{motion}_{index}"
        split_train.append(seq_name)

    for seq in test:
        seq = Path(seq)
        motion, index= seq.stem.split()
        sbj_name = seq.parents[1].name
        seq_name = f"{sbj_name}_{motion}_{index}"
        split_test.append(seq_name)

    with open (assets_path/"chi3d_train.json", "w") as f:
        json.dump(split_train, f, indent=4)
    with open (assets_path/"chi3d_test.json", "w") as f:
        json.dump(split_test, f, indent=4)
        

def preprocess(cfg):
    set_start_method('spawn')
    # convert to Path
    target_folder = Path(cfg.chi3d.target)
    chi3d_path = Path(cfg.chi3d.root)

    # list dataset sequences
    _sequences = get_sequences_list(
        "chi3d", chi3d_path)
    #print(f"preprocess_chi3dfhuman.py. Loaded sequences: {_sequences}")

    # filter sequences based on split
    if cfg.chi3d.split in ["train", "test"]:
        with open(cfg.chi3d.split_file, "r") as fp:
            split = json.load(fp)
            # print(f"Loaded split file {split}")
        sequences = split[cfg.chi3d.split]
        #print(f"preprocess_chi3d.py Loaded split sequences: {split_sequences}")
        # for seq in _sequences:
        #     print(f"{seq.name}")
        #     if seq.name in split_sequences:
        #         print(f"preprocess_chi3d.py Found matching sequence: {seq.name}")

        #print(f"preprocess_chi3d.py Filtered sequences: {sequences}")
        hdf5_name = f"dataset_{cfg.chi3d.split}"
        #print(f"hdf5name: {hdf5_name}")
    else:
        sequences = _sequences
        hdf5_name = "dataset"
    if cfg.chi3d.downsample == "10fps":
        hdf5_name += "_10fps"
    elif cfg.chi3d.downsample == "1fps":
        hdf5_name += "_1fps"
    else:
        hdf5_name += "_50fps"

    # init hdf5 file
    target_folder.mkdir(exist_ok=True, parents=True)
    if (target_folder / f"{hdf5_name}.hdf5").is_file():
        mode = "a"
    else:
        mode = "w"
    h5py_file = h5py.File(str(target_folder / f"{hdf5_name}.hdf5"), mode)

    # preprocess each sequence
    for sequence in tqdm.tqdm(sequences, total=len(sequences), ncols=80):
        sequence = Path(sequence)
        motion, index= sequence.stem.split()
        sbj_name = sequence.parents[1].name
        seq_name = f"{sbj_name}_{motion}_{index}"

        sbj_gender = "neutral"
        second_sbj_gender = "neutral"

        if sbj_name == "s02":
            sbj_gender = "male"
            second_sbj_gender = "female" 
        elif sbj_name == "s03":
            sbj_gender = "male"
            second_sbj_gender = "female"
        elif sbj_name == "s04":
            sbj_gender = "female"
            second_sbj_gender = "male"
        #print(f"subjects:{subjects}")

        # ============ 1 Load subject SMPL param
        # sequences : train/<sbj>/smplx/<motion> <index>.json

        with open(sequence, "r") as fp:
            params = json.load(fp)

        smplx_params1 = dict() 
        smplx_params2 = dict() 
        for param in ["betas", "body_pose", "left_hand_pose" , "right_hand_pose", "global_orient", "transl"]:
            # subject1
            smplx_params1[param] = np.array(params[param][0])
            # subject2
            smplx_params2[param] = np.array(params[param][1])

        T_original = len(params["transl"][0])

        #print("load subject smpl param done")
        #print(f"T_original:{T_original}")
        
        # downsample parameters if needed
        downsample_factor = 1
        if cfg.chi3d.downsample != "None":
            if cfg.chi3d.downsample == "10fps":
                downsample_factor = 5
            elif cfg.chi3d.downsample == "1fps":
                downsample_factor = 50

        if downsample_factor != 1:
            # Downsample
            for key in smplx_params1:
                smplx_params1[key] = smplx_params1[key][::downsample_factor]
                smplx_params2[key] = smplx_params2[key][::downsample_factor]

            # Split Scenario 1: specific motions for test the rest for train
            T = smplx_params1["transl"].shape[0]

            # Split scenario 2: 80% of each sequence for train, 20% for test
            # split_idx = int(0.8 * T_original)
            # if cfg.chi3d.split == "train":
            #     for key in smplx_params1:
            #         smplx_params1[key] = smplx_params1[key][:split_idx]
            #         smplx_params2[key] = smplx_params2[key][:split_idx]
            # elif cfg.chi3d.split == "test":
            #     for key in smplx_params1:
            #         smplx_params1[key] = smplx_params1[key][split_idx:]
            #         smplx_params2[key] = smplx_params2[key][split_idx:]
            # T = smplx_params1["transl"].shape[0]

            # Split scenario 3: 80% of each motion for train, 20% of each motion for test
            # T = smplx_params1["transl"].shape[0]

        else: # 50fps
            # Split Scenario 1: specific motions for test the rest for train
            T = T_original

            # Split scenario 2: 80% of each sequence for train, 20% for test
            # split_idx = int(0.8 * T_original)
            # if cfg.chi3d.split == "train":
            #     for key in smplx_params1:
            #         smplx_params1[key] = smplx_params1[key][:split_idx]
            #         smplx_params2[key] = smplx_params2[key][:split_idx]
            # elif cfg.chi3d.split == "test":
            #     for key in smplx_params1:
            #         smplx_params1[key] = smplx_params1[key][split_idx:]
            #         smplx_params2[key] = smplx_params2[key][split_idx:]
            # T = smplx_params1["transl"].shape[0]

            # Split scenario 3: 80% of each motion for train, 20% of each motion for test
            # T = T_original

        # ============ 2 extract vertices for subject 1
        preprocess_transforms = []

        # create smplx model

        sbj_model = smplx.build_layer(
            model_path=str(cfg.env.smpl_folder), model_type="smplx", gender=sbj_gender,
            use_pca=False, num_betas=10, batch_size=T
        )
        # convert parameters sbj1 
        # torch.Tensor
        body_model_params1 = {
            "betas": torch.tensor(smplx_params1['betas'],dtype=torch.float),
            "transl": torch.tensor(smplx_params1["transl"],dtype=torch.float),
            "global_orient": torch.tensor(smplx_params1['global_orient'],dtype=torch.float).reshape(T, -1, 9),
            "body_pose": torch.tensor(smplx_params1['body_pose'],dtype=torch.float).reshape(T, -1, 9),
            "left_hand_pose": torch.tensor(smplx_params1['left_hand_pose'],dtype=torch.float).reshape(T, -1, 9),
            "right_hand_pose": torch.tensor(smplx_params1['right_hand_pose'],dtype=torch.float).reshape(T, -1, 9),
        }
        if cfg.input_type == "smpl":
            body_model_params1["left_hand_pose"] = None
            body_model_params1["right_hand_pose"] = None

        # get smpl(-h) vertices
        sbj_output = sbj_model(pose2rot=False, get_skin=True, return_full_pose=True, **body_model_params1)
        sbj_verts = tensor_to_cpu(sbj_output.vertices)
        sbj_joints = tensor_to_cpu(sbj_output.joints)

        # save smpl parameters np.array
        sbj_smpl = {
            "betas": body_model_params1["betas"].numpy(),
            "transl": body_model_params1["transl"].numpy(),
            "global_orient": body_model_params1["global_orient"].numpy(),
            "body_pose": body_model_params1["body_pose"].numpy(),
            "left_hand_pose": body_model_params1["left_hand_pose"].numpy(),
            "right_hand_pose": body_model_params1["right_hand_pose"].numpy()
        }

        for i in range(T):
            # create mesh
            sbj_faces = sbj_model.faces
            sbj_mesh = trimesh.Trimesh(vertices=sbj_verts[i], faces=sbj_faces)
            # save sbj mesh
            #sbj_mesh.export(target_folder / "meshes"/ f"{seq_name}_sbj_{i}.ply")

        #print("subject1 extracted")
        # ============ 3 extract vertices for subject 2
  
        # create smplx model
        second_sbj_model = smplx.build_layer(
            model_path=str(cfg.env.smpl_folder), model_type="smplx", gender=second_sbj_gender,
            use_pca=False, num_betas=10, batch_size=T
        )

        # convert parameters sbj1 
        # torch.Tensor
        body_model_params2 = {
            "betas": torch.tensor(smplx_params2['betas'],dtype=torch.float),
            "transl": torch.tensor(smplx_params2["transl"],dtype=torch.float),
            "global_orient": torch.tensor(smplx_params2['global_orient'],dtype=torch.float).reshape(T, -1, 9),
            "body_pose": torch.tensor(smplx_params2['body_pose'],dtype=torch.float).reshape(T, -1, 9),
            "left_hand_pose": torch.tensor(smplx_params2['left_hand_pose'],dtype=torch.float).reshape(T, -1, 9),
            "right_hand_pose": torch.tensor(smplx_params2['right_hand_pose'],dtype=torch.float).reshape(T, -1, 9),
        }
        if cfg.input_type == "smpl":
            body_model_params2["left_hand_pose"] = None
            body_model_params2["right_hand_pose"] = None

        # get smpl(-h) vertices
        second_sbj_output = second_sbj_model(pose2rot=False, get_skin=True, return_full_pose=True, **body_model_params2)
        second_sbj_verts = tensor_to_cpu(second_sbj_output.vertices)
        second_sbj_joints = tensor_to_cpu(second_sbj_output.joints)

        # save smpl parameters np.array
        second_sbj_smpl = {
            "betas": body_model_params2["betas"].numpy(),
            "transl": body_model_params2["transl"].numpy(),
            "global_orient": body_model_params2["global_orient"].numpy(),
            "body_pose": body_model_params2["body_pose"].numpy(),
            "left_hand_pose": body_model_params2["left_hand_pose"].numpy(),
            "right_hand_pose": body_model_params2["right_hand_pose"].numpy()
        }

        for i in range(T):
            # create mesh
            second_sbj_faces = second_sbj_model.faces
            second_sbj_mesh = trimesh.Trimesh(vertices=second_sbj_verts[i], faces=second_sbj_faces)
            # save sbj mesh
            #second_sbj_mesh.export(target_folder / "meshes"/ f"{seq_name}_second_sbj_{i}.ply")
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
        add_meatada_to_hdf5(seq_group, seq_name, T, sbj_gender, second_sbj_gender)
        for sample in preprocess_results:
            sample.dump_hdf5(seq_group)
        #print("saved to h5")

    # ============ 9 Save global info
    suffix = f"{cfg.chi3d.split}_{cfg.chi3d.downsample}"
    OmegaConf.save(config=cfg, f=str(target_folder / f"preprocess_config_{suffix}.yaml"))
    # ===========================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess CHI3D data with 50 fps annotations')

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

#python -m tridi.preprocessing.preprocess_chi3d -c ./config/env.yaml -- chi3d.split="train" chi3d.downsample="50fps"
#python -m tridi.preprocessing.preprocess_chi3d -c ./config/env.yaml -- chi3d.split="test" chi3d.downsample="50fps"
