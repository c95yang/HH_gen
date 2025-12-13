"""
Code to preprocess 30fps annotations for the BEAHVE dataset.
"""
import argparse
import json
import pickle as pkl
import warnings
from copy import deepcopy
from itertools import compress
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

from .common import (tensor_to_cpu, estimate_transform, DatasetSample, preprocess_worker, \
    get_sequences_list, th_posemap_axisang, generate_object_meshes, contacts_worker, \
    generate_obj_keypoints_from_barycentric, trimesh_load, init_preprocessing, \
    generate_behave_canonicalized_objects, add_sequence_datasets_to_hdf5, add_meatada_to_hdf5)
from ..utils.parallel_map import parallel_map
import torch

# 你的 L/R 对称关节映射
BODY_MIRROR_MAP = [1, 0, 2, 4, 3, 5, 7, 6, 8, 10, 9,
                   11, 13, 12, 14, 16, 15, 18, 17, 20, 19]

def axis_angle_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    """aa: (B, 3) -> R: (B, 3, 3)"""
    B = aa.shape[0]
    angle = torch.linalg.norm(aa + 1e-8, dim=1, keepdim=True)  # (B, 1)
    axis = aa / angle

    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    ca = torch.cos(angle).squeeze(1)
    sa = torch.sin(angle).squeeze(1)
    C = 1.0 - ca

    R = torch.zeros(B, 3, 3, device=aa.device, dtype=aa.dtype)
    R[:, 0, 0] = ca + x * x * C
    R[:, 0, 1] = x * y * C - z * sa
    R[:, 0, 2] = x * z * C + y * sa
    R[:, 1, 0] = y * x * C + z * sa
    R[:, 1, 1] = ca + y * y * C
    R[:, 1, 2] = y * z * C - x * sa
    R[:, 2, 0] = z * x * C - y * sa
    R[:, 2, 1] = z * y * C + x * sa
    R[:, 2, 2] = ca + z * z * C
    return R

def make_face_to_face_mirror(body_model_params):
    """
    输入: 一个 sbj 的 body_model_params (含 global_orient: [B,1,9])
    输出: (params_a, params_b)
        - params_a: 原人物
        - params_b: 镜像并且自动调整到“面对面”的人物
    """
    # 深拷贝
    params_a = {k: v.clone() for k, v in body_model_params.items()}
    params_b = {k: v.clone() for k, v in body_model_params.items()}

    body_pose = params_a["body_pose"]
    B = body_pose.shape[0]
    device = body_pose.device
    dtype = body_pose.dtype

    # 左右镜像平面：x -> -x
    Mx = torch.diag(torch.tensor([-1.0,  1.0,  1.0], device=device, dtype=dtype))

    # 世界竖直轴（假设 y 向上，如果你是别的坐标系，这里要改）
    up = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)

    # 模型局部“前向”轴（很多 SMPL 系列是 Z 前，如果不对可以改成 [0,0,-1] 或 [1,0,0]）
    f_local = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)

    # ---------- 1) 局部关节旋转：镜像 + L/R 互换 ----------
    def mirror_local_pose(pose, use_body_map=False):
        # pose: (B, J, 9)
        pose = pose.view(B, -1, 3, 3)  # -> (B, J, 3, 3)
        if use_body_map:
            pose = pose[:, BODY_MIRROR_MAP, :, :]
        # R' = Mx @ R @ Mx
        pose = Mx @ pose @ Mx
        return pose.view(B, -1, 9)

    # body：镜像 + L/R 关节重排
    params_b["body_pose"] = mirror_local_pose(params_b["body_pose"], use_body_map=True)

    # hands：镜像后再左右互换
    lh = params_b["left_hand_pose"].view(B, -1, 3, 3)
    rh = params_b["right_hand_pose"].view(B, -1, 3, 3)
    lh_m = Mx @ lh @ Mx
    rh_m = Mx @ rh @ Mx
    params_b["left_hand_pose"]  = rh_m.view(B, -1, 9)
    params_b["right_hand_pose"] = lh_m.view(B, -1, 9)

    # ---------- 2) global_orient：自动算“面对面” ----------
    # A 的根关节旋转
    go_a = params_a["global_orient"].view(B, 1, 3, 3)  # (B,1,3,3)
    R0   = go_a[:, 0]                                  # (B,3,3)

    # A 的前向向量（世界坐标）
    fA = (R0 @ f_local)    # (B,3)

    # B 初始（镜像后还没对准）的根旋转
    R_mirror = Mx @ R0 @ Mx   # (B,3,3)
    fB0 = (R_mirror @ f_local)

    # 在水平面上投影，并归一化
    def project_horiz(v):
        # 去掉竖直分量
        v_proj = v - (v * up).sum(-1, keepdim=True) * up
        v_norm = v_proj.norm(dim=-1, keepdim=True) + 1e-8
        return v_proj / v_norm

    pA  = project_horiz(fA)   # (B,3) A 的水平前向
    pB0 = project_horiz(fB0)  # (B,3) B 镜像后的水平前向

    # 希望：Q * pB0 = -pA  （面对面）
    target = -pA
    cosang = (pB0 * target).sum(-1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    angle  = torch.acos(cosang)  # (B,)

    # 决定旋转方向符号（顺时针/逆时针），用叉积与 up 的点积判断
    cross  = torch.cross(pB0, target, dim=-1)  # (B,3)
    sign   = torch.sign((cross * up).sum(-1))  # (B,)
    angle  = angle * sign                      # (B,)

    # 构造绕 up 轴的 axis-angle，然后转成旋转矩阵 Q
    axis = up.unsqueeze(0).expand(B, 3)        # 所有人同一个竖直轴
    aa   = axis * angle.unsqueeze(-1)          # (B,3)
    Q    = axis_angle_to_matrix(aa)           # (B,3,3)

    # 最终 B 的根旋转
    R1 = Q @ R_mirror                         # (B,3,3)
    params_b["global_orient"] = R1.view(B, 1, 9)

    # ---------- 3) transl：你可以先不动，只改朝向 ----------
    if "transl" in params_a:
        # 这里先保持原始位置，有需要你再按自己的逻辑改
        params_b["transl"] = params_a["transl"].clone()

        t_center = params_a["transl"].clone()
        pA_dir   = pA / (pA.norm(dim=-1, keepdim=True) + 1e-8)
        dist = 2.0  
        params_a["transl"] = t_center - pA_dir * (dist / 2.0)
        params_b["transl"] = t_center + pA_dir * (dist / 2.0)

    return params_b



def apply_symmetry_sbj(body_model_params):
    # symmetrical mapping for body joints
    body_sym_map = np.array(
        [1, 0, 2, 4, 3, 5, 7, 6, 8, 10, 9, 11, 13, 12, 14, 16, 15, 18, 17, 20, 19]
    )
    #body_model_params["body_pose"] = body_model_params["body_pose"][body_sym_map]
    body_model_params["body_pose"] = body_model_params["body_pose"][:, body_sym_map, :]
    # reshape for body, hands
    body_model_params["body_pose"] = body_model_params["body_pose"].reshape(
        body_model_params["body_pose"].shape[0], -1, 3, 3
    )
    body_model_params["left_hand_pose"] = body_model_params["left_hand_pose"].reshape(
        body_model_params["left_hand_pose"].shape[0], -1, 3, 3
    )   
    body_model_params["right_hand_pose"] = body_model_params["right_hand_pose"].reshape(
        body_model_params["right_hand_pose"].shape[0], -1, 3, 3
    )


    # z y x -> -z -y x
    sign_flip = np.array([[
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0]
    ]], dtype=np.float32)

    # flip rotations
    # IMPORTANT: FLIP FIXER
    body_model_params = {k: v * sign_flip 
                         if k != "global_orient" and k != "transl" and k != "betas" 
                         else v for k, v in body_model_params.items()}
    
    #reshape back
    body_model_params["body_pose"] = body_model_params["body_pose"].reshape(
        body_model_params["body_pose"].shape[0], -1, 9
    )
    body_model_params["left_hand_pose"] = body_model_params["left_hand_pose"].reshape(
        body_model_params["left_hand_pose"].shape[0], -1, 9
    )
    body_model_params["right_hand_pose"] = body_model_params["right_hand_pose"].reshape(
        body_model_params["right_hand_pose"].shape[0], -1, 9
    )


    # flipping left and right hands in the output
    lh, rh = body_model_params["left_hand_pose"], body_model_params["right_hand_pose"]
    body_model_params["left_hand_pose"] = rh
    body_model_params["right_hand_pose"] = lh

    return body_model_params

def preprocess(cfg):
    set_start_method('spawn')

    # convert to Path
    target_folder = Path(cfg.behave.target)
    behave_30fps_path = Path(cfg.behave.full_anno_path)

    # list dataset sequences
    _sequences = get_sequences_list(
        "behave", behave_30fps_path, objects=cfg.behave.objects, subjects=cfg.behave.subjects
    )
    print(f"preprocess_behave_30fps.py 42. Loaded sequences: {_sequences}")

    # filter sequences based on split
    if cfg.behave.split in ["train", "test"]:
        with open(cfg.behave.split_file, "r") as fp:
            split = json.load(fp)
            # print(f"Loaded split file {split}")
        split_sequences = split[cfg.behave.split]
        print(f"preprocess_behave.py 50. Loaded split sequences: {split_sequences}")
        for seq in _sequences:
            print(f"{seq.name}")
            if seq.name in split_sequences:
                print(f"preprocess_behave.py 51. Found matching sequence: {seq.name}")
        sequences = [seq for seq in _sequences if seq.name in split_sequences]

        print(f"preprocess_behave_30fps.py 57. Filtered sequences: {sequences}")
        hdf5_name = f"dataset_{cfg.behave.split}"
    else:
        sequences = _sequences
        hdf5_name = "dataset"
    if cfg.behave.downsample == "10fps":
        hdf5_name += "_10fps"
    elif cfg.behave.downsample == "1fps":
        hdf5_name += "_1fps"
    else:
        hdf5_name += "_30fps"

    # init hdf5 file
    subjects = list(set([s.name.split("_")[1] for s in sequences]))

    target_folder.mkdir(exist_ok=True, parents=True)
    if (target_folder / f"{hdf5_name}.hdf5").is_file():
        mode = "a"
    else:
        mode = "w"
    h5py_file = h5py.File(str(target_folder / f"{hdf5_name}.hdf5"), mode)
    for sbj in subjects:
        group_name = f"{sbj}_{cfg.behave.split}" if cfg.behave.split in ["train", "test"] else sbj
        if not group_name in h5py_file:
            h5py_file.create_group(group_name)

    # preprocess each sequence
    for sequence in tqdm.tqdm(sequences, total=len(sequences), ncols=80):
        # load sequence info
        with (sequence / "info.json").open("r") as fp:
            sequence_info = json.load(fp)  # 'cat', 'gender'

        # parse sequence name: Date<:02d>_Sub<:02d>_<object>_<optional:action>
        sequence_name_split = sequence.name.split("_")
        seq_subject = sequence_name_split[1]

        # ============ 1 Load object poses and subject SMPL params
        smpl_params = dict(np.load(sequence / "smpl_fit_all.npz")) #['poses', 'betas', 'trans', 'save_name', 'frame_times', 'gender']
        t_stamps = smpl_params["frame_times"]
        T = len(t_stamps)

        # ============ 2 extract vertices for subject
        preprocess_transforms = []

        # create smplh model
        if not(cfg.input_type in ["smplh", "smpl"]):
            raise NotImplementedError("Only SMPL and SMPL+H are supported for the BEHAVE dataset")
        sbj_model = smplx.build_layer(
            model_path=str(cfg.env.smpl_folder), model_type="smplh", gender=sequence_info["gender"],
            use_pca=False, num_betas=10, batch_size=T
        )

        # convert parameters
        th_pose_axisangle = torch.tensor(smpl_params["poses"].reshape(T, 52, 3))
        th_pose_rotmat = th_posemap_axisang(th_pose_axisangle.reshape(T * 52, 3)).reshape(T, 52, 9)
        body_model_params = {
            "betas": torch.tensor(smpl_params['betas']),
            "transl": torch.tensor(smpl_params["trans"]),
            "global_orient": th_pose_rotmat[:, :1].reshape(T, -1, 9),
            "body_pose": th_pose_rotmat[:, 1:22].reshape(T, -1, 9),
            "left_hand_pose": th_pose_rotmat[:, 22:37].reshape(T, -1, 9),
            "right_hand_pose": th_pose_rotmat[:, 37:].reshape(T, -1, 9),
        }
        if cfg.input_type == "smpl":
            body_model_params["left_hand_pose"] = None
            body_model_params["right_hand_pose"] = None

        # get smpl(-h) vertices
        sbj_output = sbj_model(pose2rot=False, get_skin=True, return_full_pose=True, **body_model_params)
        sbj_verts = tensor_to_cpu(sbj_output.vertices)
        sbj_joints = tensor_to_cpu(sbj_output.joints)
        sbj_transl = body_model_params["transl"].numpy()
        sbj_orient = body_model_params["global_orient"].numpy().reshape(T, 3, 3)

        # save smpl parameters
        sbj_smpl = {
            "betas": body_model_params["betas"],
            "transl": sbj_transl,
            "global_orient": sbj_orient.reshape(T, 1, 9),
            "body_pose": body_model_params["body_pose"].reshape(T, -1, 9).numpy(),
            "left_hand_pose": body_model_params["left_hand_pose"].reshape(T, -1, 9).numpy(),
            "right_hand_pose": body_model_params["right_hand_pose"].reshape(T, -1, 9).numpy()
        }

        for i in range(T):
            # create mesh
            sbj_faces = sbj_model.faces
            sbj_mesh = trimesh.Trimesh(vertices=sbj_verts[i], faces=sbj_faces)
            # save sbj mesh
            sbj_mesh.export(target_folder / f"{seq_subject}_sbj_{i}_before.ply")

        # =============Second sbj==============================
        # second_body_model_params = apply_symmetry_sbj(deepcopy(body_model_params))    
        second_body_model_params = make_face_to_face_mirror(deepcopy(body_model_params))    
        second_sbj_model = smplx.build_layer(
            model_path=str(cfg.env.smpl_folder), model_type="smplh", gender=sequence_info["gender"],
            use_pca=False, num_betas=10, batch_size=T
        )
        second_sbj_output = second_sbj_model(pose2rot=False, get_skin=True, return_full_pose=True, **second_body_model_params)
        second_sbj_verts = tensor_to_cpu(second_sbj_output.vertices)
        second_sbj_joints = tensor_to_cpu(second_sbj_output.joints)
        second_sbj_transl = second_body_model_params["transl"].numpy()
        second_sbj_orient = second_body_model_params["global_orient"].numpy().reshape(T, 3, 3)

        # save smpl parameters
        second_sbj_smpl = {
            "betas": second_body_model_params["betas"],
            "transl": second_sbj_transl,
            "global_orient": second_sbj_orient.reshape(T, 1, 9),
            "body_pose": second_body_model_params["body_pose"].reshape(T, -1, 9).numpy(),
            "left_hand_pose": second_body_model_params["left_hand_pose"].reshape(T, -1, 9).numpy(),
            "right_hand_pose": second_body_model_params["right_hand_pose"].reshape(T, -1, 9).numpy()
        }

        for i in range(T):
            # create mesh
            second_sbj_faces = second_sbj_model.faces
            second_sbj_mesh = trimesh.Trimesh(vertices=second_sbj_verts[i], faces=second_sbj_faces)
            # save sbj mesh
            second_sbj_mesh.export(target_folder / f"{seq_subject}_second_sbj_{i}_before.ply")
        # ===========================================

        # ============ 5 Align the ground plane ============  
        if len(preprocess_transforms) != T:
            #dummy append
            for _ in range(T - len(preprocess_transforms)):
                preprocess_transforms.append({
                    "R": np.eye(3, dtype=np.float32),
                    "t": np.zeros(3, dtype=np.float32),
                    "rot_center": np.zeros(3, dtype=np.float32)
                })
                
        for i in range(T):
            z_min_sbj = np.min(sbj_verts[i, :, 2])
            z_min_second_sbj = np.min(second_sbj_verts[i, :, 2])
            z_min = min(z_min_sbj, z_min_second_sbj)
            t_align_z = np.array([0.0, 0.0, -z_min], dtype=np.float32)

            preprocess_transforms[i]["t"] += t_align_z

            sbj_verts[i] += t_align_z
            # obj_verts[i] += t_align_z
            sbj_joints[i] += t_align_z
            sbj_smpl["transl"][i] += t_align_z

            second_sbj_verts[i] += t_align_z
            # obj_verts[i] += t_align_z
            second_sbj_joints[i] += t_align_z
            second_sbj_smpl["transl"][i] += t_align_z
        # ==================================================

        # ============ 7 preprocess each time stamp in parallel
        # name mapping to split sequences for the same subject from different days
        if cfg.behave.split == "test":
            seq_subject = f"{seq_subject}_test"
        elif cfg.behave.split == "train":
            seq_subject = f"{seq_subject}_train"

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
                subject=seq_subject,
                #action=seq_action,
                #object=seq_object,
                t_stamp=t,
                sbj_mesh=deepcopy(sbj_mesh),
                # obj_mesh=deepcopy(obj_mesh),
                # obj_mesh = None,
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

            result.sbj_mesh.export(target_folder / f"{seq_subject}_sbj_{t}_after.ply")
            result.second_sbj_mesh.export(target_folder / f"{seq_subject}_second_sbj_{t}_after.ply")

            preprocess_results.append(result)
        # ===========================================

        # print(preprocess_results[0])

        # ============ 8 Save subject-specific data
        # contact_masks[f"{seq_subject}_{seq_object}_{seq_action}"] = contact_mask
        # seq_group_name = f"{seq_object}_{seq_action}"
        # seq_group_name = "Default"
        # if seq_group_name in h5py_file[seq_subject]:
        #     del h5py_file[seq_subject][seq_group_name]
        seq_group = h5py_file[seq_subject] #.create_group(seq_group_name)
        add_sequence_datasets_to_hdf5(seq_group, preprocess_results[0], T)
        add_meatada_to_hdf5(seq_group, seq_subject, T, sequence_info["gender"])
        for sample in preprocess_results:
            sample.dump_hdf5(seq_group)


    # ============ 9 Save global info
    suffix = f"{cfg.behave.split}_{cfg.behave.downsample}"
    # with (target_folder / f"contact_masks_{suffix}.pkl").open("wb") as fp:
    #     pkl.dump(contact_masks, fp)
    OmegaConf.save(config=cfg, f=str(target_folder / f"preprocess_config_{suffix}.yaml"))
    # ===========================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess BEHAVE data with 30 fps annotations')

    parser.add_argument('--config', "-c", type=str, nargs="*", help='Path to YAML config(-s) file.')
    parser.add_argument("overrides", type=str, nargs="*", help="Overrides for the config.")
    arguments = parser.parse_args()

    config = init_preprocessing(arguments)
    print("Preprocessing with config:")
    print(OmegaConf.to_yaml(config))

    # canonicalize objects using pre-computed transforms
    # generate_behave_canonicalized_objects(
    #     Path(config.behave.orig_objects_path),
    #     Path(config.behave.can_objects_path)
    # )
    
    # preprocess data
    preprocess(config)

    # optionally generate object keypoints
    if config.behave.generate_obj_keypoints:
        generate_object_meshes(
            config.behave.objects,
            Path(config.behave.can_objects_path),
            Path(config.behave.target)
        )

        generate_obj_keypoints_from_barycentric(
            config.behave.objects,
            Path(config.env.assets_folder) / "object_keypoints" / "behave.pkl",
            Path(config.behave.target),
        )
