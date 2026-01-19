"""
Common utilities for data preprocessing.

Functions:
    params_to_torch, tensor_to_cpu, prepare_params, parse_npz
are from https://github.com/otaheri/GRAB/blob/master/tools/utils.py

Function:
    filter_contact_frames
is from https://github.com/otaheri/GRAB/blob/master/grab/grab_preprocessing.py

License for functions mentioned above:
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
"""
import json
from pathlib import Path
import pickle as pkl
from typing import Union
from dataclasses import dataclass, field
from collections import defaultdict

import igl
import numpy as np
import torch
import trimesh
from omegaconf import OmegaConf
import re
import os

from config.preprocessing import PreprocessConfig


def init_preprocessing(arguments):
    # create default config
    structured_config = PreprocessConfig()

    # convert to omegaconf
    config = OmegaConf.structured(structured_config)

    # merge with config(-s) from file
    if arguments.config is not None:
        for config_file in arguments.config:
            config = OmegaConf.merge(config, OmegaConf.load(config_file))

    # merge with cli
    if len(arguments.overrides) > 0:
        cli_config = OmegaConf.from_dotlist(arguments.overrides)
        config = OmegaConf.merge(config, cli_config)

    OmegaConf.resolve(config)

    return config


def trimesh_load(path):
    return trimesh.load(str(path), process=False, validate=False, inspect=False)


def params_to_torch(params, dtype=torch.float32):
    return {k: torch.from_numpy(v).type(dtype) for k, v in params.items()}


def tensor_to_cpu(tensor):
    return tensor.detach().cpu().numpy()


def filter_contact_frames(cfg, seq_data):
    if cfg.grab.only_contact_frames:
        frame_mask = (seq_data['contact']['object'] > 0).any(axis=1)
    else:
        frame_mask = (seq_data['contact']['object'] > -1).any(axis=1)
    return frame_mask


def prepare_params(params, frame_mask, dtype = np.float32):
    return {k: v[frame_mask].astype(dtype) for k, v in params.items()}


def parse_npz(npz, allow_pickle=True):
    npz = np.load(npz, allow_pickle=allow_pickle)
    npz = {k: npz[k].item() for k in npz.files}
    return npz


def estimate_transform(vertices_from, vertices_to):
    """
    Based on compute_similarity_transform from https://github.com/akanazawa/hmr/blob/master/src/benchmark/eval_util.py
    """
    R, t = None, None

    centoid_from = vertices_from.mean(0, keepdims=True)
    centoid_to = vertices_to.mean(0, keepdims=True)

    vertices_from_shifted = vertices_from - centoid_from
    vertices_to_shifted = vertices_to - centoid_to

    vertices_from_shifted = vertices_from_shifted.swapaxes(0, 1)
    vertices_to_shifted = vertices_to_shifted.swapaxes(0, 1)

    H = vertices_from_shifted @ vertices_to_shifted.T

    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        # print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = (centoid_to.reshape(3, 1) - R @ centoid_from.reshape(3, 1)).reshape(1, 3)

    return R, t


def get_sequences_list(dataset, input_path, subjects=None, objects=None):
    all_seqs = []
    subjects = ["*"] if subjects is None else subjects
    if dataset == "behave":
        # sequences / Date<:02d>_Sub<:02d>_<object>_<optional:action> / t<:04d>.000 /
        if objects is not None and len(objects) > 0:
            for sbj in subjects:
                for object in objects:
                    all_seqs.extend(list(input_path.glob(f'*_{sbj}_{object}*/')))
        else:
            for sbj in subjects:
                all_seqs.extend(list(input_path.glob(f'*_{sbj}_*/')))
    elif dataset == "embody3d":
        # sequences w. 2, 3, 4 people
        not_filtered = []
        for sbj in subjects:
            not_filtered.extend(list(input_path.glob(f'{sbj}/*')))

        # pattern for subject folders
        pattern = re.compile(r"^[A-Z]{3}\d{3}$")
        # count subjects
        for sequence in not_filtered:
            count = 0
            if sequence.is_dir():
                for subfolder in sorted(os.listdir(sequence)):
                    if Path(sequence/subfolder).is_dir() and pattern.match(subfolder):
                        count += 1
                # append if 2 subjects
                if count == 2:
                    all_seqs.append(sequence)
    elif dataset == "interhuman":
        # motions / .pkl
        all_seqs.extend(list(input_path.glob(f'motions/*.pkl')))
    elif dataset == "chi3d":
        all_seqs.extend(list(input_path.glob(f'train/*/smplx/*.json')))
    return all_seqs


def add_meatada_to_hdf5(seq_group, sbj, T, gender=""):
    seq_group.attrs['sbj'] = sbj
    seq_group.attrs['T'] = T
    seq_group.attrs['gender'] = gender
    seq_group.attrs['second_sbj_gender'] = second_sbj_gender

def add_sequence_datasets_to_hdf5(seq_group, sample, T):
    # object mesh
    # if sample.obj_mesh is not None:
    #     obj_v, obj_f = sample.obj_mesh.vertices, sample.obj_mesh.faces
    #     seq_group.create_dataset("obj_v", shape=(T, obj_v.shape[0], 3))
    #     seq_group.create_dataset("obj_f", shape=(obj_f.shape[0], 3), data=obj_f)
    # subject mesh
    #sbj_v, sbj_f = sample.sbj_mesh.vertices, sample.sbj_mesh.faces
    #seq_group.create_dataset("sbj_v", shape=(T, sbj_v.shape[0], 3))
    #seq_group.create_dataset("sbj_f", shape=(sbj_f.shape[0], 3), data=sbj_f)
    # subject joints
    seq_group.create_dataset("sbj_j", shape=(T, sample.sbj_joints.shape[0], 3))
    # subject features
    if sample.sbj_smpl is not None :
        seq_group.create_dataset("sbj_smpl_betas", shape=(T, 10))
        seq_group.create_dataset("sbj_smpl_transl", shape=(T, 3))
        seq_group.create_dataset("sbj_smpl_global", shape=(T, 1, 9))
        seq_group.create_dataset("sbj_smpl_body", shape=(T, 21, 9))
        seq_group.create_dataset("sbj_smpl_lh", shape=(T, 15, 9))
        seq_group.create_dataset("sbj_smpl_rh", shape=(T, 15, 9))
    # # object features
    # if sample.obj_center is not None:
    #     seq_group.create_dataset("obj_c", shape=(T, 3))
    #     seq_group.create_dataset("obj_R", shape=(T, 9))


    # second subject mesh
    #second_sbj_v, second_sbj_f = sample.second_sbj_mesh.vertices, sample.second_sbj_mesh.faces
    #seq_group.create_dataset("second_sbj_v", shape=(T, second_sbj_v.shape[0], 3))
    #seq_group.create_dataset("second_sbj_f", shape=(second_sbj_f.shape[0], 3), data=second_sbj_f)
    # second subject joints
    seq_group.create_dataset("second_sbj_j", shape=(T, sample.second_sbj_joints.shape[0], 3))
    # second subject features
    if sample.second_sbj_smpl is not None :
        seq_group.create_dataset("second_sbj_smpl_betas", shape=(T, 10))
        seq_group.create_dataset("second_sbj_smpl_transl", shape=(T, 3))
        seq_group.create_dataset("second_sbj_smpl_global", shape=(T, 1, 9))
        seq_group.create_dataset("second_sbj_smpl_body", shape=(T, 21, 9))
        seq_group.create_dataset("second_sbj_smpl_lh", shape=(T, 15, 9))
        seq_group.create_dataset("second_sbj_smpl_rh", shape=(T, 15, 9))


    # preprocessing params
    seq_group.create_dataset("prep_R", shape=(T, 9))
    seq_group.create_dataset("prep_t", shape=(T, 3))
    seq_group.create_dataset("prep_rot_center", shape=(T, 3))
    seq_group.create_dataset("prep_s", shape=(T,))
    seq_group.create_dataset("orig_t_stamp", shape=(T,))


@dataclass
class DatasetSample:
    # General info
    subject: str
    #object: str
    #action: str
    t_stamp: int
    # Meshes and PCs
    # first subject features
    sbj_mesh: trimesh.Trimesh
    sbj_pc: np.ndarray
    sbj_joints: np.ndarray
    sbj_smpl: Union[dict, None]  # body model parameters
    # second subject features
    second_sbj_mesh: trimesh.Trimesh
    second_sbj_pc: np.ndarray
    second_sbj_joints: np.ndarray
    second_sbj_smpl: Union[dict, None]  # second body model parameters
    # Object features
    # obj_center: Union[np.ndarray, None]
    # obj_rotation: Union[np.ndarray, None]
    # Preprocessing params
    scale: float = field(init=False)
    preprocess_transforms: dict

    def dump_hdf5(self, seq_group):
        t = self.t_stamp
        # save data to hdf5
        #seq_group["sbj_v"][t] = self.sbj_mesh.vertices
        seq_group["sbj_j"][t] = self.sbj_joints
        if self.sbj_smpl is not None:
            seq_group["sbj_smpl_betas"][t] = self.sbj_smpl["betas"]
            seq_group["sbj_smpl_transl"][t] = self.sbj_smpl["transl"]
            seq_group["sbj_smpl_global"][t] = self.sbj_smpl["global_orient"]
            seq_group["sbj_smpl_body"][t] = self.sbj_smpl["body_pose"]
            seq_group["sbj_smpl_lh"][t] = self.sbj_smpl["left_hand_pose"]
            seq_group["sbj_smpl_rh"][t] = self.sbj_smpl["right_hand_pose"]

        #seq_group["second_sbj_v"][t] = self.second_sbj_mesh.vertices
        seq_group["second_sbj_j"][t] = self.second_sbj_joints

        if self.second_sbj_smpl is not None:
            seq_group["second_sbj_smpl_betas"][t] = self.second_sbj_smpl["betas"]
            seq_group["second_sbj_smpl_transl"][t] = self.second_sbj_smpl["transl"]
            seq_group["second_sbj_smpl_global"][t] = self.second_sbj_smpl["global_orient"]
            seq_group["second_sbj_smpl_body"][t] = self.second_sbj_smpl["body_pose"]
            seq_group["second_sbj_smpl_lh"][t] = self.second_sbj_smpl["left_hand_pose"]
            seq_group["second_sbj_smpl_rh"][t] = self.second_sbj_smpl["right_hand_pose"]

        seq_group["prep_R"][t] = self.preprocess_transforms["R"].flatten()
        seq_group["prep_t"][t] = self.preprocess_transforms["t"]
        seq_group["prep_rot_center"][t] = self.preprocess_transforms["rot_center"]
        seq_group["prep_s"][t] = self.scale

    def dump(self, data_path):
        output_path = data_path / self.subject / f"{self.object}_{self.action}" / f"t{self.t_stamp:05d}"
        output_path.mkdir(exist_ok=True, parents=True)

        # meshes
        _ = self.sbj_mesh.export(output_path / "subject.ply")
        if self.obj_mesh is not None:
            _ = self.obj_mesh.export(output_path / "object.ply")

        # subject features
        if self.sbj_smpl is not None:
            with open(output_path / "smpl.pkl", "wb") as fp:
                pkl.dump(self.sbj_smpl, fp)

        # preprocessing data
        with(output_path / "preprocess_transform.pkl").open("wb") as fp:
            preprocess_transforms = self.preprocess_transforms

            preprocess_transforms.update({
                "scale": self.scale,
            })
            pkl.dump(preprocess_transforms, fp)


def contacts_worker(obj_mesh, obj_verts, sbj_verts, sbj_faces, contact_threshold):
    obj_mesh.vertices = obj_verts

    obj_points = obj_mesh.sample(8000)
    # obj2sbj_d, _, _ = igl.signed_distance(obj_points, sbj_verts, sbj_faces, return_normals=False)
    P = __import__('numpy').ascontiguousarray(obj_points, dtype=__import__('numpy').float64)
    V = __import__('numpy').ascontiguousarray(sbj_verts, dtype=__import__('numpy').float64)
    F = __import__('numpy').ascontiguousarray(sbj_faces, dtype=__import__('numpy').int64)
    obj2sbj_d, _, _, _ = igl.signed_distance(P, V, F)


    if np.any(obj2sbj_d < contact_threshold):
        return True
    else:
        return False


def preprocess_worker(
    sample: DatasetSample
):  
    # sample = apply_symmetry(sample)
    # sample = apply_z_rotation(sample)

    sample.sbj_mesh.vertices = np.copy(sample.sbj_pc)
    sample.second_sbj_mesh.vertices = np.copy(sample.second_sbj_pc)

    # # invert only if symmetry augmentation is applied
    # sample.sbj_mesh.invert()
    # sample.second_sbj_mesh.invert()

    sample.scale = 1.0

    return sample


def apply_z_rotation(sample: DatasetSample) -> DatasetSample:
    angle = np.random.choice(np.arange(-np.pi, np.pi, np.pi / 36))
    #angle = np.pi /2

    cos, sin = np.cos(angle), np.sin(angle)
    R = np.array([
        [cos, -sin, 0.0],
        [sin,  cos, 0.0],
        [0.0,  0.0, 1.0]
    ], dtype=np.float32)

    def rot(x):
        return x @ R.T

    # smpl params
    sample.sbj_smpl["transl"] = rot(sample.sbj_smpl["transl"])
    sample.sbj_smpl["global_orient"] = rotate_axis_angle(
        sample.sbj_smpl["global_orient"], R
    )

    sample.second_sbj_smpl["transl"] = rot(sample.second_sbj_smpl["transl"])
    sample.second_sbj_smpl["global_orient"] = rotate_axis_angle(
        sample.second_sbj_smpl["global_orient"], R
    )

    # point clouds and joints
    sample.sbj_pc = rot(sample.sbj_pc)
    sample.sbj_joints = rot(sample.sbj_joints)
    sample.second_sbj_pc = rot(sample.second_sbj_pc)
    sample.second_sbj_joints = rot(sample.second_sbj_joints)

    return sample

def rotate_axis_angle(axis_angle, Rz):
    R_new = Rz @ axis_angle.reshape(3, 3)
    R_new = R_new.reshape(9)
    return R_new


def apply_symmetry(sample: DatasetSample) -> DatasetSample:
    body_sym_map = np.array(
        [1, 0, 2, 4, 3, 5, 7, 6, 8, 10, 9, 11, 13, 12, 14, 16, 15, 18, 17, 20, 19],
        dtype=np.int64
    )

    # (x,y,z)->(-x,y,z)
    S = np.diag([-1.0, 1.0, 1.0]).astype(np.float32)
    s_vec = np.array([-1.0, 1.0, 1.0], dtype=np.float32)  # 给 axis-angle 用

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
            if np.asarray(go).shape == (1, 9):
                params["global_orient"] = flip_global_orient_1x9(go)
            else:
                params["global_orient"] = flip_rot(go)

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
    sample.sbj_smpl = flip_params(sample.sbj_smpl)
    sample.second_sbj_smpl = flip_params(sample.second_sbj_smpl)

    # --- flip point clouds / joints ---
    sample.sbj_pc = flip_points(sample.sbj_pc)
    sample.sbj_joints = flip_points(sample.sbj_joints)
    sample.second_sbj_pc = flip_points(sample.second_sbj_pc)
    sample.second_sbj_joints = flip_points(sample.second_sbj_joints)

    return sample


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [batch_size, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [batch_size, 3, 3]
    From https://github.com/gulvarol/smplpytorch/blob/master/smplpytorch/pytorch/rodrigues_layer.py
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = \
        norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    batch_size = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([
        w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy,
        w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz,
        w2 - x2 - y2 + z2
    ],
                         dim=1).view(batch_size, 3, 3)
    return rotMat


def batch_rodrigues(axisang):
    """
    From https://github.com/gulvarol/smplpytorch/blob/master/smplpytorch/pytorch/rodrigues_layer.py
    """
    #axisang N x 3
    axisang_norm = torch.norm(axisang + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(axisang_norm, -1)
    axisang_normalized = torch.div(axisang, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * axisang_normalized], dim=1)
    rot_mat = quat2mat(quat)
    rot_mat = rot_mat.view(rot_mat.shape[0], 9)
    return rot_mat


def th_posemap_axisang(pose_vectors):
    """
    Converts axis-angle to rotmat
    pose_vectors (Tensor (batch_size x 72)): pose parameters in axis-angle representation
    From: https://github.com/gulvarol/smplpytorch/blob/master/smplpytorch/pytorch/tensutils.py
    """
    rot_nb = int(pose_vectors.shape[1] / 3)
    rot_mats = []
    for joint_idx in range(rot_nb):
        axis_ang = pose_vectors[:, joint_idx * 3:(joint_idx + 1) * 3]
        rot_mat = batch_rodrigues(axis_ang)
        rot_mats.append(rot_mat)

    rot_mats = torch.cat(rot_mats, 1)
    return rot_mats


def generate_object_meshes(objects, objects_path, output_path):
    if len(objects) == 0:
        objects = sorted(list(objects_path.glob("*")))
        objects = [obj.stem for obj in objects]

    for obj in objects:
        object_mesh = trimesh.load(str(objects_path / f"{obj}.ply"), process=False, validate=False)

        output_folder = output_path / "object_meshes"
        output_folder.mkdir(exist_ok=True, parents=True)
        _ = object_mesh.export(str(output_folder / f"{obj}.ply"))


def generate_obj_keypoints_from_barycentric(
    objects, keypoints_path, output_path
):
    # list of triangles indices - sampled_points_triangles_ids
    # barycentric coords - sampled_points_bary
    with open(keypoints_path, "rb") as fp:
        barycentric_dict = pkl.load(fp)

    if len(objects) == 0:
        objects = sorted(list((output_path / "object_meshes").glob("*")))
        objects = [obj.stem for obj in objects]

    for obj in objects:
        object_mesh = trimesh.load(
            str(output_path / "object_meshes" / f"{obj}.ply"), process=False, validate=False
        )

        # load triangle_ids and barycentric coordinates
        triangles_ids = barycentric_dict[obj]["triangles_ids"]
        barycentric = barycentric_dict[obj]["barycentric"]
        triangles = object_mesh.faces[triangles_ids]
        triangles_coords = object_mesh.vertices[triangles]

        sampled_points = \
            trimesh.triangles.barycentric_to_points(triangles_coords, barycentric)

        output_folder = output_path / "object_keypoints"
        output_folder.mkdir(exist_ok=True, parents=True)
        np.savez(output_folder / f"{obj}.npz", cartesian=sampled_points, barycentric=barycentric,
                 triangles_ids=triangles_ids)


def generate_obj_keypoints_barycentric(objects, keypoints_path, output_path, dataset):
    if len(objects) == 0:
        objects = sorted(list(keypoints_path.glob("*")))
        objects = [obj.stem for obj in objects]

    # save only barycentric coordinates and triangles ids
    barycentric_dict = dict()
    for obj in objects:
        obj_keypoints = np.load(keypoints_path / f"{obj}.npz")
        triangles_ids = obj_keypoints["triangles_ids"]
        barycentric = obj_keypoints["barycentric"]

        barycentric_dict[obj] = {
            "triangles_ids": triangles_ids,
            "barycentric": barycentric
        }

    output_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, "wb") as fp:
        pkl.dump(barycentric_dict, fp)


def generate_obj_keypoints(objects, objects_path, n_keypoints, output_path, is_center=False):
    if len(objects) == 0:
        objects = sorted(list(objects_path.glob("*")))
        objects = [obj.stem for obj in objects]

    for obj in objects:
        object_mesh = trimesh.load(str(objects_path / f"{obj}.ply"), process=False, validate=False)

        if is_center:
            _v = np.array(object_mesh.vertices)
            object_mesh.vertices = _v - np.mean(_v, axis=0)

        output_folder = output_path / "object_meshes"
        output_folder.mkdir(exist_ok=True, parents=True)
        _ = object_mesh.export(str(output_folder / f"{obj}.ply"))

        sampled_points = object_mesh.sample(n_keypoints)
        _, _, sampled_points_triangles_ids = trimesh.proximity.closest_point(object_mesh, sampled_points)
        sampled_points_triangles = object_mesh.faces[sampled_points_triangles_ids]

        sampled_points_bary = trimesh.triangles.points_to_barycentric(
            object_mesh.vertices[sampled_points_triangles],
            sampled_points
        )

        output_folder = output_path / "object_keypoints"
        output_folder.mkdir(exist_ok=True, parents=True)
        np.savez(output_folder / f"{obj}.npz", cartesian=sampled_points, barycentric=sampled_points_bary,
                triangles_ids=sampled_points_triangles_ids)


def generate_behave_split(behave_path, target_path):
    with (behave_path / "split.json").open("r") as fp:
        official_split = json.load(fp)

    # new split format is [[sbj_<split>, obj_act_date], ...]
    for split in ["train", "test"]:
        new_split = []
        sequences = official_split[split]
        for sequence in sequences:
            sequence_split = sequence.split("_")
            date = sequence_split[0]
            subject = sequence_split[1]
            object = sequence_split[2]
            action = "" if len(sequence_split) == 3 else sequence_split[3]

            new_subject = f"{subject}_{split}"
            new_obj_act = f"{object}_{action}_{date}" if len(action) > 0 else f"{object}_{date}"
            new_split.append([new_subject, new_obj_act])

        with (target_path / f"behave_split_{split}").open("w") as fp:
            json.dump(new_split, fp, indent=2)


def generate_behave_canonicalized_objects(behave_orig_objects_path, behave_can_objects_path):
    with open("./assets/behave_objects_canonicalization.pkl", "rb") as fp:
        transforms = pkl.load(fp)

    behave_can_objects_path.mkdir(exist_ok=True, parents=True)
    for obj, (R, t) in transforms.items():
        if obj in ["chairblack", "chairwood"]:
            suffix = "_f2500"
        elif obj == "tablesquare":
            suffix = "_f2000"
        elif obj == "monitor":
            suffix = "_closed_f1000"
        else:
            suffix = "_f1000"
        src_m = trimesh_load(str(behave_orig_objects_path / f"{obj}/{obj}{suffix}.ply"))
        tgt_v = np.array(src_m.vertices)
        tgt_v = tgt_v + t
        tgt_v = np.dot(R, tgt_v.T).T
        src_m.vertices = tgt_v
        _ = src_m.export(str(behave_can_objects_path / f"{obj}.ply"))


def compute_behave_object_canonicalization(behave_orig_obj_path, beahve_can_obj_path):
    src_objects = sorted(list(behave_orig_obj_path.glob("*/")))
    src_objects = [obj.name for obj in src_objects]

    transforms = {}
    for obj in src_objects:
        src = trimesh_load(behave_orig_obj_path / f"{obj}/{obj}_f1000.ply")
        tgt = trimesh_load(beahve_can_obj_path / f"{obj}.ply")

        t = (-1) * np.array(src.vertices).mean(axis=0, keepdims=True)
        _t = (-1) * np.array(tgt.vertices).mean(axis=0, keepdims=True)

        src.vertices += t
        tgt.vertices += _t

        R, _ = estimate_transform(src.vertices, tgt.vertices)

        transforms[obj] = (R, t)

    return transforms


def get_sbj_to_obj_act(path_template, grab_path, subjects, objects, actions):
    subject_to_obj_act = defaultdict(list)
    for subject in subjects:
        obj_acts = list(grab_path.glob(path_template.format(subject=subject, object="*", action="*")))
        for obj_act in obj_acts:
            obj_act = str(obj_act.name).split("_")
            obj = obj_act[0]
            act = "_".join(obj_act[1:])

            valid_action = True
            if actions is not None and len(actions) > 0:
                for exclude_action in actions:
                    if act in exclude_action:
                        valid_action = False
                        break

            if (objects is None or (len(objects) > 0 and obj in objects)) and valid_action:
                subject_to_obj_act[subject].append((obj, act))

    return subject_to_obj_act
