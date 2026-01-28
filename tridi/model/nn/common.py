"""
Common utilities for NN-based metrics (generation / reconstruction).

This version supports:
- HOI-style HDF5: /<subject>/<obj_act>/sbj_j, obj_c, obj_R, ...
- HH-style HDF5: /<subject>/sbj_j, second_sbj_j, sbj_smpl_*, second_sbj_smpl_*, attrs['T']
- Variable joint count (J) instead of hard-coded 73.
- Selecting sbj vs second_sbj via knn.sample_target.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm


def pseudo_inverse(mat: torch.Tensor) -> torch.Tensor:
    assert len(mat.shape) == 3
    tr = torch.bmm(mat.transpose(2, 1), mat)
    tr_inv = torch.inverse(tr)
    inv = torch.bmm(tr_inv, mat.transpose(2, 1))
    return inv


def init_object_orientation(src_axis: torch.Tensor, tgt_axis: torch.Tensor) -> torch.Tensor:
    pseudo = pseudo_inverse(src_axis)
    rot = torch.bmm(pseudo, tgt_axis)

    U, S, V = torch.svd(rot)
    R = torch.bmm(U, V.transpose(2, 1))
    return R

# ---------------------------
# Pose-only canonicalization (remove translation + global yaw)
# ---------------------------
def _rot_inv_about_axis(axis: str, yaw: float) -> np.ndarray:
    c = np.cos(-yaw).astype(np.float32)
    s = np.sin(-yaw).astype(np.float32)
    if axis == "y":  # y-up: rotate in x-z plane
        return np.array([[c, 0.0, s],
                         [0.0, 1.0, 0.0],
                         [-s, 0.0, c]], dtype=np.float32)
    if axis == "z":  # z-up: rotate in x-y plane
        return np.array([[c, -s, 0.0],
                         [s,  c, 0.0],
                         [0.0, 0.0, 1.0]], dtype=np.float32)
    raise ValueError(f"Unsupported axis: {axis}")

def _infer_up_axis_from_body(j: np.ndarray) -> str:
    # use shoulders - hips as a crude "up" direction
    J = j.shape[0]
    lhip, rhip = 1, 2
    lsho, rsho = 16, 17
    if J > max(lsho, rsho, lhip, rhip):
        hip_mid = 0.5 * (j[lhip] + j[rhip])
        sho_mid = 0.5 * (j[lsho] + j[rsho])
        up_vec = sho_mid - hip_mid
        # choose dominant axis between y and z (CHI3D is typically z-up)
        return "z" if abs(up_vec[2]) > abs(up_vec[1]) else "y"
    # fallback: default to z (matches your chi3d stats)
    return "z"

def _estimate_yaw(j: np.ndarray, up_axis: str) -> float:
    J = j.shape[0]
    lhip, rhip = 1, 2
    lsho, rsho = 16, 17
    if J <= max(lhip, rhip):
        return 0.0

    across = j[rhip] - j[lhip]
    if J > max(lsho, rsho):
        across2 = j[rsho] - j[lsho]
        across = 0.5 * (across + across2)

    # project across to horizontal plane
    across_h = across.copy()
    if up_axis == "y":
        across_h[1] = 0.0
        up = np.array([0.0, 1.0, 0.0], np.float32)
        # forward = up x across
        forward = np.cross(up, across_h).astype(np.float32)
        forward[1] = 0.0
        forward /= (np.linalg.norm(forward) + 1e-8)
        # yaw to align forward -> +Z
        return float(np.arctan2(forward[0], forward[2]))

    if up_axis == "z":
        across_h[2] = 0.0
        up = np.array([0.0, 0.0, 1.0], np.float32)
        forward = np.cross(up, across_h).astype(np.float32)
        forward[2] = 0.0
        forward /= (np.linalg.norm(forward) + 1e-8)
        # yaw to align forward -> +Y
        return float(np.arctan2(forward[0], forward[1]))

    return 0.0

def canonicalize_pose_only_joints(joints: np.ndarray) -> np.ndarray:
    j = joints.astype(np.float32)
    j = j - j[[0]]  # remove translation (root-center)

    up_axis = _infer_up_axis_from_body(j)
    yaw = _estimate_yaw(j, up_axis)
    R = _rot_inv_about_axis(up_axis, yaw)

    # apply rotation (row-vector points)
    j = (R @ j.T).T

    # resolve 180° ambiguity: enforce rhip.x > lhip.x after canonicalization
    J = j.shape[0]
    lhip, rhip = 1, 2
    if J > max(lhip, rhip):
        if j[rhip, 0] < j[lhip, 0]:
            Rpi = _rot_inv_about_axis(up_axis, -np.pi)  # rotate by +pi
            j = (Rpi @ j.T).T

    return j



# ---------------------------
# HDF5 structure helpers
# ---------------------------
def _is_hh_sequence_group(g: h5py.Group) -> bool:
    # HH style: group has attrs['T'] and joints datasets directly
    return ("T" in g.attrs) and (("sbj_j" in g.keys()) or ("second_sbj_j" in g.keys()))


def _resolve_sequence_group(hdf5_dataset: h5py.File, sequence: Tuple[str, str, str]) -> h5py.Group:
    """
    Return the actual group that contains datasets for this sequence.

    - HH style: sequence = (sbj, "", ""), return hdf5_dataset[sbj]
    - HOI style: sequence = (sbj, obj, act), try hdf5_dataset[sbj][f"{obj}_{act}"]
                 fallback to hdf5_dataset[sbj] if datasets exist there.
    """
    sbj, obj, act = sequence
    g_sbj = hdf5_dataset[sbj]

    # HH style direct
    if _is_hh_sequence_group(g_sbj):
        return g_sbj

    # HOI style nested
    if obj is not None and act is not None and (obj != "" or act != ""):
        key = f"{obj}_{act}" if act != "" else f"{obj}"
        if key in g_sbj:
            g = g_sbj[key]
            # some datasets store T at inner group
            return g

    # fallback
    return g_sbj


def get_sequence_from_hdf5(
    hdf5_file: Union[str, Path],
    subjects: Optional[List[str]] = None,
    actions: Optional[List[str]] = None,
    objects: Optional[List[str]] = None
) -> List[Tuple[str, str, str]]:
    """
    Returns list of sequences:
    - HH style: (sbj, "", "")
    - HOI style: (sbj, obj, act)
    """
    sequences: List[Tuple[str, str, str]] = []
    hdf5_file = str(hdf5_file)

    with h5py.File(hdf5_file, "r") as hdf5_dataset:
        for sbj in hdf5_dataset.keys():
            if subjects is not None and sbj not in subjects:
                continue

            g = hdf5_dataset[sbj]

            # HH style
            if _is_hh_sequence_group(g):
                sequences.append((sbj, "", ""))
                continue

            # HOI style: iterate obj_act groups
            obj_acts = list(g.keys())
            for obj_act in obj_acts:
                obj_act = str(obj_act)
                parts = obj_act.split("_")
                obj = parts[0]
                act = "_".join(parts[1:]) if len(parts) > 1 else ""

                if actions is not None and len(actions) > 0 and act not in actions:
                    continue
                if objects is not None and len(objects) > 0 and obj not in objects:
                    continue

                sequences.append((sbj, obj, act))

    return sequences


def get_sequences_from_split(
    split_file: Union[str, Path],
    subjects: Optional[List[str]],
    actions: Optional[List[str]],
    objects: Optional[List[str]]
) -> List[Tuple[str, str, str]]:
    sequences: List[Tuple[str, str, str]] = []

    with open(split_file, "r") as fp:
        split = json.load(fp)

    for sbj, obj_act in split:
        _oa_split = obj_act.split("_")
        obj = _oa_split[0]
        act = "_".join(_oa_split[1:])

        if subjects is not None and len(subjects) > 0 and sbj not in subjects:
            continue
        if actions is not None and len(actions) > 0 and act not in actions:
            continue
        if objects is not None and len(objects) > 0 and obj not in objects:
            continue

        sequences.append((sbj, obj, act))

    return sequences


# ---------------------------
# Feature / label extraction
# ---------------------------
def _pick_human_keys(knn) -> Dict[str, str]:
    """
    Decide which human (sbj vs second_sbj) to read based on knn.sample_target.
    """
    sample_target = getattr(knn, "sample_target", "sbj")
    prefix = "second_sbj_" if sample_target == "second_sbj" else "sbj_"

    keys = {}
    keys["j"] = prefix + "j"
    keys["global"] = prefix + "smpl_global"
    keys["body"] = prefix + "smpl_body"
    keys["lh"] = prefix + "smpl_lh"
    keys["rh"] = prefix + "smpl_rh"
    keys["transl"] = prefix + "smpl_transl"
    keys["betas"] = prefix + "smpl_betas"
    return keys

def get_data_for_sequence(
    knn,
    sequence: Tuple[str, str, str],
    dataset: str,
    hdf5_dataset: h5py.File,
    objname2classid: Optional[Dict[str, int]] = None,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, int, np.ndarray]]:
    """
    For general NN: returns (features, labels)
    For class-specific NN: returns (features, class_id, labels)
    """
    sbj, obj, act = sequence
    hdf5_sequence = _resolve_sequence_group(hdf5_dataset, sequence)

    # Time length
    if "T" in hdf5_sequence.attrs:
        T = int(hdf5_sequence.attrs["T"])
        t_stamps = range(T)
    else:
        # fallback: infer from sbj_j length if exists
        if "sbj_j" in hdf5_sequence:
            T = hdf5_sequence["sbj_j"].shape[0]
            t_stamps = range(T)
        else:
            return (np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.float32)) if objname2classid is None \
                else (np.zeros((0, 0), dtype=np.float32), -1, np.zeros((0, 0), dtype=np.float32))

    if T == 0:
        return (np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.float32)) if objname2classid is None \
            else (np.zeros((0, 0), dtype=np.float32), -1, np.zeros((0, 0), dtype=np.float32))

    # for HOI class-specific mapping
    class_id = -1
    if objname2classid is not None:
        if obj in objname2classid:
            class_id = int(objname2classid[obj])
        else:
            # unknown class -> skip
            return (np.zeros((0, 0), dtype=np.float32), -1, np.zeros((0, 0), dtype=np.float32))

    labels_list: List[np.ndarray] = []

    human_keys = _pick_human_keys(knn)

    # Allocate features
    if knn.model_features == "human_joints":
        if human_keys["j"] not in hdf5_sequence:
            raise KeyError(f"Missing joints dataset '{human_keys['j']}' in sequence group for {sbj}.")
        J = int(hdf5_sequence[human_keys["j"]].shape[1])
        feat_dim = (J - 1) * 3
        features = np.zeros((T, feat_dim), dtype=np.float32)

    elif knn.model_features == "human_parameters":
        features = np.zeros((T, 52 * 3), dtype=np.float32)

    else:
        raise RuntimeError(f"Unknown model_features {knn.model_features}")

    # Loop over time
    for t, t_stamp in enumerate(t_stamps):
        # ---- features ----
        if knn.model_features == "human_joints":
            sbj_j = np.asarray(hdf5_sequence[human_keys["j"]][t_stamp], dtype=np.float32)  # (J,3)

            if getattr(knn, "pose_only", False):
                sbj_j = canonicalize_pose_only_joints(sbj_j)   # root + yaw
            else:
                sbj_j = sbj_j - sbj_j[[0]]  # only root-center

            features[t] = sbj_j[1:].reshape(-1)


        elif knn.model_features == "human_parameters":
            # concat (1,9)+(21,9)+(15,9)+(15,9) => (52,9)
            pose9 = np.concatenate([
                np.asarray(hdf5_sequence[human_keys["global"]][t_stamp], dtype=np.float32),  # (1,9)
                np.asarray(hdf5_sequence[human_keys["body"]][t_stamp], dtype=np.float32),    # (21,9)
                np.asarray(hdf5_sequence[human_keys["lh"]][t_stamp], dtype=np.float32),      # (15,9)
                np.asarray(hdf5_sequence[human_keys["rh"]][t_stamp], dtype=np.float32),      # (15,9)
            ], axis=0)

            # pose9 can be (52,9) or already (52,3,3)
            if pose9.ndim == 2 and pose9.shape[1] == 9:
                pose_mat = pose9.reshape(-1, 3, 3)
            elif pose9.ndim == 3 and pose9.shape[1:] == (3, 3):
                pose_mat = pose9
            else:
                raise ValueError(f"Unexpected pose shape: {pose9.shape} for {sequence}")

            rotvec = np.zeros((pose_mat.shape[0], 3), dtype=np.float32)
            for i in range(pose_mat.shape[0]):
                rotvec[i] = Rotation.from_matrix(pose_mat[i]).as_rotvec()
            features[t] = rotvec.reshape(-1)

        # ---- labels ----
        if knn.model_labels == "data_source":
            label = np.array(int(dataset == "samples"), dtype=np.int32)
            labels_list.append(label)

        elif knn.model_labels == "human_parameters":
            # betas + pose(52*9) + transl(3)
            betas = np.asarray(hdf5_sequence[human_keys["betas"]][t_stamp], dtype=np.float32).reshape(-1)
            pose = np.concatenate([
                np.asarray(hdf5_sequence[human_keys["global"]][t_stamp], dtype=np.float32),
                np.asarray(hdf5_sequence[human_keys["body"]][t_stamp], dtype=np.float32),
                np.asarray(hdf5_sequence[human_keys["lh"]][t_stamp], dtype=np.float32),
                np.asarray(hdf5_sequence[human_keys["rh"]][t_stamp], dtype=np.float32),
            ], axis=0).reshape(-1)
            transl = np.asarray(hdf5_sequence[human_keys["transl"]][t_stamp], dtype=np.float32).reshape(-1)
            labels_list.append(np.concatenate([betas, pose, transl], axis=0).astype(np.float32))

        else:
            raise RuntimeError(f"Unknown model_labels {knn.model_labels}")

    labels = np.stack(labels_list, axis=0)

    if objname2classid is None:
        return features, labels
    return features, class_id, labels


# ---------------------------
# Dataset -> HDF5 path
# ---------------------------
def get_hdf5_files_for_nn(cfg, dataset_list: list) -> Dict[str, Path]:
    """
    Map dataset_name -> hdf5 path for given split.
    Must include your datasets (e.g., chi3d).
    """
    hdf5_files: Dict[str, Path] = {}

    for dataset_name, dataset_split in dataset_list:
        if dataset_name == "grab":
            fps = cfg.grab.fps_train if dataset_split == "train" else cfg.grab.fps_eval
            hdf5_files[dataset_name] = Path(cfg.grab.root) / f"dataset_{dataset_split}_{fps}fps.hdf5"

        elif dataset_name == "behave":
            fps = cfg.behave.fps_train if dataset_split == "train" else cfg.behave.fps_eval
            hdf5_files[dataset_name] = Path(cfg.behave.root) / f"dataset_{dataset_split}_{fps}fps.hdf5"

        elif dataset_name == "intercap":
            fps = cfg.intercap.fps_train if dataset_split == "train" else cfg.intercap.fps_eval
            hdf5_files[dataset_name] = Path(cfg.intercap.root) / f"dataset_{dataset_split}_{fps}fps.hdf5"

        elif dataset_name == "omomo":
            fps = cfg.omomo.fps_train if dataset_split == "train" else cfg.omomo.fps_eval
            hdf5_files[dataset_name] = Path(cfg.omomo.root) / f"dataset_{dataset_split}_{fps}fps.hdf5"

        elif dataset_name == "chi3d":
            fps = cfg.chi3d.fps_train if dataset_split == "train" else cfg.chi3d.fps_eval
            hdf5_files[dataset_name] = Path(cfg.chi3d.root) / f"dataset_{dataset_split}_{fps}fps.hdf5"

        elif dataset_name == "samples":
            hdf5_files[dataset_name] = Path(cfg.sample.samples_file)

        else:
            raise KeyError(f"Unknown dataset_name='{dataset_name}' in get_hdf5_files_for_nn()")

    return hdf5_files


def get_sequences_for_nn(cfg, dataset_list: list, hdf5_files: Dict[str, Path]) -> Dict[str, List[Tuple[str, str, str]]]:
    """
    Return dict: dataset_name -> list of sequences.
    """
    sequences: Dict[str, List[Tuple[str, str, str]]] = {}

    for dataset_name, dataset_split in dataset_list:
        if dataset_name == "behave":
            if dataset_split == "train":
                sequences["behave"] = get_sequences_from_split(
                    cfg.behave.train_split_file, cfg.behave.train_subjects,
                    cfg.behave.train_actions, cfg.behave.objects
                )
            else:
                sequences["behave"] = get_sequences_from_split(
                    cfg.behave.test_split_file, cfg.behave.test_subjects,
                    cfg.behave.test_actions, cfg.behave.objects
                )

        elif dataset_name == "samples":
            sequences["samples"] = get_sequence_from_hdf5(hdf5_files["samples"])

        elif dataset_name in ["grab", "intercap", "omomo", "chi3d"]:
            base_config = getattr(cfg, dataset_name)

            if dataset_split == "train":
                subjects = getattr(base_config, "train_subjects", None)
                actions = getattr(base_config, "train_actions", None)
            else:
                subjects = getattr(base_config, "test_subjects", None)
                actions = getattr(base_config, "test_actions", None)

            objects = getattr(base_config, "objects", None)

            sequences[dataset_name] = get_sequence_from_hdf5(
                hdf5_files[dataset_name], subjects=subjects, actions=actions, objects=objects
            )
        else:
            raise KeyError(f"Unknown dataset_name='{dataset_name}' in get_sequences_for_nn()")

    return sequences


def get_features_for_nn(
    knn, sequences: Dict[str, List[Tuple[str, str, str]]],
    hdf5_files: Dict[str, Path],
    is_train: bool = True
):
    """
    General NN loader.
    Ensures outputs are numpy arrays (not Python lists).
    """
    if is_train:
        features_list: List[np.ndarray] = []
        labels_list: List[np.ndarray] = []
        t_stamps: List[str] = []
    else:
        features_list = defaultdict(list)
        labels_list = defaultdict(list)
        t_stamps = defaultdict(list)

    for dataset_name, sequences_list in sequences.items():
        with h5py.File(hdf5_files[dataset_name], "r") as hdf5_dataset:
            for sequence in tqdm(sequences_list, ncols=80, leave=False):
                _features, _labels = get_data_for_sequence(
                    knn, sequence, dataset_name, hdf5_dataset
                )
                if _features.size == 0:
                    continue

                sbj, obj, act = sequence
                _t = [f"{sbj}/{obj}/{act}/t{t_stamp:05d}" for t_stamp in range(_labels.shape[0])]

                if is_train:
                    features_list.append(_features)
                    labels_list.append(_labels)
                    t_stamps.extend(_t)
                else:
                    features_list[dataset_name].append(_features)
                    labels_list[dataset_name].append(_labels)
                    t_stamps[dataset_name].extend(_t)

    # concatenate
    if is_train:
        if len(features_list) == 0:
            return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.float32), t_stamps
        features = np.concatenate(features_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        return features, labels, t_stamps

    # is_train == False
    features_out, labels_out = {}, {}
    for dataset_name in sequences.keys():
        if len(features_list[dataset_name]) == 0:
            features_out[dataset_name] = np.zeros((0, 0), dtype=np.float32)
            labels_out[dataset_name] = np.zeros((0, 0), dtype=np.float32)
            continue
        features_out[dataset_name] = np.concatenate(features_list[dataset_name], axis=0)
        labels_out[dataset_name] = np.concatenate(labels_list[dataset_name], axis=0)

    return features_out, labels_out, t_stamps
