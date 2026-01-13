from pathlib import Path
from typing import Union

import h5py
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from tqdm.autonotebook import tqdm

from config.config import ProjectConfig
from tridi.model.nn.common import get_hdf5_files_for_nn, get_sequences_for_nn

def _get_seq_group(h5, sbj: str, obj: str = "", act: str = ""):
   
    if obj and act:
        key = f"{obj}_{act}"
        return h5[sbj][key]
    return h5[sbj]


def _safe_T(g, key: str):
    
    if key not in g:
        return 0
    T_data = int(g[key].shape[0])
    T_attr = int(g.attrs.get("T", T_data))
    return min(T_data, T_attr)

@torch.no_grad()
def contacts_worker(sbj_verts, obj_verts, contact_threshold):
    with torch.no_grad():
        contacts = torch.cdist(
            sbj_verts,
            obj_verts,
        )
        contacts = contacts.min(dim=-1).values.cpu().numpy()
        contacts_mask = contacts <= contact_threshold
    return contacts_mask


def compute_similarity_transform(S1, S2):
    """
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale*(R.dot(mu1))

    # 7. Error:
    S1_hat = scale*R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat


def align_to_root(joints, root_idx=0):
    return joints - joints[..., root_idx:root_idx + 1, :]


def get_mpjpe(pred_joints, gt_joints):
    # root aligned
    pred_joints = align_to_root(pred_joints)
    gt_joints = align_to_root(gt_joints)

    mpjpe = np.sqrt(np.sum((gt_joints - pred_joints) ** 2, axis=-1))
    return mpjpe.mean(-1)


def get_mpjpe_pa(pred_joints, gt_joints):
    # root aligned
    pred_joints = align_to_root(pred_joints)
    gt_joints = align_to_root(gt_joints)

    # procrustes aligned
    pred_joints = compute_similarity_transform(pred_joints, gt_joints)
    mpjpe_pa = np.sqrt(np.sum((gt_joints - pred_joints) ** 2, axis=-1))
    return mpjpe_pa.mean(-1)


@torch.no_grad()
def get_contact_similarity_sampled_sbj(
    sampled_sbj_v, gt_obj_v, obj_f, gt_sbj_v, sbj_f, contact_threshold=0.05, return_pred_contacts=False
):
    T = sampled_sbj_v.shape[0]
    B = 50

    gt_obj_v = torch.from_numpy(gt_obj_v).cuda()
    gt_sbj_v = torch.from_numpy(gt_sbj_v).cuda()

    gt_input_data = [{
        "obj_verts": gt_obj_v[t : min(t + B, T)],
        "sbj_verts": gt_sbj_v[t : min(t + B, T)],
        "contact_threshold": contact_threshold,
    } for t in range(0, T, B)]

    gt_contacts_mask = [contacts_worker(**gt_input_data[t]) for t in range(len(gt_input_data))]
    gt_contacts_mask = np.concatenate(gt_contacts_mask, axis=0)
    sampled_sbj_v = torch.from_numpy(sampled_sbj_v).cuda()

    pred_input_data = [{
        "obj_verts": gt_obj_v[t : min(t + B, T)],
        "sbj_verts": sampled_sbj_v[t : min(t + B, T)],
        "contact_threshold": contact_threshold,
    } for t in range(0, T, B)]

    pred_contacts_mask = [contacts_worker(**pred_input_data[t]) for t in range(len(pred_input_data))]
    pred_contacts_mask = np.concatenate(pred_contacts_mask, axis=0)

    # Compare gt and sampled contacts
    similarity_mask = gt_contacts_mask == pred_contacts_mask  # T x 6890
    similarity_mask = similarity_mask.mean(1)  # T

    if return_pred_contacts:
        return gt_contacts_mask, similarity_mask, pred_contacts_mask
    else:
        return gt_contacts_mask, similarity_mask


def get_sbj_metrics(
    cfg: ProjectConfig,
    samples_file: Union[str, Path],
    dataset: str
):
    test_datasets = [(dataset, "test")]
    test_hdf5 = get_hdf5_files_for_nn(cfg, test_datasets)
    test_sequences = get_sequences_for_nn(cfg, test_datasets, test_hdf5)

    mpjpe_results, mpjpe_pa_results = [], []
    mpjpe_results_second_sbj, mpjpe_pa_results_second_sbj = [], []

    with h5py.File(test_hdf5[dataset], "r") as test_hdf5_dataset, \
         h5py.File(samples_file, "r") as samples_hdf5_dataset:

        for sequence in tqdm(test_sequences[dataset], ncols=80, leave=False):
            # 兼容 sequence 是 tuple 或 string
            if isinstance(sequence, (list, tuple)) and len(sequence) == 3:
                sbj, obj, act = sequence
            else:
                sbj, obj, act = str(sequence), "", ""

            # 缺序列就跳过（你那种只保留2个序列的小 h5 很常见）
            if sbj not in test_hdf5_dataset or sbj not in samples_hdf5_dataset:
                continue

            try:
                test_sequence = _get_seq_group(test_hdf5_dataset, sbj, obj, act)
                sampled_sequence = _get_seq_group(samples_hdf5_dataset, sbj, obj, act)
            except KeyError:
                # HOI 层级缺 obj_act 或者 samples 没写进去
                continue

            # 用真实长度做交集：sbj / second_sbj 都要考虑
            T_sbj = min(_safe_T(test_sequence, "sbj_j"), _safe_T(sampled_sequence, "sbj_j"))
            T_2   = min(_safe_T(test_sequence, "second_sbj_j"), _safe_T(sampled_sequence, "second_sbj_j"))

            # 两个人都要算的话，就取共同可用帧数
            T = min(T_sbj, T_2)
            if T <= 0:
                continue

            for t_stamp in range(T):
                # --- sbj ---
                test_joints = test_sequence["sbj_j"][t_stamp]
                sampled_joints = sampled_sequence["sbj_j"][t_stamp]
                mpjpe_results.append(get_mpjpe(sampled_joints, test_joints))
                mpjpe_pa_results.append(get_mpjpe_pa(sampled_joints, test_joints))

                # --- second sbj ---
                test_joints_2 = test_sequence["second_sbj_j"][t_stamp]
                sampled_joints_2 = sampled_sequence["second_sbj_j"][t_stamp]
                mpjpe_results_second_sbj.append(get_mpjpe(sampled_joints_2, test_joints_2))
                mpjpe_pa_results_second_sbj.append(get_mpjpe_pa(sampled_joints_2, test_joints_2))

    mpjpe_results = np.array(mpjpe_results, dtype=np.float32)
    mpjpe_pa_results = np.array(mpjpe_pa_results, dtype=np.float32)
    mpjpe_results_second_sbj = np.array(mpjpe_results_second_sbj, dtype=np.float32)
    mpjpe_pa_results_second_sbj = np.array(mpjpe_pa_results_second_sbj, dtype=np.float32)

    return mpjpe_results, mpjpe_pa_results, mpjpe_results_second_sbj, mpjpe_pa_results_second_sbj


@torch.no_grad()
def get_contact_similarity_sampled_obj(
    sampled_obj_v, gt_obj_v, obj_f, gt_sbj_v, sbj_f, contact_threshold=0.05, return_pred_contacts=False
):
    T = sampled_obj_v.shape[0]
    B = 50

    gt_obj_v = torch.from_numpy(gt_obj_v).cuda()
    gt_sbj_v = torch.from_numpy(gt_sbj_v).cuda()

    gt_input_data = [{
        "obj_verts": gt_obj_v[t : min(t + B, T)],
        "sbj_verts": gt_sbj_v[t : min(t + B, T)],
        "contact_threshold": contact_threshold,
    } for t in range(0, T, B)]

    gt_contacts_mask = [contacts_worker(**gt_input_data[t]) for t in range(len(gt_input_data))]
    gt_contacts_mask = np.concatenate(gt_contacts_mask, axis=0)

    sampled_obj_v = torch.from_numpy(sampled_obj_v).cuda()

    pred_input_data = [{
        "obj_verts": sampled_obj_v[t : min(t + B, T)],
        "sbj_verts": gt_sbj_v[t : min(t + B, T)],
        "contact_threshold": contact_threshold,
    } for t in range(0, T, B)]

    pred_contacts_mask = [contacts_worker(**pred_input_data[t]) for t in range(len(pred_input_data))]
    pred_contacts_mask = np.concatenate(pred_contacts_mask, axis=0)

    # Compare gt and sampled contacts
    similarity_mask = gt_contacts_mask == pred_contacts_mask  # T x 6890
    similarity_mask = similarity_mask.mean(1)  # T

    if return_pred_contacts:
        return gt_contacts_mask, similarity_mask, pred_contacts_mask
    else:
        return gt_contacts_mask, similarity_mask


def get_contact_similarity_sampled_sbjobj(
    sampled_obj_v, sampled_sbj_v, gt_obj_v, obj_f, gt_sbj_v, sbj_f, contact_threshold=0.05,
    return_pred_contacts=False
):
    T = sampled_obj_v.shape[0]
    B = 50
    # Get gt contacts
    gt_obj_v = torch.from_numpy(gt_obj_v).cuda()
    gt_sbj_v = torch.from_numpy(gt_sbj_v).cuda()

    gt_input_data = [{
        "obj_verts": gt_obj_v[t : min(t + B, T)],
        "sbj_verts": gt_sbj_v[t : min(t + B, T)],
        "contact_threshold": contact_threshold,
    } for t in range(0, T, B)]

    # gt_contacts_mask = [contacts_worker(**gt_input_data[t]) for t in range(len(gt_input_data))]
    gt_contacts_mask = [contacts_worker(**gt_input_data[t]) for t in range(len(gt_input_data))]
    gt_contacts_mask = np.concatenate(gt_contacts_mask, axis=0)

    # Get sampled contacts
    sampled_obj_v = torch.from_numpy(sampled_obj_v).cuda()
    sampled_sbj_v = torch.from_numpy(sampled_sbj_v).cuda()

    pred_input_data = [{
        "obj_verts": sampled_obj_v[t : min(t + B, T)],
        "sbj_verts": sampled_sbj_v[t : min(t + B, T)],
        "contact_threshold": contact_threshold,
    } for t in range(0, T, B)]

    pred_contacts_mask = [contacts_worker(**pred_input_data[t]) for t in range(len(pred_input_data))]
    pred_contacts_mask = np.concatenate(pred_contacts_mask, axis=0)

    # Compare gt and sampled contacts
    similarity_mask = gt_contacts_mask == pred_contacts_mask  # T x 6890
    similarity_mask = similarity_mask.mean(1)  # T

    if return_pred_contacts:
        return gt_contacts_mask, similarity_mask, pred_contacts_mask
    else:
        return gt_contacts_mask, similarity_mask


def get_obj_v2v(pred_v, gt_v):
    return np.linalg.norm(gt_v - pred_v, axis=-1).mean(-1)


def get_obj_center_distance(pred_v, gt_v):
    return np.linalg.norm(gt_v.mean(axis=1) - pred_v.mean(axis=1), axis=1)


def get_chamfer_distance(gt_mesh, pred_mesh):
    # gt nn
    gt_mesh_points = gt_mesh.sample(10000)
    gt_nn = NearestNeighbors(
        n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric="l2", n_jobs=-1
    ).fit(gt_mesh_points)
    # pred nn
    pred_mesh_points = pred_mesh.sample(10000)
    posed_nn = NearestNeighbors(
        n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric="l2", n_jobs=-1
    ).fit(pred_mesh_points)
    # distances
    pred_to_gt = np.mean(gt_nn.kneighbors(pred_mesh_points)[0])
    gt_to_pred = np.mean(posed_nn.kneighbors(gt_mesh_points)[0])
    return pred_to_gt + gt_to_pred


def get_obj_metrics(
    cfg: ProjectConfig,
    samples_file: Union[str, Path],
    dataset: str
):
    test_datasets = [(dataset, "test")]
    test_hdf5 = get_hdf5_files_for_nn(cfg, test_datasets)
    test_sequences = get_sequences_for_nn(cfg, test_datasets, test_hdf5)

    # load downsample mask
    sbj_contact_indexes = torch.from_numpy(
        np.load(Path(cfg.env.assets_folder) / "smpl_template_decimated_idxs.npy")
    ).long()

    obj_v2v, obj_center_distance = [], []
    obj_contact_mesh, obj_contact_diffused = [], []
    with h5py.File(test_hdf5[dataset], "r") as test_hdf5_dataset, \
            h5py.File(samples_file, "r") as samples_hdf5_dataset:

        for sequence in tqdm(test_sequences[dataset], ncols=80, leave=False):
            sbj, obj, act = sequence

            test_sequence = test_hdf5_dataset[sbj][f"{obj}_{act}"]
            sampled_sequence = samples_hdf5_dataset[sbj][f"{obj}_{act}"]
            T = test_sequence.attrs["T"]
            T_stamps = list(range(T))
            mask = np.ones(T, dtype=bool)

            test_obj_v = np.asarray(test_sequence["obj_v"])[mask]
            test_sbj_v = np.asarray(test_sequence["sbj_v"])[mask]
            sampled_obj_v = np.asarray(sampled_sequence["obj_v"], dtype=np.float32)[mask]
            obj_f = np.asarray(sampled_sequence["obj_f"], dtype=np.int32)
            sbj_v = np.asarray(sampled_sequence["sbj_v"], dtype=np.float32)[mask]
            sbj_f = np.asarray(sampled_sequence["sbj_f"], dtype=np.int32)


            obj_v2v.append(get_obj_v2v(sampled_obj_v, test_obj_v).flatten())
            obj_center_distance.append(get_obj_center_distance(sampled_obj_v, test_obj_v).flatten())

            # compute similarity of mesh-based contacts
            gt_contacts_mask, contacts_similarity = get_contact_similarity_sampled_obj(
                sampled_obj_v, test_obj_v, obj_f,
                test_sbj_v[:, sbj_contact_indexes], sbj_f
            )
            obj_contact_mesh.append(contacts_similarity)

            # compute similarity of directly diffused contacts
            if "sbj_contact" in sampled_sequence.keys():
                diffused_contact_mask = np.asarray(sampled_sequence['sbj_contact'][mask], dtype=np.float32)
                diffused_contact_mask = diffused_contact_mask[:, sbj_contact_indexes]
                similarity_mask = gt_contacts_mask == diffused_contact_mask  # T x 6890
                obj_contact_diffused.append(similarity_mask.mean(1))  # T
    obj_v2v = np.concatenate(obj_v2v, 0)
    obj_center_distance = np.concatenate(obj_center_distance, 0)
    obj_contact_mesh = np.concatenate(obj_contact_mesh, 0) if len(obj_contact_mesh) > 0 else [0]
    obj_contact_diffused = np.concatenate(obj_contact_diffused, 0) if len(obj_contact_diffused) > 0 else [0]

    return obj_v2v, obj_center_distance, obj_contact_mesh, obj_contact_diffused