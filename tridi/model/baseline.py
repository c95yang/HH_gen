"""
Nearest Neighbor Baseline Model - parallel implementation to TriDi.

This model acts as a standalone baseline that can generate poses by:
1. Finding nearest neighbors in training data
2. Returning the paired person's pose from training

Compatible with Sampler for both meshes and hdf5 output.
"""

from logging import getLogger
from pathlib import Path
from typing import Optional, Dict, Tuple
import json

import numpy as np
import h5py
import torch

from tridi.model.wrappers.mesh import MeshModel

try:
    import faiss
except Exception:  # pragma: no cover - optional dependency
    faiss = None

from tridi.data.hh_batch_data import HHBatchData
from config.config import ProjectConfig
from tridi.model.base import TriDiModelOutput
from tridi.utils.geometry import matrix_to_rotation_6d, rotation_6d_to_matrix

logger = getLogger(__name__)


def _adapt_retrieved_global_to_input(
    input_global_6d: np.ndarray,  # (B, 6) or (6,)
    input_transl: np.ndarray,     # (B, 3) or (3,)
    retrieved_global_6d: np.ndarray,
    retrieved_transl: np.ndarray,
    anchor_global_6d: np.ndarray,  # the input person's counterpart in retrieved frame
    anchor_transl: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Adapt retrieved person's global orient and transl to input scene using relative relationship.

    Uses the relative offset (d, R_rel) from the retrieved training frame and applies it
    to the input person's position in the test scene.
    """
    def to_mat(g6):
        g6 = np.asarray(g6, dtype=np.float32)
        if g6.ndim == 1:
            g6 = g6.reshape(1, 6)
        return rotation_6d_to_matrix(torch.from_numpy(g6)).numpy()

    def to_6d(R):
        return matrix_to_rotation_6d(torch.from_numpy(R)).numpy()

    R_input = to_mat(input_global_6d)       # (B, 3, 3)
    R_ret = to_mat(retrieved_global_6d)
    R_anchor = to_mat(anchor_global_6d)

    t_input = np.asarray(input_transl, dtype=np.float32)
    t_ret = np.asarray(retrieved_transl, dtype=np.float32)
    t_anchor = np.asarray(anchor_transl, dtype=np.float32)
    if t_input.ndim == 1:
        t_input = t_input.reshape(1, 3)
        t_ret = t_ret.reshape(1, 3)
        t_anchor = t_anchor.reshape(1, 3)

    # Relative: d = t_ret - t_anchor (world), d_local = R_anchor^T @ d
    d = t_ret - t_anchor
    d_local = np.einsum("bij,bj->bi", R_anchor.transpose(0, 2, 1), d)
    R_rel = np.einsum("bij,bjk->bik", R_ret, R_anchor.transpose(0, 2, 1))

    # Apply to input: t_new = t_input + R_input @ d_local, R_new = R_rel @ R_input
    t_new = t_input + np.einsum("bij,bj->bi", R_input, d_local)
    R_new = np.einsum("bij,bjk->bik", R_rel, R_input)

    global_6d_new = to_6d(R_new).reshape(-1, 6)
    return global_6d_new, t_new


class NNBaseline:
    """
    NN Baseline Model - generates another person poses by retrieval.
    Bidirectional retrieval of two persons' poses given one person's input.
    
    Interface compatible with TriDi for use in Sampler.
    """
    
    def __init__(self, cfg: ProjectConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_mode = self.cfg.sample.mode
        self.feature_type = "pose_shape"  # Options: "full", "pose_shape", "pose_only", "joints_only"
        
        # Store training data features and poses
        self.train_data_cache: Dict[str, Dict] = {}
        self._load_train_data()
        
        logger.info("[Baseline] Initialized Nearest Neighbor Model")

    def _load_train_data(self):
        """Pre-load training data for all datasets.

        We cache per-frame SMPL parameters and a simple feature to perform nearest-neighbor retrieval at sampling time.
        
        Feature types:
        - "full": concatenated [shape, global_rot, pose, transl]
        - "pose_only": only pose (body + hands)
        - "joints": joints positions centered on root, flattened (requires sbj_j and second_sbj_j in hdf5)
        """
        for dataset in self.cfg.run.datasets:
            dataset_cfg = getattr(self.cfg, dataset)

            train_split_file = dataset_cfg.train_split_file
            fps_train = int(dataset_cfg.fps_train)
            h5_path = Path(dataset_cfg.root) / f"dataset_train_{fps_train}fps.hdf5"

            try:
                with open(train_split_file, "r") as f:
                    train_seqs = json.load(f)
            except FileNotFoundError:
                logger.warning(f"[Baseline] Train split file not found: {train_split_file}")
                continue

            sbj_feats_list = []
            second_sbj_feats_list = []

            sbj_shape_list = []
            sbj_global_list = []
            sbj_pose_list = []
            sbj_c_list = []

            second_shape_list = []
            second_global_list = []
            second_pose_list = []
            second_c_list = []

            if not h5_path.exists():
                logger.warning(f"[Baseline] Train hdf5 not found: {h5_path}")
                continue

            with h5py.File(str(h5_path), "r") as f:
                for seq_name in train_seqs:
                    if seq_name not in f:
                        continue

                    g = f[seq_name]
                    required_keys = [
                        "sbj_j","sbj_smpl_betas", "sbj_smpl_global", "sbj_smpl_body", "sbj_smpl_lh", "sbj_smpl_rh", "sbj_smpl_transl",
                        "second_sbj_j","second_sbj_smpl_betas", "second_sbj_smpl_global", "second_sbj_smpl_body", "second_sbj_smpl_lh", "second_sbj_smpl_rh", "second_sbj_smpl_transl",
                    ]
                    if not all(k in g for k in required_keys):
                        logger.debug(f"[Baseline] Missing keys in seq {seq_name}, skip.")
                        continue

                    T = int(g.attrs.get("T", g["sbj_smpl_betas"].shape[0]))

                    for t in range(T):
                        sbj_params, second_params = self._extract_frame_params(g, t)

                        sbj_feat = self._build_feature_from_arrays(sbj_params, self.feature_type)
                        second_sbj_feat = self._build_feature_from_arrays(second_params, self.feature_type)

                        sbj_feats_list.append(sbj_feat)
                        second_sbj_feats_list.append(second_sbj_feat)
                        
                        sbj_shape_list.append(sbj_params["shape"])
                        sbj_global_list.append(sbj_params["global"])
                        sbj_pose_list.append(sbj_params["pose"])
                        sbj_c_list.append(sbj_params["c"])

                        second_shape_list.append(second_params["shape"])
                        second_global_list.append(second_params["global"])
                        second_pose_list.append(second_params["pose"])
                        second_c_list.append(second_params["c"])

            cache = {
                "sbj_feats": np.stack(sbj_feats_list, axis=0).astype(np.float32),
                "sbj_shape": np.stack(sbj_shape_list, axis=0).astype(np.float32),
                "sbj_global": np.stack(sbj_global_list, axis=0).astype(np.float32),
                "sbj_pose": np.stack(sbj_pose_list, axis=0).astype(np.float32),
                "sbj_c": np.stack(sbj_c_list, axis=0).astype(np.float32),
                "second_sbj_feats": np.stack(second_sbj_feats_list, axis=0).astype(np.float32),
                "second_sbj_shape": np.stack(second_shape_list, axis=0).astype(np.float32),
                "second_sbj_global": np.stack(second_global_list, axis=0).astype(np.float32),
                "second_sbj_pose": np.stack(second_pose_list, axis=0).astype(np.float32),
                "second_sbj_c": np.stack(second_c_list, axis=0).astype(np.float32),
            }
            
            # Optional: build faiss indices for fast NN
            if faiss is not None:
                cache["sbj_index"] = self._build_faiss_index(cache["sbj_feats"])
                cache["second_sbj_index"] = self._build_faiss_index(cache["second_sbj_feats"])
            else:
                logger.info("[Baseline] Faiss not available, using brute-force NN search.")

            self.train_data_cache[dataset] = cache

    @staticmethod
    def _build_faiss_index(features: np.ndarray):
        index = faiss.IndexFlatL2(features.shape[1])
        index.add(features.astype(np.float32))
        return index

    def _extract_frame_params(self, group, t: int):
        sbj_shape = np.asarray(group["sbj_smpl_betas"][t], dtype=np.float32).reshape(-1)
        sbj_c = np.asarray(group["sbj_smpl_transl"][t], dtype=np.float32).reshape(-1)

        sbj_global_mat = np.asarray(group["sbj_smpl_global"][t], dtype=np.float32).reshape(1, 3, 3)
        sbj_body = np.asarray(group["sbj_smpl_body"][t], dtype=np.float32)
        sbj_lh = np.asarray(group["sbj_smpl_lh"][t], dtype=np.float32)
        sbj_rh = np.asarray(group["sbj_smpl_rh"][t], dtype=np.float32)
        sbj_pose_mats = np.concatenate([sbj_body, sbj_lh, sbj_rh], axis=0).reshape(-1, 3, 3)

        sbj_global = matrix_to_rotation_6d(torch.from_numpy(sbj_global_mat)).reshape(-1).numpy()
        sbj_pose = matrix_to_rotation_6d(torch.from_numpy(sbj_pose_mats)).reshape(-1).numpy()

        sbj_joints = np.asarray(group["sbj_j"][t], dtype=np.float32)  # (J, 3)


        second_shape = np.asarray(group["second_sbj_smpl_betas"][t], dtype=np.float32).reshape(-1)
        second_c = np.asarray(group["second_sbj_smpl_transl"][t], dtype=np.float32).reshape(-1)

        second_global_mat = np.asarray(group["second_sbj_smpl_global"][t], dtype=np.float32).reshape(1, 3, 3)
        second_body = np.asarray(group["second_sbj_smpl_body"][t], dtype=np.float32)
        second_lh = np.asarray(group["second_sbj_smpl_lh"][t], dtype=np.float32)
        second_rh = np.asarray(group["second_sbj_smpl_rh"][t], dtype=np.float32)
        second_pose_mats = np.concatenate([second_body, second_lh, second_rh], axis=0).reshape(-1, 3, 3)

        second_global = matrix_to_rotation_6d(torch.from_numpy(second_global_mat)).reshape(-1).numpy()
        second_pose = matrix_to_rotation_6d(torch.from_numpy(second_pose_mats)).reshape(-1).numpy()

        second_sbj_joints = np.asarray(group["second_sbj_j"][t], dtype=np.float32)  # (J, 3)

        sbj_params = {
            "shape": sbj_shape,
            "global": sbj_global,
            "pose": sbj_pose,
            "c": sbj_c,
            "joints": sbj_joints,
        }
        second_params = {
            "shape": second_shape,
            "global": second_global,
            "pose": second_pose,
            "c": second_c,
            "joints": second_sbj_joints,
        }
        
        
        return sbj_params, second_params
    
    @staticmethod
    def split_output(output):
        if isinstance(output, TriDiModelOutput):
            return output
        return TriDiModelOutput(
            sbj_shape=output[:, :10],
            sbj_global=output[:, 10:16],
            sbj_pose=output[:, 16:16 + 51 * 6],
            sbj_c=output[:, 16 + 51 * 6:16 + 51 * 6 + 3],
            second_sbj_shape=output[:, 16 + 51 * 6 + 3:16 + 51 * 6 + 3 + 10],
            second_sbj_global=output[:, 16 + 51 * 6 + 3 + 10:16 + 51 * 6 + 3 + 10 + 6],
            second_sbj_pose=output[:, 16 + 51 * 6 + 3 + 10 + 6:16 + 2 * (51 * 6) + 3 + 10 + 6],
            second_sbj_c=output[:, -3:],
        )
    
    @staticmethod
    def _build_feature_from_batch(batch: HHBatchData, idx: int, feature_type: str) -> np.ndarray:
        shape = batch.sbj_shape[idx].cpu().numpy()
        global_rot = batch.sbj_global[idx].cpu().numpy()
        pose = batch.sbj_pose[idx].cpu().numpy()
        transl = batch.sbj_c[idx].cpu().numpy()

        second_shape = batch.second_sbj_shape[idx].cpu().numpy()
        second_global_rot = batch.second_sbj_global[idx].cpu().numpy()
        second_pose = batch.second_sbj_pose[idx].cpu().numpy()
        second_transl = batch.second_sbj_c[idx].cpu().numpy()

        if feature_type == "full":
            sbj_feat = np.concatenate([shape, global_rot, pose, transl], axis=0)
            second_sbj_feat = np.concatenate([second_shape, second_global_rot, second_pose, second_transl], axis=0)
        if feature_type == "pose_shape":
            sbj_feat = np.concatenate([shape, pose], axis=0)
            second_sbj_feat = np.concatenate([second_shape, second_pose], axis=0)
        elif feature_type == "pose_only":
            sbj_feat = pose
            second_sbj_feat = second_pose
        elif feature_type == "joints_only":
            # Get joints
            sbj_joints = batch.sbj_j[idx].cpu().numpy()  # (J, 3)
            joints_centered = sbj_joints - sbj_joints[0:1]  # (J, 3)
            sbj_feat = joints_centered[1:].reshape
            second_sbj_joints = batch.second_sbj_j[idx].cpu().numpy()  # (J, 3)
            second_joints_centered = second_sbj_joints - second_sbj_joints[0:1]  # (J, 3)
            second_sbj_feat = second_joints_centered[1:].reshape(-1)
        else:
            raise ValueError(f"Invalid feature type {feature_type} for NNBaseline")

        return sbj_feat, second_sbj_feat
    
    @staticmethod
    def _build_feature_from_arrays(params: Dict, feature_type: str) -> np.ndarray:
        shape = params["shape"]
        global_rot = params["global"]
        pose = params["pose"]
        transl = params["c"]
        # feat = np.concatenate([shape, global_rot, pose, transl], axis=0)
        if feature_type == "full":
            feat = np.concatenate([shape, global_rot, pose, transl], axis=0)
        elif feature_type == "pose_shape":
            feat =  np.concatenate([shape, pose], axis=0)
        elif feature_type == "pose_only":
            feat = pose
        elif feature_type == "joints_only":
            joints = params["joints"]  # (J, 3)
            joints_centered = joints - joints[0:1]  # (J, 3)
            feat = joints_centered[1:].reshape(-1)
        else:
            raise ValueError(f"Invalid feature type {feature_type} for NNBaseline")
        return feat
    
    def to(self, device):
        """Move model to device (no-op for baseline)."""
        self.device = device
        return self
    
    def eval(self):
        """Set to eval mode (no-op for baseline)."""
        return self
    
    def set_mesh_model(self, mesh_model):
        """Set mesh model (compatible with Sampler)."""
        self.mesh_model = mesh_model
    
    def _get_cache_for_batch(self, batch: HHBatchData) -> Optional[Dict]:
        dataset_name = None
        if hasattr(batch, "meta") and isinstance(batch.meta, dict):
            dataset_name = batch.meta.get("dataset", None)

        if dataset_name not in self.train_data_cache:
            dataset_name = next(iter(self.train_data_cache.keys()), None)

        return self.train_data_cache.get(dataset_name, None)

    @staticmethod
    def _search_neighbors(cache: Dict, feats: np.ndarray, mode: str) -> np.ndarray:
        if mode == "sbj":
            train_feats = cache["sbj_feats"]
            index = cache.get("sbj_index", None)
        elif mode == "second_sbj":
            train_feats = cache["second_sbj_feats"]
            index = cache.get("second_sbj_index", None)
        else:
            raise ValueError(f"Invalid mode {mode} for neighbor search")    
        if faiss is not None and index is not None:
            _, nn_idx = index.search(feats.astype(np.float32), k=1)
            return nn_idx[:, 0]
        dists = np.linalg.norm(feats[:, None, :] - train_feats[None, :, :], axis=-1)
        nn_idx = np.argmin(dists, axis=-1)
        return nn_idx

    def _batch_to_output(self, batch: HHBatchData) -> TriDiModelOutput:
        to_device = lambda t: t.to(self.device) if t is not None else None
        return TriDiModelOutput(
            sbj_shape=to_device(batch.sbj_shape),
            sbj_global=to_device(batch.sbj_global),
            sbj_pose=to_device(batch.sbj_pose),
            sbj_c=to_device(batch.sbj_c),
            second_sbj_shape=to_device(batch.second_sbj_shape),
            second_sbj_global=to_device(batch.second_sbj_global),
            second_sbj_pose=to_device(batch.second_sbj_pose),
            second_sbj_c=to_device(batch.second_sbj_c),
        )

    @torch.no_grad()
    def forward_sample(self, sample_mode: Optional[Tuple], batch: Optional[HHBatchData]):
        if isinstance(batch, dict):
            batch = HHBatchData(**batch)

        B = batch.batch_size()
        cache = self._get_cache_for_batch(batch)

        if cache is None:
            logger.warning("[Baseline] No cache available, returning GT batch")
            out = self._batch_to_output(batch)
            return out, out  # (output, raw_output) for unpacking

        sbj_feats, second_feats = zip(*[self._build_feature_from_batch(batch, i, self.feature_type) for i in range(B)])
        sbj_feats = np.stack(sbj_feats, axis=0)
        second_sbj_feats = np.stack(second_feats, axis=0)

        if sample_mode[0] == "1" and sample_mode[1] == "0":
            # second_sbj is given, only save and don't touch it. Sampe sbj from nearest.
            second_shape = batch.second_sbj_shape.to(self.device)
            second_pose = batch.second_sbj_pose.to(self.device)
            second_global = batch.second_sbj_global.to(self.device)
            second_c = batch.second_sbj_c.to(self.device)

            nn_idx = self._search_neighbors(cache, second_sbj_feats, mode="second_sbj")

            retrieved_sbj_shape = torch.from_numpy(cache["sbj_shape"][nn_idx]).to(self.device)
            retrieved_sbj_pose = torch.from_numpy(cache["sbj_pose"][nn_idx]).to(self.device)
            retrieved_sbj_global_np, retrieved_sbj_c_np = _adapt_retrieved_global_to_input(
                input_global_6d=batch.second_sbj_global.cpu().numpy(),
                input_transl=batch.second_sbj_c.cpu().numpy(),
                retrieved_global_6d=cache["sbj_global"][nn_idx],
                retrieved_transl=cache["sbj_c"][nn_idx],
                anchor_global_6d=cache["second_sbj_global"][nn_idx],
                anchor_transl=cache["second_sbj_c"][nn_idx],
            )
            retrieved_sbj_global = torch.from_numpy(retrieved_sbj_global_np).to(self.device)
            retrieved_sbj_c = torch.from_numpy(retrieved_sbj_c_np).to(self.device)

            output = TriDiModelOutput(
                sbj_shape=retrieved_sbj_shape,
                sbj_global=retrieved_sbj_global,
                sbj_pose=retrieved_sbj_pose,
                sbj_c=retrieved_sbj_c,
                second_sbj_shape=second_shape,
                second_sbj_global=second_global,
                second_sbj_pose=second_pose,
                second_sbj_c=second_c,
            )

            retrieved_output = TriDiModelOutput(
                sbj_shape=torch.from_numpy(cache["sbj_shape"][nn_idx]).to(self.device),
                sbj_global=torch.from_numpy(cache["sbj_global"][nn_idx]).to(self.device),
                sbj_pose=torch.from_numpy(cache["sbj_pose"][nn_idx]).to(self.device),
                sbj_c=torch.from_numpy(cache["sbj_c"][nn_idx]).to(self.device),
                second_sbj_shape=torch.from_numpy(cache["second_sbj_shape"][nn_idx]).to(self.device),
                second_sbj_global=torch.from_numpy(cache["second_sbj_global"][nn_idx]).to(self.device),
                second_sbj_pose=torch.from_numpy(cache["second_sbj_pose"][nn_idx]).to(self.device),
                second_sbj_c=torch.from_numpy(cache["second_sbj_c"][nn_idx]).to(self.device),
            )

        elif sample_mode[0] == "0" and sample_mode[1] == "1":
            # sbj is given, only save and don't touch it. Sample second_sbj from nearest.
            sbj_shape = batch.sbj_shape.to(self.device)
            sbj_pose = batch.sbj_pose.to(self.device)
            sbj_global = batch.sbj_global.to(self.device)
            sbj_c = batch.sbj_c.to(self.device)

            nn_idx = self._search_neighbors(cache, sbj_feats, mode="sbj")

            retrieved_second_sbj_shape = torch.from_numpy(cache["second_sbj_shape"][nn_idx]).to(self.device)
            retrieved_second_sbj_pose = torch.from_numpy(cache["second_sbj_pose"][nn_idx]).to(self.device)
            retrieved_second_sbj_global_np, retrieved_second_sbj_c_np = _adapt_retrieved_global_to_input(
                input_global_6d=batch.sbj_global.cpu().numpy(),
                input_transl=batch.sbj_c.cpu().numpy(),
                retrieved_global_6d=cache["second_sbj_global"][nn_idx],
                retrieved_transl=cache["second_sbj_c"][nn_idx],
                anchor_global_6d=cache["sbj_global"][nn_idx],
                anchor_transl=cache["sbj_c"][nn_idx],
            )
            retrieved_second_sbj_global = torch.from_numpy(retrieved_second_sbj_global_np).to(self.device)
            retrieved_second_sbj_c = torch.from_numpy(retrieved_second_sbj_c_np).to(self.device)

            output = TriDiModelOutput(
                sbj_shape=sbj_shape,
                sbj_global=sbj_global,
                sbj_pose=sbj_pose,
                sbj_c=sbj_c,
                second_sbj_shape=retrieved_second_sbj_shape,
                second_sbj_global=retrieved_second_sbj_global,
                second_sbj_pose=retrieved_second_sbj_pose,
                second_sbj_c=retrieved_second_sbj_c,
            )

            retrieved_output = TriDiModelOutput(
                sbj_shape=torch.from_numpy(cache["sbj_shape"][nn_idx]).to(self.device),
                sbj_global=torch.from_numpy(cache["sbj_global"][nn_idx]).to(self.device),
                sbj_pose=torch.from_numpy(cache["sbj_pose"][nn_idx]).to(self.device),
                sbj_c=torch.from_numpy(cache["sbj_c"][nn_idx]).to(self.device),
                second_sbj_shape=torch.from_numpy(cache["second_sbj_shape"][nn_idx]).to(self.device),
                second_sbj_global=torch.from_numpy(cache["second_sbj_global"][nn_idx]).to(self.device),
                second_sbj_pose=torch.from_numpy(cache["second_sbj_pose"][nn_idx]).to(self.device),
                second_sbj_c=torch.from_numpy(cache["second_sbj_c"][nn_idx]).to(self.device),
            )

        else:
            raise ValueError(f"Invalid sample_mode {sample_mode}")
        
        return output, retrieved_output

    def forward(self, batch: HHBatchData, mode='sample', sample_type: Optional[Tuple] = None, **kwargs):
        if mode != 'sample':
            raise NotImplementedError('NNBaseline supports only sample mode')
        return self.forward_sample(sample_type, batch)

    def __call__(self, batch: HHBatchData, mode='sample', sample_type: Optional[Tuple] = None, **kwargs):
        return self.forward(batch, mode=mode, sample_type=sample_type, **kwargs)
