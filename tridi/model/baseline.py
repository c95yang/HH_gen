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

try:
    import faiss
except ImportError:
    faiss = None

from tridi.data.hh_batch_data import HHBatchData
from config.config import ProjectConfig
from tridi.model.base import TriDiModelOutput
from tridi.utils.geometry import matrix_to_rotation_6d

logger = getLogger(__name__)


class NearestNeighborBaselineModel:
    """
    NN Baseline Model - generates another person poses by retrieval.
    
    Interface compatible with TriDi for use in Sampler.
    """
    
    def __init__(self, cfg: ProjectConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Store training data features and poses
        self.train_data_cache: Dict[str, Dict] = {}
        self._load_train_data()
        
        logger.info("[Baseline] Initialized Nearest Neighbor Model")
    
    def _load_train_data(self):
        """Pre-load training data for all datasets.

        We cache per-frame SMPL parameters and a simple feature (flattened betas + global + pose + transl)
        to perform nearest-neighbor retrieval at sampling time.
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

            train_feats = []
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
                        "sbj_smpl_betas", "sbj_smpl_global", "sbj_smpl_body", "sbj_smpl_lh", "sbj_smpl_rh", "sbj_smpl_transl",
                        "second_sbj_smpl_betas", "second_sbj_smpl_global", "second_sbj_smpl_body", "second_sbj_smpl_lh", "second_sbj_smpl_rh", "second_sbj_smpl_transl",
                    ]
                    if not all(k in g for k in required_keys):
                        logger.debug(f"[Baseline] Missing keys in seq {seq_name}, skip.")
                        continue

                    T = int(g.attrs.get("T", g["sbj_smpl_betas"].shape[0]))

                    for t in range(T):
                        sbj_params, second_params = self._extract_frame_params(g, t)
                        feat = self._build_feature_from_arrays(sbj_params)

                        train_feats.append(feat)
                        sbj_shape_list.append(sbj_params["shape"])
                        sbj_global_list.append(sbj_params["global"])
                        sbj_pose_list.append(sbj_params["pose"])
                        sbj_c_list.append(sbj_params["c"])

                        second_shape_list.append(second_params["shape"])
                        second_global_list.append(second_params["global"])
                        second_pose_list.append(second_params["pose"])
                        second_c_list.append(second_params["c"])

            if len(train_feats) == 0:
                logger.warning(f"[Baseline] No training data found for {dataset}")
                continue

            sbj_feats_arr = np.stack(train_feats, axis=0).astype(np.float32)
            cache = {
                "sbj_feats": sbj_feats_arr,
                "sbj_shape": np.stack(sbj_shape_list, axis=0).astype(np.float32),
                "sbj_global": np.stack(sbj_global_list, axis=0).astype(np.float32),
                "sbj_pose": np.stack(sbj_pose_list, axis=0).astype(np.float32),
                "sbj_c": np.stack(sbj_c_list, axis=0).astype(np.float32),
                "second_sbj_shape": np.stack(second_shape_list, axis=0).astype(np.float32),
                "second_sbj_global": np.stack(second_global_list, axis=0).astype(np.float32),
                "second_sbj_pose": np.stack(second_pose_list, axis=0).astype(np.float32),
                "second_sbj_c": np.stack(second_c_list, axis=0).astype(np.float32),
                "faiss_index": None,
            }

            if faiss is not None:
                index = faiss.IndexFlatL2(sbj_feats_arr.shape[1])
                index.add(sbj_feats_arr)
                cache["faiss_index"] = index

            self.train_data_cache[dataset] = cache
            logger.info(f"[Baseline] Cached {len(train_feats)} frames for {dataset}")

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

        second_shape = np.asarray(group["second_sbj_smpl_betas"][t], dtype=np.float32).reshape(-1)
        second_c = np.asarray(group["second_sbj_smpl_transl"][t], dtype=np.float32).reshape(-1)

        second_global_mat = np.asarray(group["second_sbj_smpl_global"][t], dtype=np.float32).reshape(1, 3, 3)
        second_body = np.asarray(group["second_sbj_smpl_body"][t], dtype=np.float32)
        second_lh = np.asarray(group["second_sbj_smpl_lh"][t], dtype=np.float32)
        second_rh = np.asarray(group["second_sbj_smpl_rh"][t], dtype=np.float32)
        second_pose_mats = np.concatenate([second_body, second_lh, second_rh], axis=0).reshape(-1, 3, 3)

        second_global = matrix_to_rotation_6d(torch.from_numpy(second_global_mat)).reshape(-1).numpy()
        second_pose = matrix_to_rotation_6d(torch.from_numpy(second_pose_mats)).reshape(-1).numpy()

        sbj_params = {
            "shape": sbj_shape,
            "global": sbj_global,
            "pose": sbj_pose,
            "c": sbj_c,
        }
        second_params = {
            "shape": second_shape,
            "global": second_global,
            "pose": second_pose,
            "c": second_c,
        }
        return sbj_params, second_params

    @staticmethod
    def _build_feature_from_batch(batch: HHBatchData, idx: int) -> np.ndarray:
        shape = batch.sbj_shape[idx].cpu().numpy()
        global_rot = batch.sbj_global[idx].cpu().numpy()
        pose = batch.sbj_pose[idx].cpu().numpy()
        transl = batch.sbj_c[idx].cpu().numpy()
        feat = np.concatenate([shape, global_rot, pose, transl], axis=0)
        return feat
    
    @staticmethod
    def _build_feature_from_arrays(params: Dict) -> np.ndarray:
        shape = params["shape"]
        global_rot = params["global"]
        pose = params["pose"]
        transl = params["c"]
        feat = np.concatenate([shape, global_rot, pose, transl], axis=0)
        return feat
    
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
    def _search_neighbors(cache: Dict, feats: np.ndarray) -> np.ndarray:
        if cache.get("faiss_index") is not None:
            _, idx = cache["faiss_index"].search(feats, 1)
            return idx[:, 0]

        dists = np.linalg.norm(cache["sbj_feats"][None] - feats[:, None, :], axis=2)
        return np.argmin(dists, axis=1).astype(np.int64)

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
    def forward_sample(self, sample_mode: Optional[Tuple] = None, batch: Optional[HHBatchData] = None):
        if batch is None:
            raise ValueError("Batch is required for NN retrieval baseline")

        if isinstance(batch, dict):
            batch = HHBatchData(**batch)
        sample_mode = tuple(str(m) for m in sample_mode)

        B = batch.batch_size()
        cache = self._get_cache_for_batch(batch)

        if cache is None:
            logger.warning("[Baseline] No cache available, returning GT batch")
            return self._batch_to_output(batch)

        feats = np.stack([self._build_feature_from_batch(batch, i) for i in range(B)], axis=0)
        nn_idx = self._search_neighbors(cache, feats)

        def pick(batch_attr: str, cache_key: str, mode_flag: str):
            if mode_flag == "1":
                return torch.from_numpy(cache[cache_key][nn_idx]).to(self.device)
            value = getattr(batch, batch_attr, None)
            return value.to(self.device) if value is not None else None

        sbj_shape = pick("sbj_shape", "sbj_shape", sample_mode[0])
        sbj_global = pick("sbj_global", "sbj_global", sample_mode[0])
        sbj_pose = pick("sbj_pose", "sbj_pose", sample_mode[0])
        sbj_c = pick("sbj_c", "sbj_c", sample_mode[0])

        second_shape = pick("second_sbj_shape", "second_sbj_shape", sample_mode[1])
        second_global = pick("second_sbj_global", "second_sbj_global", sample_mode[1])
        second_pose = pick("second_sbj_pose", "second_sbj_pose", sample_mode[1])
        second_c = pick("second_sbj_c", "second_sbj_c", sample_mode[1])

        return TriDiModelOutput(
            sbj_shape=sbj_shape,
            sbj_global=sbj_global,
            sbj_pose=sbj_pose,
            sbj_c=sbj_c,
            second_sbj_shape=second_shape,
            second_sbj_global=second_global,
            second_sbj_pose=second_pose,
            second_sbj_c=second_c,
        )

    def forward(self, batch: HHBatchData, mode='sample', sample_type: Optional[Tuple] = None, **kwargs):
        if mode != 'sample':
            raise NotImplementedError('NearestNeighborBaselineModel supports only sample mode')
        return self.forward_sample(sample_type, batch)

    def __call__(self, batch: HHBatchData, mode='sample', sample_type: Optional[Tuple] = None, **kwargs):
        return self.forward(batch, mode=mode, sample_type=sample_type, **kwargs)
