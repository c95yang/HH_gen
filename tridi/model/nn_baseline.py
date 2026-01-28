# tridi/model/nn_baseline.py
from __future__ import annotations

from dataclasses import is_dataclass
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import h5py
import torch
import torch.nn as nn

try:
    import faiss  # faiss-cpu
except Exception:
    faiss = None

from config.config import ProjectConfig
from tridi.model.base import TriDiModelOutput
from tridi.model.nn.common import get_hdf5_files_for_nn, get_sequences_for_nn
from tridi.data.hh_batch_data import HHBatchData
logger = getLogger(__name__)


# -----------------------------
# helpers: mode parsing
# -----------------------------
def _parse_mode_2bits(mode: Any) -> Tuple[bool, bool]:
    """
    Accepts:
      "10", "01", "11"
      or "sample_10" (endswith 10)
      or list/tuple like ["1","0"]
    """
    s = str(mode)
    # try take the last two 0/1
    import re
    m = re.search(r"([01])([01])$", s)
    if m:
        a, b = m.group(1), m.group(2)
        return (a == "1"), (b == "1")
    # fallback: index
    if len(s) >= 2 and s[0] in "01" and s[1] in "01":
        return (s[0] == "1"), (s[1] == "1")
    raise ValueError(f"Cannot parse sample.mode into 2 bits: {mode}")


def _mode_to_sample_folder_name(mode: Any) -> str:
    sample_sbj, sample_second = _parse_mode_2bits(mode)
    parts = []
    if sample_sbj:
        parts.append("sbj")
    if sample_second:
        parts.append("second_sbj")
    return "_".join(parts) if len(parts) > 0 else "none"


# -----------------------------
# helpers: hdf5 read (supports 6D or R(3x3))
# -----------------------------
def _as_f32(x):
    return np.asarray(x, dtype=np.float32)


def _mat9_to_mat33(x: np.ndarray) -> np.ndarray:
    # (...,9) -> (...,3,3)
    return x.reshape(*x.shape[:-1], 3, 3)


def _mat33_to_6d(R: np.ndarray) -> np.ndarray:
    # (...,3,3) -> (...,6) take first 2 columns
    return R[..., :, :2].reshape(*R.shape[:-2], 6).astype(np.float32)


def _read_global6(g: h5py.Group, prefix: str, T: int) -> np.ndarray:
    k6 = f"{prefix}_global"
    kR = f"{prefix}_smpl_global"

    if k6 in g:
        x = _as_f32(g[k6][:T])
        if x.ndim == 3:  # (T,1,6) -> (T,6)
            x = x.reshape(T, -1)
        return x

    if kR in g:
        x = _as_f32(g[kR][:T])
        # expected (T,1,9) or (T,9)
        if x.ndim == 3:
            x = x.reshape(T, -1)
        R = _mat9_to_mat33(x.reshape(T, 1, 9))  # (T,1,3,3)
        g6 = _mat33_to_6d(R).reshape(T, 6)
        return g6

    raise KeyError(f"Missing {k6} or {kR} in group {g.name}")


def _read_pose6(g: h5py.Group, prefix: str, T: int) -> np.ndarray:
    k6 = f"{prefix}_pose"
    kb = f"{prefix}_smpl_body"
    kl = f"{prefix}_smpl_lh"
    kr = f"{prefix}_smpl_rh"

    if k6 in g:
        x = _as_f32(g[k6][:T])
        # allow (T,51,6) or (T,306)
        if x.ndim == 3:
            x = x.reshape(T, -1)
        return x

    if kb in g and kl in g and kr in g:
        body = _as_f32(g[kb][:T])  # (T,21,9)
        lh   = _as_f32(g[kl][:T])  # (T,15,9)
        rh   = _as_f32(g[kr][:T])  # (T,15,9)

        bodyR = _mat9_to_mat33(body)  # (T,21,3,3)
        lhR   = _mat9_to_mat33(lh)
        rhR   = _mat9_to_mat33(rh)

        body6 = _mat33_to_6d(bodyR)  # (T,21,6)
        lh6   = _mat33_to_6d(lhR)    # (T,15,6)
        rh6   = _mat33_to_6d(rhR)    # (T,15,6)

        pose6 = np.concatenate([body6, lh6, rh6], axis=1)  # (T,51,6)
        return pose6.reshape(T, -1).astype(np.float32)

    raise KeyError(f"Missing {k6} or (smpl_body/lh/rh) in group {g.name}")


def _read_shape(g: h5py.Group, prefix: str, T: int) -> np.ndarray:
    k = f"{prefix}_shape"
    kb = f"{prefix}_smpl_betas"
    if k in g:
        x = _as_f32(g[k][:T])
        return x.reshape(T, -1)
    if kb in g:
        x = _as_f32(g[kb][:T])
        return x.reshape(T, -1)
    raise KeyError(f"Missing {k} or {kb} in group {g.name}")


def _read_transl(g: h5py.Group, prefix: str, T: int) -> np.ndarray:
    k = f"{prefix}_c"
    kt = f"{prefix}_smpl_transl"
    if k in g:
        x = _as_f32(g[k][:T])
        return x.reshape(T, 3)
    if kt in g:
        x = _as_f32(g[kt][:T])
        return x.reshape(T, 3)
    raise KeyError(f"Missing {k} or {kt} in group {g.name}")


def _safe_T(g: h5py.Group, any_key: str) -> int:
    if any_key not in g:
        return 0
    T_data = int(g[any_key].shape[0])
    T_attr = int(g.attrs.get("T", T_data))
    return min(T_data, T_attr)


# -----------------------------
# checkpoint format
# -----------------------------
def is_nn_baseline_checkpoint(path_or_dict) -> bool:
    if isinstance(path_or_dict, (str, Path)):
        ckpt = torch.load(str(path_or_dict), map_location="cpu")
    else:
        ckpt = path_or_dict
    return str(ckpt.get("model_type", "")).lower() in ["nn_baseline", "nnbaseline", "knn_baseline"]


def create_nn_baseline_checkpoint(out_path: str, ref_split="train", top_k=5, feature="pose6"):
    """
    生成一个 baseline checkpoint 文件（里面不放 state_dict，只放超参/缓存位）。
    第一次 sample 时会自动 build index（也可以你自己手动提前 build）。
    """
    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_type": "nn_baseline",
        "nn_baseline": {
            "ref_split": str(ref_split),
            "top_k": int(top_k),
            "feature": str(feature),  # 目前固定用 pose6 (flatten)
            # cache（可空）：按 dataset、方向存 faiss index + target params
            "cache": {}
        }
    }
    torch.save(ckpt, out_path)
    logger.info(f"[NNBaseline] created checkpoint: {out_path}")


# -----------------------------
# NNBaselineModel: behaves like a "trained model"
# -----------------------------
class NNBaselineModel(nn.Module):
    """
    Acts like diffusion model for sampler.py:
      forward(batch, "sample", sample_type=...) -> TriDiModelOutput-like object

    It uses KNN over reference split (default train) to retrieve target person's SMPL params.
    """

    def __init__(self, cfg: ProjectConfig, checkpoint_path: str):
        super().__init__()
        if faiss is None:
            raise RuntimeError("faiss is required for NNBaselineModel. Please `pip install faiss-cpu`.")

        self.cfg = cfg
        self.checkpoint_path = str(checkpoint_path)
        self.is_nn_baseline = True

        ckpt = torch.load(self.checkpoint_path, map_location="cpu")
        if not is_nn_baseline_checkpoint(ckpt):
            raise ValueError(f"Not an nn_baseline checkpoint: {self.checkpoint_path}")

        self.ckpt = ckpt
        nncfg = ckpt["nn_baseline"]
        self.ref_split = str(nncfg.get("ref_split", "train"))
        self.top_k = int(nncfg.get("top_k", 5))
        self.feature = str(nncfg.get("feature", "pose6"))

        self._cache: Dict[str, Any] = nncfg.setdefault("cache", {})  # will be saved back optionally
        self._current_dataset: Optional[str] = None
        self._call_id = 0  # to diversify reps without changing sampler.py

        # mesh_model will be injected by sampler.py, but baseline doesn't require it
        self.mesh_model = None

    # sampler.py will call this
    def set_mesh_model(self, mesh_model):
        self.mesh_model = mesh_model

    # sampler.py expects this method exists on diffusion model
    def split_output(self, output):
        return output

    def set_current_dataset(self, dataset_name: str):
        self._current_dataset = str(dataset_name)

    # ---- build cache ----
    def _build_cache_for_dataset(self, dataset: str):
        """
        Build two directions cache for this dataset:
          cache[dataset]["sbj"]        : cond=second_sbj -> tgt=sbj
          cache[dataset]["second_sbj"] : cond=sbj        -> tgt=second_sbj
        """
        logger.info(f"[NNBaseline] building cache for dataset={dataset}, ref_split={self.ref_split}")

        ref_datasets = [(dataset, self.ref_split)]
        ref_hdf5 = get_hdf5_files_for_nn(self.cfg, ref_datasets)
        ref_sequences = get_sequences_for_nn(self.cfg, ref_datasets, ref_hdf5)[dataset]

        # load reference h5 once
        feats_sbj = []
        feats_second = []
        tgt_sbj = {"global": [], "pose": [], "shape": [], "c": []}
        tgt_second = {"global": [], "pose": [], "shape": [], "c": []}

        with h5py.File(ref_hdf5[dataset], "r") as f:
            for (sbj, obj, act) in ref_sequences:
                g = f[sbj] if (not obj and not act) else f[sbj][f"{obj}_{act}"]

                # Need both persons present
                # We'll use pose keys existence to decide T
                # sbj direction: cond=second_sbj_pose, tgt=sbj_*
                # second direction: cond=sbj_pose, tgt=second_sbj_*
                # Determine T by whichever key exists
                # Use second_sbj_pose/global to define T (more stable)
                try:
                    T = _safe_T(g, "sbj_pose" if "sbj_pose" in g else ("sbj_smpl_body" if "sbj_smpl_body" in g else "sbj_j"))
                except Exception:
                    T = 0
                if T <= 0:
                    continue

                # read cond poses
                try:
                    sbj_pose6 = _read_pose6(g, "sbj", T)
                    second_pose6 = _read_pose6(g, "second_sbj", T)
                except Exception:
                    continue

                # read targets (sbj params)
                try:
                    sbj_g6 = _read_global6(g, "sbj", T)
                    sbj_sh = _read_shape(g, "sbj", T)
                    sbj_c  = _read_transl(g, "sbj", T)

                    second_g6 = _read_global6(g, "second_sbj", T)
                    second_sh = _read_shape(g, "second_sbj", T)
                    second_c  = _read_transl(g, "second_sbj", T)
                except Exception:
                    continue

                # feature = pose-only (flatten), dim should match batch pose6 flatten
                feats_sbj.append(second_pose6)     # cond second -> sample sbj
                feats_second.append(sbj_pose6)     # cond sbj -> sample second

                tgt_sbj["global"].append(sbj_g6)
                tgt_sbj["pose"].append(sbj_pose6)
                tgt_sbj["shape"].append(sbj_sh)
                tgt_sbj["c"].append(sbj_c)

                tgt_second["global"].append(second_g6)
                tgt_second["pose"].append(second_pose6)
                tgt_second["shape"].append(second_sh)
                tgt_second["c"].append(second_c)

        if len(feats_sbj) == 0:
            raise RuntimeError(f"[NNBaseline] empty reference features for dataset={dataset} split={self.ref_split}")

        feats_sbj = np.concatenate(feats_sbj, axis=0).astype(np.float32)
        feats_second = np.concatenate(feats_second, axis=0).astype(np.float32)

        def cat_dict(d):
            return {k: np.concatenate(v, axis=0).astype(np.float32) for k, v in d.items()}

        tgt_sbj = cat_dict(tgt_sbj)
        tgt_second = cat_dict(tgt_second)

        # build faiss indices
        index_sbj = faiss.IndexFlatL2(feats_sbj.shape[1])
        index_sbj.add(feats_sbj)

        index_second = faiss.IndexFlatL2(feats_second.shape[1])
        index_second.add(feats_second)

        # store in cache (in-memory)
        self._cache.setdefault(dataset, {})
        self._cache[dataset]["sbj"] = {
            "index": index_sbj,
            "tgt": tgt_sbj,
        }
        self._cache[dataset]["second_sbj"] = {
            "index": index_second,
            "tgt": tgt_second,
        }

        # optionally persist cache metadata back to checkpoint (index itself is not trivially torch-serializable)
        # If you really want persistence, we can serialize index bytes + arrays, but file may get huge.
        logger.info(f"[NNBaseline] cache ready: feats_sbj={len(feats_sbj)}, feats_second={len(feats_second)}")

    def _ensure_cache(self, dataset: str):
        if dataset not in self._cache or "sbj" not in self._cache[dataset] or "second_sbj" not in self._cache[dataset]:
            self._build_cache_for_dataset(dataset)

    # ---- make output ----
    def _make_output(self, B: int, device: torch.device, **fields):
        """
        Robustly construct TriDiModelOutput even if its __init__ has more fields.
        Unprovided fields filled with None.
        """
        try:
            if is_dataclass(TriDiModelOutput):
                import dataclasses
                names = [f.name for f in dataclasses.fields(TriDiModelOutput) if f.init]
                for n in names:
                    fields.setdefault(n, None)
                return TriDiModelOutput(**fields)
            else:
                import inspect
                sig = inspect.signature(TriDiModelOutput)
                for n in sig.parameters.keys():
                    fields.setdefault(n, None)
                return TriDiModelOutput(**fields)
        except Exception:
            # fallback: return a simple object with attributes
            from types import SimpleNamespace
            return SimpleNamespace(**fields)

    @torch.no_grad()
    def forward(self, batch, phase: str, sample_type=None):
        if phase != "sample":
            raise RuntimeError("NNBaselineModel is only for sampling")
        
        if isinstance(batch, dict):
            batch = HHBatchData(**batch)
        if self._current_dataset is None:
            # best effort: if only one dataset in cfg.run.datasets
            if len(self.cfg.run.datasets) == 1:
                self._current_dataset = str(self.cfg.run.datasets[0])
            else:
                raise RuntimeError("NNBaselineModel needs current dataset name. Call model.set_current_dataset(name).")

        dataset = self._current_dataset
        self._ensure_cache(dataset)

        sbj_pose_cond = getattr(batch, "sbj_pose")
        device = sbj_pose_cond.device
        dtype = sbj_pose_cond.dtype

        self._call_id += 1
        rng = np.random.RandomState((self._call_id * 1000003) % (2**31 - 1))

        sample_sbj, sample_second = _parse_mode_2bits(self.cfg.sample.mode)

        # batch provides GT params in 6D already
        def _pose_flat(x):
            if x is None:
                return None
            if x.ndim == 3:
                return x.reshape(x.shape[0], -1)
            return x
        sbj_pose_in = _pose_flat(getattr(batch, "sbj_pose"))
        second_pose_in = _pose_flat(getattr(batch, "second_sbj_pose"))
        B = int(sbj_pose_in.shape[0])
        # start from GT
        sbj_pose, second_pose = sbj_pose_in, second_pose_in
        sbj_shape  = _pose_flat(getattr(batch, "sbj_shape"))
        second_shape = _pose_flat(getattr(batch, "second_sbj_shape"))
        sbj_global = _pose_flat(getattr(batch, "sbj_global"))
        second_global = _pose_flat(getattr(batch, "second_sbj_global"))
        sbj_c = getattr(batch, "sbj_c")
        second_c = getattr(batch, "second_sbj_c")

        # ---- sample sbj (cond = second_sbj pose) ----
        if sample_sbj and sample_second:
            N_ref = int(self._cache[dataset]["N_ref"])
            chosen = rng.randint(0, N_ref, size=(B,))

            tgt_sbj = self._cache[dataset]["sbj"]["tgt"]
            tgt_second = self._cache[dataset]["second_sbj"]["tgt"]

            sbj_global = torch.from_numpy(tgt_sbj["global"][chosen]).to(device)
            sbj_pose   = torch.from_numpy(tgt_sbj["pose"][chosen]).to(device)
            sbj_shape  = torch.from_numpy(tgt_sbj["shape"][chosen]).to(device)
            sbj_c      = torch.from_numpy(tgt_sbj["c"][chosen]).to(device)

            second_global = torch.from_numpy(tgt_second["global"][chosen]).to(device)
            second_pose   = torch.from_numpy(tgt_second["pose"][chosen]).to(device)
            second_shape  = torch.from_numpy(tgt_second["shape"][chosen]).to(device)
            second_c      = torch.from_numpy(tgt_second["c"][chosen]).to(device)

        else:
            # --------- 10: cond second -> sample sbj ----------
            if sample_sbj:
                cond = second_pose_in.detach().float().cpu().numpy().astype(np.float32)
                cond = np.ascontiguousarray(cond)
                idx_pack = self._cache[dataset]["sbj"]
                index = idx_pack["index"]
                tgt = idx_pack["tgt"]

                k = max(1, int(self.top_k))
                _, nn = index.search(cond, k)
                if k == 1:
                    chosen = nn[:, 0]
                else:
                    pick = rng.randint(0, k, size=(B,))
                    chosen = nn[np.arange(B), pick]

                sbj_global = torch.from_numpy(tgt["global"][chosen]).to(device)
                sbj_pose   = torch.from_numpy(tgt["pose"][chosen]).to(device)
                sbj_shape  = torch.from_numpy(tgt["shape"][chosen]).to(device)
                sbj_c      = torch.from_numpy(tgt["c"][chosen]).to(device)

            # --------- 01: cond sbj -> sample second ----------
            if sample_second:
                cond = sbj_pose_in.detach().float().cpu().numpy().astype(np.float32)
                cond = np.ascontiguousarray(cond)
                idx_pack = self._cache[dataset]["second_sbj"]
                index = idx_pack["index"]
                tgt = idx_pack["tgt"]

                k = max(1, int(self.top_k))
                _, nn = index.search(cond, k)
                if k == 1:
                    chosen = nn[:, 0]
                else:
                    pick = rng.randint(0, k, size=(B,))
                    chosen = nn[np.arange(B), pick]

                second_global = torch.from_numpy(tgt["global"][chosen]).to(device)
                second_pose   = torch.from_numpy(tgt["pose"][chosen]).to(device)
                second_shape  = torch.from_numpy(tgt["shape"][chosen]).to(device)
                second_c      = torch.from_numpy(tgt["c"][chosen]).to(device)


        # ensure shapes: global (B,6), pose (B,306)
        I6 = torch.tensor([1,0,0,0,1,0], device=device, dtype=dtype).view(1,6).repeat(B,1)
        sbj_global = I6
        second_global = I6
        sbj_c = torch.zeros((B,3), device=device, dtype=dtype)
        second_c = torch.zeros((B,3), device=device, dtype=dtype)

        out = self._make_output(
            B=B, device=device,
            sbj_global=sbj_global, sbj_pose=_pose_flat(sbj_pose), sbj_shape=sbj_shape, sbj_c=sbj_c,
            second_sbj_global=second_global, second_sbj_pose=_pose_flat(second_pose), second_sbj_shape=second_shape, second_sbj_c=second_c,
        )
        return out

