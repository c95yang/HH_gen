#!/usr/bin/env python3
"""
how to use:
1) make sure cases.json is generated (see select_cases_from_hdf5.py)
2) export ply:
python tridi/tools/vis/export_cases_to_ply.py \
  --samples_dir /media/uv/Data/workspace/HH_gen/experiments/021_chi3d/artifacts/step_20000_samples/chi3d/sbj \
  --dataset_root /media/uv/Data/workspace/HH_gen/tridi/data/preprocessed/chi3d_smplx \
    --mode 10 \
    --smplx_model_dir /path/to/smplx_models
"""

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import argparse
import glob
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Any, Set

import h5py
import numpy as np
import trimesh
import torch
import smplx

from config.config import ProjectConfig
from tridi.data import get_eval_dataloader
from tridi.data.hh_batch_data import HHBatchData

logger = logging.getLogger(__name__)


def _fallback_compute_similarity_transform(S1: np.ndarray, S2: np.ndarray) -> np.ndarray:
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert S2.shape[1] == S1.shape[1]

    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    var1 = np.sum(X1 ** 2)
    K = X1.dot(X2.T)

    U, _, Vh = np.linalg.svd(K)
    V = Vh.T
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    R = V.dot(Z.dot(U.T))

    scale = np.trace(R.dot(K)) / var1
    t = mu2 - scale * (R.dot(mu1))
    S1_hat = scale * R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T
    return S1_hat


try:
    from tridi.utils.metrics.reconstruction import get_mpjpe_pa as _repo_get_mpjpe_pa

    def _get_mpjpe_pa(pred_joints: np.ndarray, gt_joints: np.ndarray) -> float:
        val = _repo_get_mpjpe_pa(pred_joints, gt_joints)
        return float(np.asarray(val))

except Exception as e:  # noqa: BLE001
    logger.warning("Falling back to local MPJPE_PA implementation, reason: %s", e)

    def _get_mpjpe_pa(pred_joints: np.ndarray, gt_joints: np.ndarray) -> float:
        pred = np.asarray(pred_joints, dtype=np.float32)
        gt = np.asarray(gt_joints, dtype=np.float32)

        pred = pred - pred[[0]]
        gt = gt - gt[[0]]

        pred_aligned = _fallback_compute_similarity_transform(pred, gt)
        mpjpe_pa = np.sqrt(np.sum((gt - pred_aligned) ** 2, axis=-1)).mean(-1)
        return float(mpjpe_pa)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export selected cases to ply files")
    parser.add_argument("--samples_dir", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--cases_json", type=str, default="")
    parser.add_argument("--mode", type=str, default="10", choices=["10", "01", "11"])
    parser.add_argument("--k_preds", type=int, default=3)
    parser.add_argument(
        "--rep_strategy",
        type=str,
        default="best_median_worst",
        choices=["best_k", "first_k", "best_median_worst"],
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--smplx_model_dir", type=str, default="")
    parser.add_argument("--default_gender", type=str, default="neutral", choices=["neutral", "male", "female"])
    parser.add_argument(
        "--default_sbj_gender",
        type=str,
        default=None,
        choices=["neutral", "male", "female"],
        help="Default gender for prefix=sbj when dataset does not provide gender.",
    )
    parser.add_argument(
        "--default_second_sbj_gender",
        type=str,
        default=None,
        choices=["neutral", "male", "female"],
        help="Default gender for prefix=second_sbj when dataset does not provide gender.",
    )
    parser.add_argument(
        "--use_batch_gender",
        dest="use_batch_gender",
        action="store_true",
        default=True,
        help="Use eval dataloader batch gender lookup as first-priority source (default: enabled).",
    )
    parser.add_argument(
        "--no_use_batch_gender",
        dest="use_batch_gender",
        action="store_false",
        help="Disable eval dataloader batch gender lookup.",
    )
    parser.add_argument(
        "--config_env",
        type=str,
        default="",
        help="Path to env config yaml (e.g., config/env.yaml) for building eval dataloader.",
    )
    parser.add_argument(
        "--scenario_yaml",
        type=str,
        default="",
        help="Path to scenario yaml (e.g., scenarios/chi3d.yaml) for building eval dataloader.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="",
        help="Optional run name used in temporary config context for dataloader build.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="chi3d",
        help="Datasets for eval dataloader build. Supports comma-separated or JSON list (default: chi3d).",
    )
    return parser.parse_args()


def _infer_run_name_from_samples_dir(samples_dir: Path) -> str:
    parts = list(samples_dir.parts)
    if "experiments" in parts:
        i = parts.index("experiments")
        if i + 1 < len(parts):
            return parts[i + 1]
    return samples_dir.name


def _parse_datasets_arg(v: str) -> List[str]:
    s = (v or "").strip()
    if not s:
        return ["chi3d"]
    if s.startswith("["):
        try:
            arr = json.loads(s)
            if isinstance(arr, list):
                out = [str(x).strip() for x in arr if str(x).strip()]
                return out if out else ["chi3d"]
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to parse --datasets as JSON list (%s), fallback to comma split.", e)
    out = [x.strip() for x in s.split(",") if x.strip()]
    return out if out else ["chi3d"]


def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _to_scalar(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        if x.numel() == 0:
            return None
        return x.detach().cpu().reshape(-1)[0].item()
    if isinstance(x, np.ndarray):
        if x.size == 0:
            return None
        return x.reshape(-1)[0].item()
    return x


def _to_bool_gender(x: Any) -> Optional[bool]:
    v = _to_scalar(x)
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer, float, np.floating)):
        return bool(int(v))
    s = str(v).strip().lower()
    if s in {"1", "true", "female", "f"}:
        return True
    if s in {"0", "false", "male", "m"}:
        return False
    return None


def _to_int(x: Any) -> Optional[int]:
    v = _to_scalar(x)
    if v is None:
        return None
    try:
        return int(v)
    except Exception:  # noqa: BLE001
        return None


def _to_str(x: Any) -> Optional[str]:
    v = _to_scalar(x)
    if v is None:
        return None
    if isinstance(v, bytes):
        v = v.decode("utf-8", errors="ignore")
    s = str(v).strip()
    return s if s else None


def _build_eval_cfg_for_gender_lookup(
    split: str,
    samples_dir: Path,
    dataset_root: Path,
    config_env: str,
    scenario_yaml: str,
    run_name: str,
    datasets_raw: str,
):
    try:
        from omegaconf import OmegaConf
    except Exception as e:  # noqa: BLE001
        logger.warning("Cannot import OmegaConf for batch-gender lookup: %s", e)
        return None

    config_files = []
    if config_env:
        config_files.append(Path(config_env))
    if scenario_yaml:
        config_files.append(Path(scenario_yaml))
    if not config_files:
        logger.warning("Batch-gender lookup enabled but no --config_env/--scenario_yaml provided. Fallback to defaults.")
        return None

    cfg = OmegaConf.structured(ProjectConfig())
    loaded_any = False
    for cf in config_files:
        if not cf.exists():
            logger.warning("Config file for batch-gender lookup not found: %s", cf)
            continue
        cfg = OmegaConf.merge(cfg, OmegaConf.load(str(cf)))
        loaded_any = True

    if not loaded_any:
        logger.warning("No valid config files loaded for batch-gender lookup. Fallback to defaults.")
        return None

    cfg.run.job = "sample"
    cfg.sample.split = str(split)
    cfg.run.datasets = _parse_datasets_arg(datasets_raw)
    cfg.run.name = run_name or _infer_run_name_from_samples_dir(samples_dir)
    cfg.dataloader.workers = 0

    if "chi3d" in cfg.run.datasets:
        cfg.chi3d.root = str(dataset_root)
        split_file = dataset_root / f"chi3d_{split}.json"
        if split_file.exists():
            setattr(cfg.chi3d, f"{split}_split_file", str(split_file))

    OmegaConf.resolve(cfg)
    return cfg


def build_gender_map_from_eval_dataloader(
    targets: Set[Tuple[str, int]],
    split: str,
    samples_dir: Path,
    dataset_root: Path,
    config_env: str,
    scenario_yaml: str,
    run_name: str,
    datasets_raw: str,
) -> Dict[Tuple[str, int], Tuple[bool, bool]]:
    gender_map: Dict[Tuple[str, int], Tuple[bool, bool]] = {}
    if not targets:
        return gender_map

    cfg = _build_eval_cfg_for_gender_lookup(
        split=split,
        samples_dir=samples_dir,
        dataset_root=dataset_root,
        config_env=config_env,
        scenario_yaml=scenario_yaml,
        run_name=run_name,
        datasets_raw=datasets_raw,
    )
    if cfg is None:
        return gender_map

    try:
        dataloaders = get_eval_dataloader(cfg)
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to build eval dataloader for batch-gender lookup: %s", e)
        return gender_map

    missing_gender_warned = False
    for dataloader in dataloaders:
        for batch in dataloader:
            if isinstance(batch, dict):
                batch = HHBatchData(**batch)

            seqs = _as_list(getattr(batch, "sbj", None))
            if not seqs:
                seqs = _as_list(getattr(batch, "sequence", None))
            if not seqs:
                seqs = _as_list(getattr(batch, "name", None))

            ts = _as_list(getattr(batch, "t_stamp", None))
            g1 = _as_list(getattr(batch, "sbj_gender", None))
            g2 = _as_list(getattr(batch, "second_sbj_gender", None))

            if not g1 or not g2:
                if not missing_gender_warned:
                    logger.warning("Eval dataloader batch has no sbj_gender/second_sbj_gender. Fallback to defaults.")
                    missing_gender_warned = True
                continue

            B = max(len(seqs), len(ts), len(g1), len(g2))
            for i in range(B):
                seq_val = seqs[i] if i < len(seqs) else (seqs[0] if len(seqs) == 1 else None)
                t_val = ts[i] if i < len(ts) else (ts[0] if len(ts) == 1 else None)
                g1_val = g1[i] if i < len(g1) else (g1[0] if len(g1) == 1 else None)
                g2_val = g2[i] if i < len(g2) else (g2[0] if len(g2) == 1 else None)

                seq = _to_str(seq_val)
                t_stamp = _to_int(t_val)
                sbj_bool = _to_bool_gender(g1_val)
                second_bool = _to_bool_gender(g2_val)

                if seq is None or t_stamp is None or sbj_bool is None or second_bool is None:
                    continue
                key = (seq, int(t_stamp))
                if key in targets and key not in gender_map:
                    gender_map[key] = (bool(sbj_bool), bool(second_bool))

            if len(gender_map) >= len(targets):
                logger.info("Batch-gender lookup completed early: hit %d/%d targets", len(gender_map), len(targets))
                return gender_map

    logger.info("Batch-gender lookup matched %d/%d target cases", len(gender_map), len(targets))
    return gender_map


def _gender_bool_to_str(v: bool) -> str:
    # Dataset convention: True=female, False=male
    return "female" if bool(v) else "male"


def _read_gender_from_gt_group(gt_group: h5py.Group, prefix: str) -> Optional[Any]:
    role_gender_key = "sbj_gender" if prefix == "sbj" else "second_sbj_gender"
    gender_val = None

    if role_gender_key in gt_group:
        try:
            gender_raw = gt_group[role_gender_key][()]
            if isinstance(gender_raw, np.ndarray):
                if gender_raw.ndim == 0:
                    gender_raw = gender_raw.item()
                elif gender_raw.size > 0:
                    gender_raw = gender_raw.reshape(-1)[0]
            gender_val = gender_raw
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to read %s from %s: %s", role_gender_key, gt_group.name, e)

    if gender_val is None and role_gender_key in gt_group.attrs:
        gender_val = gt_group.attrs.get(role_gender_key)

    return gender_val


def _resolve_smplx_model_root(p: Path) -> Path:
    """
    smplx.create(model_path, model_type="smplx") will look for
    model_path/"smplx"/SMPLX_*.npz internally.
    So this function always returns the parent root that contains "smplx/".
    """
    p = Path(p)

    # case 1: user passes .../smplx_models (recommended)
    if (p / "smplx").is_dir():
        return p

    # case 2: user passes .../smplx_models/smplx
    if p.name == "smplx" and p.is_dir():
        return p.parent

    raise FileNotFoundError(
        f"Invalid smplx_model_dir={p}. Expected either <root> containing 'smplx/' "
        f"or the 'smplx/' folder itself."
    )


def _resolve_smplx_model_dir(cli_value: str) -> Path:
    val = (cli_value or "").strip()
    if not val:
        val = os.environ.get("SMPLX_MODEL_DIR", "").strip()
    if not val:
        raise RuntimeError(
            "SMPL-X model directory is required. "
            "Please provide --smplx_model_dir or set SMPLX_MODEL_DIR environment variable."
        )
    p = Path(val)
    if not p.exists():
        raise FileNotFoundError(f"SMPL-X model directory does not exist: {p}")
    return _resolve_smplx_model_root(p)


def _find_rep_files(samples_dir: Path) -> List[Path]:
    rep_files = sorted(Path(p) for p in glob.glob(str(samples_dir / "samples_rep_*.hdf5")))
    assert rep_files, (
        f"No rep files found at {samples_dir}/samples_rep_*.hdf5. "
        "pls make sure if sampling output correct."
    )
    return rep_files


def _rep_id_from_path(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def _required_joint_keys(mode: str) -> List[str]:
    if mode == "10":
        return ["sbj_j"]
    if mode == "01":
        return ["second_sbj_j"]
    if mode == "11":
        return ["sbj_j", "second_sbj_j"]
    raise ValueError(f"Unsupported mode: {mode}")


def _assert_group_has_keys(group: h5py.Group, keys: List[str], context: str) -> None:
    missing = [k for k in keys if k not in group]
    assert not missing, (
        f"HDF5 structure doesnt match: {context} missing key={missing}. "
        f"available keys={sorted(list(group.keys()))}"
    )


def _get_group_T(group: h5py.Group, joint_keys: List[str]) -> int:
    _assert_group_has_keys(group, joint_keys, context="sequence group")
    t_from_data = min(int(group[k].shape[0]) for k in joint_keys)
    t_attr = int(group.attrs.get("T", t_from_data))
    return min(t_from_data, t_attr)


def _score_case_for_rep(gt_group: h5py.Group, pred_group: h5py.Group, t: int, mode: str) -> float:
    if mode == "10":
        return _get_mpjpe_pa(pred_group["sbj_j"][t], gt_group["sbj_j"][t])
    if mode == "01":
        return _get_mpjpe_pa(pred_group["second_sbj_j"][t], gt_group["second_sbj_j"][t])
    if mode == "11":
        sbj_score = _get_mpjpe_pa(pred_group["sbj_j"][t], gt_group["sbj_j"][t])
        second_score = _get_mpjpe_pa(pred_group["second_sbj_j"][t], gt_group["second_sbj_j"][t])
        return 0.5 * (sbj_score + second_score)
    raise ValueError(f"Unsupported mode: {mode}")


def _compute_case_rep_scores(
    seq: str,
    t: int,
    mode: str,
    gt_h5: h5py.File,
    rep_h5_map: Dict[int, h5py.File],
) -> Dict[int, float]:
    if seq not in gt_h5:
        return {}
    gt_group = gt_h5[seq]
    req_keys = _required_joint_keys(mode)
    _assert_group_has_keys(gt_group, req_keys, f"GT[{seq}]")

    scores = {}
    for rep_id, rep_h5 in rep_h5_map.items():
        if seq not in rep_h5:
            continue
        pred_group = rep_h5[seq]
        try:
            _assert_group_has_keys(pred_group, req_keys, f"REP{rep_id}[{seq}]")
        except AssertionError as e:
            logger.warning("Skip rep=%d seq=%s because key mismatch: %s", rep_id, seq, e)
            continue

        t_gt = _get_group_T(gt_group, req_keys)
        t_pred = _get_group_T(pred_group, req_keys)
        if t >= min(t_gt, t_pred):
            continue

        scores[rep_id] = _score_case_for_rep(gt_group, pred_group, t, mode)
    return scores


def _compose_two_meshes(v1: np.ndarray, f1: np.ndarray, v2: np.ndarray, f2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    v = np.concatenate([v1, v2], axis=0)
    f2_shifted = f2 + v1.shape[0]
    f = np.concatenate([f1, f2_shifted], axis=0)
    return v, f


def _export_mesh(vertices: np.ndarray, faces: np.ndarray, out_path: Path) -> None:
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.export(str(out_path))


def _normalize_gender(g: Optional[str], default_gender: str) -> str:
    if g is None:
        return default_gender
    if isinstance(g, bytes):
        g = g.decode("utf-8", errors="ignore")
    g = str(g).lower().strip()
    if g in {"male", "m", "0", "false"}:
        return "male"
    if g in {"female", "f", "1", "true"}:
        return "female"
    if g in {"neutral", "n"}:
        return "neutral"
    return default_gender


class _SMPLXCache:
    def __init__(self, smplx_model_dir: str, device: str = "cpu"):
        self.device = device
        self.model_dir = Path(smplx_model_dir)
        self.model_root = _resolve_smplx_model_root(self.model_dir)
        self.models: Dict[str, object] = {}
        logger.info("[SMPLXCache] model_root resolved to: %s", self.model_root)

    def get_model(self, gender: str):
        g = _normalize_gender(gender, "neutral")
        if g not in self.models:
            p = Path(self.model_dir)
            if p.name.lower() == "smplx":
                p = p.parent
            model_path = str(p)
            logger.info("[SMPLXCache] Loading SMPL-X gender=%s from %s", g, model_path)
            self.models[g] = smplx.create(
                model_path=model_path,
                model_type="smplx",
                gender=g,
                ext="npz",
                use_pca=False,
            ).to(self.device)
        return self.models[g]


def _reshape_pose9(x: np.ndarray, n: int) -> np.ndarray:
    a = np.asarray(x, dtype=np.float32)
    if a.shape == (n, 9):
        return a.reshape(n, 3, 3)
    if a.shape == (n, 3, 3):
        return a
    raise ValueError(f"Unexpected pose shape {a.shape}, expected ({n},9) or ({n},3,3)")


def _load_smpl_params_from_group(group: h5py.Group, prefix: str, t: int) -> Dict[str, np.ndarray]:
    keys = [
        f"{prefix}_smpl_betas",
        f"{prefix}_smpl_transl",
        f"{prefix}_smpl_global",
        f"{prefix}_smpl_body",
        f"{prefix}_smpl_lh",
        f"{prefix}_smpl_rh",
    ]
    _assert_group_has_keys(group, keys, f"GT[{group.name}]/{prefix} smpl params")

    betas = np.asarray(group[f"{prefix}_smpl_betas"][t], dtype=np.float32).reshape(1, 10)
    transl = np.asarray(group[f"{prefix}_smpl_transl"][t], dtype=np.float32).reshape(1, 3)
    global_orient = np.asarray(group[f"{prefix}_smpl_global"][t], dtype=np.float32).reshape(1, 3, 3)
    body_pose = _reshape_pose9(group[f"{prefix}_smpl_body"][t], 21).reshape(1, 21, 3, 3)
    left_hand_pose = _reshape_pose9(group[f"{prefix}_smpl_lh"][t], 15).reshape(1, 15, 3, 3)
    right_hand_pose = _reshape_pose9(group[f"{prefix}_smpl_rh"][t], 15).reshape(1, 15, 3, 3)

    return {
        "betas": betas,
        "transl": transl,
        "global_orient": global_orient,
        "body_pose": body_pose,
        "left_hand_pose": left_hand_pose,
        "right_hand_pose": right_hand_pose,
    }


def _infer_prep_transform(
    joints_raw: np.ndarray,
    joints_target: np.ndarray,
    prep_R: Optional[np.ndarray],
    prep_t: Optional[np.ndarray],
    prep_s: Optional[float],
    prep_rot_center: Optional[np.ndarray],
) -> Tuple[Callable[[np.ndarray], np.ndarray], str, float]:
    """
    Infer the best transform variant from common preprocessing formulas.
    Returns transform_fn, best_name, best_err.
    """
    if prep_R is None or prep_t is None or prep_s is None or prep_rot_center is None:
        logger.warning("prep_* missing, fallback to identity transform.")
        return (lambda x: x), "identity", float("inf")

    R = np.asarray(prep_R, dtype=np.float32).reshape(3, 3)
    t = np.asarray(prep_t, dtype=np.float32).reshape(3)
    s = float(np.asarray(prep_s).reshape(()))
    c = np.asarray(prep_rot_center, dtype=np.float32).reshape(3)

    def A(x):
        return s * ((R @ (x - c).T).T) + t

    def B(x):
        return s * ((x - c) @ R.T) + t

    def C(x):
        return ((R @ x.T).T) * s + t

    def D(x):
        return (x @ R.T) * s + t

    candidates = {"A": A, "B": B, "C": C, "D": D}
    jr = np.asarray(joints_raw, dtype=np.float32)
    jt = np.asarray(joints_target, dtype=np.float32)
    n = min(jr.shape[0], jt.shape[0])
    if n <= 0:
        logger.warning("No valid joints to infer prep transform, fallback identity.")
        return (lambda x: x), "identity", float("inf")
    jr = jr[:n]
    jt = jt[:n]

    best_name = "identity"
    best_err = float("inf")
    best_fn = lambda x: x
    for name, fn in candidates.items():
        try:
            pred = fn(jr)
            err = float(np.linalg.norm(pred - jt, axis=-1).mean())
            if err < best_err:
                best_err = err
                best_name = name
                best_fn = fn
        except Exception as e:  # noqa: BLE001
            logger.warning("Transform candidate %s failed: %s", name, e)

    return best_fn, best_name, best_err


def _rotmat9_to_mat(x) -> torch.Tensor:
    """Convert pose input to float32 torch rotmat (...,3,3)."""
    t_in = torch.as_tensor(np.asarray(x), dtype=torch.float32)
    if t_in.ndim == 0:
        raise ValueError("Invalid scalar rotmat input")
    if t_in.shape[-1] == 9:
        t_in = t_in.reshape(*t_in.shape[:-1], 3, 3)
    elif t_in.ndim >= 2 and t_in.shape[-2:] == (3, 3):
        pass
    else:
        raise ValueError(f"Unsupported rotmat input shape: {tuple(t_in.shape)}")
    return t_in


def _mat_to_axis_angle(R: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Convert rotation matrix tensor (...,3,3) to axis-angle (...,3) in torch."""
    if R.ndim < 2 or R.shape[-2:] != (3, 3):
        raise ValueError(f"Expected (...,3,3), got {tuple(R.shape)}")

    orig_shape = R.shape[:-2]
    Rf = R.reshape(-1, 3, 3)

    trace = Rf[:, 0, 0] + Rf[:, 1, 1] + Rf[:, 2, 2]
    cos_angle = torch.clamp((trace - 1.0) * 0.5, -1.0, 1.0)
    angle = torch.acos(cos_angle)
    sin_angle = torch.sin(angle)

    vee = torch.stack(
        [
            Rf[:, 2, 1] - Rf[:, 1, 2],
            Rf[:, 0, 2] - Rf[:, 2, 0],
            Rf[:, 1, 0] - Rf[:, 0, 1],
        ],
        dim=-1,
    )

    denom = (2.0 * sin_angle).unsqueeze(-1)
    axis = vee / (denom + eps)
    aa = axis * angle.unsqueeze(-1)

    small = (angle.abs() < 1e-4).unsqueeze(-1)
    aa = torch.where(small, 0.5 * vee, aa)

    return aa.reshape(*orig_shape, 3)


_SMPL_FALLBACK_SHAPE_LOGGED = False
_SMPL_FALLBACK_MODELINFO_LOGGED = False


def _generate_gt_mesh_from_smpl(
    gt_group: h5py.Group,
    prefix: str,
    t: int,
    gender: str,
    smplx_cache: _SMPLXCache,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate GT mesh from SMPL-X params, then align with prep_* inferred transform."""
    global _SMPL_FALLBACK_SHAPE_LOGGED, _SMPL_FALLBACK_MODELINFO_LOGGED

    params = _load_smpl_params_from_group(gt_group, prefix, t)
    model = smplx_cache.get_model(gender)
    device = next(model.parameters()).device

    global_R = _rotmat9_to_mat(params["global_orient"]).reshape(-1, 3, 3)[:1]
    body_R = _rotmat9_to_mat(params["body_pose"]).reshape(-1, 21, 3, 3)[:1]
    lh_R = _rotmat9_to_mat(params["left_hand_pose"]).reshape(-1, 15, 3, 3)[:1]
    rh_R = _rotmat9_to_mat(params["right_hand_pose"]).reshape(-1, 15, 3, 3)[:1]

    global_aa = _mat_to_axis_angle(global_R)
    body_aa = _mat_to_axis_angle(body_R)
    lh_aa = _mat_to_axis_angle(lh_R)
    rh_aa = _mat_to_axis_angle(rh_R)

    num_betas = int(getattr(model, "num_betas", 10))
    num_expr = int(getattr(model, "num_expression_coeffs", 0))

    betas_raw = np.asarray(gt_group[f"{prefix}_smpl_betas"][t], np.float32).reshape(-1)
    betas_np = np.zeros((1, num_betas), np.float32)
    n = min(num_betas, betas_raw.shape[0])
    betas_np[0, :n] = betas_raw[:n]
    betas = torch.from_numpy(betas_np).to(device)

    transl = torch.as_tensor(np.asarray(params["transl"]), dtype=torch.float32, device=device).reshape(-1, 3)[:1]

    kwargs = {}
    if num_expr > 0:
        kwargs["expression"] = torch.zeros((1, num_expr), dtype=betas.dtype, device=betas.device)

    if not _SMPL_FALLBACK_MODELINFO_LOGGED:
        shapedirs_last = int(model.shapedirs.shape[-1]) if hasattr(model, "shapedirs") else -1
        expr_shape = tuple(kwargs["expression"].shape) if "expression" in kwargs else None
        logger.info(
            "SMPL-X fallback coeff dims num_betas=%d num_expr=%d shapedirs_last=%d betas_shape=%s expression_shape=%s",
            num_betas,
            num_expr,
            shapedirs_last,
            tuple(betas.shape),
            expr_shape,
        )
        _SMPL_FALLBACK_MODELINFO_LOGGED = True

    if not _SMPL_FALLBACK_SHAPE_LOGGED:
        logger.info(
            "SMPL-X fallback shapes global=%s body=%s lh=%s rh=%s",
            tuple(global_aa.shape), tuple(body_aa.shape), tuple(lh_aa.shape), tuple(rh_aa.shape)
        )
        _SMPL_FALLBACK_SHAPE_LOGGED = True

    with torch.no_grad():
        out = model(
            betas=betas,
            global_orient=global_aa.to(device),
            body_pose=body_aa.to(device),
            left_hand_pose=lh_aa.to(device),
            right_hand_pose=rh_aa.to(device),
            transl=transl,
            return_verts=True,
            **kwargs,
        )

    v_raw = out.vertices[0].detach().cpu().numpy().astype(np.float32)
    j_raw = out.joints[0].detach().cpu().numpy().astype(np.float32)
    f = np.asarray(model.faces, dtype=np.int64)

    # prep_* may not exist in all datasets
    prep_R = prep_t = prep_s = prep_rot_center = None
    try:
        if "prep_R" in gt_group and "prep_t" in gt_group and "prep_s" in gt_group and "prep_rot_center" in gt_group:
            prep_R = np.asarray(gt_group["prep_R"][t], dtype=np.float32).reshape(3, 3)
            prep_t = np.asarray(gt_group["prep_t"][t], dtype=np.float32).reshape(3)
            prep_s = float(np.asarray(gt_group["prep_s"][t]).reshape(()))
            prep_rot_center = np.asarray(gt_group["prep_rot_center"][t], dtype=np.float32).reshape(3)
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed reading prep_* for %s/%s t=%d: %s", gt_group.name, prefix, t, e)

    j_target_key = f"{prefix}_j"
    if j_target_key in gt_group:
        j_target = np.asarray(gt_group[j_target_key][t], dtype=np.float32)
    else:
        j_target = j_raw
        logger.warning("Missing %s in GT group %s, cannot infer prep transform; using identity.", j_target_key, gt_group.name)

    try:
        tf_fn, tf_name, tf_err = _infer_prep_transform(j_raw, j_target, prep_R, prep_t, prep_s, prep_rot_center)
        logger.info("GT fallback transform for %s/%s t=%d -> %s (mean joint err=%.6f)", gt_group.name, prefix, t, tf_name, tf_err)
        v = tf_fn(v_raw).astype(np.float32)
    except Exception as e:  # noqa: BLE001
        logger.warning("Prep transform inference failed for %s/%s t=%d, using raw vertices: %s", gt_group.name, prefix, t, e)
        v = v_raw

    return v, f


def _get_gt_vertices_faces_with_fallback(
    gt_group: h5py.Group,
    prefix: str,
    seq: str,
    t: int,
    smplx_cache: _SMPLXCache,
    default_gender: str,
    default_sbj_gender: Optional[str] = None,
    default_second_sbj_gender: Optional[str] = None,
    use_batch_gender: bool = True,
    gender_map: Optional[Dict[Tuple[str, int], Tuple[bool, bool]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Read mesh from GT hdf5 if available, else generate from SMPL-X params."""
    v_key = f"{prefix}_v"
    f_key = f"{prefix}_f"
    if v_key in gt_group and f_key in gt_group:
        v = np.asarray(gt_group[v_key][t], dtype=np.float32)
        f = np.asarray(gt_group[f_key], dtype=np.int32)
        return v, f

    logger.warning(
        "GT keys (%s,%s) missing in %s. Falling back to SMPL-X generation from params.",
        v_key, f_key, gt_group.name,
    )
    # 1) First-priority: batch-gender lookup from eval dataloader
    gender_val = None
    if use_batch_gender and gender_map is not None:
        pair = gender_map.get((seq, int(t)))
        if pair is not None:
            sbj_b, second_b = pair
            gender_val = _gender_bool_to_str(sbj_b if prefix == "sbj" else second_b)

    # 2) Second-priority: GT group/attrs if present
    if gender_val is None:
        gender_val = _read_gender_from_gt_group(gt_group, prefix)

    # 3) Third-priority: default roles / default gender fallback
    if gender_val is None:
        if prefix == "sbj" and default_sbj_gender is not None:
            gender_val = default_sbj_gender
        elif prefix == "second_sbj" and default_second_sbj_gender is not None:
            gender_val = default_second_sbj_gender
        else:
            gender_val = default_gender

    gender = _normalize_gender(gender_val, default_gender)
    return _generate_gt_mesh_from_smpl(gt_group, prefix, t, gender, smplx_cache)


def _select_rep_ids(
    rep_scores: Dict[int, float],
    strategy: str,
    k_preds: int,
) -> Tuple[List[int], Dict[int, str]]:
    assert k_preds > 0, "k_preds must be > 0"
    assert rep_scores, "current case has no available rep scores"

    sorted_by_score = sorted(rep_scores.items(), key=lambda kv: kv[1])
    rep_ids_sorted_score = [rid for rid, _ in sorted_by_score]
    reasons: Dict[int, str] = {}

    if strategy == "best_k":
        chosen = rep_ids_sorted_score[:k_preds]
        for rid in chosen:
            reasons[rid] = "best_k"
        return chosen, reasons

    if strategy == "first_k":
        chosen = sorted(rep_scores.keys())[:k_preds]
        for rid in chosen:
            reasons[rid] = "first_k"
        return chosen, reasons

    # best_median_worst
    n = len(sorted_by_score)
    best = sorted_by_score[0][0]
    median = sorted_by_score[n // 2][0]
    worst = sorted_by_score[-1][0]

    chosen = []
    for rid in [best, median, worst]:
        if rid not in chosen:
            chosen.append(rid)

    if best in chosen:
        reasons[best] = "best(min MPJPE_PA)"
    if median in chosen:
        reasons[median] = "median(MPJPE_PA rank median)"
    if worst in chosen:
        reasons[worst] = "worst(max MPJPE_PA)"

    # if k_preds not 3 ，fill（ score from best to worst）
    if len(chosen) < k_preds:
        for rid in rep_ids_sorted_score:
            if rid not in chosen:
                chosen.append(rid)
                reasons[rid] = "fill_by_score"
            if len(chosen) >= k_preds:
                break

    chosen = chosen[:k_preds]
    return chosen, reasons


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    )
    args = _parse_args()

    try:
        smplx_model_dir = _resolve_smplx_model_dir(args.smplx_model_dir)
    except Exception as e:  # noqa: BLE001
        logger.error("Failed to resolve SMPL-X model directory: %s", e)
        return

    smplx_cache = _SMPLXCache(smplx_model_dir)

    samples_dir = Path(args.samples_dir)
    dataset_root = Path(args.dataset_root)
    cases_json = Path(args.cases_json) if args.cases_json else (samples_dir / "_viz" / "cases.json")

    assert cases_json.exists(), f"cases_json does not exist: {cases_json}"
    gt_hdf5_path = dataset_root / f"dataset_{args.split}_25fps.hdf5"
    assert gt_hdf5_path.exists(), f"GT hdf5 does not exist: {gt_hdf5_path}"

    with cases_json.open("r", encoding="utf-8") as f:
        cases_payload = json.load(f)

    selected_cases = []
    for key in ["random", "best", "worst"]:
        if key in cases_payload:
            selected_cases.extend(cases_payload[key])

    # 去重但保留顺序
    seen = set()
    dedup_cases = []
    for item in selected_cases:
        seq = item["seq"]
        t = int(item["t"])
        key = (seq, t)
        if key in seen:
            continue
        seen.add(key)
        dedup_cases.append(key)

    assert dedup_cases, "theres no case in cases.json that is available for export"
    logger.info("Loaded %d unique cases from %s", len(dedup_cases), cases_json)

    gender_map: Dict[Tuple[str, int], Tuple[bool, bool]] = {}
    if args.use_batch_gender:
        try:
            targets = set((seq, int(t)) for seq, t in dedup_cases)
            gender_map = build_gender_map_from_eval_dataloader(
                targets=targets,
                split=args.split,
                samples_dir=samples_dir,
                dataset_root=dataset_root,
                config_env=args.config_env,
                scenario_yaml=args.scenario_yaml,
                run_name=args.run_name,
                datasets_raw=args.datasets,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Batch-gender lookup failed, fallback to default logic: %s", e)
            gender_map = {}
    else:
        logger.info("Batch-gender lookup disabled via --no_use_batch_gender")

    rep_files = _find_rep_files(samples_dir)
    rep_h5_map: Dict[int, h5py.File] = {}
    ply_root = samples_dir / "_viz" / "ply"
    ply_root.mkdir(parents=True, exist_ok=True)

    try:
        for path in rep_files:
            rep_h5_map[_rep_id_from_path(path)] = h5py.File(path, "r")

        with h5py.File(gt_hdf5_path, "r") as gt_h5:
            for case_i, (seq, t) in enumerate(dedup_cases):
                logger.info("[%d/%d] Exporting seq=%s t=%d", case_i + 1, len(dedup_cases), seq, t)
                assert seq in gt_h5, f" cant find sequence in GT: {seq}"
                gt_group = gt_h5[seq]

                req_keys = _required_joint_keys(args.mode)
                _assert_group_has_keys(gt_group, req_keys, f"GT[{seq}]")

                rep_scores = _compute_case_rep_scores(seq, t, args.mode, gt_h5, rep_h5_map)
                assert rep_scores, (
                    f"case(seq={seq}, t={t}) isnt available, no rep has valid score. "
                    "pls make sure cases.json is generated from the same samples_rep_*.hdf5."
                )

                chosen_reps, reasons = _select_rep_ids(rep_scores, args.rep_strategy, args.k_preds)
                case_dir = ply_root / f"{seq}_t{t:05d}"
                case_dir.mkdir(parents=True, exist_ok=True)

                # faces / vertices key 探测与校验
                # mode=10: target sbj, condition second_sbj
                # mode=01: target second_sbj, condition sbj
                # mode=11: target 两者合并，不导 condition
                if args.mode == "10":
                    target_v_key, target_f_key = "sbj_v", "sbj_f"
                    cond_v_key, cond_f_key = "second_sbj_v", "second_sbj_f"

                    # condition.ply
                    cond_v, cond_f = _get_gt_vertices_faces_with_fallback(
                        gt_group, "second_sbj", seq, t,
                        smplx_cache=smplx_cache,
                        default_gender=args.default_gender,
                        default_sbj_gender=args.default_sbj_gender,
                        default_second_sbj_gender=args.default_second_sbj_gender,
                        use_batch_gender=args.use_batch_gender,
                        gender_map=gender_map,
                    )
                    _export_mesh(cond_v, cond_f, case_dir / "condition.ply")

                    # gt.ply
                    gt_v, gt_f = _get_gt_vertices_faces_with_fallback(
                        gt_group, "sbj", seq, t,
                        smplx_cache=smplx_cache,
                        default_gender=args.default_gender,
                        default_sbj_gender=args.default_sbj_gender,
                        default_second_sbj_gender=args.default_second_sbj_gender,
                        use_batch_gender=args.use_batch_gender,
                        gender_map=gender_map,
                    )
                    _export_mesh(gt_v, gt_f, case_dir / "gt.ply")

                    # pred_repXX.ply
                    for rep_id in chosen_reps:
                        rep_group = rep_h5_map[rep_id][seq]
                        _assert_group_has_keys(rep_group, [target_v_key, target_f_key], f"REP{rep_id}[{seq}]")
                        pred_v = np.asarray(rep_group[target_v_key][t], dtype=np.float32)
                        pred_f = np.asarray(rep_group[target_f_key], dtype=np.int32)
                        _export_mesh(pred_v, pred_f, case_dir / f"pred_rep{rep_id:02d}.ply")

                elif args.mode == "01":
                    target_v_key, target_f_key = "second_sbj_v", "second_sbj_f"
                    cond_v_key, cond_f_key = "sbj_v", "sbj_f"

                    cond_v, cond_f = _get_gt_vertices_faces_with_fallback(
                        gt_group, "sbj", seq, t,
                        smplx_cache=smplx_cache,
                        default_gender=args.default_gender,
                        default_sbj_gender=args.default_sbj_gender,
                        default_second_sbj_gender=args.default_second_sbj_gender,
                        use_batch_gender=args.use_batch_gender,
                        gender_map=gender_map,
                    )
                    _export_mesh(cond_v, cond_f, case_dir / "condition.ply")

                    gt_v, gt_f = _get_gt_vertices_faces_with_fallback(
                        gt_group, "second_sbj", seq, t,
                        smplx_cache=smplx_cache,
                        default_gender=args.default_gender,
                        default_sbj_gender=args.default_sbj_gender,
                        default_second_sbj_gender=args.default_second_sbj_gender,
                        use_batch_gender=args.use_batch_gender,
                        gender_map=gender_map,
                    )
                    _export_mesh(gt_v, gt_f, case_dir / "gt.ply")

                    for rep_id in chosen_reps:
                        rep_group = rep_h5_map[rep_id][seq]
                        _assert_group_has_keys(rep_group, [target_v_key, target_f_key], f"REP{rep_id}[{seq}]")
                        pred_v = np.asarray(rep_group[target_v_key][t], dtype=np.float32)
                        pred_f = np.asarray(rep_group[target_f_key], dtype=np.int32)
                        _export_mesh(pred_v, pred_f, case_dir / f"pred_rep{rep_id:02d}.ply")

                else:  # mode == "11"
                    # 不导出 condition.ply；gt/pred 都合并两个人体 mesh
                    keys = ["sbj_v", "sbj_f", "second_sbj_v", "second_sbj_f"]
                    gt_v1, gt_f1 = _get_gt_vertices_faces_with_fallback(
                        gt_group, "sbj", seq, t,
                        smplx_cache=smplx_cache,
                        default_gender=args.default_gender,
                        default_sbj_gender=args.default_sbj_gender,
                        default_second_sbj_gender=args.default_second_sbj_gender,
                        use_batch_gender=args.use_batch_gender,
                        gender_map=gender_map,
                    )
                    gt_v2, gt_f2 = _get_gt_vertices_faces_with_fallback(
                        gt_group, "second_sbj", seq, t,
                        smplx_cache=smplx_cache,
                        default_gender=args.default_gender,
                        default_sbj_gender=args.default_sbj_gender,
                        default_second_sbj_gender=args.default_second_sbj_gender,
                        use_batch_gender=args.use_batch_gender,
                        gender_map=gender_map,
                    )
                    gt_v, gt_f = _compose_two_meshes(gt_v1, gt_f1, gt_v2, gt_f2)
                    _export_mesh(gt_v, gt_f, case_dir / "gt.ply")

                    for rep_id in chosen_reps:
                        rep_group = rep_h5_map[rep_id][seq]
                        _assert_group_has_keys(rep_group, keys, f"REP{rep_id}[{seq}]")
                        pred_v1 = np.asarray(rep_group["sbj_v"][t], dtype=np.float32)
                        pred_f1 = np.asarray(rep_group["sbj_f"], dtype=np.int32)
                        pred_v2 = np.asarray(rep_group["second_sbj_v"][t], dtype=np.float32)
                        pred_f2 = np.asarray(rep_group["second_sbj_f"], dtype=np.int32)
                        pred_v, pred_f = _compose_two_meshes(pred_v1, pred_f1, pred_v2, pred_f2)
                        _export_mesh(pred_v, pred_f, case_dir / f"pred_rep{rep_id:02d}.ply")

                # case meta
                score_items = [{"rep": int(k), "mpjpe_pa": float(v)} for k, v in sorted(rep_scores.items())]
                meta = {
                    "seq": seq,
                    "t": int(t),
                    "mode": args.mode,
                    "rep_strategy": args.rep_strategy,
                    "k_preds": args.k_preds,
                    "exported_reps": [int(r) for r in chosen_reps],
                    "rep_scores_mpjpe_pa": score_items,
                    "selection_reasons": [{"rep": int(r), "reason": reasons.get(r, "")}
                                          for r in chosen_reps],
                }
                with (case_dir / "meta.json").open("w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)

    finally:
        for h5 in rep_h5_map.values():
            h5.close()

    logger.info("Done. Export root: %s", ply_root)


if __name__ == "__main__":
    main()
