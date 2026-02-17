import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import logging
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import trimesh
from PIL import Image


logger = logging.getLogger(__name__)


_REP_RE = re.compile(r"pred_rep(\d+)", flags=re.IGNORECASE)


def find_case_dirs(ply_root: Path) -> List[Path]:
    """Recursively discover case directories under ply_root."""
    ply_root = Path(ply_root)
    if not ply_root.exists():
        raise FileNotFoundError(f"ply_root does not exist: {ply_root}")

    case_dirs: List[Path] = []
    for d in sorted([p for p in ply_root.rglob("*") if p.is_dir()]):
        has_meta = (d / "meta.json").exists()
        has_gt = any(d.glob("gt*.ply"))
        if has_meta or has_gt:
            case_dirs.append(d)

    # Also include root-level cases if user points directly to a single case dir
    if (ply_root / "meta.json").exists() or any(ply_root.glob("gt*.ply")):
        if ply_root not in case_dirs:
            case_dirs.insert(0, ply_root)

    return case_dirs


def _sorted_nicely(paths: List[Path]) -> List[Path]:
    return sorted(paths, key=lambda p: p.name)


def scan_case_assets(case_dir: Path) -> Dict:
    """Scan one case folder and return robust mesh groups."""
    case_dir = Path(case_dir)
    ply_files = _sorted_nicely(list(case_dir.glob("*.ply")))

    condition = _sorted_nicely([p for p in ply_files if p.name.lower().startswith("condition")])
    gt = _sorted_nicely([p for p in ply_files if p.name.lower().startswith("gt")])

    preds_by_rep: Dict[int, List[Path]] = {}
    for p in ply_files:
        name = p.name.lower()
        if not name.startswith("pred_rep"):
            continue
        m = _REP_RE.search(name)
        if m is None:
            logger.warning("Skip pred file without rep id: %s", p)
            continue
        rep = int(m.group(1))
        preds_by_rep.setdefault(rep, []).append(p)

    for rep in list(preds_by_rep.keys()):
        preds_by_rep[rep] = _sorted_nicely(preds_by_rep[rep])

    meta_path = case_dir / "meta.json"
    return {
        "case_dir": case_dir,
        "meta_path": meta_path if meta_path.exists() else None,
        "condition": condition,
        "gt": gt,
        "preds_by_rep": dict(sorted(preds_by_rep.items(), key=lambda kv: kv[0])),
    }


def choose_pred_reps(preds_by_rep: Dict[int, List[Path]], k_preds: int, seed: int = 42) -> List[int]:
    rep_ids = sorted(preds_by_rep.keys())
    if not rep_ids:
        return []
    if len(rep_ids) <= k_preds:
        return rep_ids

    # deterministic selection: first K after stable sort
    # (kept deterministic for reproducibility; seed is accepted for future random strategy)
    _ = seed
    return rep_ids[:k_preds]


def load_mesh_from_ply(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, process=False, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.dump()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"Unsupported mesh type loaded from {path}: {type(mesh)}")
    return mesh


def collect_bbox_from_paths(paths: List[Path]) -> Tuple[np.ndarray, np.ndarray]:
    mins, maxs = [], []
    for p in paths:
        try:
            mesh = load_mesh_from_ply(p)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed loading mesh for bbox (%s): %s", p, e)
            continue
        mins.append(np.asarray(mesh.vertices).min(axis=0))
        maxs.append(np.asarray(mesh.vertices).max(axis=0))

    if not mins:
        return np.array([-1.0, -1.0, -1.0], dtype=np.float32), np.array([1.0, 1.0, 1.0], dtype=np.float32)
    return np.min(np.stack(mins, axis=0), axis=0), np.max(np.stack(maxs, axis=0), axis=0)


def build_camera_spec(all_mesh_paths: List[Path], camera_mode: str, width: int, height: int) -> Dict:
    if camera_mode not in {"auto", "fixed"}:
        raise ValueError(f"Unknown camera mode: {camera_mode}")

    if camera_mode == "fixed":
        return {
            "resolution": (width, height),
            "fov_x": np.deg2rad(40.0),
            "translation": (4.5, -4.5, 2.8),
            "look_at": (0.0, 0.0, 0.8),
        }

    bb_min, bb_max = collect_bbox_from_paths(all_mesh_paths)
    center = 0.5 * (bb_min + bb_max)
    extent = float(np.linalg.norm(bb_max - bb_min))
    extent = max(extent, 1.0)

    translation = center + np.array([0.0, -2.4 * extent, 1.3 * extent], dtype=np.float32)
    return {
        "resolution": (width, height),
        "fov_x": np.deg2rad(38.0),
        "translation": tuple(translation.tolist()),
        "look_at": tuple(center.tolist()),
    }


def _vary_color(base_rgb: Tuple[float, float, float], idx: int, total: int, scale: float = 0.22) -> Tuple[float, float, float]:
    if total <= 1:
        return base_rgb
    t = idx / max(total - 1, 1)
    delta = (t - 0.5) * 2.0 * scale
    c = np.clip(np.asarray(base_rgb, dtype=np.float32) + delta, 0.0, 1.0)
    return float(c[0]), float(c[1]), float(c[2])


def color_condition(idx: int, total: int) -> Tuple[float, float, float]:
    return _vary_color((0.22, 0.45, 0.95), idx, total, scale=0.14)


def color_prediction(idx: int, total: int) -> Tuple[float, float, float]:
    return _vary_color((0.20, 0.78, 0.34), idx, total, scale=0.20)


def color_gt(idx: int, total: int) -> Tuple[float, float, float]:
    return _vary_color((0.95, 0.83, 0.20), idx, total, scale=0.10)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def copy_outputs_to_inplace(case_dir: Path, src_dir: Path, names: List[str]) -> None:
    dst_dir = ensure_dir(case_dir / "renders")
    for n in names:
        src = src_dir / n
        if src.exists():
            shutil.copy2(src, dst_dir / n)


def make_image_grid(image_paths: List[Path], out_path: Path, cols: Optional[int] = None, bg=(245, 245, 245, 255)) -> None:
    valid = [Path(p) for p in image_paths if Path(p).exists()]
    if not valid:
        logger.warning("No images for grid: %s", out_path)
        return

    imgs = [Image.open(p).convert("RGBA") for p in valid]
    w = max(i.width for i in imgs)
    h = max(i.height for i in imgs)

    n = len(imgs)
    if cols is None:
        cols = min(3, n)
    cols = max(1, cols)
    rows = int(np.ceil(n / cols))

    canvas = Image.new("RGBA", (cols * w, rows * h), color=bg)
    for i, img in enumerate(imgs):
        r = i // cols
        c = i % cols
        x = c * w + (w - img.width) // 2
        y = r * h + (h - img.height) // 2
        canvas.alpha_composite(img, (x, y))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
