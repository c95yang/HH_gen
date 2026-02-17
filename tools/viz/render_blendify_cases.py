#!/usr/bin/env python3
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

from tools.viz.utils import (
    find_case_dirs,
    scan_case_assets,
    choose_pred_reps,
    build_camera_spec,
    color_condition,
    color_prediction,
    color_gt,
    ensure_dir,
    copy_outputs_to_inplace,
    make_image_grid,
)

logger = logging.getLogger(__name__)


def _safe_import_blendify():
    try:
        from blendify import scene
        from blendify.colors import UniformColors
        from blendify.materials import PrincipledBSDFMaterial
        import trimesh
        return scene, UniformColors, PrincipledBSDFMaterial, trimesh
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "Failed to import blendify stack. Please activate viz env and install tools/viz/requirements_viz.txt"
        ) from e


def _add_meshes_to_scene(scene, UniformColors, PrincipledBSDFMaterial, trimesh_mod, mesh_specs: List[Tuple[Path, Tuple[float, float, float], float]]):
    for p, rgb, alpha in mesh_specs:
        if not p.exists():
            logger.warning("Missing mesh file, skip: %s", p)
            continue
        try:
            mesh = trimesh_mod.load(p, process=False, force="mesh")
            if isinstance(mesh, trimesh_mod.Scene):
                mesh = trimesh_mod.util.concatenate(tuple(mesh.dump()))
            vertices = np.asarray(mesh.vertices, dtype=np.float32)
            faces = np.asarray(mesh.faces, dtype=np.int32)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed loading mesh %s: %s", p, e)
            continue

        mat = PrincipledBSDFMaterial(alpha=float(alpha), roughness=0.45, metallic=0.0, specular_ior=0.5)
        col = UniformColors(tuple(float(x) for x in rgb))
        obj = scene.renderables.add_mesh(vertices, faces, material=mat, colors=col, tag=p.stem)
        try:
            obj.set_smooth(True)
        except Exception:  # noqa: BLE001
            pass


def _setup_camera_and_lights(scene, cam_spec: dict):
    scene.set_perspective_camera(
        cam_spec["resolution"],
        fov_x=float(cam_spec["fov_x"]),
        rotation_mode="look_at",
        rotation=tuple(cam_spec["look_at"]),
        translation=tuple(cam_spec["translation"]),
    )

    # simple robust lighting setup
    try:
        scene.lights.set_background_light(0.03)
    except Exception:  # noqa: BLE001
        pass
    scene.lights.add_point(strength=900, shadow_soft_size=1.0, translation=(4.0, -3.0, 4.5))
    scene.lights.add_sun(strength=2.0)



def _render(scene, out_path: Path, use_gpu: bool, samples: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scene.render(filepath=str(out_path), use_gpu=use_gpu, samples=int(samples), use_denoiser=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render HH_gen _viz/ply cases with blendify")
    parser.add_argument("--ply_root", type=str, required=True)
    parser.add_argument("--out_root", type=str, default="")
    parser.add_argument("--k_preds", type=int, default=3)

    parser.add_argument("--res", type=int, default=1024)
    parser.add_argument("--W", type=int, default=0)
    parser.add_argument("--H", type=int, default=0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--camera", type=str, default="auto", choices=["auto", "fixed"])
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--samples", type=int, default=128)

    parser.add_argument("--condition_alpha", type=float, default=0.35)
    parser.add_argument("--pred_alpha", type=float, default=0.60)
    parser.add_argument("--gt_alpha", type=float, default=0.90)

    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")
    args = parse_args()

    ply_root = Path(args.ply_root)
    if args.out_root:
        out_root = Path(args.out_root)
    else:
        out_root = (ply_root / ".." / "renders").resolve()

    W = int(args.W) if int(args.W) > 0 else int(args.res)
    H = int(args.H) if int(args.H) > 0 else int(args.res)

    case_dirs = find_case_dirs(ply_root)
    logger.info("Found %d case folders under %s", len(case_dirs), ply_root)
    if args.dry_run:
        for c in case_dirs:
            logger.info("[DRY] %s", c)
        return

    scene, UniformColors, PrincipledBSDFMaterial, trimesh_mod = _safe_import_blendify()
    use_gpu = str(args.device).lower() != "cpu"

    for i, case_dir in enumerate(case_dirs, start=1):
        logger.info("[%d/%d] Rendering case: %s", i, len(case_dirs), case_dir.name)
        assets = scan_case_assets(case_dir)

        condition_paths = assets["condition"]
        gt_paths = assets["gt"]
        preds_by_rep = assets["preds_by_rep"]

        if len(gt_paths) == 0:
            logger.warning("No GT ply in case %s, skip.", case_dir)
            continue
        if len(preds_by_rep) == 0:
            logger.warning("No prediction plys in case %s, skip.", case_dir)
            continue

        selected_reps = choose_pred_reps(preds_by_rep, k_preds=args.k_preds, seed=args.seed)
        if not selected_reps:
            logger.warning("No selected reps for case %s, skip.", case_dir)
            continue

        all_for_camera: List[Path] = []
        all_for_camera.extend(condition_paths)
        all_for_camera.extend(gt_paths)
        for rep in selected_reps:
            all_for_camera.extend(preds_by_rep.get(rep, []))
        cam_spec = build_camera_spec(all_for_camera, args.camera, W, H)

        out_case = ensure_dir(out_root / case_dir.name)

        # ---------- overlay: condition + K predictions ----------
        overlay_specs = []
        for ci, p in enumerate(condition_paths):
            overlay_specs.append((p, color_condition(ci, len(condition_paths)), float(args.condition_alpha)))

        for ri, rep in enumerate(selected_reps):
            rep_files = preds_by_rep.get(rep, [])
            if not rep_files:
                logger.warning("Rep %02d has no files in case %s", rep, case_dir)
                continue
            rep_color = color_prediction(ri, len(selected_reps))
            for p in rep_files:
                overlay_specs.append((p, rep_color, float(args.pred_alpha)))

        scene.clear()
        _add_meshes_to_scene(scene, UniformColors, PrincipledBSDFMaterial, trimesh_mod, overlay_specs)
        _setup_camera_and_lights(scene, cam_spec)
        overlay_png = out_case / "overlay.png"
        _render(scene, overlay_png, use_gpu=use_gpu, samples=args.samples)

        # ---------- gt ----------
        gt_specs = []
        for gi, p in enumerate(gt_paths):
            gt_specs.append((p, color_gt(gi, len(gt_paths)), float(args.gt_alpha)))

        scene.clear()
        _add_meshes_to_scene(scene, UniformColors, PrincipledBSDFMaterial, trimesh_mod, gt_specs)
        _setup_camera_and_lights(scene, cam_spec)
        gt_png = out_case / "gt.png"
        _render(scene, gt_png, use_gpu=use_gpu, samples=args.samples)

        # ---------- pred individual and grid ----------
        pred_images = []
        for ri, rep in enumerate(selected_reps):
            rep_specs = []
            rep_files = preds_by_rep.get(rep, [])
            rep_color = color_prediction(ri, len(selected_reps))
            for p in rep_files:
                rep_specs.append((p, rep_color, float(args.pred_alpha)))
            if not rep_specs:
                continue

            scene.clear()
            _add_meshes_to_scene(scene, UniformColors, PrincipledBSDFMaterial, trimesh_mod, rep_specs)
            _setup_camera_and_lights(scene, cam_spec)
            pred_png = out_case / f"pred_rep{rep:02d}.png"
            _render(scene, pred_png, use_gpu=use_gpu, samples=args.samples)
            pred_images.append(pred_png)

        if pred_images:
            make_image_grid(pred_images, out_case / "pred_grid.png", cols=min(3, len(pred_images)))
        else:
            logger.warning("No per-rep prediction images to build grid for case %s", case_dir)

        # Keep compatibility with the requested per-case-folder 'renders/' layout
        copy_outputs_to_inplace(
            case_dir,
            out_case,
            ["overlay.png", "gt.png", "pred_grid.png"] + [f"pred_rep{rep:02d}.png" for rep in selected_reps],
        )

    logger.info("Done. Global render root: %s", out_root)


if __name__ == "__main__":
    main()
