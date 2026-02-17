# HH_gen Blendify Visualization Toolchain (Independent of train/sample/eval)

This folder provides a **minimal-intrusion** rendering pipeline:
- It only reads exported PLY files from `_viz/ply/...`
- It does **not** import `ProjectConfig`, `tridi`, `HHBatchData`, or training code
- It is intended to run in a separate **Python 3.11** env

---

## 1) Select cases from sampling hdf5

First, generate a compact case list from `samples_rep_*.hdf5`:

```bash
python tridi/tools/vis/select_cases_from_hdf5.py \
  --samples_dir "/media/uv/Data/workspace/HH_gen/experiments/021_chi3d/artifacts/step_30000_samples/chi3d/sbj" \
  --dataset_root "/media/uv/Data/workspace/HH_gen/tridi/data/preprocessed/chi3d_smplx" \
  --mode 10
```

This writes:
- `<samples_dir>/_viz/cases.json`

---

## 2) Export selected cases to PLY

```bash
python tridi/tools/vis/export_cases_to_ply.py \
  --samples_dir "/media/uv/Data/workspace/HH_gen/experiments/021_chi3d/artifacts/step_30000_samples/chi3d/sbj" \
  --dataset_root "/media/uv/Data/workspace/HH_gen/tridi/data/preprocessed/chi3d_smplx" \
  --smplx_model_dir "/media/uv/Data/workspace/HH_gen/tridi/data/smplx_models" \
  --mode 10 \
  --k_preds 3 \
  --rep_strategy best_median_worst \
  --use_batch_gender \
  --config_env "/media/uv/Data/workspace/HH_gen/config/env.yaml" \
  --scenario_yaml "/media/uv/Data/workspace/HH_gen/scenarios/chi3d.yaml" \
  --datasets "chi3d" \
  --default_sbj_gender male \
  --default_second_sbj_gender female \
  --default_gender neutral
```

Notes:
- `--smplx_model_dir` is required when GT hdf5 does not contain `sbj_v/sbj_f` and mesh fallback from SMPL-X params is needed.
- If `<smplx_model_dir>/smplx` exists, loader will prefer that subfolder automatically.
- `--use_batch_gender` (default enabled) builds a gender lookup from eval dataloader batches for cases in `cases.json`.
- If batch-gender lookup cannot be built (missing config files / missing gender fields), script logs warning and falls back automatically.
- Gender priority for fallback mesh generation (per role):
  1) batch gender map (`sbj_gender` / `second_sbj_gender` from eval dataloader; True=female, False=male)
  2) hdf5 group dataset/attr `sbj_gender` or `second_sbj_gender`
  3) `--default_sbj_gender` / `--default_second_sbj_gender`
  4) `--default_gender` (final fallback)
- You can alternatively set environment variable `SMPLX_MODEL_DIR` instead of passing `--smplx_model_dir`.

This writes case folders under:
- `<samples_dir>/_viz/ply/<seq>_tXXXXX/`

---

## 3) Create visualization environment (Python 3.11)

```bash
conda create -n hhgen_viz python=3.11 -y
conda activate hhgen_viz
pip install -r tools/viz/requirements_viz.txt
```

---

## 4) Run batch rendering

```bash
python tools/viz/render_blendify_cases.py \
  --ply_root "/media/uv/Data/workspace/HH_gen/experiments/021_chi3d/artifacts/step_30000_samples/chi3d/sbj/_viz/ply" \
  --out_root "/media/uv/Data/workspace/HH_gen/experiments/021_chi3d/artifacts/step_30000_samples/chi3d/sbj/_viz/renders" \
  --k_preds 3 \
  --res 1024 \
  --camera auto \
  --device cpu
```

### Useful options
- `--camera {auto,fixed}` (default: `auto`)
- `--W/--H` to override square resolution from `--res`
- `--dry_run` to print discovered case folders without rendering
- `--samples` rendering samples (default `128`)

---

## 5) Optional: rebuild prediction grids only

This script now scans recursively under `--renders_root` (when root is a parent folder),
groups images by case directory, and writes one grid per case.

```bash
python tools/viz/render_blendify_grid.py \
  --renders_root "/media/uv/Data/workspace/HH_gen/experiments/021_chi3d/artifacts/step_30000_samples/chi3d/sbj/_viz/renders" \
  --pattern "pred_rep*.png" \
  --cols 3 \
  --out_name "pred_grid.png"
```

If `--renders_root` points directly to one case folder, it will build only that case's grid.

Useful options:
- `--out_name` output filename inside each case folder (default: `pred_grid.png`)
- `--dry_run` only print discovered case folders and matched image counts, without writing files

---

## Input assumptions

Each case folder in `_viz/ply/<case_name>/` may contain (robustly handled if missing):
- `condition*.ply`
- `gt*.ply`
- `pred_repXX*.ply`
- `meta.json`

Missing assets are skipped with warnings.

---

## Output layout

Primary output is written to:
- `<out_root>/<case_name>/overlay.png`
- `<out_root>/<case_name>/gt.png`
- `<out_root>/<case_name>/pred_repXX.png`
- `<out_root>/<case_name>/pred_grid.png`

For compatibility with per-case layout, files are also copied to:
- `<ply_root>/<case_name>/renders/...`

---

## Visual convention 

- Condition: blue shades
- Predictions: green shades (different shade per rep)
- GT: yellow shades
- Same camera angle is reused for all images of one case (`overlay`, `gt`, `pred_repXX`, `pred_grid`)
