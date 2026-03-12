# Training
```bash
python main.py -c config/env.yaml scenarios/chi3d.yaml -- \
  run.name=chi3d run.job=train
```

## Current trained main model (source of truth)

The finalized run is [experiments/029_chi3d_hhi_full/config.yaml](experiments/029_chi3d_hhi_full/config.yaml) with:
- full trilateral diffusion (`H1/H2/Interaction`)
- `model.use_interaction_diffusion=true`
- `model_conditioning.use_interaction_conditioning=true`
- `model.data_interaction_channels=128`
- CHI3D split files currently used by that run:
  - `chi3d_train.json`, `chi3d_val.json`, `chi3d_test.json`

### Full train command (same setup family)

```bash
python main.py -c config/env.yaml scenarios/chi3d.yaml -- \
  run.job=train \
  run.name=031_chi3d_hhi_split2 \
  chi3d.root=/media/uv/Data/workspace/HH_gen/tridi/data/preprocessed/chi3d_smplx \
  chi3d.train_split_file=/media/uv/Data/workspace/HH_gen/tridi/data/preprocessed/chi3d_smplx/chi3d_train.json \
  chi3d.val_split_file=/media/uv/Data/workspace/HH_gen/tridi/data/preprocessed/chi3d_smplx/chi3d_val.json \
  chi3d.test_split_file=/media/uv/Data/workspace/HH_gen/tridi/data/preprocessed/chi3d_smplx/chi3d_test.json \
  model.use_interaction_diffusion=true \
  model_conditioning.use_interaction_conditioning=true \
  model_conditioning.interaction_source=both \
  train.losses.denoise_interaction=5.0
```

### Optional: prototype-aligned interaction latent training

This keeps the existing diffusion interaction branch, but adds an auxiliary
prototype classification loss on the interaction latent space:
- `train.losses.interaction_proto_ce`: weighted CE on cosine-similarity logits
- prototypes come from the same shared text/contact-conditioned interaction space
- both the target interaction latent and the predicted interaction latent are supervised

Recommended starter run:

```bash
python main.py -c config/env.yaml scenarios/chi3d.yaml -- \
  run.job=train \
  run.name=032_chi3d_hhi_split1_protoce \
  chi3d.root=/media/uv/Data/workspace/HH_gen/tridi/data/preprocessed/chi3d_smplx \
  chi3d.train_split_file=/media/uv/Data/workspace/HH_gen/tridi/data/preprocessed/chi3d_smplx/chi3d_train.json \
  chi3d.val_split_file=/media/uv/Data/workspace/HH_gen/tridi/data/preprocessed/chi3d_smplx/chi3d_val.json \
  chi3d.test_split_file=/media/uv/Data/workspace/HH_gen/tridi/data/preprocessed/chi3d_smplx/chi3d_test.json \
  model.use_interaction_diffusion=true \
  model_conditioning.use_interaction_conditioning=true \
  model_conditioning.interaction_source=both \
  model_conditioning.interaction_proto_temperature=0.07 \
  model_conditioning.interaction_proto_normalize=true \
  train.losses.denoise_interaction=5.0 \
  train.losses.interaction_proto_ce=1.0
```

# Sample

Mode semantics with interaction diffusion enabled:
- `sample.mode="100"`: sample `sbj` (condition on `second_sbj` + interaction)
- `sample.mode="010"`: sample `second_sbj` (condition on `sbj` + interaction)
- `sample.mode="001"`: sample `interaction` (condition on `sbj` + `second_sbj`)
- `sample.mode="110"`: sample both humans (condition on interaction)
- `sample.mode="111"`: sample all three branches

Note for leading-zero modes: pass an explicit string literal in dotlist overrides,
e.g. `sample.mode='"010"'` and `sample.mode='"001"'`.

## Current main-model sampling commands (for eval inputs)

Generate `sbj` samples hdf5:
```bash
python main.py -c config/env.yaml scenarios/chi3d.yaml -- \
  run.job=sample \
  run.name=032_chi3d_hhi_split1_protoce \
  sample.target=hdf5 \
  resume.checkpoint="experiments/032_chi3d_hhi_split1_protoce/checkpoints/checkpoint-step-0035000.pth" \
  resume.step=35000 \
  dataloader.batch_size=1024 \
  sample.mode="100" \
  run.datasets='["chi3d"]' \
  sample.dataset=normal \
  sample.repetitions=3
```

Generate `second_sbj` samples hdf5:
```bash
python main.py -c config/env.yaml scenarios/chi3d.yaml -- \
  run.job=sample \
  run.name=032_chi3d_hhi_split1_protoce \
  sample.target=hdf5 \
  resume.checkpoint="experiments/032_chi3d_hhi_split1_protoce/checkpoints/checkpoint-step-0035000.pth" \
  resume.step=35000 \
  dataloader.batch_size=1024 \
  sample.mode='"010"' \
  run.datasets='["chi3d"]' \
  sample.dataset=normal \
  sample.repetitions=3
```

Optional qualitative mesh export (sample both humans):
```bash
python main.py -c config/env.yaml scenarios/chi3d.yaml -- \
  run.job=sample \
  run.name=029_chi3d_hhi_full \
  sample.target=meshes \
  resume.checkpoint="experiments/029_chi3d_hhi_full/checkpoints/checkpoint-step-0030000.pth" \
  resume.step=30000 \
  dataloader.batch_size=4 \
  sample.mode="110" \
  run.datasets='["chi3d"]' \
  sample.dataset=normal \
  sample.repetitions=1
```

## Interaction-only inference check (mode=001)

Generate interaction branch only (`interaction_latent`) from fixed humans:
```bash
python main.py -c config/env.yaml scenarios/chi3d.yaml -- \
  run.job=sample \
  run.name=032_chi3d_hhi_split1_protoce \
  sample.target=hdf5 \
  resume.checkpoint="experiments/032_chi3d_hhi_split1_protoce/checkpoints/checkpoint-step-0035000.pth" \
  resume.step=35000 \
  run.datasets='["chi3d"]' \
  sample.mode='"001"' \
  sample.split=test \
  sample.dataset=normal \
  sample.repetitions=1 \
  dataloader.batch_size=256 \
  logging.wandb=false
```

Decode predicted `interaction_latent` to action labels (nearest cosine to action prototypes):
```bash
python -m tridi.tools.inspect_mode001_interaction \
  -c config/env.yaml scenarios/chi3d.yaml \
  run.name=032_chi3d_hhi_split1_protoce \
  resume.checkpoint=experiments/032_chi3d_hhi_split1_protoce/checkpoints/checkpoint-step-0035000.pth \
  resume.step=35000 \
  run.datasets='["chi3d"]' \
  sample.split=test \
  dataloader.batch_size=256 \
  --sampling_strategy stratified \
  --seed 42 \
  --max_items 100 \
  --topk 3
```

`--sampling_strategy` options:
- `sequential`: first `max_items` in dataloader order
- `random`: shuffle all candidates, then keep `max_items`
- `stratified` (default): balance by GT label before truncation

Decoded rows are saved to:
- `experiments/029_chi3d_hhi_full/artifacts/mode001_decode_step_25000.jsonl`

Each row includes:
- GT action label (`gt_label`)
- predicted top-1 label (`pred_label_top1`)
- top-k labels/scores (`pred_topk_labels`, `pred_topk_scores`)
- raw predicted latent (`interaction_latent`)

Evaluate decode quality (match/mismatch, accuracy, confusion, top-k hit rate):
```bash
python -m tridi.tools.evaluate_mode001_decode \
  experiments/032_chi3d_hhi_split1_protoce/artifacts/mode001_decode_step_35000.jsonl
```

This writes:
- `experiments/029_chi3d_hhi_full/artifacts/mode001_decode_step_25000_summary.json`
- `experiments/029_chi3d_hhi_full/artifacts/mode001_decode_step_25000_confusion.csv`

To compare a new prototype-aligned run against the baseline, reuse the same mode=001 tools
with the new run/checkpoint/step. Example:

```bash
python -m tridi.tools.inspect_mode001_interaction \
  -c config/env.yaml scenarios/chi3d.yaml \
  run.name=032_chi3d_hhi_split2_protoce \
  resume.checkpoint=experiments/032_chi3d_hhi_split2_protoce/checkpoints/checkpoint-step-0040000.pth \
  resume.step=40000 \
  run.datasets='["chi3d"]' \
  sample.split=test \
  dataloader.batch_size=256 \
  --sampling_strategy stratified \
  --seed 42 \
  --max_items 100 \
  --topk 3

python -m tridi.tools.evaluate_mode001_decode \
  experiments/032_chi3d_hhi_split2_protoce/artifacts/mode001_decode_step_40000.jsonl

python -m tridi.tools.analyze_mode001_decode \
  -c config/env.yaml scenarios/chi3d.yaml \
  run.name=032_chi3d_hhi_split2_protoce \
  resume.checkpoint=experiments/032_chi3d_hhi_split2_protoce/checkpoints/checkpoint-step-0040000.pth \
  resume.step=40000 \
  run.datasets='["chi3d"]' \
  sample.split=test \
  dataloader.batch_size=256 \
  --jsonl experiments/032_chi3d_hhi_split2_protoce/artifacts/mode001_decode_step_40000.jsonl \
  --max_gt_items 500 \
  --seed 42
```

# Eval

Evaluate from generated hdf5 samples:
```bash
python main.py -c config/env.yaml \
  scenarios/chi3d.yaml -- \
  run.job=eval \
  run.name=032_chi3d_hhi_split1_protoce \
  resume.step=35000 \
  run.datasets='["chi3d"]' \
  eval.samples_folder="experiments/032_chi3d_hhi_split1_protoce/artifacts/step_35000_samples" \
  eval.sampling_target='["sbj","second_sbj"]' \
  eval.pose_only_like_baseline=false \
  eval.use_gen_metrics=true eval.use_rec_metrics=true
```

# Prepocessing chi3d

Before running prepocesssing, put chi3d raw data in this folder structure
```
chi3d
└── raw
    ├── chi3d_info.json
    ├── split.json
    ├── test
    └── train
HH_gen
└── ...
```

```bash
python -m tridi.preprocessing.preprocess_chi3d -c ./config/env.yaml -- chi3d.split="train" chi3d.root="../chi3d/raw" chi3d.assets="./data/preprocessed/chi3d_smplx" chi3d.downsample="25fps"
python -m tridi.preprocessing.preprocess_chi3d -c ./config/env.yaml -- chi3d.split="val" chi3d.root="../chi3d/raw" chi3d.assets="./data/preprocessed/chi3d_smplx" chi3d.downsample="25fps"
python -m tridi.preprocessing.preprocess_chi3d -c ./config/env.yaml -- chi3d.split="test" chi3d.root="../chi3d/raw" chi3d.assets="./data/preprocessed/chi3d_smplx" chi3d.downsample="25fps"

# Optional: interaction-aware split5 (uses interaction_contact_signature.json)
python -m tridi.preprocessing.preprocess_chi3d -c ./config/env.yaml -- chi3d.split="train" chi3d.split_strategy="split5" chi3d.split5_min_contact_edges=1 chi3d.split5_use_active_window=true chi3d.split5_window_margin_ratio=0.3 chi3d.split5_window_min_frames=32 chi3d.split5_keep_full_if_no_signal=true chi3d.root="/media/uv/Data/workspace/HH_gen/tridi/data/raw/chi3d" chi3d.assets="/media/uv/Data/workspace/HH_gen/tridi/data/preprocessed/chi3d_smplx" chi3d.downsample="25fps"
python -m tridi.preprocessing.preprocess_chi3d -c ./config/env.yaml -- chi3d.split="val" chi3d.split_strategy="split5" chi3d.split5_min_contact_edges=1 chi3d.split5_use_active_window=true chi3d.split5_window_margin_ratio=0.3 chi3d.split5_window_min_frames=32 chi3d.split5_keep_full_if_no_signal=true chi3d.root="/media/uv/Data/workspace/HH_gen/tridi/data/raw/chi3d" chi3d.assets="/media/uv/Data/workspace/HH_gen/tridi/data/preprocessed/chi3d_smplx" chi3d.downsample="25fps"
python -m tridi.preprocessing.preprocess_chi3d -c ./config/env.yaml -- chi3d.split="test" chi3d.split_strategy="split5" chi3d.split5_min_contact_edges=1 chi3d.split5_use_active_window=true chi3d.split5_window_margin_ratio=0.3 chi3d.split5_window_min_frames=32 chi3d.split5_keep_full_if_no_signal=true chi3d.root="/media/uv/Data/workspace/HH_gen/tridi/data/raw/chi3d" chi3d.assets="/media/uv/Data/workspace/HH_gen/tridi/data/preprocessed/chi3d_smplx" chi3d.downsample="25fps"

# split5 active-window rule:
# keep [start_fr, end_fr] from interaction_contact_signature.json,
# then expand by margin_ratio * active_length and enforce split5_window_min_frames.
```

After prepocessing, data folder structure should like this.
```
HH_gen
└── data
    ├── preprocessed
    │   └── chi3d_smplx
    │       ├── chi3d_test.json
    │       ├── chi3d_test_split5.json
    │       ├── chi3d_train.json
    │       ├── chi3d_train_split5.json
    │       ├── chi3d_val.json
    │       ├── chi3d_val_split5.json
    │       ├── dataset_test_25fps.hdf5
    │       ├── dataset_train_25fps.hdf5
    │       ├── dataset_val_25fps.hdf5
    │       ├── preprocess_config_test_25fps.yaml
    │       ├── preprocess_config_train_25fps.yaml
    │       └── preprocess_config_val_25fps.yaml
    └── smplx_models
        ├── ...
```