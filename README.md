# Training
```bash
python main.py -c config/env.yaml scenarios/chi3d.yaml -- \
  run.name=chi3d run.job=train
```
# Prepare a baseline checkpoint
```bash
python - <<'PY'
from tridi.model.nn_baseline import create_nn_baseline_checkpoint
create_nn_baseline_checkpoint(
    "experiments/nn_baseline/checkpoints/checkpoint-step-0000000.pth",
    ref_split="train",
    top_k=1,     # diversity：>1；normally 1
    feature="pose6",
)
print("done")
PY

```

# Sample

MESHES
```bash
python main.py -c \
  config/env.yaml \
  scenarios/chi3d.yaml -- \
  run.job=sample \
  run.name=001_chi3d_aug \
  sample.target=meshes \
  resume.checkpoint="experiments/001_chi3d_aug/checkpoints/checkpoint-step-0050000.pth" \
  resume.step=50000 \
  dataloader.batch_size=4096 \
  sample.mode=sample_10 \
  run.datasets='[chi3d]' \
  sample.dataset=normal \
  sample.repetitions=3
```

HDF5
```bash
python main.py -c \
  config/env.yaml \
  scenarios/chi3d.yaml -- \
  run.job=sample \
  run.name=000_chi3d \
  sample.target=hdf5 \
  resume.checkpoint="experiments/000_chi3d/checkpoints/checkpoint-step-0020000.pth" \
  resume.step=20000 \
  dataloader.batch_size=4096 \
  sample.mode=sample_10 \
  run.datasets='[chi3d]' \
  sample.dataset=normal \
  sample.repetitions=3
```

```bash
python main.py -c config/env.yaml scenarios/chi3d.yaml -- \
  run.job=sample \
  run.name=020_nn_000_chi3d \
  sample.target=hdf5 \
  resume.checkpoint="experiments/nn_baseline/checkpoints/checkpoint-step-0000000.pth" \
  resume.step=0 \
  dataloader.batch_size=4096 \
  sample.mode=sample_10 \
  run.datasets='[chi3d]' \
  sample.dataset=normal \
  sample.repetitions=3
```
# Eval
```bash
python main.py -c config/env.yaml \
  scenarios/chi3d.yaml -- \
  run.job=eval \
  run.name=008_chi3d  \
  resume.step=5000 \
  run.datasets=["chi3d"] \
  eval.sampling_target=["sbj","second_sbj"] \
  eval.use_gen_metrics=true eval.use_rec_metrics=true
```

compare with nn-baseline

```bash
python main.py -c config/env.yaml scenarios/chi3d.yaml -- \
  run.job=eval \
  run.name=019_chi3d \
  resume.step=70000 \
  run.datasets='["chi3d"]' \
  eval.sampling_target='["sbj","second_sbj"]' \
  eval.use_gen_metrics=true \
  eval.use_rec_metrics=true \
  eval.method_name=hhgen
```

```bash
python main.py -c config/env.yaml scenarios/chi3d.yaml -- \
  run.job=eval \
  run.name=020_nn_000_chi3d \
  resume.step=0 \
  run.datasets='["chi3d"]' \
  eval.sampling_target='["sbj","second_sbj"]' \
  eval.use_gen_metrics=true \
  eval.use_rec_metrics=true \
  eval.method_name=NNBaseline
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
```

After prepocessing, data folder structure should like this.
```
HH_gen
└── data
    ├── preprocessed
    │   └── chi3d_smplx
    │       ├── chi3d_test.json
    │       ├── chi3d_train.json
    │       ├── chi3d_val.json
    │       ├── dataset_test_25fps.hdf5
    │       ├── dataset_train_25fps.hdf5
    │       ├── dataset_val_25fps.hdf5
    │       ├── preprocess_config_test_25fps.yaml
    │       ├── preprocess_config_train_25fps.yaml
    │       └── preprocess_config_val_25fps.yaml
    └── smplx_models
        ├── ...
```