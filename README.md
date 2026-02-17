# Training
```bash
python main.py -c config/env.yaml scenarios/chi3d.yaml -- \
  run.name=chi3d run.job=train
```

# Sample

MESHES
```bash
python main.py -c \
  config/env.yaml \
  scenarios/chi3d.yaml -- \
  run.job=sample \
  run.name=019_chi3d \
  sample.target=meshes \
  resume.checkpoint="experiments/021_chi3d/checkpoints/checkpoint-step-0030000.pth" \
  resume.step=30000 \
  dataloader.batch_size=4 \
  sample.mode=sample_01 \
  run.datasets='[chi3d]' \
  sample.dataset=normal \
  sample.repetitions=1
```

HDF5
```bash
python main.py -c \
  config/env.yaml \
  scenarios/chi3d.yaml -- \
  run.job=sample \
  run.name=021_chi3d \
  sample.target=hdf5 \
  resume.checkpoint="experiments/021_chi3d/checkpoints/checkpoint-step-0030000.pth" \
  resume.step=30000 \
  dataloader.batch_size=4096 \
  sample.mode=sample_01 \
  run.datasets='[chi3d]' \
  sample.dataset=normal \
  sample.repetitions=3
```

force gender
```bash
python main.py -c config/env.yaml scenarios/chi3d.yaml -- \
  run.job=sample \
  run.name=020_chi3d \
  model_conditioning.use_gender_conditioning=true \
  sample.target=meshes \
  resume.checkpoint="experiments/021_chi3d/checkpoints/checkpoint-step-0030000.pth" \
  resume.step=30000 \
  dataloader.batch_size=4 \
  sample.mode=sample_10 \
  run.datasets='["chi3d"]' \
  sample.dataset=normal \
  sample.repetitions=1 \
  sample.sbj_gender=male \
  sample.second_sbj_gender=male
```

NN_comparable 
```bash
python main.py -c config/env.yaml scenarios/chi3d.yaml -- \
  run.job=sample \
  run.name=022_chi3d \
  sample.target=hdf5 \
  sample.pose_only_like_baseline=true \
  resume.checkpoint="experiments/021_chi3d/checkpoints/checkpoint-step-0030000.pth" \
  resume.step=30000 \
  dataloader.batch_size=4096 \
  sample.mode=sample_01 \
  run.datasets='["chi3d"]' \
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
  eval.pose_only_like_baseline=true \
  eval.use_gen_metrics=true eval.use_rec_metrics=true
```

with sanity check

```bash
python main.py -c config/env.yaml \
  scenarios/chi3d.yaml -- \
  run.job=eval \
  run.name=022_chi3d \
  resume.step=30000 \
  'run.datasets=["chi3d"]' \
  'eval.sampling_target=["sbj","second_sbj"]' \
  eval.sanity_gt_train_test=true \
  'eval.sanity_gt_test_test=true' \
  eval.pose_only_like_baseline=true \
  eval.sanity_seed=42 \
  eval.sanity_max_n=20000 \
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