## Training

```bash
python main.py -c config/env.yaml scenarios/chi3d.yaml -- \
  run.name=chi3d run.job=train
```

## Sample

MESHES
```bash
python main.py -c \
  config/env.yaml \
  scenarios/chi3d.yaml -- \
  run.job=sample \
  run.name=008_chi3d \
  sample.target=meshes \
  resume.checkpoint="experiments/008_chi3d/checkpoints/checkpoint-step-0005000.pth" \
  resume.step=5000 \
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
  run.name=008_chi3d \
  sample.target=hdf5 \
  resume.checkpoint="experiments/008_chi3d/checkpoints/checkpoint-step-0005000.pth" \
  resume.step=5000 \
  dataloader.batch_size=4096 \
  sample.mode=sample_01 \
  run.datasets='[chi3d]' \
  sample.dataset=normal \
  sample.repetitions=3
```

## Eval

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


## Prepocessing chi3d

```bash
python -m tridi.preprocessing.preprocess_chi3d -c ./config/env.yaml -- chi3d.split="train" chi3d.root="../chi3d/raw" chi3d.assets="./data/preprocessed/chi3d_smplx" chi3d.downsample="25fps"
python -m tridi.preprocessing.preprocess_chi3d -c ./config/env.yaml -- chi3d.split="val" chi3d.root="../chi3d/raw" chi3d.assets="./data/preprocessed/chi3d_smplx" chi3d.downsample="25fps"
python -m tridi.preprocessing.preprocess_chi3d -c ./config/env.yaml -- chi3d.split="test" chi3d.root="../chi3d/raw" chi3d.assets="./data/preprocessed/chi3d_smplx" chi3d.downsample="25fps"
```