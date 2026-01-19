## Training

```bash
python main.py -c config/env.yaml scenarios/chi3d_overfit.yaml -- \
  run.name=004_chi3d_overfit run.job=train
```

## Sample

MESHES
```bash
python main.py -c \
  config/env.yaml \
  scenarios/chi3d_overfit_aug.yaml -- \
  run.job=sample \
  run.name=005_chi3d_overfit_aug \
  sample.target=meshes \
  resume.checkpoint="experiments/005_chi3d_overfit_aug/checkpoints/checkpoint-step-0018000.pth" \
  resume.step=18000 \
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
  scenarios/chi3d_overfit_aug.yaml -- \
  run.job=sample \
  run.name=005_chi3d_overfit_aug \
  sample.target=hdf5 \
  resume.checkpoint="experiments/005_chi3d_overfit_aug/checkpoints/checkpoint-step-0018000.pth" \
  resume.step=18000 \
  dataloader.batch_size=4096 \
  sample.mode=sample_01 \
  run.datasets='[chi3d]' \
  sample.dataset=normal \
  sample.repetitions=3
```

## Eval

```bash
python main.py -c config/env.yaml \
  scenarios/chi3d_overfit_aug.yaml -- \
  run.job=eval \
  run.name=005_chi3d_overfit_aug  \
  resume.step=18000 \
  run.datasets=["chi3d"] \
  eval.sampling_target=["sbj","second_sbj"] \
  eval.use_gen_metrics=true eval.use_rec_metrics=true
```
