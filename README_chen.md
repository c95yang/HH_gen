## Training

```bash
python main.py -c config/env.yaml scenarios/chi3d_overfit.yaml -- \
  run.name=004_chi3d_overfit run.job=train
```

## Sample

MESHES
```bash
python main.py -c config/env.yaml \
  scenarios/chi3d_full.yaml -- \
  run.job=sample \
  run.name=002_chi3d_full \
  sample.target=meshes \
  resume.checkpoint="experiments/002_chi3d_full/checkpoints/checkpoint-step-0008000.pth" \
  resume.step=8000 \
  dataloader.batch_size=4096 \
  sample.mode=sample_01 \
  run.datasets='[chi3d]' \
  sample.dataset=normal \
  sample.repetitions=3
```
HDF5
```bash
python main.py -c config/env.yaml \
  scenarios/chi3d_overfit_aug.yaml -- \
  run.job=sample \
  run.name=005_chi3d_overfit_aug \
  sample.target=hdf5 \
  resume.checkpoint="experiments/005_chi3d_overfit_aug/checkpoints/checkpoint-step-0008000.pth" \
  resume.step=8000 \
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
  run.name=005_chi3d_overfit_aug \
  resume.step=8000 \
  run.datasets=["chi3d"] \
  eval.sampling_target=["sbj","second_sbj"] \
  eval.use_gen_metrics=true eval.use_rec_metrics=true
```
