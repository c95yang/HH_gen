## Architecture

```
Query Person A (test)
        ↓
Extract Features
        ↓
Find Nearest Match in Training Data
        ↓
Retrieve Paired Person B from Training
        ↓
Output: Person B's Features (as meshes or HDF5)
```

## Usage

### Generate Baseline Samples (Meshes)

```bash
python main.py -c config/env.yaml scenarios/chi3d.yaml -- \
  run.job=baseline \
  run.name=002_baseline_poseshape \
  sample.target=meshes \
  dataloader.batch_size=4096 \
  sample.mode=sample_10 \
  run.datasets='["chi3d"]' \
  sample.repetitions=1
```

### Generate Baseline Samples (HDF5)

```bash
python main.py -c config/env.yaml scenarios/chi3d.yaml -- \
  run.job=baseline \
  run.name=002_baseline_poseshape \
  sample.target=hdf5 \
  dataloader.batch_size=4096 \
  sample.mode=sample_10 \
  run.datasets='["chi3d"]' \
  sample.repetitions=1
```

### Sampler Integration

The `Sampler` class treats baseline model same as TriDi:
- Calls `model.forward(batch)` to get predictions
- Converts joints to meshes (if target=meshes)
- Saves to HDF5 (if target=hdf5)
- No modifications needed!

## Evaluation

After generating baseline samples, evaluate:

```bash
python main.py -c config/env.yaml scenarios/chi3d.yaml -- \
  run.job=eval \
  run.name=002_baseline_poseshape \
  'run.datasets=["chi3d"]' \
  'eval.sampling_target=["sbj","second_sbj"]' \
  eval.use_gen_metrics=true \
  eval.use_rec_metrics=true
```