```bash
python main.py -c config/env.yaml scenarios/chi3d.yaml -- \
  run.job=sample run.name=000_chi3d sample.target=meshes \
  resume.checkpoint="experiments/000_chi3d/checkpoints/checkpoint-step-0050000.pth" resume.step=50000 \
  dataloader.batch_size=2048 sample.mode="sample_10" \
  run.datasets=["chi3d"] sample.dataset=normal sample.repetitions=3
```

```bash
python main.py -c config/env.yaml scenarios/chi3d.yaml -- \
  run.job=sample \
  run.name=chi3d \
  sample.target=hdf5 \
  resume.checkpoint="experiments/chi3d/checkpoints/checkpoint-step-0050000.pth" \
  resume.step=50000 \
  dataloader.batch_size=2048 \
  sample.mode=sample_01 \
  run.datasets='[chi3d]' \
  sample.dataset=normal \
  sample.repetitions=3
```

```bash
python main.py -c config/env.yaml scenarios/chi3d.yaml -- \
  run.job=sample \
  run.name=chi3d2 \
  sample.target=meshes \
  resume.checkpoint="experiments/000_chi3d2/checkpoints/checkpoint-step-0050000.pth" \
  resume.step=50000 \
  dataloader.batch_size=2048 \
  sample.mode=sample_10 \
  run.datasets='[chi3d]' \
  sample.dataset=normal \
  sample.repetitions=3
```

Use the command below to run evaluation on the generated samples. The `eval.sampling_target` parameter controls 
which modalities are evaluated (possible values: `sbj_contact`, `obj_contact`,):
```bash
python main.py -c config/env.yaml scenarios/mirror.yaml -- \
  run.job=eval run.name=001_01_mirror resume.step=-1 eval.sampling_target=["sbj","second_sbj"] 
```


evaluate gen_metrics together,but reconstruction metrics separately
```bash
python main.py -c config/env.yaml scenarios/chi3d.yaml -- \
  run.job=eval run.name=008_chi3d2_1fps resume.step=20000 \
  'run.datasets=["chi3d"]' \
  'eval.sampling_target=["sbj","second_sbj"]' \
  eval.use_gen_metrics=true eval.use_rec_metrics=false \
  eval.samples_folder=experiments/008_chi3d2_1fps/artifacts/step_20000_samples
```

```bash
python main.py -c config/env.yaml scenarios/chi3d.yaml -- \
  run.job=eval run.name=008_chi3d2_1fps resume.step=20000 \
  'run.datasets=["chi3d"]' \
  'eval.sampling_target=["sbj"]' \
  eval.use_gen_metrics=false eval.use_rec_metrics=true \
  eval.samples_folder=experiments/008_chi3d2_1fps/artifacts/step_20000_samples
```

```bash
python main.py -c config/env.yaml scenarios/chi3d.yaml -- \
  run.job=eval run.name=008_chi3d2_1fps resume.step=20000 \
  'run.datasets=["chi3d"]' \
  'eval.sampling_target=["second_sbj"]' \
  eval.use_gen_metrics=false eval.use_rec_metrics=true \
  eval.samples_folder=experiments/008_chi3d2_1fps/artifacts/step_20000_samples
```

## Training
Use the following command to run the training:

python main.py -c config/env.yaml scenarios/gb_main.yaml -- \
  run.name=001_gb_main run.job=train

```bash
python main.py -c config/env.yaml scenarios/chi3d.yaml -- \
  run.name=chi3d run.job=train
```

```bash
python main.py -c config/env.yaml scenarios/chi3d.yaml -- \
  run.job=sample run.name=chi3d sample.target=meshes \
  resume.checkpoint="experiments/chi3d/checkpoints/checkpoint-step-0020000.pth" resume.step=20000 \
  dataloader.batch_size=512 sample.mode="sample_10" \
  run.datasets=["chi3d"] sample.dataset=normal sample.repetitions=3
```
