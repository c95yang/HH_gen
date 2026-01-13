python main.py -c config/env.yaml scenarios/chi3d.yaml -- \
  run.job=sample \
  run.name=chi3d2 \
  sample.target=meshes \
  resume.checkpoint="experiments/003_chi3d2/checkpoints/checkpoint-step-0020000.pth" \
  resume.step=20000 \
  dataloader.batch_size=2048 \
  sample.mode=sample_01 \
  run.datasets='[chi3d]' \
  sample.dataset=normal \
  sample.repetitions=3

python main.py -c config/env.yaml scenarios/chi3d.yaml -- \
  run.job=sample \
  run.name=chi3d2_1fps \
  sample.target=hdf5 \
  resume.checkpoint="experiments/003_chi3d2/checkpoints/checkpoint-step-0020000.pth" \
  resume.step=20000 \
  run.datasets='["chi3d"]' \
  chi3d.fps_eval=1 \
  chi3d.test_split_file="/media/uv/Data/workspace/HH_gen/tridi/data/preprocessed/chi3d_smplx/chi3d_test_1fps_split.json" \
  dataloader.batch_size=256 \
  dataloader.workers=0 \
  sample.mode=sample_01 \
  sample.dataset=normal \
  sample.repetitions=3

python main.py -c config/env.yaml scenarios/chi3d.yaml -- \
  run.job=eval run.name=chi3d resume.step=50000 \
  'run.datasets=["chi3d"]' \
  'eval.sampling_target=["sbj","second_sbj"]' \
  eval.use_gen_metrics=true eval.use_rec_metrics=false