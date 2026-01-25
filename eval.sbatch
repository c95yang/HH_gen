#!/bin/bash
#SBATCH -p NORMAL
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#####SBATCH -w node4,node5,node8,node9,node10,node11,node12,node13,node14,node15,node16,node17,node18,node19,node20,node21
#SBATCH -w node4,node5,node9,node11,node12,node13,node14,node15,node16,node17,node18,node19,node20,node21

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tridi

mkdir -p logs

echo "JOB=$SLURM_JOB_ID HOST=$(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

scenario=chi3d.yaml
run_name=001_chi3d_aug
step=50000
ckpt=experiments/${run_name}/checkpoints/checkpoint-step-00${step}.pth
target=hdf5

common_args=(
  -c config/env.yaml scenarios/${scenario} --
  run.job=sample
  run.name=${run_name}
  sample.target=${target}
  resume.checkpoint=${ckpt}
  resume.step=${step}
  dataloader.batch_size=4096
  run.datasets='[chi3d]'
  sample.dataset=normal
  sample.repetitions=3
)

echo "==> Run sample_10 ..."
python main.py "${common_args[@]}" sample.mode=sample_10

echo "==> Run sample_01 ..."
python main.py "${common_args[@]}" sample.mode=sample_01

echo "==> Run eval ..."
python main.py -c config/env.yaml scenarios/${scenario} -- \
  run.job=eval \
  run.name="${run_name}" \
  resume.step="${step}" \
  run.datasets='["chi3d"]' \
  eval.sampling_target='["sbj","second_sbj"]' \
  eval.use_gen_metrics=true \
  eval.use_rec_metrics=true \
  eval.nn_baseline=true eval.nn_baseline_ref_split=train eval.nn_baseline_k=1

echo "==> Done."
