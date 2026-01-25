#!/bin/bash
#SBATCH -p NORMAL
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -o logs/output_%A_%a.txt
#SBATCH -e logs/error_%A_%a.txt
#####SBATCH -w node4,node5,node8,node9,node10,node11,node12,node13,node14,node15,node16,node17,node18,node19,node20,node21
#SBATCH -w node4,node5,node9,node11,node12,node13,node14,node15,node16,node17,node18,node19,node20,node21

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tridi

echo "JOB=$SLURM_JOB_ID ARRAY=$SLURM_ARRAY_TASK_ID HOST=$(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi


if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
  python main.py -c config/env.yaml scenarios/chi3d.yaml -- run.name=000_chi3d run.job=train
else
  python main.py -c config/env.yaml scenarios/chi3d_aug.yaml  -- run.name=001_chi3d_aug  run.job=train
fi

