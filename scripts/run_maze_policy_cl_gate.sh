#!/usr/bin/env bash
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name=binh_job
#SBATCH --output=/mnt/data/binhnt6/trm_runs/logs/binh_job_%j.out
#SBATCH --error=/mnt/data/binhnt6/trm_runs/logs/binh_job_%j.err

set -euo pipefail

cd /home/binhnt6/TinyRecursiveModels

RUN_ID="${RUN_ID:-binh_job_$(date +%Y%m%d_%H%M%S)}"
SOURCE_MAZE_DATA="${SOURCE_MAZE_DATA:-/mnt/data/binhnt6/trm_data/maze-30x30-hard-1k-noaug}"
POLICY_DATA="${POLICY_DATA:-/mnt/data/binhnt6/trm_data/maze-30x30-hard-1k-noaug-next-hop-policy}"

mkdir -p /mnt/data/binhnt6/trm_runs/logs
mkdir -p /mnt/data/binhnt6/trm_runs/wandb /mnt/data/binhnt6/cache /mnt/data/binhnt6/tmp

export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DIR="${WANDB_DIR:-/mnt/data/binhnt6/trm_runs/wandb}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/mnt/data/binhnt6/cache}"
export TMPDIR="${TMPDIR:-/mnt/data/binhnt6/tmp}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/mnt/data/binhnt6/cache/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/mnt/data/binhnt6/cache/triton}"

python -m dataset.build_maze_policy_dataset \
  --input-dir "${SOURCE_MAZE_DATA}" \
  --output-dir "${POLICY_DATA}" \
  --overwrite

export TASK_A_DATA="${SOURCE_MAZE_DATA}"
export TASK_B_DATA="${POLICY_DATA}"
export RUN_PREFIX="${RUN_ID}"
export CL_DATA_DIR="${CL_DATA_DIR:-/mnt/data/binhnt6/trm_data/cl_two_task/${RUN_ID}}"
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/mnt/data/binhnt6/trm_runs/checkpoints/${RUN_ID}}"
export RESULT_ROOT="${RESULT_ROOT:-/mnt/data/binhnt6/trm_runs/results/${RUN_ID}}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
export EPOCHS_A="${EPOCHS_A:-250}"
export EPOCHS_B="${EPOCHS_B:-250}"
export EPOCHS_JOINT="${EPOCHS_JOINT:-250}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-250}"
export EMA="${EMA:-False}"
export EXTRA_PRETRAIN_ARGS="${EXTRA_PRETRAIN_ARGS:-arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=4 arch.mlp_t=False lr_warmup_steps=0 lr_min_ratio=1.0}"

bash scripts/run_two_dataset_cl_learnability_gate.sh
