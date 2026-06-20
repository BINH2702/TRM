#!/usr/bin/env bash
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --job-name=maze_cl_gate
#SBATCH --output=/mnt/data/binhnt6/trm_runs/logs/maze_cl_gate_%j.out
#SBATCH --error=/mnt/data/binhnt6/trm_runs/logs/maze_cl_gate_%j.err

set -euo pipefail

cd /home/binhnt6/TinyRecursiveModels

RUN_ID="${RUN_ID:-maze_cl_gate_pathlen_$(date +%Y%m%d_%H%M%S)}"

export DISABLE_COMPILE="${DISABLE_COMPILE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-/mnt/data/binhnt6/trm_runs/wandb}"

export SOURCE_DATA="${SOURCE_DATA:-/mnt/data/binhnt6/trm_data/maze-30x30-hard-1k-noaug}"
export CL_DATA_DIR="${CL_DATA_DIR:-/mnt/data/binhnt6/trm_data/cl_splits/${RUN_ID}}"
export RUN_PREFIX="${RUN_PREFIX:-${RUN_ID}}"
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/mnt/data/binhnt6/trm_runs/checkpoints/${RUN_ID}}"
export RESULT_ROOT="${RESULT_ROOT:-/mnt/data/binhnt6/trm_runs/results/${RUN_ID}}"
export RESULT_CSV="${RESULT_CSV:-${RESULT_ROOT}/learnability_gate_endpoint_matrix.csv}"

export SPLIT_KEY="${SPLIT_KEY:-maze_path_length}"
export TASK_A_SIDE="${TASK_A_SIDE:-low}"
export TASK_A_FRACTION="${TASK_A_FRACTION:-0.5}"
export MAX_TRAIN_GROUPS="${MAX_TRAIN_GROUPS:-1000}"
export MAX_TEST_GROUPS="${MAX_TEST_GROUPS:-400}"

export ARCH="${ARCH:-trm}"
export EPOCHS_A="${EPOCHS_A:-2500}"
export EPOCHS_B="${EPOCHS_B:-2500}"
export EPOCHS_JOINT="${EPOCHS_JOINT:-2500}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-2500}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
export EMA="${EMA:-False}"
export LR="${LR:-1e-4}"
export PUZZLE_EMB_LR="${PUZZLE_EMB_LR:-1e-4}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-1.0}"
export PUZZLE_EMB_WEIGHT_DECAY="${PUZZLE_EMB_WEIGHT_DECAY:-1.0}"
export EXTRA_PRETRAIN_ARGS="${EXTRA_PRETRAIN_ARGS:-arch.mlp_t=False arch.L_layers=2 lr_warmup_steps=0 lr_min_ratio=1.0}"

mkdir -p /mnt/data/binhnt6/trm_runs/logs "${RESULT_ROOT}"
echo "${RUN_ID}" > "${RESULT_ROOT}/RUN_ID.txt"

echo "RUN_ID=${RUN_ID}"
echo "SOURCE_DATA=${SOURCE_DATA}"
echo "CL_DATA_DIR=${CL_DATA_DIR}"
echo "CHECKPOINT_ROOT=${CHECKPOINT_ROOT}"
echo "RESULT_CSV=${RESULT_CSV}"

bash scripts/run_single_dataset_cl_learnability_gate.sh
