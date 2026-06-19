#!/usr/bin/env bash
set -euo pipefail

# Minimal single-dataset continual-learning scaffold for TRM.
#
# This script keeps Task A and Task B in the same dataset family by splitting one
# prepared puzzle dataset into two compatible dataset roots, then trains:
#   1. Task A from scratch
#   2. Task B from the Task A checkpoint
#
# Required:
#   SOURCE_DATA=data/maze-30x30-hard-1k bash scripts/run_single_dataset_cl_finetune.sh

SOURCE_DATA="${SOURCE_DATA:?Set SOURCE_DATA to a prepared puzzle dataset root.}"
CL_DATA_DIR="${CL_DATA_DIR:-data/cl_single_dataset}"
TASK_A_NAME="${TASK_A_NAME:-task_a}"
TASK_B_NAME="${TASK_B_NAME:-task_b}"
TASK_A_FRACTION="${TASK_A_FRACTION:-0.5}"
SEED="${SEED:-0}"
SPLIT_KEY="${SPLIT_KEY:-random}"
TASK_A_SIDE="${TASK_A_SIDE:-low}"
MAX_TRAIN_GROUPS="${MAX_TRAIN_GROUPS:-}"
MAX_TEST_GROUPS="${MAX_TEST_GROUPS:-}"

ARCH="${ARCH:-trm}"
RUN_PREFIX="${RUN_PREFIX:-trm_cl_single_dataset}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/cl_single_dataset}"
TASK_A_CKPT_DIR="${TASK_A_CKPT_DIR:-${CHECKPOINT_ROOT}/${RUN_PREFIX}_${TASK_A_NAME}}"
TASK_B_CKPT_DIR="${TASK_B_CKPT_DIR:-${CHECKPOINT_ROOT}/${RUN_PREFIX}_${TASK_B_NAME}_finetune}"

EPOCHS_A="${EPOCHS_A:-1000}"
EPOCHS_B="${EPOCHS_B:-1000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
LR="${LR:-1e-4}"
PUZZLE_EMB_LR="${PUZZLE_EMB_LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1.0}"
PUZZLE_EMB_WEIGHT_DECAY="${PUZZLE_EMB_WEIGHT_DECAY:-1.0}"
EMA="${EMA:-True}"
EXTRA_PRETRAIN_ARGS="${EXTRA_PRETRAIN_ARGS:-}"

TASK_A_DIR="${CL_DATA_DIR}/${TASK_A_NAME}"
TASK_B_DIR="${CL_DATA_DIR}/${TASK_B_NAME}"

export WANDB_MODE="${WANDB_MODE:-offline}"

SPLIT_ARGS=()
if [[ -n "${MAX_TRAIN_GROUPS}" ]]; then
  SPLIT_ARGS+=(--max-train-groups "${MAX_TRAIN_GROUPS}")
fi
if [[ -n "${MAX_TEST_GROUPS}" ]]; then
  SPLIT_ARGS+=(--max-test-groups "${MAX_TEST_GROUPS}")
fi

python -m dataset.split_cl_dataset \
  --input-dir "${SOURCE_DATA}" \
  --output-dir "${CL_DATA_DIR}" \
  --task-a-name "${TASK_A_NAME}" \
  --task-b-name "${TASK_B_NAME}" \
  --task-a-fraction "${TASK_A_FRACTION}" \
  --split-key "${SPLIT_KEY}" \
  --task-a-side "${TASK_A_SIDE}" \
  --seed "${SEED}" \
  --overwrite \
  "${SPLIT_ARGS[@]}"

python pretrain.py \
  arch="${ARCH}" \
  data_paths="[${TASK_A_DIR}]" \
  evaluators="[]" \
  epochs="${EPOCHS_A}" \
  eval_interval="${EVAL_INTERVAL}" \
  global_batch_size="${GLOBAL_BATCH_SIZE}" \
  lr="${LR}" \
  puzzle_emb_lr="${PUZZLE_EMB_LR}" \
  weight_decay="${WEIGHT_DECAY}" \
  puzzle_emb_weight_decay="${PUZZLE_EMB_WEIGHT_DECAY}" \
  +run_name="${RUN_PREFIX}_${TASK_A_NAME}" \
  +checkpoint_path="${TASK_A_CKPT_DIR}" \
  checkpoint_every_eval=True \
  ema="${EMA}" \
  ${EXTRA_PRETRAIN_ARGS}

TASK_A_CHECKPOINT="$(
  find "${TASK_A_CKPT_DIR}" -maxdepth 1 -type f -name 'step_*' ! -name '*all_preds*' \
    | sort -V \
    | tail -1
)"

if [[ -z "${TASK_A_CHECKPOINT}" ]]; then
  echo "No Task A checkpoint found in ${TASK_A_CKPT_DIR}" >&2
  exit 1
fi

echo "Loading Task A checkpoint for Task B: ${TASK_A_CHECKPOINT}"

python pretrain.py \
  arch="${ARCH}" \
  data_paths="[${TASK_B_DIR}]" \
  data_paths_test="[${TASK_A_DIR}]" \
  evaluators="[]" \
  epochs="${EPOCHS_B}" \
  eval_interval="${EVAL_INTERVAL}" \
  global_batch_size="${GLOBAL_BATCH_SIZE}" \
  lr="${LR}" \
  puzzle_emb_lr="${PUZZLE_EMB_LR}" \
  weight_decay="${WEIGHT_DECAY}" \
  puzzle_emb_weight_decay="${PUZZLE_EMB_WEIGHT_DECAY}" \
  +run_name="${RUN_PREFIX}_${TASK_B_NAME}_finetune" \
  +checkpoint_path="${TASK_B_CKPT_DIR}" \
  +load_checkpoint="${TASK_A_CHECKPOINT}" \
  checkpoint_every_eval=True \
  ema="${EMA}" \
  ${EXTRA_PRETRAIN_ARGS}

echo "Task A dataset: ${TASK_A_DIR}"
echo "Task B dataset: ${TASK_B_DIR}"
echo "Task A checkpoint dir: ${TASK_A_CKPT_DIR}"
echo "Task B checkpoint dir: ${TASK_B_CKPT_DIR}"
