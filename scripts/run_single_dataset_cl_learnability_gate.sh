#!/usr/bin/env bash
set -euo pipefail

# Single-dataset continual-learning learnability gate for TRM.
#
# This script splits one prepared dataset into Task A and Task B, then trains:
#   1. A-only from scratch
#   2. B-only from scratch
#   3. Joint A+B from scratch
#   4. Sequential A -> B, where B starts from the A-only checkpoint
#
# Required:
#   SOURCE_DATA=/path/to/prepared_dataset RUN_PREFIX=my_gate \
#     bash scripts/run_single_dataset_cl_learnability_gate.sh

SOURCE_DATA="${SOURCE_DATA:?Set SOURCE_DATA to a prepared puzzle dataset root.}"
RUN_PREFIX="${RUN_PREFIX:?Set RUN_PREFIX for checkpoint/result names.}"

CL_DATA_DIR="${CL_DATA_DIR:-data/cl_single_dataset/${RUN_PREFIX}}"
TASK_A_NAME="${TASK_A_NAME:-task_a}"
TASK_B_NAME="${TASK_B_NAME:-task_b}"
TASK_A_FRACTION="${TASK_A_FRACTION:-0.5}"
SEED="${SEED:-0}"
SPLIT_KEY="${SPLIT_KEY:-random}"
TASK_A_SIDE="${TASK_A_SIDE:-low}"
MAX_TRAIN_GROUPS="${MAX_TRAIN_GROUPS:-}"
MAX_TEST_GROUPS="${MAX_TEST_GROUPS:-}"

ARCH="${ARCH:-trm}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/cl_learnability_gate/${RUN_PREFIX}}"
RESULT_ROOT="${RESULT_ROOT:-results/cl_learnability_gate/${RUN_PREFIX}}"

EPOCHS_A="${EPOCHS_A:-1000}"
EPOCHS_B="${EPOCHS_B:-1000}"
EPOCHS_JOINT="${EPOCHS_JOINT:-1000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-${GLOBAL_BATCH_SIZE}}"
LR="${LR:-1e-4}"
PUZZLE_EMB_LR="${PUZZLE_EMB_LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1.0}"
PUZZLE_EMB_WEIGHT_DECAY="${PUZZLE_EMB_WEIGHT_DECAY:-1.0}"
EMA="${EMA:-False}"
EXTRA_PRETRAIN_ARGS="${EXTRA_PRETRAIN_ARGS:-}"

TASK_A_DIR="${CL_DATA_DIR}/${TASK_A_NAME}"
TASK_B_DIR="${CL_DATA_DIR}/${TASK_B_NAME}"
TASK_AB_NAME="${TASK_AB_NAME:-task_ab}"
TASK_AB_DIR="${CL_DATA_DIR}/${TASK_AB_NAME}"

A_ONLY_DIR="${CHECKPOINT_ROOT}/a_only"
B_ONLY_DIR="${CHECKPOINT_ROOT}/b_only"
JOINT_DIR="${CHECKPOINT_ROOT}/joint_ab"
SEQUENTIAL_DIR="${CHECKPOINT_ROOT}/sequential_b"
RESULT_CSV="${RESULT_CSV:-${RESULT_ROOT}/learnability_gate_endpoint_matrix.csv}"

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

TASK_A_DIR="${TASK_A_DIR}" TASK_B_DIR="${TASK_B_DIR}" TASK_AB_DIR="${TASK_AB_DIR}" python - <<'PY'
import json
import os
import shutil
from pathlib import Path

import numpy as np

FIELDS = ("inputs", "labels", "puzzle_identifiers")


def _load(base: Path, split: str, set_name: str, field: str) -> np.ndarray:
    return np.load(base / split / f"{set_name}__{field}.npy")


def _merge_offsets(a_idx: np.ndarray, b_idx: np.ndarray, offset: int) -> np.ndarray:
    return np.concatenate([a_idx, b_idx[1:] + offset]).astype(a_idx.dtype, copy=False)


task_a = Path(os.environ["TASK_A_DIR"])
task_b = Path(os.environ["TASK_B_DIR"])
task_ab = Path(os.environ["TASK_AB_DIR"])

if task_ab.exists():
    shutil.rmtree(task_ab)
task_ab.mkdir(parents=True)

for filename in ("identifiers.json", "test_puzzles.json"):
    src = task_a / filename
    if src.exists():
        shutil.copy2(src, task_ab / filename)

for split in ("train", "test"):
    with open(task_a / split / "dataset.json", "r") as f:
        meta_a = json.load(f)
    with open(task_b / split / "dataset.json", "r") as f:
        meta_b = json.load(f)

    if meta_a["sets"] != meta_b["sets"]:
        raise ValueError(f"Task A/B set mismatch: {meta_a['sets']} vs {meta_b['sets']}")

    split_out = task_ab / split
    split_out.mkdir(parents=True, exist_ok=True)

    total_examples = 0
    total_puzzles = 0
    total_groups = 0
    for set_name in meta_a["sets"]:
        for field in FIELDS:
            merged = np.concatenate(
                [_load(task_a, split, set_name, field), _load(task_b, split, set_name, field)],
                axis=0,
            )
            np.save(split_out / f"{set_name}__{field}.npy", merged)

        a_puzzle = _load(task_a, split, set_name, "puzzle_indices")
        b_puzzle = _load(task_b, split, set_name, "puzzle_indices")
        merged_puzzle = _merge_offsets(a_puzzle, b_puzzle, int(a_puzzle[-1]))
        np.save(split_out / f"{set_name}__puzzle_indices.npy", merged_puzzle)

        a_group = _load(task_a, split, set_name, "group_indices")
        b_group = _load(task_b, split, set_name, "group_indices")
        merged_group = _merge_offsets(a_group, b_group, int(a_group[-1]))
        np.save(split_out / f"{set_name}__group_indices.npy", merged_group)

        total_examples += int(merged_puzzle[-1])
        total_puzzles += int(merged_puzzle.size - 1)
        total_groups += int(merged_group.size - 1)

    meta = dict(meta_a)
    meta.update(
        {
            "total_groups": total_groups,
            "total_puzzles": total_puzzles,
            "mean_puzzle_examples": total_examples / max(total_puzzles, 1),
        }
    )
    with open(split_out / "dataset.json", "w") as f:
        json.dump(meta, f)

print(f"Merged joint Task A+B dataset: {task_ab}")
PY

run_pretrain() {
  local run_name="$1"
  local checkpoint_dir="$2"
  local data_paths="$3"
  local epochs="$4"
  local load_checkpoint="${5:-}"

  local load_args=()
  if [[ -n "${load_checkpoint}" ]]; then
    load_args+=(+load_checkpoint="${load_checkpoint}")
  fi

  python pretrain.py \
    arch="${ARCH}" \
    data_paths="${data_paths}" \
    evaluators="[]" \
    epochs="${epochs}" \
    eval_interval="${EVAL_INTERVAL}" \
    global_batch_size="${GLOBAL_BATCH_SIZE}" \
    lr="${LR}" \
    puzzle_emb_lr="${PUZZLE_EMB_LR}" \
    weight_decay="${WEIGHT_DECAY}" \
    puzzle_emb_weight_decay="${PUZZLE_EMB_WEIGHT_DECAY}" \
    +run_name="${run_name}" \
    +checkpoint_path="${checkpoint_dir}" \
    checkpoint_every_eval=True \
    ema="${EMA}" \
    "${load_args[@]}" \
    ${EXTRA_PRETRAIN_ARGS}
}

latest_checkpoint() {
  find "$1" -maxdepth 1 -type f -name 'step_*' ! -name '*all_preds*' \
    | sort -V \
    | tail -1
}

echo "=== A-only ==="
run_pretrain "${RUN_PREFIX}_a_only" "${A_ONLY_DIR}" "[${TASK_A_DIR}]" "${EPOCHS_A}"

echo "=== B-only ==="
run_pretrain "${RUN_PREFIX}_b_only" "${B_ONLY_DIR}" "[${TASK_B_DIR}]" "${EPOCHS_B}"

echo "=== Joint A+B ==="
run_pretrain "${RUN_PREFIX}_joint_ab" "${JOINT_DIR}" "[${TASK_AB_DIR}]" "${EPOCHS_JOINT}"

A_ONLY_CKPT="$(latest_checkpoint "${A_ONLY_DIR}")"
if [[ -z "${A_ONLY_CKPT}" ]]; then
  echo "No A-only checkpoint found in ${A_ONLY_DIR}" >&2
  exit 1
fi

echo "=== Sequential A->B ==="
echo "Loading A-only checkpoint for sequential B: ${A_ONLY_CKPT}"
run_pretrain "${RUN_PREFIX}_sequential_b" "${SEQUENTIAL_DIR}" "[${TASK_B_DIR}]" "${EPOCHS_B}" "${A_ONLY_CKPT}"

python scripts/eval_cl_learnability_gate.py \
  --a-only-checkpoint-dir "${A_ONLY_DIR}" \
  --b-only-checkpoint-dir "${B_ONLY_DIR}" \
  --joint-checkpoint-dir "${JOINT_DIR}" \
  --sequential-checkpoint-dir "${SEQUENTIAL_DIR}" \
  --task-a-data "${TASK_A_DIR}" \
  --task-b-data "${TASK_B_DIR}" \
  --batch-size "${EVAL_BATCH_SIZE}" \
  --output-csv "${RESULT_CSV}"

echo "Task A dataset: ${TASK_A_DIR}"
echo "Task B dataset: ${TASK_B_DIR}"
echo "Checkpoint root: ${CHECKPOINT_ROOT}"
echo "Endpoint matrix: ${RESULT_CSV}"
