#!/usr/bin/env bash
set -euo pipefail

# Continual-learning learnability gate for two prepared TRM-compatible datasets.
#
# Required:
#   TASK_A_DATA=/path/to/task_a TASK_B_DATA=/path/to/task_b RUN_PREFIX=my_gate \
#     bash scripts/run_two_dataset_cl_learnability_gate.sh

TASK_A_DATA="${TASK_A_DATA:?Set TASK_A_DATA to a prepared Task A dataset root.}"
TASK_B_DATA="${TASK_B_DATA:?Set TASK_B_DATA to a prepared Task B dataset root.}"
RUN_PREFIX="${RUN_PREFIX:?Set RUN_PREFIX for checkpoint/result names.}"

CL_DATA_DIR="${CL_DATA_DIR:-data/cl_two_dataset/${RUN_PREFIX}}"
TASK_A_NAME="${TASK_A_NAME:-task_a}"
TASK_B_NAME="${TASK_B_NAME:-task_b}"
TASK_AB_NAME="${TASK_AB_NAME:-task_ab}"
TASK_B_REPLAY_NAME="${TASK_B_REPLAY_NAME:-task_b_replay}"
TASK_A_DIR="${CL_DATA_DIR}/${TASK_A_NAME}"
TASK_B_DIR="${CL_DATA_DIR}/${TASK_B_NAME}"
TASK_AB_DIR="${CL_DATA_DIR}/${TASK_AB_NAME}"
TASK_B_REPLAY_DIR="${CL_DATA_DIR}/${TASK_B_REPLAY_NAME}"
TASK_A_ID="${TASK_A_ID:-0}"
TASK_B_ID="${TASK_B_ID:-1}"
NUM_TASK_IDENTIFIERS="${NUM_TASK_IDENTIFIERS:-2}"
REPLAY_MEMORY="${REPLAY_MEMORY:-0}"
REPLAY_LAMBDA="${REPLAY_LAMBDA:-1.0}"
REPLAY_SEED="${REPLAY_SEED:-0}"

ARCH="${ARCH:-trm}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/cl_two_dataset_gate/${RUN_PREFIX}}"
RESULT_ROOT="${RESULT_ROOT:-results/cl_two_dataset_gate/${RUN_PREFIX}}"

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

A_ONLY_DIR="${CHECKPOINT_ROOT}/a_only"
B_ONLY_DIR="${CHECKPOINT_ROOT}/b_only"
JOINT_DIR="${CHECKPOINT_ROOT}/joint_ab"
SEQUENTIAL_DIR="${CHECKPOINT_ROOT}/sequential_b"
SEQUENTIAL_REPLAY_DIR="${CHECKPOINT_ROOT}/sequential_replay_m${REPLAY_MEMORY}"
RESULT_CSV="${RESULT_CSV:-${RESULT_ROOT}/learnability_gate_endpoint_matrix.csv}"

export WANDB_MODE="${WANDB_MODE:-offline}"

TASK_A_DATA="${TASK_A_DATA}" TASK_B_DATA="${TASK_B_DATA}" TASK_A_DIR="${TASK_A_DIR}" TASK_B_DIR="${TASK_B_DIR}" TASK_AB_DIR="${TASK_AB_DIR}" TASK_B_REPLAY_DIR="${TASK_B_REPLAY_DIR}" TASK_A_ID="${TASK_A_ID}" TASK_B_ID="${TASK_B_ID}" NUM_TASK_IDENTIFIERS="${NUM_TASK_IDENTIFIERS}" REPLAY_MEMORY="${REPLAY_MEMORY}" REPLAY_LAMBDA="${REPLAY_LAMBDA}" REPLAY_SEED="${REPLAY_SEED}" python - <<'PY'
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np

FIELDS = ("inputs", "labels", "puzzle_identifiers")
ROOT_METADATA_FILES = ("test_puzzles.json",)


def _load(base: Path, split: str, set_name: str, field: str) -> np.ndarray:
    return np.load(base / split / f"{set_name}__{field}.npy")


def _merge_offsets(a_idx: np.ndarray, b_idx: np.ndarray, offset: int) -> np.ndarray:
    return np.concatenate([a_idx, b_idx[1:] + offset]).astype(a_idx.dtype, copy=False)


def _identifiers(num_identifiers: int) -> list[str]:
    defaults = ["task_a", "task_b"]
    return [defaults[i] if i < len(defaults) else f"task_{i}" for i in range(num_identifiers)]


def _copy_task_with_id(source: Path, dest: Path, task_id: int, num_identifiers: int) -> None:
    if task_id < 0 or task_id >= num_identifiers:
        raise ValueError(f"task_id must be in [0, {num_identifiers}), got {task_id}")

    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    with open(dest / "identifiers.json", "w") as f:
        json.dump(_identifiers(num_identifiers), f)
    for filename in ROOT_METADATA_FILES:
        src = source / filename
        if src.exists():
            shutil.copy2(src, dest / filename)

    for split in ("train", "test"):
        with open(source / split / "dataset.json", "r") as f:
            meta = json.load(f)

        split_out = dest / split
        split_out.mkdir(parents=True, exist_ok=True)

        for set_name in meta["sets"]:
            for field in ("inputs", "labels", "puzzle_indices", "group_indices"):
                shutil.copy2(source / split / f"{set_name}__{field}.npy", split_out / f"{set_name}__{field}.npy")

            source_ids = np.load(source / split / f"{set_name}__puzzle_identifiers.npy")
            np.save(
                split_out / f"{set_name}__puzzle_identifiers.npy",
                np.full_like(source_ids, task_id, dtype=np.int32),
            )

        meta = dict(meta)
        meta["num_puzzle_identifiers"] = num_identifiers
        meta["blank_identifier_id"] = 0
        with open(split_out / "dataset.json", "w") as f:
            json.dump(meta, f)


def _copy_root_metadata(source: Path, dest: Path) -> None:
    with open(dest / "identifiers.json", "w") as f:
        json.dump(_identifiers(num_task_identifiers), f)
    for filename in ROOT_METADATA_FILES:
        src = source / filename
        if src.exists():
            shutil.copy2(src, dest / filename)


def _write_example_level_split(dest: Path, split: str, meta_template: dict, arrays: dict[str, np.ndarray]) -> None:
    split_out = dest / split
    split_out.mkdir(parents=True, exist_ok=True)

    total_examples = 0
    total_puzzles = 0
    total_groups = 0
    for set_name, fields in arrays.items():
        n = int(fields["inputs"].shape[0])
        for field in FIELDS:
            np.save(split_out / f"{set_name}__{field}.npy", fields[field])

        indices = np.arange(n + 1, dtype=np.int32)
        np.save(split_out / f"{set_name}__puzzle_indices.npy", indices)
        np.save(split_out / f"{set_name}__group_indices.npy", indices)
        total_examples += n
        total_puzzles += n
        total_groups += n

    meta = dict(meta_template)
    meta.update(
        {
            "num_puzzle_identifiers": num_task_identifiers,
            "blank_identifier_id": 0,
            "total_groups": total_groups,
            "total_puzzles": total_puzzles,
            "mean_puzzle_examples": total_examples / max(total_puzzles, 1),
        }
    )
    with open(split_out / "dataset.json", "w") as f:
        json.dump(meta, f)


def _build_replay_dataset(task_a: Path, task_b: Path, dest: Path, replay_memory: int, replay_lambda: float, seed: int) -> None:
    if replay_memory <= 0:
        if dest.exists():
            shutil.rmtree(dest)
        return

    if replay_lambda <= 0:
        raise ValueError(f"REPLAY_LAMBDA must be positive when REPLAY_MEMORY > 0, got {replay_lambda}")

    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    _copy_root_metadata(task_a, dest)

    rng = np.random.default_rng(seed)
    for split in ("train", "test"):
        with open(task_b / split / "dataset.json", "r") as f:
            meta_b = json.load(f)

        split_arrays = {}
        for set_name in meta_b["sets"]:
            b_inputs = _load(task_b, split, set_name, "inputs")
            b_labels = _load(task_b, split, set_name, "labels")
            b_ids = _load(task_b, split, set_name, "puzzle_identifiers")

            if split == "train":
                a_inputs = _load(task_a, split, set_name, "inputs")
                a_labels = _load(task_a, split, set_name, "labels")
                a_ids = _load(task_a, split, set_name, "puzzle_identifiers")

                memory_size = min(replay_memory, int(a_inputs.shape[0]))
                memory_indices = rng.choice(a_inputs.shape[0], size=memory_size, replace=False)
                replay_rows = max(memory_size, int(round(replay_lambda * b_inputs.shape[0])))
                replay_indices = rng.choice(memory_indices, size=replay_rows, replace=True)

                inputs = np.concatenate([b_inputs, a_inputs[replay_indices]], axis=0)
                labels = np.concatenate([b_labels, a_labels[replay_indices]], axis=0)
                puzzle_identifiers = np.concatenate([b_ids, a_ids[replay_indices]], axis=0)

                order = rng.permutation(inputs.shape[0])
                inputs = inputs[order]
                labels = labels[order]
                puzzle_identifiers = puzzle_identifiers[order]
            else:
                inputs = b_inputs
                labels = b_labels
                puzzle_identifiers = b_ids

            split_arrays[set_name] = {
                "inputs": inputs,
                "labels": labels,
                "puzzle_identifiers": puzzle_identifiers,
            }

        _write_example_level_split(dest, split, meta_b, split_arrays)


source_task_a = Path(os.environ["TASK_A_DATA"])
source_task_b = Path(os.environ["TASK_B_DATA"])
task_a = Path(os.environ["TASK_A_DIR"])
task_b = Path(os.environ["TASK_B_DIR"])
task_ab = Path(os.environ["TASK_AB_DIR"])
task_b_replay = Path(os.environ["TASK_B_REPLAY_DIR"])
task_a_id = int(os.environ["TASK_A_ID"])
task_b_id = int(os.environ["TASK_B_ID"])
num_task_identifiers = int(os.environ["NUM_TASK_IDENTIFIERS"])
replay_memory = int(os.environ["REPLAY_MEMORY"])
replay_lambda = float(os.environ["REPLAY_LAMBDA"])
replay_seed = int(os.environ["REPLAY_SEED"])

_copy_task_with_id(source_task_a, task_a, task_a_id, num_task_identifiers)
_copy_task_with_id(source_task_b, task_b, task_b_id, num_task_identifiers)

if task_ab.exists():
    shutil.rmtree(task_ab)
task_ab.mkdir(parents=True, exist_ok=True)

with open(task_ab / "identifiers.json", "w") as f:
    json.dump(_identifiers(num_task_identifiers), f)
for filename in ROOT_METADATA_FILES:
    src = task_a / filename
    if src.exists():
        shutil.copy2(src, task_ab / filename)

for split in ("train", "test"):
    with open(task_a / split / "dataset.json", "r") as f:
        meta_a = json.load(f)
    with open(task_b / split / "dataset.json", "r") as f:
        meta_b = json.load(f)

    for key in ("seq_len", "vocab_size", "pad_id", "ignore_label_id", "blank_identifier_id", "sets"):
        if meta_a[key] != meta_b[key]:
            raise ValueError(f"Task metadata mismatch for {key}: {meta_a[key]} vs {meta_b[key]}")

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
            "num_puzzle_identifiers": num_task_identifiers,
            "blank_identifier_id": 0,
            "total_groups": total_groups,
            "total_puzzles": total_puzzles,
            "mean_puzzle_examples": total_examples / max(total_puzzles, 1),
        }
    )
    with open(split_out / "dataset.json", "w") as f:
        json.dump(meta, f)

print(f"Merged Task A+B dataset: {task_ab}")
_build_replay_dataset(task_a, task_b, task_b_replay, replay_memory, replay_lambda, replay_seed)
if replay_memory > 0:
    print(f"Built Task B + Task A replay dataset: {task_b_replay}")
PY

TASK_A_DATA="${TASK_A_DIR}"
TASK_B_DATA="${TASK_B_DIR}"

if [[ "${PREPARE_ONLY:-0}" == "1" ]]; then
  echo "Prepared Task A dataset: ${TASK_A_DATA}"
  echo "Prepared Task B dataset: ${TASK_B_DATA}"
  echo "Prepared joint dataset: ${TASK_AB_DIR}"
  if [[ "${REPLAY_MEMORY}" != "0" ]]; then
    echo "Prepared replay dataset: ${TASK_B_REPLAY_DIR}"
  fi
  exit 0
fi

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
run_pretrain "${RUN_PREFIX}_a_only" "${A_ONLY_DIR}" "[${TASK_A_DATA}]" "${EPOCHS_A}"

echo "=== B-only ==="
run_pretrain "${RUN_PREFIX}_b_only" "${B_ONLY_DIR}" "[${TASK_B_DATA}]" "${EPOCHS_B}"

echo "=== Joint A+B ==="
run_pretrain "${RUN_PREFIX}_joint_ab" "${JOINT_DIR}" "[${TASK_AB_DIR}]" "${EPOCHS_JOINT}"

A_ONLY_CKPT="$(latest_checkpoint "${A_ONLY_DIR}")"
if [[ -z "${A_ONLY_CKPT}" ]]; then
  echo "No A-only checkpoint found in ${A_ONLY_DIR}" >&2
  exit 1
fi

echo "=== Sequential A->B ==="
echo "Loading A-only checkpoint for sequential B: ${A_ONLY_CKPT}"
run_pretrain "${RUN_PREFIX}_sequential_b" "${SEQUENTIAL_DIR}" "[${TASK_B_DATA}]" "${EPOCHS_B}" "${A_ONLY_CKPT}"

replay_eval_args=()
if [[ "${REPLAY_MEMORY}" != "0" ]]; then
  echo "=== Sequential A->B with endpoint replay m=${REPLAY_MEMORY}, lambda=${REPLAY_LAMBDA} ==="
  echo "Loading A-only checkpoint for replay B: ${A_ONLY_CKPT}"
  run_pretrain "${RUN_PREFIX}_sequential_replay_m${REPLAY_MEMORY}" "${SEQUENTIAL_REPLAY_DIR}" "[${TASK_B_REPLAY_DIR}]" "${EPOCHS_B}" "${A_ONLY_CKPT}"
  replay_eval_args+=(--replay-checkpoint-dir "${SEQUENTIAL_REPLAY_DIR}" --replay-name "sequential_replay_m${REPLAY_MEMORY}")
fi

eval_args=()
if [[ "${MAZE_FUNCTIONAL_EVAL:-0}" == "1" ]]; then
  eval_args+=(--maze-functional)
fi

python scripts/eval_cl_learnability_gate.py \
  --a-only-checkpoint-dir "${A_ONLY_DIR}" \
  --b-only-checkpoint-dir "${B_ONLY_DIR}" \
  --joint-checkpoint-dir "${JOINT_DIR}" \
  --sequential-checkpoint-dir "${SEQUENTIAL_DIR}" \
  "${replay_eval_args[@]}" \
  --task-a-data "${TASK_A_DATA}" \
  --task-b-data "${TASK_B_DATA}" \
  --batch-size "${EVAL_BATCH_SIZE}" \
  --output-csv "${RESULT_CSV}" \
  "${eval_args[@]}"

echo "Task A dataset: ${TASK_A_DATA}"
echo "Task B dataset: ${TASK_B_DATA}"
echo "Joint dataset: ${TASK_AB_DIR}"
if [[ "${REPLAY_MEMORY}" != "0" ]]; then
  echo "Replay dataset: ${TASK_B_REPLAY_DIR}"
fi
echo "Checkpoint root: ${CHECKPOINT_ROOT}"
echo "Endpoint matrix: ${RESULT_CSV}"
