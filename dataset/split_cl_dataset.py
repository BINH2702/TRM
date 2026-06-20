import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np

from dataset.common import PuzzleDatasetMetadata


FIELDS = ("inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices")
ROOT_METADATA_FILES = ("identifiers.json", "test_puzzles.json")
MAZE_PATH_TOKEN_ID = 5


def _load_metadata(dataset_dir: Path, split: str) -> PuzzleDatasetMetadata:
    with open(dataset_dir / split / "dataset.json", "r") as f:
        return PuzzleDatasetMetadata(**json.load(f))


def _load_set_arrays(dataset_dir: Path, split: str, set_name: str) -> dict[str, np.ndarray]:
    split_dir = dataset_dir / split
    return {field: np.load(split_dir / f"{set_name}__{field}.npy") for field in FIELDS}


def _select_group_ids(num_groups: int, task_a_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if num_groups <= 0:
        raise ValueError("Cannot split a dataset with no groups.")

    rng = np.random.default_rng(seed)
    group_ids = rng.permutation(num_groups)

    task_a_count = int(round(num_groups * task_a_fraction))
    if num_groups > 1:
        task_a_count = min(max(task_a_count, 1), num_groups - 1)
    else:
        task_a_count = 1

    task_a = np.sort(group_ids[:task_a_count])
    task_b = np.sort(group_ids[task_a_count:])

    if task_b.size == 0:
        task_b = task_a.copy()

    return task_a, task_b


def _score_groups_sudoku_blanks(arrays: dict[str, np.ndarray]) -> np.ndarray:
    inputs = arrays["inputs"]
    puzzle_indices = arrays["puzzle_indices"]
    group_indices = arrays["group_indices"]

    scores = []
    for group_id in range(group_indices.size - 1):
        puzzle_start = int(group_indices[group_id])
        puzzle_end = int(group_indices[group_id + 1])
        group_scores = []

        for puzzle_id in range(puzzle_start, puzzle_end):
            example_start = int(puzzle_indices[puzzle_id])
            example_end = int(puzzle_indices[puzzle_id + 1])
            # Sudoku builder stores original zero/blank cells as token 1 after
            # shifting tokens by +1.
            group_scores.extend((inputs[example_start:example_end] == 1).sum(axis=1).tolist())

        scores.append(float(np.mean(group_scores)))

    return np.asarray(scores, dtype=np.float32)


def _score_groups_maze_path_length(arrays: dict[str, np.ndarray]) -> np.ndarray:
    labels = arrays["labels"]
    puzzle_indices = arrays["puzzle_indices"]
    group_indices = arrays["group_indices"]

    scores = []
    for group_id in range(group_indices.size - 1):
        puzzle_start = int(group_indices[group_id])
        puzzle_end = int(group_indices[group_id + 1])
        group_scores = []

        for puzzle_id in range(puzzle_start, puzzle_end):
            example_start = int(puzzle_indices[puzzle_id])
            example_end = int(puzzle_indices[puzzle_id + 1])
            # Maze builder maps CHARSET="# SGo" to token ids 1..5, so the
            # solution path character "o" is token 5 in the label grid.
            group_scores.extend((labels[example_start:example_end] == MAZE_PATH_TOKEN_ID).sum(axis=1).tolist())

        scores.append(float(np.mean(group_scores)))

    return np.asarray(scores, dtype=np.float32)


def _select_ordered_group_ids(
    arrays: dict[str, np.ndarray],
    task_a_fraction: float,
    split_key: str,
    task_a_side: str,
) -> tuple[np.ndarray, np.ndarray]:
    num_groups = int(arrays["group_indices"].size - 1)
    if split_key == "sudoku_blanks":
        scores = _score_groups_sudoku_blanks(arrays)
    elif split_key == "maze_path_length":
        scores = _score_groups_maze_path_length(arrays)
    else:
        raise ValueError(f"Unsupported ordered split key: {split_key}")
    group_ids = np.arange(num_groups)
    order = np.lexsort((group_ids, scores))
    if task_a_side == "high":
        order = order[::-1]
    elif task_a_side != "low":
        raise ValueError(f"Unsupported --task-a-side: {task_a_side}")

    task_a_count = int(round(num_groups * task_a_fraction))
    if num_groups > 1:
        task_a_count = min(max(task_a_count, 1), num_groups - 1)
    else:
        task_a_count = 1

    task_a = np.sort(order[:task_a_count])
    task_b = np.sort(order[task_a_count:])
    if task_b.size == 0:
        task_b = task_a.copy()

    return task_a, task_b


def _subset_by_groups(arrays: dict[str, np.ndarray], group_ids: np.ndarray) -> dict[str, np.ndarray]:
    inputs = arrays["inputs"]
    labels = arrays["labels"]
    puzzle_identifiers = arrays["puzzle_identifiers"]
    puzzle_indices = arrays["puzzle_indices"]
    group_indices = arrays["group_indices"]

    selected_inputs = []
    selected_labels = []
    selected_identifiers = []
    new_puzzle_indices = [0]
    new_group_indices = [0]
    example_count = 0
    puzzle_count = 0

    for group_id in group_ids.tolist():
        puzzle_start = int(group_indices[group_id])
        puzzle_end = int(group_indices[group_id + 1])

        for puzzle_id in range(puzzle_start, puzzle_end):
            example_start = int(puzzle_indices[puzzle_id])
            example_end = int(puzzle_indices[puzzle_id + 1])

            selected_inputs.append(inputs[example_start:example_end])
            selected_labels.append(labels[example_start:example_end])
            selected_identifiers.append(puzzle_identifiers[puzzle_id])

            example_count += example_end - example_start
            puzzle_count += 1
            new_puzzle_indices.append(example_count)

        new_group_indices.append(puzzle_count)

    if selected_inputs:
        out_inputs = np.concatenate(selected_inputs, axis=0)
        out_labels = np.concatenate(selected_labels, axis=0)
    else:
        out_inputs = inputs[:0].copy()
        out_labels = labels[:0].copy()

    return {
        "inputs": out_inputs,
        "labels": out_labels,
        "puzzle_identifiers": np.asarray(selected_identifiers, dtype=np.int32),
        "puzzle_indices": np.asarray(new_puzzle_indices, dtype=np.int32),
        "group_indices": np.asarray(new_group_indices, dtype=np.int32),
    }


def _write_set_arrays(output_dir: Path, split: str, set_name: str, arrays: dict[str, np.ndarray]) -> None:
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    for field, array in arrays.items():
        np.save(split_dir / f"{set_name}__{field}.npy", array)


def _write_metadata(output_dir: Path, split: str, source_metadata: PuzzleDatasetMetadata, set_stats: list[dict[str, int]]) -> None:
    total_examples = sum(stat["examples"] for stat in set_stats)
    total_puzzles = sum(stat["puzzles"] for stat in set_stats)
    total_groups = sum(stat["groups"] for stat in set_stats)

    metadata = source_metadata.model_copy(
        update={
            "total_groups": total_groups,
            "mean_puzzle_examples": total_examples / max(total_puzzles, 1),
            "total_puzzles": total_puzzles,
        }
    )

    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    with open(split_dir / "dataset.json", "w") as f:
        json.dump(metadata.model_dump(), f)


def _split_one_split(
    source_dir: Path,
    task_a_dir: Path,
    task_b_dir: Path,
    split: str,
    task_a_fraction: float,
    seed: int,
    max_groups: int | None = None,
    split_key: str = "random",
    task_a_side: str = "low",
) -> None:
    metadata = _load_metadata(source_dir, split)
    task_stats = {"task_a": [], "task_b": []}

    for set_offset, set_name in enumerate(metadata.sets):
        arrays = _load_set_arrays(source_dir, split, set_name)
        num_groups = int(arrays["group_indices"].size - 1)
        if split_key == "random":
            task_a_groups, task_b_groups = _select_group_ids(num_groups, task_a_fraction, seed + set_offset)
        else:
            task_a_groups, task_b_groups = _select_ordered_group_ids(
                arrays=arrays,
                task_a_fraction=task_a_fraction,
                split_key=split_key,
                task_a_side=task_a_side,
            )

        if max_groups is not None:
            max_a = max(1, int(round(max_groups * task_a_fraction)))
            max_b = max(1, max_groups - max_a)
            task_a_groups = task_a_groups[:max_a]
            task_b_groups = task_b_groups[:max_b]

        for task_name, task_dir, group_ids in (
            ("task_a", task_a_dir, task_a_groups),
            ("task_b", task_b_dir, task_b_groups),
        ):
            subset = _subset_by_groups(arrays, group_ids)
            _write_set_arrays(task_dir, split, set_name, subset)
            task_stats[task_name].append(
                {
                    "examples": int(subset["inputs"].shape[0]),
                    "puzzles": int(subset["puzzle_identifiers"].shape[0]),
                    "groups": int(subset["group_indices"].shape[0] - 1),
                }
            )

    _write_metadata(task_a_dir, split, metadata, task_stats["task_a"])
    _write_metadata(task_b_dir, split, metadata, task_stats["task_b"])


def _copy_root_metadata(source_dir: Path, task_a_dir: Path, task_b_dir: Path) -> None:
    for filename in ROOT_METADATA_FILES:
        src = source_dir / filename
        if not src.exists():
            continue

        for task_dir in (task_a_dir, task_b_dir):
            task_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, task_dir / filename)


def split_dataset(args: argparse.Namespace) -> None:
    source_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    task_a_dir = output_dir / args.task_a_name
    task_b_dir = output_dir / args.task_b_name

    if not source_dir.exists():
        raise FileNotFoundError(f"Input dataset directory does not exist: {source_dir}")

    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Output directory already exists: {output_dir}. Use --overwrite to replace it.")

    if output_dir.exists():
        shutil.rmtree(output_dir)

    for split in args.splits:
        if not (source_dir / split / "dataset.json").exists():
            if args.allow_missing_splits:
                continue
            raise FileNotFoundError(f"Missing split metadata: {source_dir / split / 'dataset.json'}")

        max_groups = None
        if split == "train":
            max_groups = args.max_train_groups
        elif split == "test":
            max_groups = args.max_test_groups

        _split_one_split(
            source_dir=source_dir,
            task_a_dir=task_a_dir,
            task_b_dir=task_b_dir,
            split=split,
            task_a_fraction=args.task_a_fraction,
            seed=args.seed + (0 if split == "train" else 100000),
            max_groups=max_groups,
            split_key=args.split_key,
            task_a_side=args.task_a_side,
        )

    _copy_root_metadata(source_dir, task_a_dir, task_b_dir)

    print(f"Wrote Task A dataset: {task_a_dir}")
    print(f"Wrote Task B dataset: {task_b_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split one prepared puzzle dataset into Task A and Task B for CL experiments.")
    parser.add_argument("--input-dir", required=True, help="Prepared dataset root containing train/test split folders.")
    parser.add_argument("--output-dir", required=True, help="Output root for task_a and task_b dataset directories.")
    parser.add_argument("--task-a-name", default="task_a")
    parser.add_argument("--task-b-name", default="task_b")
    parser.add_argument("--task-a-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--split-key", choices=["random", "sudoku_blanks", "maze_path_length"], default="random")
    parser.add_argument("--task-a-side", choices=["low", "high"], default="low", help="For ordered splits, choose whether Task A gets low-score or high-score groups.")
    parser.add_argument("--max-train-groups", type=int, default=None, help="Optional cap for train groups before writing each task split.")
    parser.add_argument("--max-test-groups", type=int, default=None, help="Optional cap for test groups before writing each task split.")
    parser.add_argument("--allow-missing-splits", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.task_a_fraction < 1.0:
        raise ValueError("--task-a-fraction must be between 0 and 1.")

    split_dataset(args)


if __name__ == "__main__":
    main()
