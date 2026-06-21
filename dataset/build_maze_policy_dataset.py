import argparse
import json
import shutil
from collections import deque
from pathlib import Path

import numpy as np

try:
    from dataset.common import PuzzleDatasetMetadata
except ImportError:
    from common import PuzzleDatasetMetadata


WALL = 1
GOAL = 4

IGNORE = 0
POLICY_GOAL = 1
POLICY_UP = 2
POLICY_DOWN = 3
POLICY_LEFT = 4
POLICY_RIGHT = 5

DIRS = (
    (-1, 0, POLICY_UP),
    (1, 0, POLICY_DOWN),
    (0, -1, POLICY_LEFT),
    (0, 1, POLICY_RIGHT),
)

ROOT_METADATA_FILES = ("identifiers.json", "test_puzzles.json")


def _load_metadata(dataset_dir: Path, split: str) -> PuzzleDatasetMetadata:
    with open(dataset_dir / split / "dataset.json", "r") as f:
        return PuzzleDatasetMetadata(**json.load(f))


def _distance_to_goal(input_grid: np.ndarray) -> np.ndarray:
    n, m = input_grid.shape
    goals = np.argwhere(input_grid == GOAL)
    if goals.shape[0] != 1:
        raise ValueError(f"Expected exactly one goal cell, found {goals.shape[0]}")

    dist = np.full((n, m), -1, dtype=np.int32)
    q: deque[tuple[int, int]] = deque()
    gr, gc = map(int, goals[0])
    dist[gr, gc] = 0
    q.append((gr, gc))

    while q:
        r, c = q.popleft()
        for dr, dc, _ in DIRS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < n and 0 <= nc < m):
                continue
            if input_grid[nr, nc] == WALL or dist[nr, nc] >= 0:
                continue
            dist[nr, nc] = dist[r, c] + 1
            q.append((nr, nc))

    return dist


def _policy_labels(input_seq: np.ndarray) -> np.ndarray:
    side = int(round(input_seq.size ** 0.5))
    if side * side != input_seq.size:
        raise ValueError(f"Maze input length must be square, got {input_seq.size}")

    grid = input_seq.reshape(side, side)
    dist = _distance_to_goal(grid)
    labels = np.full_like(grid, IGNORE, dtype=np.uint8)

    for r in range(side):
        for c in range(side):
            if grid[r, c] == WALL or dist[r, c] < 0:
                continue
            if dist[r, c] == 0:
                labels[r, c] = POLICY_GOAL
                continue

            next_label = IGNORE
            for dr, dc, label in DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < side and 0 <= nc < side and dist[nr, nc] == dist[r, c] - 1:
                    next_label = label
                    break
            if next_label == IGNORE:
                raise ValueError("Reachable non-goal cell has no next-hop neighbor.")
            labels[r, c] = next_label

    return labels.reshape(-1)


def _write_identifiers(output_dir: Path, num_puzzle_identifiers: int) -> None:
    identifiers = ["<blank>"] + [f"task_{i}" for i in range(1, num_puzzle_identifiers)]
    with open(output_dir / "identifiers.json", "w") as f:
        json.dump(identifiers[:num_puzzle_identifiers], f)


def convert_split(input_dir: Path, output_dir: Path, split: str, task_id: int, num_puzzle_identifiers: int) -> None:
    metadata = _load_metadata(input_dir, split)
    split_in = input_dir / split
    split_out = output_dir / split
    split_out.mkdir(parents=True, exist_ok=True)

    total_examples = 0
    for set_name in metadata.sets:
        inputs = np.load(split_in / f"{set_name}__inputs.npy")
        labels = np.stack([_policy_labels(example) for example in inputs], axis=0)
        source_puzzle_identifiers = np.load(split_in / f"{set_name}__puzzle_identifiers.npy")

        np.save(split_out / f"{set_name}__inputs.npy", inputs)
        np.save(split_out / f"{set_name}__labels.npy", labels)
        np.save(
            split_out / f"{set_name}__puzzle_identifiers.npy",
            np.full_like(source_puzzle_identifiers, task_id, dtype=np.int32),
        )

        for field in ("puzzle_indices", "group_indices"):
            shutil.copy2(split_in / f"{set_name}__{field}.npy", split_out / f"{set_name}__{field}.npy")
        total_examples += int(inputs.shape[0])

    out_metadata = metadata.model_copy(
        update={
            "vocab_size": 6,
            "ignore_label_id": IGNORE,
            "num_puzzle_identifiers": num_puzzle_identifiers,
        }
    )
    with open(split_out / "dataset.json", "w") as f:
        json.dump(out_metadata.model_dump(), f)

    print(f"Wrote {split} policy labels: {total_examples} examples")


def build_policy_dataset(input_dir: Path, output_dir: Path, overwrite: bool, task_id: int, num_puzzle_identifiers: int) -> None:
    if task_id < 0 or task_id >= num_puzzle_identifiers:
        raise ValueError(f"task_id must be in [0, {num_puzzle_identifiers}), got {task_id}")

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename in ROOT_METADATA_FILES:
        src = input_dir / filename
        if src.exists():
            shutil.copy2(src, output_dir / filename)
    _write_identifiers(output_dir, num_puzzle_identifiers)

    for split in ("train", "test"):
        if (input_dir / split / "dataset.json").exists():
            convert_split(input_dir, output_dir, split, task_id, num_puzzle_identifiers)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Maze next-hop policy target dataset from a prepared Maze dataset.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--num-puzzle-identifiers", type=int, default=1)
    args = parser.parse_args()

    build_policy_dataset(
        Path(args.input_dir),
        Path(args.output_dir),
        args.overwrite,
        args.task_id,
        args.num_puzzle_identifiers,
    )


if __name__ == "__main__":
    main()
