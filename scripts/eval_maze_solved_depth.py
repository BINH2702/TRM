import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.common import PuzzleDatasetMetadata
from pretrain import PretrainConfig, TrainState, create_model


IGNORE_LABEL_ID = -100
MAZE_PATH_TOKEN_ID = 5


def load_base_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg["evaluators"] = []
    cfg["eval_save_outputs"] = []
    cfg["checkpoint_path"] = None
    cfg["ema"] = False
    return cfg


def load_metadata(data_path: Path, split: str) -> dict:
    with open(data_path / split / "dataset.json", "r") as f:
        return json.load(f)


def build_config(
    base_config: dict,
    checkpoint: Path,
    data_path: Path,
    batch_size: int,
    halt_max_steps: int,
) -> PretrainConfig:
    cfg_raw = copy.deepcopy(base_config)
    cfg_raw["data_paths"] = [str(data_path)]
    cfg_raw["data_paths_test"] = []
    cfg_raw["load_checkpoint"] = str(checkpoint)
    cfg_raw["global_batch_size"] = batch_size
    cfg_raw["arch"]["halt_max_steps"] = halt_max_steps
    return PretrainConfig(**cfg_raw)


def load_arrays(data_path: Path, split: str, set_name: str) -> dict[str, np.ndarray]:
    split_dir = data_path / split
    return {
        "inputs": np.load(split_dir / f"{set_name}__inputs.npy"),
        "labels": np.load(split_dir / f"{set_name}__labels.npy"),
        "puzzle_identifiers": np.load(split_dir / f"{set_name}__puzzle_identifiers.npy"),
        "puzzle_indices": np.load(split_dir / f"{set_name}__puzzle_indices.npy"),
    }


def puzzle_ids_for_examples(puzzle_indices: np.ndarray, start: int, end: int) -> np.ndarray:
    puzzle_ids = np.searchsorted(puzzle_indices, np.arange(start, end), side="right") - 1
    return puzzle_ids.astype(np.int64, copy=False)


def pad_batch(
    inputs: np.ndarray,
    labels: np.ndarray,
    puzzle_identifiers: np.ndarray,
    batch_size: int,
    pad_id: int,
    ignore_label_id: int | None,
    blank_identifier_id: int,
) -> tuple[dict[str, torch.Tensor], int]:
    valid_size = inputs.shape[0]
    labels = labels.astype(np.int32, copy=True)
    if ignore_label_id is not None:
        labels[labels == ignore_label_id] = IGNORE_LABEL_ID

    if valid_size < batch_size:
        pad_size = batch_size - valid_size
        inputs = np.pad(inputs, ((0, pad_size), (0, 0)), constant_values=pad_id)
        labels = np.pad(labels, ((0, pad_size), (0, 0)), constant_values=IGNORE_LABEL_ID)
        puzzle_identifiers = np.pad(
            puzzle_identifiers,
            (0, pad_size),
            constant_values=blank_identifier_id,
        )

    batch = {
        "inputs": torch.from_numpy(inputs.astype(np.int32, copy=False)).cuda(),
        "labels": torch.from_numpy(labels.astype(np.int32, copy=False)).cuda(),
        "puzzle_identifiers": torch.from_numpy(puzzle_identifiers.astype(np.int32, copy=False)).cuda(),
    }
    return batch, valid_size


def compute_rows(
    preds: torch.Tensor,
    labels: torch.Tensor,
    global_start: int,
    valid_size: int,
    horizon: int,
    checkpoint_name: str,
    eval_name: str,
) -> list[dict[str, float | int | str]]:
    preds_np = preds[:valid_size].detach().cpu().numpy()
    labels_np = labels[:valid_size].detach().cpu().numpy()

    rows = []
    for local_i in range(valid_size):
        label = labels_np[local_i]
        pred = preds_np[local_i]
        valid = label != IGNORE_LABEL_ID

        correct = (pred == label) & valid
        valid_count = int(valid.sum())
        exact = int(correct.sum() == valid_count)
        token_acc = float(correct.sum() / max(valid_count, 1))
        incorrect_count = int(valid_count - correct.sum())

        pred_path = (pred == MAZE_PATH_TOKEN_ID) & valid
        true_path = (label == MAZE_PATH_TOKEN_ID) & valid
        path_tp = int((pred_path & true_path).sum())
        path_fp = int((pred_path & ~true_path).sum())
        path_fn = int((~pred_path & true_path).sum())
        path_precision = path_tp / max(path_tp + path_fp, 1)
        path_recall = path_tp / max(path_tp + path_fn, 1)
        path_f1 = (2 * path_precision * path_recall / max(path_precision + path_recall, 1e-12))

        rows.append(
            {
                "checkpoint": checkpoint_name,
                "eval_split": eval_name,
                "example_index": global_start + local_i,
                "horizon": horizon,
                "exact": exact,
                "token_accuracy": token_acc,
                "incorrect_count": incorrect_count,
                "path_length": int(true_path.sum()),
                "pred_path_length": int(pred_path.sum()),
                "path_precision": path_precision,
                "path_recall": path_recall,
                "path_f1": path_f1,
            }
        )
    return rows


def evaluate_horizon(
    base_config: dict,
    checkpoint: Path,
    checkpoint_name: str,
    data_path: Path,
    eval_name: str,
    batch_size: int,
    horizon: int,
    split: str,
) -> list[dict[str, float | int | str]]:
    metadata = load_metadata(data_path, split)
    cfg = build_config(base_config, checkpoint, data_path, batch_size, horizon)
    model, _, _ = create_model(cfg, train_metadata=PuzzleDatasetMetadata(**metadata), rank=0, world_size=1)
    state = TrainState(model=model, optimizers=[], optimizer_lrs=[], carry=None, step=0, total_steps=0)
    state.model.eval()

    all_rows = []
    with torch.inference_mode():
        for set_name in metadata["sets"]:
            arrays = load_arrays(data_path, split, set_name)
            total = arrays["inputs"].shape[0]
            for start in range(0, total, batch_size):
                end = min(start + batch_size, total)
                puzzle_ids = puzzle_ids_for_examples(arrays["puzzle_indices"], start, end)
                puzzle_identifiers = arrays["puzzle_identifiers"][puzzle_ids]
                batch, valid_size = pad_batch(
                    arrays["inputs"][start:end],
                    arrays["labels"][start:end],
                    puzzle_identifiers,
                    batch_size=batch_size,
                    pad_id=metadata["pad_id"],
                    ignore_label_id=metadata["ignore_label_id"],
                    blank_identifier_id=metadata["blank_identifier_id"],
                )

                with torch.device("cuda"):
                    carry = state.model.initial_carry(batch)  # type: ignore

                while True:
                    carry, _loss, _metrics, preds, all_finish = state.model(
                        carry=carry,
                        batch=batch,
                        return_keys=["preds"],
                    )
                    if all_finish:
                        break

                all_rows.extend(
                    compute_rows(
                        preds["preds"],
                        batch["labels"],
                        global_start=start,
                        valid_size=valid_size,
                        horizon=horizon,
                        checkpoint_name=checkpoint_name,
                        eval_name=eval_name,
                    )
                )

                del carry, batch, preds, all_finish

    del state, model
    torch.cuda.empty_cache()
    return all_rows


def summarize(rows: list[dict[str, float | int | str]]) -> dict[str, float | int | str]:
    exact = np.asarray([row["exact"] for row in rows], dtype=np.float64)
    token_acc = np.asarray([row["token_accuracy"] for row in rows], dtype=np.float64)
    path_f1 = np.asarray([row["path_f1"] for row in rows], dtype=np.float64)
    path_length = np.asarray([row["path_length"] for row in rows], dtype=np.float64)
    incorrect = np.asarray([row["incorrect_count"] for row in rows], dtype=np.float64)
    first = rows[0]
    return {
        "checkpoint": first["checkpoint"],
        "eval_split": first["eval_split"],
        "horizon": first["horizon"],
        "count": len(rows),
        "exact_accuracy": float(exact.mean()),
        "token_accuracy": float(token_acc.mean()),
        "path_f1": float(path_f1.mean()),
        "incorrect_count_mean": float(incorrect.mean()),
        "path_length_mean": float(path_length.mean()),
        "path_length_min": int(path_length.min()),
        "path_length_max": int(path_length.max()),
    }


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TRM Maze solved depth over multiple halt horizons.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-name", default="teacher")
    parser.add_argument("--eval-data", action="append", required=True, help="NAME=PATH, can be repeated.")
    parser.add_argument("--horizons", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    base_config = load_base_config(Path(args.config))
    checkpoint = Path(args.checkpoint)
    output_dir = Path(args.output_dir)

    per_example_rows = []
    summary_rows = []
    for eval_spec in args.eval_data:
        if "=" not in eval_spec:
            raise ValueError(f"--eval-data must be NAME=PATH, got {eval_spec}")
        eval_name, eval_path = eval_spec.split("=", 1)
        for horizon in args.horizons:
            print(f"Evaluating {args.checkpoint_name} on {eval_name} at horizon {horizon}")
            rows = evaluate_horizon(
                base_config=base_config,
                checkpoint=checkpoint,
                checkpoint_name=args.checkpoint_name,
                data_path=Path(eval_path),
                eval_name=eval_name,
                batch_size=args.batch_size,
                horizon=horizon,
                split=args.split,
            )
            per_example_rows.extend(rows)
            summary = summarize(rows)
            summary_rows.append(summary)
            print(
                "SUMMARY",
                summary["checkpoint"],
                summary["eval_split"],
                f"K={summary['horizon']}",
                f"exact={summary['exact_accuracy']:.6f}",
                f"token={summary['token_accuracy']:.6f}",
                f"path_f1={summary['path_f1']:.6f}",
            )

    write_csv(output_dir / "solved_depth_per_example.csv", per_example_rows)
    write_csv(output_dir / "solved_depth_summary.csv", summary_rows)
    print(f"Wrote {output_dir / 'solved_depth_per_example.csv'}")
    print(f"Wrote {output_dir / 'solved_depth_summary.csv'}")


if __name__ == "__main__":
    main()
