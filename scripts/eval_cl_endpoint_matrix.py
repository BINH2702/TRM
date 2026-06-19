import argparse
import copy
import csv
import os
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pretrain import PretrainConfig, TrainState, create_dataloader, create_model, evaluate


def latest_checkpoint(checkpoint_dir: Path) -> Path:
    candidates = sorted(
        [p for p in checkpoint_dir.glob("step_*") if "all_preds" not in p.name],
        key=lambda p: int(p.name.split("_")[-1]),
    )
    if not candidates:
        raise FileNotFoundError(f"No step_* checkpoint found in {checkpoint_dir}")
    return candidates[-1]


def load_base_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg["evaluators"] = []
    cfg["eval_save_outputs"] = []
    cfg["checkpoint_path"] = None
    cfg["ema"] = False
    return cfg


def eval_one(base_config: dict, checkpoint: Path, data_path: Path, batch_size: int) -> dict:
    cfg_raw = copy.deepcopy(base_config)
    cfg_raw["data_paths"] = [str(data_path)]
    cfg_raw["data_paths_test"] = []
    cfg_raw["load_checkpoint"] = str(checkpoint)
    cfg_raw["global_batch_size"] = batch_size

    cfg = PretrainConfig(**cfg_raw)
    loader, metadata = create_dataloader(
        cfg,
        "test",
        rank=0,
        world_size=1,
        test_set_mode=True,
        epochs_per_iter=1,
        global_batch_size=cfg.global_batch_size,
    )
    model, _, _ = create_model(cfg, metadata, rank=0, world_size=1)
    state = TrainState(model=model, optimizers=[], optimizer_lrs=[], carry=None, step=0, total_steps=0)
    model.eval()

    metrics = evaluate(cfg, state, loader, metadata, [], rank=0, world_size=1, cpu_group=None)
    del model, state, loader
    torch.cuda.empty_cache()
    return metrics["all"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Task A/B checkpoints on Task A/B test splits.")
    parser.add_argument("--task-a-checkpoint-dir", required=True)
    parser.add_argument("--task-b-checkpoint-dir", required=True)
    parser.add_argument("--task-a-data", required=True)
    parser.add_argument("--task-b-data", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    task_a_ckpt_dir = Path(args.task_a_checkpoint_dir)
    task_b_ckpt_dir = Path(args.task_b_checkpoint_dir)
    task_a_ckpt = latest_checkpoint(task_a_ckpt_dir)
    task_b_ckpt = latest_checkpoint(task_b_ckpt_dir)

    base_config = load_base_config(task_a_ckpt_dir / "all_config.yaml")

    rows = []
    for checkpoint_name, checkpoint in (
        ("task_a_checkpoint", task_a_ckpt),
        ("task_b_checkpoint", task_b_ckpt),
    ):
        for split_name, data_path in (
            ("task_a_test", Path(args.task_a_data)),
            ("task_b_test", Path(args.task_b_data)),
        ):
            metrics = eval_one(base_config, checkpoint, data_path, args.batch_size)
            row = {
                "checkpoint": checkpoint_name,
                "checkpoint_path": str(checkpoint),
                "eval_split": split_name,
                "data_path": str(data_path),
                "accuracy": float(metrics.get("accuracy", 0.0)),
                "exact_accuracy": float(metrics.get("exact_accuracy", 0.0)),
                "lm_loss": float(metrics.get("lm_loss", 0.0)),
                "steps": float(metrics.get("steps", 0.0)),
            }
            rows.append(row)
            print(
                "RESULT",
                row["checkpoint"],
                row["eval_split"],
                f"accuracy={row['accuracy']:.6f}",
                f"exact_accuracy={row['exact_accuracy']:.6f}",
                f"lm_loss={row['lm_loss']:.6f}",
                f"steps={row['steps']:.3f}",
            )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote endpoint matrix: {output_csv}")


if __name__ == "__main__":
    main()
