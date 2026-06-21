from collections import deque
from typing import Dict, Optional

import numpy as np
import torch


WALL = 1
START = 3
GOAL = 4
PATH = 5

POLICY_GOAL = 1
POLICY_UP = 2
POLICY_DOWN = 3
POLICY_LEFT = 4
POLICY_RIGHT = 5

DIRS = {
    POLICY_UP: (-1, 0),
    POLICY_DOWN: (1, 0),
    POLICY_LEFT: (0, -1),
    POLICY_RIGHT: (0, 1),
}


def _as_square(seq: np.ndarray) -> np.ndarray:
    side = int(round(seq.size ** 0.5))
    if side * side != seq.size:
        raise ValueError(f"Maze sequence length must be square, got {seq.size}")
    return seq.reshape(side, side)


def _single_pos(grid: np.ndarray, token: int) -> tuple[int, int]:
    positions = np.argwhere(grid == token)
    if positions.shape[0] != 1:
        raise ValueError(f"Expected exactly one token {token}, found {positions.shape[0]}")
    return tuple(map(int, positions[0]))


def _shortest_distance(input_grid: np.ndarray) -> int:
    start = _single_pos(input_grid, START)
    goal = _single_pos(input_grid, GOAL)
    q: deque[tuple[int, int, int]] = deque([(start[0], start[1], 0)])
    seen = {start}

    while q:
        r, c, d = q.popleft()
        if (r, c) == goal:
            return d
        for dr, dc in DIRS.values():
            nr, nc = r + dr, c + dc
            if not (0 <= nr < input_grid.shape[0] and 0 <= nc < input_grid.shape[1]):
                continue
            if input_grid[nr, nc] == WALL or (nr, nc) in seen:
                continue
            seen.add((nr, nc))
            q.append((nr, nc, d + 1))
    return -1


class MazePathFunctional:
    required_outputs = {"preds"}

    def __init__(self):
        self._rows = []

    def begin_eval(self):
        self._rows = []

    def update_batch(self, batch: Dict[str, torch.Tensor], preds: Dict[str, torch.Tensor]):
        inputs = batch["inputs"].detach().cpu().numpy()
        labels = batch["labels"].detach().cpu().numpy()
        pred_tokens = preds["preds"].detach().cpu().numpy()

        for input_seq, label_seq, pred_seq in zip(inputs, labels, pred_tokens):
            if np.all(label_seq == -100):
                continue
            input_grid = _as_square(input_seq)
            label_grid = _as_square(label_seq)
            pred_grid = _as_square(pred_seq)

            target_path = label_grid == PATH
            pred_path = pred_grid == PATH
            tp = int(np.logical_and(pred_path, target_path).sum())
            fp = int(np.logical_and(pred_path, np.logical_not(target_path)).sum())
            fn = int(np.logical_and(np.logical_not(pred_path), target_path).sum())

            wall_cross = bool(np.logical_and(pred_path, input_grid == WALL).any())
            start = _single_pos(input_grid, START)
            goal = _single_pos(input_grid, GOAL)
            allowed = pred_path.copy()
            allowed[start] = True
            allowed[goal] = True

            connected = False
            if not wall_cross:
                q: deque[tuple[int, int]] = deque([start])
                seen = {start}
                while q:
                    r, c = q.popleft()
                    if (r, c) == goal:
                        connected = True
                        break
                    for dr, dc in DIRS.values():
                        nr, nc = r + dr, c + dc
                        if not (0 <= nr < input_grid.shape[0] and 0 <= nc < input_grid.shape[1]):
                            continue
                        if (nr, nc) in seen or not allowed[nr, nc] or input_grid[nr, nc] == WALL:
                            continue
                        seen.add((nr, nc))
                        q.append((nr, nc))

            target_len = max(int(target_path.sum()), 1)
            self._rows.append(
                {
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "valid": float(connected and not wall_cross),
                    "connected": float(connected),
                    "wall_cross": float(wall_cross),
                    "pred_len": float(pred_path.sum()),
                    "target_len": float(target_len),
                }
            )

    def result(self, save_path: Optional[str], rank: int, world_size: int, group: Optional[torch.distributed.ProcessGroup] = None):
        if rank != 0:
            return None
        if not self._rows:
            return {}

        tp = sum(row["tp"] for row in self._rows)
        fp = sum(row["fp"] for row in self._rows)
        fn = sum(row["fn"] for row in self._rows)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)

        return {
            "maze_path_valid_rate": float(np.mean([row["valid"] for row in self._rows])),
            "maze_path_connected_rate": float(np.mean([row["connected"] for row in self._rows])),
            "maze_path_wall_cross_rate": float(np.mean([row["wall_cross"] for row in self._rows])),
            "maze_path_precision": precision,
            "maze_path_recall": recall,
            "maze_path_f1": f1,
            "maze_path_length_ratio": float(
                np.mean([row["pred_len"] / max(row["target_len"], 1.0) for row in self._rows])
            ),
        }


class MazePolicyFunctional:
    required_outputs = {"preds"}

    def __init__(self, max_steps_multiplier: int = 4):
        self.max_steps_multiplier = max_steps_multiplier
        self._rows = []

    def begin_eval(self):
        self._rows = []

    def update_batch(self, batch: Dict[str, torch.Tensor], preds: Dict[str, torch.Tensor]):
        inputs = batch["inputs"].detach().cpu().numpy()
        labels = batch["labels"].detach().cpu().numpy()
        pred_tokens = preds["preds"].detach().cpu().numpy()

        for input_seq, label_seq, pred_seq in zip(inputs, labels, pred_tokens):
            if np.all(label_seq == -100):
                continue
            input_grid = _as_square(input_seq)
            label_grid = _as_square(label_seq)
            pred_grid = _as_square(pred_seq)

            reachable = label_grid != -100
            reachable_count = max(int(reachable.sum()), 1)
            reachable_acc = float(np.logical_and(pred_grid == label_grid, reachable).sum() / reachable_count)

            start = _single_pos(input_grid, START)
            goal = _single_pos(input_grid, GOAL)
            shortest = _shortest_distance(input_grid)
            max_steps = self.max_steps_multiplier * input_grid.size

            status = "max_steps"
            r, c = start
            seen = set()
            steps = 0
            for steps in range(max_steps + 1):
                if (r, c) == goal:
                    status = "success"
                    break
                if (r, c) in seen:
                    status = "cycle"
                    break
                seen.add((r, c))

                action = int(pred_grid[r, c])
                if action not in DIRS:
                    status = "invalid_action"
                    break
                dr, dc = DIRS[action]
                nr, nc = r + dr, c + dc
                if not (0 <= nr < input_grid.shape[0] and 0 <= nc < input_grid.shape[1]):
                    status = "offgrid"
                    break
                if input_grid[nr, nc] == WALL:
                    status = "wall"
                    break
                r, c = nr, nc

            self._rows.append(
                {
                    "reachable_acc": reachable_acc,
                    "success": float(status == "success"),
                    "cycle": float(status == "cycle"),
                    "wall": float(status == "wall"),
                    "offgrid": float(status == "offgrid"),
                    "invalid_action": float(status == "invalid_action"),
                    "length_ratio": float(steps / max(shortest, 1)) if status == "success" else 0.0,
                }
            )

    def result(self, save_path: Optional[str], rank: int, world_size: int, group: Optional[torch.distributed.ProcessGroup] = None):
        if rank != 0:
            return None
        if not self._rows:
            return {}

        return {
            "maze_policy_reachable_acc": float(np.mean([row["reachable_acc"] for row in self._rows])),
            "maze_policy_rollout_success": float(np.mean([row["success"] for row in self._rows])),
            "maze_policy_cycle_rate": float(np.mean([row["cycle"] for row in self._rows])),
            "maze_policy_wall_rate": float(np.mean([row["wall"] for row in self._rows])),
            "maze_policy_offgrid_rate": float(np.mean([row["offgrid"] for row in self._rows])),
            "maze_policy_invalid_action_rate": float(np.mean([row["invalid_action"] for row in self._rows])),
            "maze_policy_success_length_ratio": float(
                np.mean([row["length_ratio"] for row in self._rows if row["success"]]) if any(row["success"] for row in self._rows) else 0.0
            ),
        }
