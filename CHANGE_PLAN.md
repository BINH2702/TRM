# Change Plan: Process Forgetting in TRM

This file tracks the plan for modifying this repo to study process forgetting in
iterative reasoners such as TRM, and records what has changed over time.

## 1. Current Goal

Implement experiments and diagnostics for **process forgetting**: a failure mode
where an iterative reasoner keeps old endpoint accuracy but loses the old
free-running solution process that moves latent states toward correct answers.

The target method is **Free-Running Reasoning Flow Consolidation (FR-RFC)**.
During continual learning, FR-RFC should preserve the old model's reasoning-flow
distribution under the current model's own rollout, not only match final
answers or teacher-forced hidden states.

## 2. Completed Work And Current Results

This section summarizes what has been done in this TRM repo so far and what the
first results mean.

### Implemented

1. Added a dataset splitter for same-family continual learning:

```text
dataset/split_cl_dataset.py
```

It splits one prepared TRM `PuzzleDataset` into compatible `task_a` and
`task_b` roots while preserving sequence length, vocabulary, token IDs, and
puzzle identifier metadata.

2. Added a sequential fine-tuning runner:

```text
scripts/run_single_dataset_cl_finetune.sh
```

It runs:

```text
Task A training from scratch
Task B fine-tuning from the Task A checkpoint
Task B evaluation on old Task A test data
```

3. Added an endpoint-matrix evaluator:

```text
scripts/eval_cl_endpoint_matrix.py
```

It evaluates:

```text
Task A checkpoint on Task A test
Task A checkpoint on Task B test
Task B checkpoint on Task A test
Task B checkpoint on Task B test
```

4. Prepared local datasets under `/mnt/data/binhnt6/trm_data`:

```text
/mnt/data/binhnt6/trm_data/sudoku-extreme-1k-noaug
/mnt/data/binhnt6/trm_data/maze-30x30-hard-1k-noaug
```

5. Ran three TRM-native Sudoku CL experiments:

```text
sudoku_cl_smoke_20260619_035939
sudoku_cl_real_20260619_040846
sudoku_cl_hard_blanks_20260619_063039
```

### Main Result So Far

The current CL setup runs successfully end to end, but the Sudoku experiments
have not yet produced a clean forgetting case.

Summary:

| Run | Split | What happened | Scientific use |
| --- | --- | --- | --- |
| `sudoku_cl_smoke_20260619_035939` | tiny random Sudoku split | Pipeline completed, but accuracy was too low to interpret. | Smoke test only. |
| `sudoku_cl_real_20260619_040846` | random Sudoku A/B split | Old Task A exact accuracy increased from 0.065 to 0.080 after Task B fine-tuning. | First endpoint baseline; random split is too homogeneous. |
| `sudoku_cl_hard_blanks_20260619_063039` | easy Sudoku by blank count -> harder Sudoku by blank count | Old Task A exact accuracy increased from 0.100 to 0.110 after Task B fine-tuning, while Task B exact accuracy stayed at 0.005. | Harder distribution split, but not a clean forgetting test because Task B did not learn. |

Main conclusion:

```text
The TRM-native CL pipeline is working, but no endpoint forgetting has appeared
yet. The harder blank-count split creates distribution shift, but Task B is too
hard under the current budget, so the result is not yet a strong
stability-plasticity test.
```

The longer Sudoku run is the first useful endpoint baseline.

Endpoint matrix:

| Checkpoint | Eval split | Token accuracy | Exact accuracy | LM loss |
| --- | --- | ---: | ---: | ---: |
| Task A checkpoint | Task A test | 0.602 | 0.065 | 1.881 |
| Task A checkpoint | Task B test | 0.617 | 0.075 | 1.733 |
| Task B checkpoint | Task A test | 0.610 | 0.080 | 2.456 |
| Task B checkpoint | Task B test | 0.593 | 0.060 | 2.599 |

Interpretation:

```text
No endpoint forgetting is visible in this first Sudoku split.
Old Task A exact accuracy goes from 0.065 to 0.080 after Task B fine-tuning.
```

This does not prove TRM has no forgetting. It means this first Sudoku split is
not yet a strong forgetting setup because Task A and Task B are random splits of
the same Sudoku distribution.

The harder blank-count split also did not show endpoint forgetting:

| Checkpoint | Eval split | Token accuracy | Exact accuracy | LM loss |
| --- | --- | ---: | ---: | ---: |
| Task A checkpoint | Task A test | 0.640 | 0.100 | 2.236 |
| Task A checkpoint | Task B test | 0.546 | 0.005 | 2.818 |
| Task B checkpoint | Task A test | 0.644 | 0.110 | 1.791 |
| Task B checkpoint | Task B test | 0.547 | 0.005 | 2.234 |

Interpretation:

```text
No endpoint forgetting is visible in the harder easy-to-hard split either.
Old Task A exact accuracy goes from 0.100 to 0.110 after Task B fine-tuning.
Task B remains hard: exact accuracy stays at 0.005.
```

### Important Bug Fixed

The endpoint evaluator originally failed with:

```text
ModuleNotFoundError: No module named 'pretrain'
```

Cause:

```text
Running scripts/eval_cl_endpoint_matrix.py made Python search from scripts/
instead of the repo root.
```

Fix:

```text
Add the repo root to sys.path inside scripts/eval_cl_endpoint_matrix.py.
```

The training checkpoints were valid; only the final evaluator step failed. The
endpoint matrix was computed successfully afterward without retraining.

### Current Conclusion

The repo now has a working TRM-native CL endpoint baseline:

```text
Sudoku split A -> Sudoku split B
```

The next scientific step is not another random Sudoku split. The next step
should be one of:

1. Add TRM process-trace diagnostics for the saved checkpoints.
2. Increase Task B learning or use a more learnable hard split.
3. Run Maze split A -> Maze split B after confirming compute budget.

## 3. TRM-Native Hypothesis

This repo should test the proposal on **Tiny Recursive Models (TRM)** directly,
not on the previous graph-model benchmark.

The basic TRM-native continual-learning question is:

```text
Train TRM on Task A from one dataset family.
Save the Task A checkpoint.
Fine-tune the same TRM on Task B from the same dataset family.
Evaluate whether the Task B model preserves Task A endpoint accuracy and the
Task A recursive reasoning process.
```

The first hypothesis is endpoint forgetting:

```text
Sequential fine-tuning on Task B may reduce Task A accuracy.
```

The stronger process-forgetting hypothesis is:

```text
Even when Task A endpoint accuracy is partly preserved, the recursive TRM
process can drift: hidden states, update vectors, halt behavior, confidence, or
extra-rollout stability may differ from the Task A teacher.
```

The first test should stay inside one dataset family:

```text
Maze split A -> Maze split B
ARC split A -> ARC split B
Sudoku split A -> Sudoku split B
```

Do not start with `Sudoku -> Maze`, because cross-dataset training changes
sequence length, vocabulary, and output structure. That would mix continual
learning with architecture compatibility problems.

The minimum success criterion for the first TRM-native test is:

1. Train a Task A checkpoint.
2. Fine-tune it on Task B.
3. Evaluate old Task A and new Task B endpoint accuracy.
4. Save enough checkpoint structure to later compare Task A teacher rollouts
   against Task B student rollouts.

## 4. Research Idea Summary

TRM does not produce an answer in one pass. It repeatedly updates latent states
before predicting. Because of this, old-task knowledge may live in the update
field:

```text
z_{k+1} = F_theta(z_k, x)
V_theta(z, x) = F_theta(z, x) - z
```

Standard continual-learning methods usually preserve endpoints:

```text
x -> final answer
```

For iterative reasoners, this is incomplete. A model can still answer old replay
examples correctly while its old latent trajectories, update vectors, residuals,
or solution basins have changed. This hidden damage is process forgetting.

The repo should be extended to measure and reduce this process drift.

## 5. Main Method

Train on the new task with four parts:

1. New-task loss on current examples.
2. Endpoint preservation loss on old memory examples.
3. Free-running flow loss between old-model and current-model rollouts.
4. Basin or residual preservation loss for old-task solution regions.

The important design choice is **free-running matching**:

- Teacher-forced matching checks the student at old teacher states.
- Free-running matching checks the states the current model actually visits.
- FR-RFC should use free-running flow as the main objective.
- Teacher-forced velocity matching can remain as a baseline or warm-up.

## 6. Diagnostics To Add

1. **Endpoint accuracy**
   - Old replay accuracy.
   - New-task accuracy.
   - OOD or stress accuracy where possible.

2. **Transport / field drift**
   - Compare old and current update vectors along old-task rollouts.
   - Track drift on teacher states and student free-running states separately.

3. **Path drift**
   - Compare latent trajectories over iteration steps.
   - Include iteration index so the comparison is time-aware.

4. **Residual drift**
   - Measure `||F_theta(z, x) - z||` near old late-stage states.
   - Detect whether old solution states remain approximate fixed points.

5. **Basin robustness**
   - Perturb old late-stage states.
   - Roll out the current model.
   - Check whether it returns to a correct old-task solution.

6. **Extra-rollout stability**
   - Evaluate old tasks at `K`, `2K`, and `4K` iterations.
   - Detect cases where extra inference hurts old-task behavior.

## 7. Numbered Implementation Plan

1. **Map TRM state access**
   - Identify where `z_H`, `z_L`, logits, halt scores, and residual-like updates
     can be exposed in `models/recursive_reasoning/trm.py`.
   - Decide the minimal return format for collecting process traces.

2. **Add rollout tracing**
   - Add an optional trace mode that records selected iteration states,
     projected states, update vectors, residuals, and step indices.
   - Keep default training behavior unchanged.

3. **Add process metrics**
   - Implement field drift, path drift, residual drift, and extra-rollout
     stability metrics.
   - Start with coordinate-anchored comparisons because this repo resumes from
     old checkpoints.

4. **Add memory / teacher support**
   - Store or recompute old-model traces for memory examples.
   - Support frozen teacher model rollouts during continual training.

5. **Implement losses**
   - Endpoint replay/distillation baseline.
   - Teacher-forced velocity matching baseline.
   - Free-running flow matching loss.
   - Optional basin recovery loss.

6. **Run small controlled tests**
   - First verify on a small synthetic or toy dataset.
   - Then run Sudoku/Maze/TRM-compatible experiments if compute allows.

7. **Record every change**
   - Update the change log below with files changed, summary, and verification.

## 8. Next Step Update: Single-Dataset CL First

The next concrete implementation direction is to test continual learning within
one dataset family before attempting cross-dataset transfer.

Do not start with:

```text
Sudoku -> Maze
```

because those datasets have different sequence lengths and vocabularies. A
cross-dataset run would mix continual learning with architecture/head changes.

Start with one same-format split:

```text
Maze split A -> Maze split B
ARC split A -> ARC split B
Sudoku split A -> Sudoku split B
```

The first implemented baseline is:

```text
finetune: Task A checkpoint -> Task B training with no old-task protection
```

Implemented files:

```text
dataset/split_cl_dataset.py
scripts/run_single_dataset_cl_finetune.sh
```

The splitter creates compatible `task_a` and `task_b` dataset roots from one
prepared PuzzleDataset while preserving metadata such as `seq_len`, `vocab_size`,
token IDs, and puzzle identifier count.

Example:

```bash
python -m dataset.split_cl_dataset \
  --input-dir data/maze-30x30-hard-1k \
  --output-dir data/maze-cl \
  --task-a-fraction 0.5 \
  --seed 0 \
  --overwrite
```

Sequential finetune scaffold:

```bash
SOURCE_DATA=data/maze-30x30-hard-1k \
CL_DATA_DIR=data/maze-cl \
RUN_PREFIX=maze_cl \
EPOCHS_A=1000 \
EPOCHS_B=1000 \
GLOBAL_BATCH_SIZE=128 \
bash scripts/run_single_dataset_cl_finetune.sh
```

The next implementation step after this baseline is a TRM process-trace
evaluator for saved checkpoints. It should compare the Task A teacher checkpoint
and the Task B finetuned checkpoint on old Task A examples.

## 9. Key Risks

- Latent-coordinate matching is not fully representation invariant.
- Free-running teacher/student rollouts add compute cost.
- Full training runs may be expensive on this repo.
- Process metrics must be implemented without breaking normal `pretrain.py`.
- Same-dataset splits may be too easy and endpoint accuracy may stay saturated.
- Cross-dataset CL is more interesting but requires unified heads or task heads.
- `torch.compile` can make trace extraction harder; use `DISABLE_COMPILE=1` for
  early diagnostic runs.

## 10. Implementation Notes

- Prefer optional flags and return keys instead of changing default model output.
- Keep trace tensors detached unless a loss explicitly needs gradients.
- Start with low-dimensional projections for state and velocity traces.
- Use cached teacher traces where possible to reduce training cost.
- Keep teacher-forced velocity matching as a baseline, not the main method.
- Keep CL tests inside one dataset family until the baseline and evaluator work.

## 11. Change Log

| Date | Files Changed | Summary | Verification |
| --- | --- | --- | --- |
| 2026-06-19 | `CHANGE_PLAN.md` | Added this plan and change tracking file. | Not run; documentation-only change. |
| 2026-06-19 | `CHANGE_PLAN.md` | Summarized the process-forgetting proposal and converted the file into an implementation plan for FR-RFC. | Not run; documentation-only change. |
| 2026-06-19 | `CHANGE_PLAN.md` | Added a TRM-native hypothesis test plan to avoid relying on the previous graph-model experiments. | Not run; documentation-only change. |
| 2026-06-19 | `CHANGE_PLAN.md`, `dataset/split_cl_dataset.py`, `scripts/run_single_dataset_cl_finetune.sh` | Restored the previous plan history and added the first single-dataset CL setup. | `python3 -m py_compile dataset/split_cl_dataset.py`; `bash -n scripts/run_single_dataset_cl_finetune.sh`; temporary PuzzleDataset split smoke test. |
| 2026-06-19 | `CHANGE_PLAN.md` | Added explicit numbered sections and a TRM-native hypothesis section. | `git diff --check`. |
| 2026-06-19 | `CHANGE_PLAN.md`, `dataset/split_cl_dataset.py`, `scripts/run_single_dataset_cl_finetune.sh` | Added smoke-run split caps, configurable EMA, and recorded the Sudoku CL smoke run. | `python3 -m py_compile dataset/split_cl_dataset.py`; `bash -n scripts/run_single_dataset_cl_finetune.sh`; Slurm MIG run `sudoku_cl_smoke_20260619_035939`. |
| 2026-06-19 | `CHANGE_PLAN.md`, `scripts/eval_cl_endpoint_matrix.py` | Added endpoint-matrix evaluator and recorded the longer Sudoku CL run. | Slurm MIG run `sudoku_cl_real_20260619_040846`; standalone endpoint evaluation wrote `/mnt/data/binhnt6/trm_runs/results/sudoku_cl_real_20260619_040846_endpoint_matrix.csv`. |
| 2026-06-19 | `CHANGE_PLAN.md`, `dataset/split_cl_dataset.py`, `scripts/run_single_dataset_cl_finetune.sh` | Added Sudoku blank-count split support and recorded the harder easy-to-hard Sudoku CL run. | `python3 -m py_compile dataset/split_cl_dataset.py scripts/eval_cl_endpoint_matrix.py`; `bash -n scripts/run_single_dataset_cl_finetune.sh`; Slurm MIG run `sudoku_cl_hard_blanks_20260619_063039`. |

## 12. Smoke Run Log

### 2026-06-19: Sudoku CL Smoke Run

Run id:

```text
sudoku_cl_smoke_20260619_035939
```

Purpose:

```text
Verify the TRM-native single-dataset continual-learning pipeline.
```

Dataset:

```text
source: /mnt/data/binhnt6/trm_data/sudoku-extreme-1k-noaug
split:  /mnt/data/binhnt6/trm_data/cl_splits/sudoku_cl_smoke_20260619_035939
```

Split sizes:

| Task | Split | Examples | Sequence length | Vocab size |
| --- | --- | ---: | ---: | ---: |
| Task A | train | 500 | 81 | 11 |
| Task A | test | 100 | 81 | 11 |
| Task B | train | 500 | 81 | 11 |
| Task B | test | 100 | 81 | 11 |

Training configuration:

```text
partition: mig
gpu: 1
epochs_a: 20
epochs_b: 20
eval_interval: 20
global_batch_size: 128
DISABLE_COMPILE: 1
EMA: False
hidden_size: 128
H_cycles: 1
L_cycles: 2
halt_max_steps: 4
```

Artifacts:

```text
log: /mnt/data/binhnt6/trm_runs/logs/sudoku_cl_smoke_20260619_035939.log
task_a_checkpoint: /mnt/data/binhnt6/trm_runs/checkpoints/sudoku_cl_smoke_20260619_035939_task_a/step_78
task_b_checkpoint: /mnt/data/binhnt6/trm_runs/checkpoints/sudoku_cl_smoke_20260619_035939_task_b_finetune/step_78
```

Outcome:

```text
Task A trained and saved a checkpoint.
Task B loaded the Task A checkpoint, trained, evaluated on Task A test data, and
saved a checkpoint.
```

Endpoint smoke metrics:

| Checkpoint | Eval split | Token accuracy | Exact accuracy | LM loss | Inference steps |
| --- | --- | ---: | ---: | ---: | ---: |
| Task A checkpoint | Task A test | 0.131 | 0.000 | 2.293 | 4.0 |
| Task A checkpoint | Task B test | 0.135 | 0.000 | 2.279 | 4.0 |
| Task B checkpoint | Task A test | 0.207 | 0.000 | 2.161 | 4.0 |
| Task B checkpoint | Task B test | 0.210 | 0.000 | 2.150 | 4.0 |

This is a pipeline smoke test only. It is not a scientific CL result because
the model was trained for only 78 optimization steps per phase.

## 13. Longer Sudoku CL Run Log

### 2026-06-19: Sudoku CL Real Run

Run id:

```text
sudoku_cl_real_20260619_040846
```

Purpose:

```text
Run a first non-smoke TRM-native single-dataset CL test:
Sudoku Task A -> Sudoku Task B.
```

Dataset:

```text
source: /mnt/data/binhnt6/trm_data/sudoku-extreme-1k-noaug
split:  /mnt/data/binhnt6/trm_data/cl_splits/sudoku_cl_real_20260619_040846
```

Split sizes:

| Task | Split | Examples | Sequence length | Vocab size |
| --- | --- | ---: | ---: | ---: |
| Task A | train | 500 | 81 | 11 |
| Task A | test | 200 | 81 | 11 |
| Task B | train | 500 | 81 | 11 |
| Task B | test | 200 | 81 | 11 |

Training configuration:

```text
partition: mig
gpu: 1
epochs_a: 5000
epochs_b: 5000
steps per phase: 19531
global_batch_size: 128
DISABLE_COMPILE: 1
EMA: False
H_cycles: 3
L_cycles: 6
halt_max_steps: 16
```

Artifacts:

```text
log: /mnt/data/binhnt6/trm_runs/logs/sudoku_cl_real_20260619_040846.log
task_a_checkpoint: /mnt/data/binhnt6/trm_runs/checkpoints/sudoku_cl_real_20260619_040846_task_a/step_19531
task_b_checkpoint: /mnt/data/binhnt6/trm_runs/checkpoints/sudoku_cl_real_20260619_040846_task_b_finetune/step_19531
endpoint_matrix: /mnt/data/binhnt6/trm_runs/results/sudoku_cl_real_20260619_040846_endpoint_matrix.csv
```

Endpoint matrix:

| Checkpoint | Eval split | Token accuracy | Exact accuracy | LM loss | Inference steps |
| --- | --- | ---: | ---: | ---: | ---: |
| Task A checkpoint | Task A test | 0.602 | 0.065 | 1.881 | 16.0 |
| Task A checkpoint | Task B test | 0.617 | 0.075 | 1.733 | 16.0 |
| Task B checkpoint | Task A test | 0.610 | 0.080 | 2.456 | 16.0 |
| Task B checkpoint | Task B test | 0.593 | 0.060 | 2.599 | 16.0 |

Outcome:

```text
Both Task A and Task B training completed and saved checkpoints.
The original Slurm job exited with code 1 because the endpoint evaluator could
not import repo-root pretrain.py when launched as scripts/eval_cl_endpoint_matrix.py.
The import path was fixed and the endpoint matrix was computed afterward without
retraining.
```

Initial interpretation:

```text
No clear endpoint forgetting is visible in this split: Task B checkpoint has
slightly higher old Task A exact accuracy than the Task A checkpoint.

However, Task B fine-tuning also reduced Task B test exact accuracy from the
Task A checkpoint baseline, suggesting this small split/checkpoint is not yet a
clean CL-forgetting setting. The next useful step is process-trace diagnostics
and/or a harder split where Task A and Task B are more distinct.
```

## 14. Harder Sudoku Blank-Count Split Run Log

### 2026-06-19: Sudoku Easy-To-Hard CL Run

Run id:

```text
sudoku_cl_hard_blanks_20260619_063039
```

Purpose:

```text
Create a harder same-dataset CL split by assigning easier Sudoku puzzles to
Task A and harder Sudoku puzzles to Task B.
```

Split rule:

```text
split_key: sudoku_blanks
task_a_side: low
```

Task A gets lower-blank puzzles. Task B gets higher-blank puzzles.

Dataset:

```text
source: /mnt/data/binhnt6/trm_data/sudoku-extreme-1k-noaug
split:  /mnt/data/binhnt6/trm_data/cl_splits/sudoku_cl_hard_blanks_20260619_063039
```

Blank-count separation:

| Task | Split | Examples | Blank min | Blank max | Blank mean |
| --- | --- | ---: | ---: | ---: | ---: |
| Task A | train | 500 | 46 | 56 | 54.46 |
| Task A | test | 200 | 46 | 56 | 55.07 |
| Task B | train | 500 | 56 | 64 | 57.13 |
| Task B | test | 200 | 57 | 64 | 57.57 |

Training configuration:

```text
partition: mig
gpu: 1
epochs_a: 5000
epochs_b: 5000
steps per phase: 19531
global_batch_size: 128
DISABLE_COMPILE: 1
EMA: False
H_cycles: 3
L_cycles: 6
halt_max_steps: 16
```

Artifacts:

```text
log: /mnt/data/binhnt6/trm_runs/logs/sudoku_cl_hard_blanks_20260619_063039.log
task_a_checkpoint: /mnt/data/binhnt6/trm_runs/checkpoints/sudoku_cl_hard_blanks_20260619_063039_task_a/step_19531
task_b_checkpoint: /mnt/data/binhnt6/trm_runs/checkpoints/sudoku_cl_hard_blanks_20260619_063039_task_b_finetune/step_19531
endpoint_matrix: /mnt/data/binhnt6/trm_runs/results/sudoku_cl_hard_blanks_20260619_063039_endpoint_matrix.csv
```

Endpoint matrix:

| Checkpoint | Eval split | Token accuracy | Exact accuracy | LM loss | Inference steps |
| --- | --- | ---: | ---: | ---: | ---: |
| Task A checkpoint | Task A test | 0.640 | 0.100 | 2.236 | 16.0 |
| Task A checkpoint | Task B test | 0.546 | 0.005 | 2.818 | 16.0 |
| Task B checkpoint | Task A test | 0.644 | 0.110 | 1.791 | 16.0 |
| Task B checkpoint | Task B test | 0.547 | 0.005 | 2.234 | 16.0 |

Outcome:

```text
The harder easy-to-hard split ran successfully and wrote the endpoint matrix.
No endpoint forgetting is visible: old Task A exact accuracy changes from 0.100
to 0.110 after Task B fine-tuning.
```

Interpretation:

```text
The harder blank-count split creates clear distribution shift: Task A is easier
and Task B is harder. The Task A checkpoint has almost no exact generalization
to the harder Task B test split, and Task B fine-tuning does not improve Task B
exact accuracy under this budget.

This is still not a clean endpoint-forgetting setup. It is useful because it
shows that random Sudoku splitting was too homogeneous, but Task B learning is
too weak to create a strong stability-plasticity test.
```

Next step from this result:

```text
Use this hard split for process diagnostics, or increase Task B learning budget
/ use a more learnable split before testing forgetting claims.
```

## 15. Notes

- Keep entries short and factual.
- Include commands used for verification when possible.
- Mention any skipped tests and the reason they were skipped.
