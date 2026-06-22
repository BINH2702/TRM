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

4. Added a four-run learnability-gate evaluator and runner:

```text
scripts/eval_cl_learnability_gate.py
scripts/run_single_dataset_cl_learnability_gate.sh
```

It runs and evaluates:

```text
A-only
B-only
joint A+B
sequential A -> B
```

5. Added a Maze path-length CL gate wrapper:

```text
scripts/run_maze_cl_gate.sh
```

It splits Maze by old solution-path length and can be launched later with:

```bash
sbatch scripts/run_maze_cl_gate.sh
```

6. Prepared local datasets under `/mnt/data/binhnt6/trm_data`:

```text
/mnt/data/binhnt6/trm_data/sudoku-extreme-1k-noaug
/mnt/data/binhnt6/trm_data/maze-30x30-hard-1k-noaug
```

7. Ran four TRM-native Sudoku CL experiments:

```text
sudoku_cl_smoke_20260619_035939
sudoku_cl_real_20260619_040846
sudoku_cl_hard_blanks_20260619_063039
sudoku_cl_gate_1gpu_blanks_mlp_t_20260619_085517
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
| `sudoku_cl_gate_1gpu_blanks_mlp_t_20260619_085517` | same blank-count split, `arch.mlp_t=True` | A-only, B-only, joint A+B, and sequential A->B completed. Best B exact is sequential at 0.040; best A exact is sequential at 0.135. | Learnability gate; still not strong enough for scientific CL forgetting because B-only and joint B exact remain very low. |

Main conclusion:

```text
The TRM-native CL pipeline is working, but no endpoint forgetting has appeared
yet. The harder blank-count split creates distribution shift, but Task B remains
too weak under the current no-augmentation budget, even with the Sudoku-style
`arch.mlp_t=True` setting. The result is not yet a strong stability-plasticity
test.
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

1. Build a more learnable Sudoku split or use augmented Sudoku.
2. Increase Task B learning strength before making forgetting claims.
3. Add TRM process-trace diagnostics only after endpoint learnability improves.
4. Run Maze split A -> Maze split B after confirming compute budget.

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
| 2026-06-20 | `CHANGE_PLAN.md` | Recorded the one-GPU Sudoku learnability gate with A-only, B-only, joint, and sequential runs. | Slurm MIG job `20898`; output `/mnt/data/binhnt6/trm_runs/results/sudoku_cl_gate_1gpu_blanks_mlp_t_20260619_085517/learnability_gate_endpoint_matrix.csv`. |
| 2026-06-20 | `CHANGE_PLAN.md`, `dataset/split_cl_dataset.py`, `scripts/eval_cl_learnability_gate.py`, `scripts/run_single_dataset_cl_learnability_gate.sh`, `scripts/run_maze_cl_gate.sh` | Added Maze path-length split support and a reproducible Maze CL learnability gate. | `python3 -m py_compile dataset/split_cl_dataset.py scripts/eval_cl_learnability_gate.py scripts/eval_cl_endpoint_matrix.py`; `bash -n scripts/run_single_dataset_cl_learnability_gate.sh scripts/run_maze_cl_gate.sh`; Maze split smoke test. |

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

## 15. One-GPU Sudoku Learnability Gate

### 2026-06-19: Blank-Count Learnability Gate

Run id:

```text
sudoku_cl_gate_1gpu_blanks_mlp_t_20260619_085517
```

Purpose:

```text
Check whether the current blank-count Sudoku split is learnable enough for a
meaningful continual-learning experiment.
```

This is a gate, not yet a process-forgetting experiment. The gate asks:

```text
Can Task A learn by itself?
Can Task B learn by itself?
Can Task A and Task B learn jointly?
Can Task B be learned after Task A in a sequential A -> B run?
```

Slurm job:

```text
job_id: 20898
partition: mig
gpu: 1 MIG GPU
cpus_per_task: 4
mem: 20G
state: COMPLETED
exit_code: 0
elapsed: 03:48:33
```

Dataset:

```text
source: /mnt/data/binhnt6/trm_data/sudoku-extreme-1k-noaug
split:  /mnt/data/binhnt6/trm_data/cl_splits/sudoku_cl_gate_1gpu_blanks_mlp_t_20260619_085517
```

Split rule:

```text
split_key: sudoku_blanks
task_a_side: low
```

Task A gets easier Sudoku puzzles with fewer blank cells. Task B gets harder
Sudoku puzzles with more blank cells.

Blank-count separation:

| Task | Split | Examples | Blank min | Blank max | Blank mean |
| --- | --- | ---: | ---: | ---: | ---: |
| Task A | train | 500 | 46 | 56 | 54.46 |
| Task A | test | 200 | 46 | 56 | 55.07 |
| Task B | train | 500 | 56 | 64 | 57.13 |
| Task B | test | 200 | 57 | 64 | 57.57 |

Model and training configuration:

```text
arch: trm
arch.mlp_t: True
H_cycles: 3
L_cycles: 6
halt_max_steps: 16
hidden_size: 512
epochs per single-task phase: 5000
global_batch_size: 128
lr: 1e-4
puzzle_emb_lr: 1e-4
weight_decay: 1.0
puzzle_emb_weight_decay: 1.0
EMA: False
DISABLE_COMPILE: 1
```

The `arch.mlp_t=True` setting was used because the TRM README uses the
Sudoku-style MLP transition setting for strong Sudoku runs.

Runs completed:

| Run | Meaning | Training steps |
| --- | --- | ---: |
| `a_only` | Train from scratch on Task A only. | 19,531 |
| `b_only` | Train from scratch on Task B only. | 19,531 |
| `joint_ab` | Train from scratch on Task A and Task B together. | 39,062 |
| `sequential_b` | Train Task A, then fine-tune on Task B. | 19,531 for Task B after Task A |

Artifacts:

```text
log: /mnt/data/binhnt6/trm_runs/logs/trm_cl_gate_1gpu_20898.out
stderr/progress: /mnt/data/binhnt6/trm_runs/logs/trm_cl_gate_1gpu_20898.err
result_csv: /mnt/data/binhnt6/trm_runs/results/sudoku_cl_gate_1gpu_blanks_mlp_t_20260619_085517/learnability_gate_endpoint_matrix.csv
a_only_checkpoint: /mnt/data/binhnt6/trm_runs/checkpoints/sudoku_cl_gate_1gpu_blanks_mlp_t_20260619_085517/a_only/step_19531
b_only_checkpoint: /mnt/data/binhnt6/trm_runs/checkpoints/sudoku_cl_gate_1gpu_blanks_mlp_t_20260619_085517/b_only/step_19531
joint_checkpoint: /mnt/data/binhnt6/trm_runs/checkpoints/sudoku_cl_gate_1gpu_blanks_mlp_t_20260619_085517/joint_ab/step_39062
sequential_checkpoint: /mnt/data/binhnt6/trm_runs/checkpoints/sudoku_cl_gate_1gpu_blanks_mlp_t_20260619_085517/sequential_b/step_19531
```

Endpoint matrix:

| Checkpoint | Eval split | Token accuracy | Exact accuracy | LM loss | Inference steps |
| --- | --- | ---: | ---: | ---: | ---: |
| `a_only` | Task A test | 0.634 | 0.090 | 2.118 | 16.0 |
| `a_only` | Task B test | 0.545 | 0.015 | 2.621 | 16.0 |
| `b_only` | Task A test | 0.635 | 0.075 | 1.396 | 16.0 |
| `b_only` | Task B test | 0.538 | 0.010 | 1.766 | 16.0 |
| `joint_ab` | Task A test | 0.646 | 0.095 | 1.570 | 16.0 |
| `joint_ab` | Task B test | 0.556 | 0.020 | 1.947 | 16.0 |
| `sequential_b` | Task A test | 0.670 | 0.135 | 1.295 | 16.0 |
| `sequential_b` | Task B test | 0.569 | 0.040 | 1.669 | 16.0 |

Outcome:

```text
The one-GPU learnability gate completed successfully.
No endpoint forgetting is visible: sequential A -> B has the best Task A exact
accuracy, 0.135, compared with 0.090 for A-only.
```

Main interpretation:

```text
The current no-augmentation blank-count Sudoku split is still not a clean
continual-learning forgetting setup. Task B remains weak even when trained
from scratch or jointly.
```

Evidence:

```text
B-only Task B exact:      0.010
Joint A+B Task B exact:   0.020
Sequential Task B exact:  0.040
```

The sequential run improves Task B somewhat, but 4% exact accuracy is still too
low for a scientific process-forgetting experiment. A weak Task B learner cannot
produce a meaningful stability-plasticity test.

Current conclusion:

```text
The TRM-native CL infrastructure works, and the one-GPU gate confirms that the
current Sudoku setup is the bottleneck. The next step should be to make the
Sudoku split more learnable, likely with augmentation or a milder difficulty
split, before adding process-flow diagnostics or FR-RFC losses.
```

## 16. Maze Path-Length CL Gate

### 2026-06-20: Added Maze Gate Setup

Purpose:

```text
Move the next TRM-native hypothesis test to a dataset with a cleaner process
depth axis, while keeping Sudoku as a separate branch.
```

The current Sudoku blank-count gate showed that the infrastructure works, but
the current no-augmentation Sudoku split is not yet learnable enough for a
scientific process-forgetting test. Maze is the next candidate because the
solution path length gives a direct old-task stress axis:

```text
shorter paths vs longer paths
normal rollout vs longer rollout
endpoint accuracy vs path/process stability
```

Implemented:

```text
dataset/split_cl_dataset.py
scripts/eval_cl_learnability_gate.py
scripts/run_single_dataset_cl_learnability_gate.sh
scripts/run_maze_cl_gate.sh
scripts/submit_maze_sequential_after_a.sh
```

Maze split rule:

```text
split_key: maze_path_length
task_a_side: low
```

Task A receives shorter-path mazes. Task B receives longer-path mazes.

The Maze dataset encodes the solution path character `o` as token id `5`, so
the path-length score is:

```text
(labels == 5).sum(axis=1)
```

Local source dataset:

```text
/mnt/data/binhnt6/trm_data/maze-30x30-hard-1k-noaug
```

Approximate half-split path-length stats:

| Split | Task | Examples | Path min | Path median | Path mean | Path max |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| train | Task A low half | 500 | 108 | 109 | 109.258 | 111 |
| train | Task B high half | 500 | 111 | 115 | 116.870 | 145 |
| test | Task A low half | 500 | 108 | 109 | 109.208 | 111 |
| test | Task B high half | 500 | 111 | 114 | 115.830 | 152 |

The separation is real but not huge. This means the first Maze run should be
treated as a learnability/interference gate, not as a final process-forgetting
experiment.

Maze gate command:

```bash
sbatch scripts/run_maze_cl_gate.sh
```

Default run configuration:

```text
source_data: /mnt/data/binhnt6/trm_data/maze-30x30-hard-1k-noaug
split_key: maze_path_length
task_a_side: low
max_train_groups: 1000
max_test_groups: 400
arch: trm
arch.mlp_t: False
arch.L_layers: 2
halt_max_steps: 16
epochs_a: 2500
epochs_b: 2500
epochs_joint: 2500
global_batch_size: 64
EMA: False
DISABLE_COMPILE: 1
```

The gate runs:

| Run | Meaning |
| --- | --- |
| `a_only` | Train from scratch on short-path Task A only. |
| `b_only` | Train from scratch on long-path Task B only. |
| `joint_ab` | Train from scratch on Task A and Task B together. |
| `sequential_b` | Train Task A, then fine-tune on Task B. |

Pass criteria before process diagnostics:

```text
A-only should learn Task A.
B-only should learn Task B.
Joint should learn both reasonably.
Sequential A -> B should learn Task B.
Then check whether old Task A endpoint, longer-rollout behavior, or path
process metrics degrade after Task B.
```

Caveat:

```text
Maze sequence length is 900, compared with Sudoku sequence length 81. The
official README reports Maze as a longer run, so the first job should be
monitored for MIG throughput and memory before scaling to multiple seeds.
The first 128-batch MIG attempt OOMed immediately, so the script defaults were
changed to batch size 64 and 2500 epochs per phase. This keeps the same rough
optimizer-step budget as 5000 epochs with batch size 128.
```

Verification completed:

```text
python3 -m py_compile dataset/split_cl_dataset.py scripts/eval_cl_learnability_gate.py scripts/eval_cl_endpoint_matrix.py
bash -n scripts/run_single_dataset_cl_learnability_gate.sh scripts/run_maze_cl_gate.sh
python3 -m dataset.split_cl_dataset --input-dir /mnt/data/binhnt6/trm_data/maze-30x30-hard-1k-noaug --output-dir /mnt/data/binhnt6/trm_data/cl_splits/maze_pathlen_split_smoke --split-key maze_path_length --task-a-side low --task-a-fraction 0.5 --max-train-groups 20 --max-test-groups 20 --overwrite
```

### 2026-06-20: First Maze Gate Execution and Partial Results

Run id:

```text
maze_cl_gate_pathlen_20260620_015643
```

Main paths:

```text
split_dir: /mnt/data/binhnt6/trm_data/cl_splits/maze_cl_gate_pathlen_20260620_015643
checkpoint_root: /mnt/data/binhnt6/trm_runs/checkpoints/maze_cl_gate_pathlen_20260620_015643
result_root: /mnt/data/binhnt6/trm_runs/results/maze_cl_gate_pathlen_20260620_015643
partial_result_csv: /mnt/data/binhnt6/trm_runs/results/maze_cl_gate_pathlen_20260620_015643/partial_done_endpoint_matrix.csv
```

Execution notes:

1. The first Maze job with `global_batch_size=128` failed immediately with CUDA
   OOM on a 40GB MIG GPU.
2. The script defaults were changed to `global_batch_size=64` and
   `epochs=2500` per phase.
3. The serial gate was split into parallel branches after `a_only` completed:
   `b_only_parallel`, `joint_ab_parallel`, and `sequential_b_parallel`.
4. `a_only`, `b_only_parallel`, and `sequential_b_parallel` completed.
5. The first `joint_ab_parallel` job completed training but failed during
   built-in validation with:

```text
KeyError: 'all1'
```

Cause:

```text
TRM training accepted data_paths=[task_a,task_b], but validation created set
names all and all1 while the evaluator expected only the metadata set all.
```

Fix:

```text
Created a merged Task A+B dataset:
/mnt/data/binhnt6/trm_data/cl_splits/maze_cl_gate_pathlen_20260620_015643/task_ab
```

The joint-only rerun is now using:

```text
data_paths=[/mnt/data/binhnt6/trm_data/cl_splits/maze_cl_gate_pathlen_20260620_015643/task_ab]
```

The generic learnability-gate runner was also updated to create this merged
`task_ab` dataset automatically and to use it for future joint A+B runs. This
prevents the same `all1` validation failure in later runs.

Current branch status:

| Branch | Slurm job | Status | Checkpoint |
| --- | ---: | --- | --- |
| `a_only` | `21005` | completed before serial job was canceled | `a_only/step_19531` |
| `b_only_parallel` | `21015` | completed | `b_only_parallel/step_19531` |
| `sequential_b_parallel` | `21021` | completed | `sequential_b_parallel/step_19531` |
| `joint_ab_parallel` | `21016` | failed after training during validation | none |
| `joint_ab_rerun` | `21096` | running | pending |

Completed-branch endpoint matrix:

| Checkpoint | Eval split | Token accuracy | Exact accuracy | LM loss | Steps |
| --- | --- | ---: | ---: | ---: | ---: |
| `a_only` | Task A test | 0.976 | 0.290 | 0.089 | 16.0 |
| `a_only` | Task B test | 0.974 | 0.295 | 0.095 | 16.0 |
| `b_only` | Task A test | 0.970 | 0.130 | 0.088 | 16.0 |
| `b_only` | Task B test | 0.969 | 0.115 | 0.092 | 16.0 |
| `sequential_b` | Task A test | 0.984 | 0.480 | 0.064 | 16.0 |
| `sequential_b` | Task B test | 0.986 | 0.585 | 0.054 | 16.0 |

Current interpretation:

```text
The completed Maze branches do not show forgetting.
Sequential A -> B improves both Task A and Task B compared with A-only/B-only.
```

This means the first Maze split is currently behaving like positive transfer or
curriculum learning, not like a process-forgetting setting. The split is useful
because TRM learns Maze much better than the first Sudoku gate, but the current
short-path / long-path split is probably too similar or too easy to transfer
across.

Important comparison:

```text
A-only Task A exact:       0.290
A-only Task B exact:       0.295
Sequential Task A exact:   0.480
Sequential Task B exact:   0.585
```

The A-only model already performs about equally on Task A and Task B, so this
path-length split does not yet create a clean old/new task separation.

Next steps after the joint rerun finishes:

1. Evaluate the full matrix including `joint_ab_rerun`.
2. Record whether joint training is better than sequential training.
3. If joint and sequential both improve both tasks, mark this Maze split as a
   learnability-positive but forgetting-negative gate.
4. Try a sharper Maze split if needed, for example larger path-length gap,
   different maze-generation regime, or train on long-path first then
   fine-tune on short-path to test shallow-solver interference.

## 17. Maze Solved-Depth Sentinel

### 2026-06-20: A-Teacher and Sequential Horizon Probes

Motivation:

```text
The plain Maze path-length split produced positive transfer, not forgetting.
The next proposed direction was to select examples by TRM solving depth rather
than raw path length, then test a deep-solver -> fast-shallow adaptation.
```

Implemented:

```text
scripts/eval_maze_solved_depth.py
```

The script evaluates a checkpoint at multiple TRM inference horizons and writes:

```text
solved_depth_per_example.csv
solved_depth_summary.csv
```

Metrics:

```text
exact accuracy
token accuracy
path-token F1
incorrect cell count
path length
per-example solved bucket
```

Run directories:

```text
A-only teacher:
/mnt/data/binhnt6/trm_runs/results/maze_cl_gate_pathlen_20260620_015643/solved_depth_a_teacher

Sequential checkpoint:
/mnt/data/binhnt6/trm_runs/results/maze_cl_gate_pathlen_20260620_015643/solved_depth_sequential_b
```

The first sentinel job failed because Maze labels are stored as `uint8` and the
script tried to write `-100` ignore labels before casting to `int32`. This was
fixed in the script, and the reruns completed.

### A-Only Teacher Horizon Result

| Eval split | K=4 exact | K=8 exact | K=16 exact | K=32 exact | K=16 path F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Task A | 0.285 | 0.295 | 0.290 | 0.280 | 0.903 |
| Task B | 0.275 | 0.290 | 0.295 | 0.290 | 0.901 |

Solved-depth buckets:

| Eval split | solved@4 | solved@8 | solved@16 | solved@32 | unsolved@32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Task A | 57 | 6 | 0 | 0 | 137 |
| Task B | 55 | 4 | 1 | 0 | 140 |

Reading:

```text
The A-only teacher does not have a useful late-solved population.
Most solved examples are already solved at K=4, and most unsolved examples
remain unsolved at K=32.
```

So the current A-only checkpoint is not suitable for selecting:

```text
deep examples = unsolved at K=4/8 but solved at K=16/32
```

### Sequential Checkpoint Horizon Result

| Eval split | K=4 exact | K=8 exact | K=16 exact | K=32 exact | K=16 path F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Task A | 0.495 | 0.495 | 0.480 | 0.490 | 0.934 |
| Task B | 0.590 | 0.600 | 0.585 | 0.605 | 0.945 |

Solved-depth buckets:

| Eval split | solved@4 | solved@8 | solved@16 | solved@32 | unsolved@32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Task A | 99 | 7 | 2 | 2 | 90 |
| Task B | 118 | 8 | 2 | 1 | 71 |

Reading:

```text
The stronger sequential checkpoint also does not show a meaningful late-solved
depth axis. Extra inference from K=4 to K=16/K=32 changes exact accuracy only
slightly.
```

Current conclusion:

```text
The next TRM process-forgetting run should not be launched yet.
The current Maze checkpoints do not expose the desired deep-vs-fast-shallow
structure.
```

The most likely issue is that the current training setup is not close enough to
the strong Maze regime in the TRM paper:

```text
current run:
  no augmentation
  2500 epochs per phase
  ema=False
  L_cycles=6

README/paper-style Maze:
  maze-30x30-hard-1k with 8 augmentations
  50000 epochs
  ema=True
  H_cycles=3, L_cycles=4
```

Recommended next step:

```text
Train or obtain a stronger Maze teacher first, closer to the paper recipe.
Then rerun solved-depth evaluation at K=4,8,16,32.
Only if there are enough examples solved late should we build the
Deep@16 -> FastShallow@4/8 CL gate.
```

### 2026-06-20: K=1/K=2 Early-Horizon Probe

Purpose:

```text
Check whether the depth axis is hidden below K=4. With H_cycles=3 and L_cycles=6,
K=4 may already contain enough internal computation to solve most examples that
the current model can solve.
```

Run:

```text
Slurm job: 21175
Output root:
/mnt/data/binhnt6/trm_runs/results/maze_cl_gate_pathlen_20260620_015643

A-only teacher:
solved_depth_k12_a_teacher

Sequential checkpoint:
solved_depth_k12_sequential_b
```

A-only teacher:

| Eval split | K=1 exact | K=2 exact | K=4 exact | K=8 exact | K=16 exact | K=32 exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Task A | 0.000 | 0.170 | 0.285 | 0.295 | 0.290 | 0.280 |
| Task B | 0.000 | 0.180 | 0.275 | 0.290 | 0.295 | 0.290 |

Sequential checkpoint:

| Eval split | K=1 exact | K=2 exact | K=4 exact | K=8 exact | K=16 exact | K=32 exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Task A | 0.000 | 0.400 | 0.495 | 0.495 | 0.480 | 0.490 |
| Task B | 0.000 | 0.445 | 0.590 | 0.600 | 0.585 | 0.605 |

Solved-depth buckets with K=1/K=2 included:

| Checkpoint | Eval split | solved@1 | solved@2 | solved@4 | solved@8 | solved@16 | solved@32 | unsolved@32 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A-only | Task A | 0 | 34 | 24 | 6 | 0 | 0 | 136 |
| A-only | Task B | 0 | 36 | 21 | 4 | 1 | 0 | 138 |
| Sequential | Task A | 0 | 80 | 27 | 6 | 2 | 0 | 85 |
| Sequential | Task B | 0 | 89 | 33 | 5 | 2 | 1 | 70 |

Reading:

```text
There is an early-depth axis from K=1 to K=2/K=4.
There is still almost no late-depth axis after K=4.
```

This changes the possible fast-shallow design:

```text
K_B=4 is probably not a strong fast-shallow pressure because K=4 is already
near saturation for the current checkpoint family.

If we keep a depth-based Maze direction, the only plausible pressure is K_B=1
or K_B=2, not K_B=4/K_B=8.
```

But K=1 is too hard in the current checkpoints:

```text
exact@1 = 0.000 for both A-only and sequential checkpoints.
```

So the next practical recommendation remains:

```text
Do not launch deep->fast-shallow CL yet.
First either train a stronger/paper-style Maze teacher and rerun this probe,
or move to a real same-format Maze task-conflict gate such as shortest-path ->
distance-map / next-hop-policy.
```

## 18. Maze Full Endpoint Matrix And Joint Result

### 2026-06-21: Full A/B Endpoint Evaluation

After the `joint_ab` rerun completed, a short GPU evaluation was launched to
evaluate all four Maze gate checkpoints on the Task A and Task B test splits.

Run:

```text
Slurm job: 21191
job name: maze_full_eval
resource: 1 MIG GPU
elapsed: 00:02:59
output csv:
/mnt/data/binhnt6/trm_runs/results/maze_cl_gate_pathlen_20260620_015643/learnability_gate_endpoint_matrix.csv
```

Full endpoint matrix:

| Checkpoint | Task A token acc | Task A exact | Task B token acc | Task B exact |
| --- | ---: | ---: | ---: | ---: |
| `a_only` | 0.976 | 0.290 | 0.974 | 0.295 |
| `b_only` | 0.970 | 0.130 | 0.969 | 0.115 |
| `joint_ab` | 0.964 | 0.000 | 0.963 | 0.000 |
| `sequential_b` | 0.984 | 0.480 | 0.986 | 0.585 |

### Meaning Of `joint_ab`

`joint_ab` is a control baseline, not a continual-learning run.

```text
a_only: train from scratch on Task A only
b_only: train from scratch on Task B only
joint_ab: train from scratch on Task A + Task B together
sequential_b: train Task A, then continue training on Task B
```

For this run:

```text
Task A = shorter-path Maze examples
Task B = longer-path Maze examples
joint_ab = merged Task A + Task B training set
```

The original TRM repo has the normal model and training loop, but it did not
already include this single-dataset CL gate. The Task A/B split, merged
`task_ab` dataset, four-branch gate, and endpoint matrix evaluation were added
for this project.

### Why Joint Exact Is Zero

The joint model is not random:

```text
joint Task A token accuracy: 0.964
joint Task B token accuracy: 0.963
```

However, Maze exact accuracy requires the whole 30x30 output to be correct.
With 900 cells, around 96% token accuracy can still mean many wrong cells per
maze. If every test maze has at least one wrong cell, exact accuracy is zero.

Sanity checks:

```text
task_ab metadata has the expected merged sizes.
task_ab uses the same single Maze puzzle identifier as Task A and Task B.
checkpoint puzzle_emb shape is [1, 512] for a_only, b_only, joint_ab, and sequential_b.
```

So the zero exact result is not obviously caused by a puzzle-id or checkpoint
shape bug. The most likely reading is that the short no-augmentation joint run
underfits exact Maze solving.

### Why This Is Not Comparable To The TRM Paper Maze Result

This `joint_ab` run is normal TRM training on the merged split, but it is not
the official-strength Maze recipe.

| Item | TRM README Maze recipe | Current Maze gate |
| --- | --- | --- |
| Dataset | `maze-30x30-hard-1k` with 8 augmentations | `maze-30x30-hard-1k-noaug` |
| Training | `epochs=50000` | `epochs=2500` |
| EMA | `ema=True` | `ema=False` |
| Cycles | `H_cycles=3`, `L_cycles=4` | `H_cycles=3`, `L_cycles=6` |
| Purpose | final Maze training recipe | quick CL learnability gate |

Therefore this gate should not be used to claim that TRM cannot solve Maze.
It only says that the current short no-augmentation path-length gate is not a
good scientific CL/process-forgetting setup.

### Current Conclusion

The completed Maze path-length gate shows:

```text
sequential A->B improves both Task A and Task B.
joint A+B underfits exact Maze solving in this short setup.
the split does not create forgetting.
the split does not expose a useful late recursive-depth axis.
```

Current decision:

```text
Do not use this path-length Maze split as the main TRM validation.
```

Next meaningful options:

```text
1. Train a stronger official-style Maze teacher and rerun horizon probes.
2. Move to a same-format Maze task-conflict gate:
   shortest path -> distance map / next-hop policy.
```

## 19. Maze Path -> Next-Hop Policy Quick Gate

### Purpose

The path-length Maze split did not create a useful continual-learning failure
mode. It behaved like more training on the same shortest-path task. The next
gate therefore changed Task B while keeping the same Maze input format and TRM
architecture:

```text
Task A: Maze shortest-path output.
Task B: Maze next-hop policy output.
```

This is still a TRM-native Maze reasoning setup, but Task B asks for a different
recursive target. The goal of this gate was not to prove process forgetting yet.
It was to check whether the task pair is learnable and separated enough to
justify a real CL experiment.

### Code Implementation

New implementation files:

```text
dataset/build_maze_policy_dataset.py
scripts/run_two_dataset_cl_learnability_gate.sh
scripts/run_maze_policy_cl_gate.sh
```

`dataset/build_maze_policy_dataset.py` builds a derived Maze policy dataset from
the existing prepared Maze data.

Policy labels:

| Label | Meaning |
| ---: | --- |
| 0 | ignored wall / unreachable / padding |
| 1 | goal cell |
| 2 | move up |
| 3 | move down |
| 4 | move left |
| 5 | move right |

For each Maze input, the script runs BFS from the goal over non-wall cells. Each
reachable non-goal cell receives a direction label pointing to a neighbor with
distance one step closer to the goal. This creates a global next-hop policy
target using the same `seq_len=900` and `vocab_size=6` shape as the original
Maze task.

`scripts/run_two_dataset_cl_learnability_gate.sh` is a generic two-dataset CL
gate. It trains:

```text
a_only:       train from scratch on Task A only
b_only:       train from scratch on Task B only
joint_ab:     train from scratch on Task A + Task B merged together
sequential_b: train Task A, then continue on Task B
```

It also evaluates all four checkpoints on both test splits and writes an
endpoint matrix.

`scripts/run_maze_policy_cl_gate.sh` is the Maze-specific Slurm wrapper. It now
uses the generic public job name:

```text
binh_job
```

and redirects W&B/cache/temp output to:

```text
/mnt/data/binhnt6
```

so future runs do not write Hydra/W&B/cache data under the full `/home`
filesystem.

### Run Setting

Completed quick gate:

```text
run id: maze_policy_gate_20260621_103704
allocation: 21301 / binh_job
device: 1 full H100 allocation
Task A data: /mnt/data/binhnt6/trm_data/maze-30x30-hard-1k-noaug
Task B data: /mnt/data/binhnt6/trm_data/maze-30x30-hard-1k-noaug-next-hop-policy
epochs A/B: 250
epochs joint: 250
batch size: 64
eval horizon: 16 steps
EMA: false
```

Output matrix:

```text
/mnt/data/binhnt6/trm_runs/results/maze_policy_gate_20260621_103704/learnability_gate_endpoint_matrix.csv
```

Checkpoint root:

```text
/mnt/data/binhnt6/trm_runs/checkpoints/maze_policy_gate_20260621_103704
```

### Result

| Checkpoint | Task A token acc | Task A exact | Task B token acc | Task B exact |
| --- | ---: | ---: | ---: | ---: |
| `a_only` | 0.961 | 0.026 | 0.288 | 0.000 |
| `b_only` | 0.175 | 0.000 | 0.914 | 0.018 |
| `joint_ab` | 0.769 | 0.000 | 0.499 | 0.000 |
| `sequential_b` | 0.540 | 0.000 | 0.972 | 0.046 |

### Interpretation

This gate is much more separated than the previous path-length Maze split:

```text
a_only learns Task A token structure but does not solve Task B.
b_only learns Task B token structure but does not solve Task A.
sequential_b learns Task B best but damages Task A strongly.
```

The sequential result is the first useful TRM-Maze interference signal in this
branch:

```text
Task A token acc drops from 0.961 to 0.540 after Task B training.
Task B token acc rises to 0.972 after Task B training.
```

However, exact accuracy is still too low for a final process-forgetting claim:

```text
best Task A exact: 0.026
best Task B exact: 0.046
joint exact: 0.000 on both tasks
```

So this is a useful engineering and interference gate, not yet a scientific CL
result.

### Current Decision

The path -> next-hop-policy pair is worth continuing because it creates real
task separation and sequential interference.

The next real run should not add FR-flow yet. It should first strengthen this
gate:

```text
1. Run a longer full-GPU learnability/interference gate.
2. Add A->B endpoint replay after sequential training.
3. Check whether replay can preserve old Task A endpoint/token behavior while
   Task B remains learned.
4. Only after that add TRM process diagnostics and flow preservation.
```

Pass condition for the next run:

```text
A-only learns Task A clearly.
B-only learns Task B clearly.
Sequential A->B learns Task B and hurts old Task A.
Endpoint replay recovers old Task A enough to create a process-forgetting test.
```

Fail condition:

```text
If exact accuracy remains near zero even after longer training, this task pair
is useful only as a token-level interference diagnostic, not as the main TRM
validation.
```

## 20. Maze Path -> Policy Task-IL Correction

### Motivation

The first path -> next-hop-policy quick gate showed real task separation, but it
had a task-interface confound:

```text
Task A path dataset:   puzzle_identifier = 0
Task B policy dataset: puzzle_identifier = 0
merged task_ab:        puzzle_identifier = 0 only
```

That means the joint dataset asked the model to map the same Maze input and the
same task condition to two different outputs. Therefore `joint_ab` was not a
clean upper bound, and sequential forgetting could be partly ordinary
label/head interference rather than recursive-process forgetting.

### Implementation Update

Added explicit Task-IL dataset handling:

```text
Task A path id:   0
Task B policy id: 1
num_puzzle_identifiers: 2
```

Updated files:

```text
dataset/build_maze_policy_dataset.py
scripts/run_two_dataset_cl_learnability_gate.sh
scripts/run_maze_policy_cl_gate.sh
scripts/eval_cl_learnability_gate.py
evaluators/maze.py
```

`dataset/build_maze_policy_dataset.py` now supports:

```text
--task-id
--num-puzzle-identifiers
```

so a policy dataset can be generated directly as Task B with identifier `1`.

`scripts/run_two_dataset_cl_learnability_gate.sh` now creates normalized copies
inside the run data directory:

```text
CL_DATA_DIR/task_a
CL_DATA_DIR/task_b
CL_DATA_DIR/task_ab
```

These normalized datasets are what training and evaluation use. This avoids the
checkpoint-shape problem where A-only checkpoints had one puzzle embedding but
B-only/sequential checkpoints used two puzzle embeddings.

The merged `task_ab` dataset now contains both task IDs:

```text
Task A examples -> puzzle_identifier 0
Task B examples -> puzzle_identifier 1
```

### Functional Maze Metrics

Added `evaluators/maze.py` with two evaluator classes:

```text
MazePathFunctional
MazePolicyFunctional
```

Path metrics include:

```text
maze_path_valid_rate
maze_path_connected_rate
maze_path_wall_cross_rate
maze_path_precision
maze_path_recall
maze_path_f1
maze_path_length_ratio
```

Policy metrics include:

```text
maze_policy_reachable_acc
maze_policy_rollout_success
maze_policy_cycle_rate
maze_policy_wall_rate
maze_policy_offgrid_rate
maze_policy_invalid_action_rate
maze_policy_success_length_ratio
```

These are needed because exact grid accuracy is too harsh for Maze and raw
token accuracy does not tell whether the predicted path or policy is
functionally usable.

`scripts/eval_cl_learnability_gate.py` now supports:

```text
--maze-functional
```

The Maze wrapper enables this by default.

### What This Fixes

Resolved:

```text
same-input / same-task-id label ambiguity in joint_ab
inconsistent puzzle-embedding shape between A-only and B/sequential checkpoints
lack of path/policy functional metrics
public Slurm wrapper naming now uses binh_job
W&B/cache/temp output is redirected to /mnt/data/binhnt6
```

Still not implemented:

```text
task-specific output heads
freezing the old Task A head during Task B
endpoint replay
TRM hidden/process-flow diagnostics
```

The current decision is to rerun the quick gate after this correction before
launching a long 24h training job. If the Task-IL corrected quick gate shows
that A-only, B-only, and joint are learnable and sequential still creates
interference, the next real run should add endpoint replay.

## 21. Corrected Maze Path -> Policy Task-IL Quick Gate

### Run

The corrected Task-IL quick gate was run inside the 24h full-H100 Slurm
allocation:

```text
Slurm job: 21301
public job name: binh_job
run id: binh_job_taskil_debug_20260621_122254
```

All run outputs were written under `/mnt/data/binhnt6`:

```text
data:
  /mnt/data/binhnt6/trm_data/cl_two_task/binh_job_taskil_debug_20260621_122254

checkpoints:
  /mnt/data/binhnt6/trm_runs/checkpoints/binh_job_taskil_debug_20260621_122254

result CSV:
  /mnt/data/binhnt6/trm_runs/results/binh_job_taskil_debug_20260621_122254/learnability_gate_endpoint_matrix.csv
```

The first training/evaluation pass completed successfully. During review, the
functional evaluator metrics were found to run but not appear in the CSV because
`pretrain.evaluate` returns evaluator metrics at the top level, while the CL
evaluation script was only returning the `"all"` split metrics. This was fixed
in:

```text
scripts/eval_cl_learnability_gate.py
```

Then an evaluation-only pass was rerun on the same saved checkpoints. No
retraining was needed.

### Endpoint And Functional Result

| Checkpoint | Eval | Token acc | Exact | Loss | Path valid | Path F1 | Policy reachable acc | Policy rollout success |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `a_only` | Task A path | 0.962 | 0.002 | 0.110 | 0.066 | 0.854 | n/a | n/a |
| `a_only` | Task B policy | 0.283 | 0.000 | 4.531 | n/a | n/a | 0.283 | 0.000 |
| `b_only` | Task A path | 0.186 | 0.000 | 5.170 | 0.000 | 0.179 | n/a | n/a |
| `b_only` | Task B policy | 0.925 | 0.027 | 0.294 | n/a | n/a | 0.925 | 0.616 |
| `joint_ab` | Task A path | 0.964 | 0.016 | 0.106 | 0.046 | 0.854 | n/a | n/a |
| `joint_ab` | Task B policy | 0.956 | 0.042 | 0.192 | n/a | n/a | 0.956 | 0.712 |
| `sequential_b` | Task A path | 0.542 | 0.000 | 2.836 | 0.000 | 0.229 | n/a | n/a |
| `sequential_b` | Task B policy | 0.935 | 0.035 | 0.297 | n/a | n/a | 0.935 | 0.622 |

Additional functional details:

```text
a_only Task A path wall-cross rate:       0.000
joint_ab Task A path wall-cross rate:     0.000
sequential_b Task A path wall-cross rate: 0.004

b_only Task B policy cycle rate:          0.145
joint_ab Task B policy cycle rate:        0.181
sequential_b Task B policy cycle rate:    0.204
```

### Interpretation

This corrected gate is meaningful and should replace the previous confounded
path-policy quick gate.

The Task-IL correction fixed the main joint-training ambiguity:

```text
previous joint_ab exact was zero on both tasks;
corrected joint_ab now learns both tasks:
  Task A token acc = 0.964, exact = 0.016
  Task B token acc = 0.956, exact = 0.042
  Task B policy rollout success = 0.712
```

The task separation is real:

```text
a_only does not solve Task B policy:
  policy rollout success = 0.000

b_only does not solve Task A path:
  path valid rate = 0.000
```

Sequential A -> B creates strong interference:

```text
Task A path token acc:
  a_only       0.962
  sequential_b 0.542

Task A path F1:
  a_only       0.854
  sequential_b 0.229

Task A valid path rate:
  a_only       0.066
  sequential_b 0.000
```

Task B is learned after sequential training:

```text
sequential_b Task B token acc = 0.935
sequential_b Task B rollout success = 0.622
```

Current scientific status:

```text
This is a useful TRM-native Task-IL/interference gate.
It is not yet process forgetting.
```

It is not yet process forgetting because old Task A endpoint and functional
behavior are not preserved after Task B. This is ordinary sequential
interference. The next experiment should add old Task A endpoint replay so that
the old endpoint/functional behavior is partially preserved before measuring
TRM hidden/process drift.

### Next Run

The next real run should be:

```text
Maze Path -> Policy Task-IL Replay Gate
```

Required additions:

```text
1. Train Task A path as before.
2. Train Task B policy with old Task A endpoint replay.
3. Keep explicit task IDs:
   Task A path id = 0
   Task B policy id = 1
4. Evaluate:
   Task A path exact, valid path rate, path F1
   Task B policy reachable accuracy and rollout success
5. Only after endpoint replay preserves old Task A enough, add TRM process
   diagnostics and free-running flow preservation.
```

Recommended first replay setting:

```text
old replay memory: 128 path examples
same quick-gate training budget first
then a longer 24h run if the replay gate works
```

## 22. Maze Path -> Policy Replay Gate

### Implementation

Implemented the first endpoint-replay gate for the corrected TRM-native
Path -> Policy Task-IL setup.

Code changes:

```text
dataset/build_maze_policy_dataset.py
  - added explicit task-id support for policy datasets
  - policy task now uses puzzle/task id 1
  - metadata can declare num_puzzle_identifiers = 2

scripts/run_two_dataset_cl_learnability_gate.sh
  - normalizes Task A to task id 0 and Task B to task id 1
  - builds a joint Task A+B dataset with both task identifiers
  - optionally builds task_b_replay from all Task B rows plus sampled old
    Task A memory rows
  - trains sequential_replay_m{memory} from the Task A checkpoint

scripts/run_maze_policy_cl_gate.sh
  - default replay setting: REPLAY_MEMORY=128, REPLAY_LAMBDA=1.0
  - public Slurm job name: binh_job
  - outputs under /mnt/data/binhnt6

scripts/eval_cl_learnability_gate.py
  - added optional replay-checkpoint evaluation
  - keeps Maze path/policy functional evaluators enabled

evaluators/maze.py
  - added MazePathFunctional and MazePolicyFunctional
```

Replay construction:

```text
Task A path memory: 128 examples
Task B policy rows: all Task B training examples
Replay branch train set: all Task B rows + replayed Task A rows
Replay lambda: 1.0
Replay implementation: data-level oversampling, not a separate weighted loss
```

Important caveat:

```text
The current run still uses the shared TRM output head. It is a useful endpoint
replay gate, but it is not yet the cleanest frozen-old-head Task-IL setup.
```

### Run

Run id:

```text
binh_job_replay_m128_20260621_132805
```

Paths:

```text
data root:
  /mnt/data/binhnt6/trm_data/cl_two_task/binh_job_replay_m128_20260621_132805

checkpoint root:
  /mnt/data/binhnt6/trm_runs/checkpoints/binh_job_replay_m128_20260621_132805

result csv:
  /mnt/data/binhnt6/trm_runs/results/binh_job_replay_m128_20260621_132805/learnability_gate_endpoint_matrix.csv
```

The run used the active 24h full-H100 allocation:

```text
Slurm job: 21301
Job name: binh_job
```

The run completed successfully. The log contains harmless NFS temporary
directory cleanup warnings, but training and evaluation completed and wrote the
endpoint matrix.

### Result

| Checkpoint | Eval task | Token acc | Exact | Loss | Path valid | Path F1 | Policy reachable | Policy rollout |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `a_only` | Task A path | 0.965 | 0.029 | 0.108 | 0.188 | 0.859 | n/a | n/a |
| `a_only` | Task B policy | 0.287 | 0.000 | 4.527 | n/a | n/a | 0.287 | 0.000 |
| `b_only` | Task A path | 0.180 | 0.000 | 4.829 | 0.000 | 0.181 | n/a | n/a |
| `b_only` | Task B policy | 0.922 | 0.046 | 0.308 | n/a | n/a | 0.922 | 0.582 |
| `joint_ab` | Task A path | 0.965 | 0.074 | 0.100 | 0.247 | 0.859 | n/a | n/a |
| `joint_ab` | Task B policy | 0.900 | 0.025 | 0.297 | n/a | n/a | 0.900 | 0.447 |
| `sequential_b` | Task A path | 0.369 | 0.000 | 3.355 | 0.000 | 0.175 | n/a | n/a |
| `sequential_b` | Task B policy | 0.961 | 0.036 | 0.162 | n/a | n/a | 0.961 | 0.726 |
| `sequential_replay_m128` | Task A path | 0.961 | 0.029 | 0.126 | 0.285 | 0.847 | n/a | n/a |
| `sequential_replay_m128` | Task B policy | 0.931 | 0.006 | 0.278 | n/a | n/a | 0.931 | 0.433 |

### Interpretation

The replay gate passes the immediate endpoint-preservation precondition better
than the previous sequential run.

Old Task A path behavior:

```text
a_only Task A path F1:                 0.859
sequential_b Task A path F1:           0.175
sequential_replay_m128 Task A path F1: 0.847
```

Old Task A valid-path behavior:

```text
a_only Task A valid path:                 0.188
sequential_b Task A valid path:           0.000
sequential_replay_m128 Task A valid path: 0.285
```

So endpoint replay restores old Task A path structure and even improves valid
path rate relative to the A-only checkpoint in this draw.

Task B remains partially learned, but replay creates a stability-plasticity
tradeoff:

```text
sequential_b Task B rollout success:           0.726
sequential_replay_m128 Task B rollout success: 0.433
```

Reading:

```text
The run establishes a usable TRM endpoint-replay baseline:
old Task A path is preserved while Task B policy remains above zero.

It is still not process forgetting yet, because we have not measured hidden
TRM trajectory drift, extra-rollout behavior, or perturbed-carry recovery under
matched old endpoints.
```

### Next Step

The next TRM experiment should add process diagnostics on this replay setting,
not FR-flow yet.

Immediate diagnostic targets:

```text
1. Evaluate a_only and sequential_replay_m128 on Task A at K = 4, 8, 16, 32.
2. Log z_H, z_L, logits, q_head, and residual curves if possible.
3. Measure whether Task A path F1 / valid path at K=16 is preserved while
   longer rollout, overthinking, or process drift worsens.
4. If process damage appears under endpoint replay, then add teacher-output
   multihorizon and free-running TRM-flow preservation.
```

## 23. Maze Path -> Policy Horizon and Process Diagnostic

### Goal

After the replay gate passed the endpoint-preservation precondition, the next
question was:

```text
Does endpoint replay preserve visible Task A path behavior while the old TRM
recursive process or longer-rollout behavior is still damaged?
```

This diagnostic used the existing checkpoints only. No training was run.

### Implementation

Added evaluation-only scripts:

```text
scripts/eval_maze_policy_horizon_diag.py
  - evaluates saved Path -> Policy checkpoints at K = 1, 2, 4, 8, 16, 32, 64
  - computes Task A path functional metrics and Task B policy functional metrics
  - writes per-example rows, horizon summaries, output drift vs a_only, and
    matched-valid Task A subset summaries

scripts/eval_maze_trace_drift.py
  - compares Task A hidden/logit traces against the a_only reference
  - reports z_H, z_L, delta-z, logits, q_halt, and q_continue drift
  - reports both all Task A examples and the matched-valid K=16 subset
```

Output directory:

```text
/mnt/data/binhnt6/trm_runs/results/binh_job_replay_m128_20260621_132805/horizon_diag_v1
```

Files:

```text
horizon_summary.csv
horizon_per_example.csv
output_drift_vs_reference.csv
matched_valid_task_a_summary.csv
matched_valid_task_a_indices.txt
trace_drift_task_a.csv
```

Matched-valid subset:

```text
Task A examples where both a_only and sequential_replay_m128 are valid at K=16:
79 examples
```

### Task A Horizon Result

Task A path metrics:

| Checkpoint | K=16 valid | K=64 valid | K=16 F1 | K=64 F1 | K=16 token | K=64 token |
|---|---:|---:|---:|---:|---:|---:|
| `a_only` | 0.188 | 0.191 | 0.858 | 0.858 | 0.964 | 0.964 |
| `joint_ab` | 0.247 | 0.249 | 0.856 | 0.857 | 0.965 | 0.966 |
| `sequential_b` | 0.000 | 0.000 | 0.175 | 0.175 | 0.369 | 0.369 |
| `sequential_replay_m128` | 0.285 | 0.278 | 0.847 | 0.847 | 0.961 | 0.961 |

Reading:

```text
Replay preserves old Task A path behavior at K=16 and remains mostly stable
through K=64. There is no strong overthinking collapse in the all-example
Task A metrics.
```

### Task B Horizon Result

Task B policy rollout:

| Checkpoint | K=16 rollout | K=64 rollout | K=16 reachable | K=64 reachable |
|---|---:|---:|---:|---:|
| `a_only` | 0.000 | 0.000 | 0.287 | 0.287 |
| `joint_ab` | 0.447 | 0.438 | 0.900 | 0.899 |
| `sequential_b` | 0.726 | 0.726 | 0.961 | 0.962 |
| `sequential_replay_m128` | 0.433 | 0.442 | 0.931 | 0.931 |

Reading:

```text
Replay keeps Task B partially learned and does not show a long-rollout collapse.
The tradeoff from Section 22 remains: replay protects old Task A but reduces
Task B rollout success relative to plain sequential_b.
```

### Matched-Valid Subset

For the 79 examples where both `a_only` and `sequential_replay_m128` are valid
at K=16:

| Checkpoint | K=16 valid | K=64 valid | K=16 F1 | K=64 F1 | K=16 token | K=64 token |
|---|---:|---:|---:|---:|---:|---:|
| `a_only` | 1.000 | 0.962 | 0.949 | 0.950 | 0.987 | 0.987 |
| `joint_ab` | 0.405 | 0.418 | 0.907 | 0.906 | 0.978 | 0.978 |
| `sequential_b` | 0.000 | 0.000 | 0.159 | 0.158 | 0.358 | 0.357 |
| `sequential_replay_m128` | 1.000 | 0.924 | 0.937 | 0.939 | 0.984 | 0.984 |

Reading:

```text
On matched-valid examples, replay is endpoint-matched at K=16 but loses a small
amount of validity under longer rollout:
  a_only: 1.000 -> 0.962
  replay: 1.000 -> 0.924

This is a weak overthinking/stability signal, not a strong functional collapse.
```

### Output Drift vs a_only

Task A output drift at K=16:

| Checkpoint | Token disagreement | Path-mask disagreement | q_halt abs diff |
|---|---:|---:|---:|
| `joint_ab` | 0.042 | 0.042 | 2.929 |
| `sequential_b` | 0.631 | 0.270 | 3.060 |
| `sequential_replay_m128` | 0.046 | 0.046 | 2.121 |

Reading:

```text
Replay has low output/path-mask drift, close to joint training and far below
plain sequential_b. The q_halt drift is nontrivial despite endpoint recovery.
```

### Hidden Trace Drift

Task A hidden/logit drift against `a_only` at K=16, all examples:

| Checkpoint | z_H RMSE | z_L RMSE | delta-z_H RMSE | delta-z_L RMSE | logits RMSE | q_halt abs diff |
|---|---:|---:|---:|---:|---:|---:|
| `joint_ab` | 0.738 | 1.363 | 0.053 | 0.085 | 21.262 | 2.929 |
| `sequential_b` | 1.285 | 1.067 | 0.141 | 0.188 | 41.409 | 3.060 |
| `sequential_replay_m128` | 0.482 | 0.969 | 0.099 | 0.216 | 21.053 | 2.121 |

Task A hidden/logit drift against `a_only` at K=16, matched-valid subset:

| Checkpoint | z_H RMSE | z_L RMSE | delta-z_H RMSE | delta-z_L RMSE | logits RMSE | q_halt abs diff |
|---|---:|---:|---:|---:|---:|---:|
| `joint_ab` | 0.743 | 1.364 | 0.039 | 0.066 | 20.792 | 2.794 |
| `sequential_b` | 1.283 | 1.069 | 0.131 | 0.178 | 41.336 | 2.181 |
| `sequential_replay_m128` | 0.436 | 0.948 | 0.070 | 0.173 | 20.436 | 2.167 |

Reading:

```text
Replay has much lower drift than plain sequential_b and is close to joint on
output/logit drift. It still uses a different hidden and q-head trajectory than
a_only even when visible Task A endpoints are matched.
```

### Scientific Interpretation

This diagnostic gives a partial but not final process-forgetting signal.

Supported:

```text
1. Endpoint replay preserves old Task A path behavior.
2. Plain sequential_b causes severe output and hidden-process drift.
3. replay_m128 strongly reduces output drift and preserves Task A across
   horizons.
4. replay_m128 still has nontrivial hidden/q-head drift relative to a_only on
   matched-valid examples.
```

Not yet supported:

```text
The current replay_m128 setting does not show strong endpoint-preserved
functional collapse under longer rollout. The K=16 -> K=64 drop is small.
```

Best current reading:

```text
This is a real TRM endpoint-preserved process-drift diagnostic, but not yet the
strong paper phenomenon. The setting is currently too protected by replay_m128
to expose a large hard-stress failure.
```

### Next Step

Run a replay boundary sweep rather than adding FR-flow immediately.

Recommended next sweep:

```text
REPLAY_MEMORY = 16, 32, 64, 128
same Path -> Policy Task-IL setup
same K = 1, 2, 4, 8, 16, 32, 64 diagnostics
```

Target boundary:

```text
old Task A path F1 / valid path mostly preserved at K=16
Task B policy still learned
but old Task A longer-rollout stability, q-head behavior, or hidden trace drift
is worse than in replay_m128
```

If all replay settings are stable, the next pressure knob should be replay
ratio / explicit weighted old loss rather than FR-flow.

## 24. Maze Path -> Policy Replay Boundary and Perturbed-Carry Stress

### Goal

The replay-m128 diagnostic in Section 23 preserved old Task A too well. The
next decisive question was:

```text
Is there a replay-strength boundary where old Task A path behavior is preserved
at the normal horizon, Task B policy is still learned, but the old TRM process
or basin stability fails under stronger stress?
```

This run keeps the corrected Task-IL setup:

```text
Task A: 30x30 maze shortest-path output
Task B: 30x30 maze next-hop policy output
task ids: path = 0, policy = 1
replay memory: 128 Task A examples
```

### Implementation

Added a perturbed-carry evaluator:

```text
scripts/eval_maze_perturbed_carry.py
```

It evaluates Task A path recovery after perturbing the recursive carry:

```text
perturbed states: z_H and z_L
perturb steps: 4 and 8
sigma: 0, 0.01, 0.03, 0.10 times carry RMS
recovery horizons: 16, 32, 64
subsets: all examples and top-25% path-length hard examples
```

Added a boundary training wrapper:

```text
scripts/run_maze_policy_replay_boundary.sh
```

It reuses the existing `a_only` checkpoint from Section 22 and trains only the
missing Task-B replay branches.

Also fixed the data-level replay-lambda construction in:

```text
scripts/run_two_dataset_cl_learnability_gate.sh
```

Previous behavior:

```text
replay_rows = max(memory_size, round(lambda * task_b_rows))
```

This made small lambdas such as `0.03` and `0.1` still use at least 128 replay
rows, so they were not truly weak replay settings.

Updated behavior:

```text
replay_rows = max(1, round(lambda * task_b_rows))
```

This makes the current boundary run an actual replay-frequency sweep. It is
still data-level oversampling rather than a clean explicit weighted loss.

### Run

The run used the active full-H100 allocation:

```text
Slurm job id: 21301
Slurm name: binh_job
time limit: 24:00:00
node: worker-1
```

Boundary run id:

```text
binh_job_boundary_m128_20260621_153314
```

Training branches:

```text
lambda = 0.03
lambda = 0.1
lambda = 0.3
```

Existing branches reused for comparison:

```text
lambda = 0      -> sequential_b from Section 22
lambda = 1.0    -> sequential_replay_m128 from Section 22
a_only          -> Task A reference from Section 22
```

Output directories:

```text
/mnt/data/binhnt6/trm_runs/checkpoints/binh_job_boundary_m128_20260621_153314
/mnt/data/binhnt6/trm_runs/results/binh_job_boundary_m128_20260621_153314
```

Diagnostics written:

```text
horizon_diag_boundary/horizon_summary.csv
horizon_diag_boundary/output_drift_vs_reference.csv
horizon_diag_boundary/trace_drift_boundary_task_a.csv
perturbed_carry_v1/perturbed_carry_summary.csv
```

### Endpoint Boundary Result

Task A path and Task B policy at the normal horizon K=16:

| Method | Task A F1 | Task A valid path | Task B policy rollout | Endpoint gate |
|---|---:|---:|---:|---|
| `a_only` | 0.858 | 0.188 | 0.000 | no Task B |
| `lambda=0` | 0.175 | 0.000 | 0.726 | old A forgotten |
| `lambda=0.03` | 0.718 | 0.000 | 0.664 | old A validity too low |
| `lambda=0.1` | 0.792 | 0.013 | 0.633 | old A validity too low |
| `lambda=0.3` | 0.872 | 0.315 | 0.748 | pass |
| `lambda=1.0` | 0.847 | 0.285 | 0.433 | Task B too weak |

Pre-registered endpoint gate:

```text
Task A F1 >= 0.82
Task A valid path >= 0.15
Task B policy rollout >= 0.55
```

Only `lambda=0.3` passes all three conditions. This is the useful boundary
setting:

```text
old Task A path behavior is preserved,
Task B policy is learned,
and the setting is less over-protected than lambda=1.0.
```

### Longer-Rollout Result

For `lambda=0.3`:

| Horizon | Task A F1 | Task A valid path | Task B policy rollout |
|---:|---:|---:|---:|
| 16 | 0.872 | 0.315 | 0.748 |
| 32 | 0.871 | 0.308 | 0.753 |
| 64 | 0.871 | 0.311 | 0.750 |

Reading:

```text
The useful boundary setting does not overthink or collapse under K=64.
Old Task A and new Task B remain stable across longer rollout.
```

### Perturbed-Carry Stress

Perturbing at step 8 with sigma `0.10`:

| Method | Horizon | Task A F1 | Task A valid path | Token acc |
|---|---:|---:|---:|---:|
| `a_only` | 16 | 0.859 | 0.186 | 0.965 |
| `a_only` | 64 | 0.859 | 0.195 | 0.965 |
| `lambda=0.3` | 16 | 0.872 | 0.306 | 0.966 |
| `lambda=0.3` | 64 | 0.871 | 0.310 | 0.965 |
| `lambda=1.0` | 16 | 0.847 | 0.285 | 0.961 |
| `lambda=1.0` | 64 | 0.847 | 0.287 | 0.961 |

Hard top-25% path-length subset, perturb step 8, sigma `0.10`:

| Method | Horizon | Task A F1 | Task A valid path | Token acc |
|---|---:|---:|---:|---:|
| `a_only` | 16 | 0.871 | 0.133 | 0.967 |
| `a_only` | 64 | 0.871 | 0.133 | 0.967 |
| `lambda=0.3` | 16 | 0.883 | 0.236 | 0.968 |
| `lambda=0.3` | 64 | 0.882 | 0.236 | 0.968 |
| `lambda=1.0` | 16 | 0.863 | 0.246 | 0.964 |
| `lambda=1.0` | 64 | 0.862 | 0.246 | 0.964 |

Reading:

```text
Perturbed-carry recovery does not expose endpoint-preserved old-task damage.
lambda=0.3 is at least as stable as a_only on the measured Task A path metrics.
```

### Trace Drift

Task A hidden/logit trace drift versus `a_only`, all examples at K=16:

| Method | z_H RMSE | z_L RMSE | delta-z_H RMSE | delta-z_L RMSE | logits RMSE | q_halt abs diff |
|---|---:|---:|---:|---:|---:|---:|
| `lambda=0` | 1.285 | 1.067 | 0.141 | 0.188 | 41.409 | 3.060 |
| `lambda=0.03` | 0.713 | 1.147 | 0.037 | 0.066 | 16.774 | 1.895 |
| `lambda=0.1` | 0.560 | 1.025 | 0.074 | 0.130 | 15.077 | 1.196 |
| `lambda=0.3` | 0.500 | 0.921 | 0.049 | 0.080 | 15.060 | 1.527 |
| `lambda=1.0` | 0.482 | 0.969 | 0.099 | 0.216 | 21.053 | 2.121 |

Reading:

```text
The endpoint-preserving lambda=0.3 checkpoint still uses a different hidden,
logit, and q-head trajectory than a_only. But this drift does not currently
produce a functional stress failure under K64 or perturbed-carry recovery.
```

### Scientific Interpretation

This run is a useful negative result for the current TRM Path -> Policy branch.

Supported:

```text
1. The replay boundary exists.
2. lambda=0.3 preserves old Task A path behavior and learns Task B policy.
3. hidden/process drift is measurable even when old Task A endpoints are
   preserved.
```

Not supported:

```text
The current TRM Path -> Policy boundary does not show endpoint-preserved
functional process forgetting. Longer rollout and perturbed-carry stress are
stable at the only endpoint-passing boundary setting.
```

Best current reading:

```text
TRM Path -> Policy is now a valid Task-IL interference and replay-recovery
benchmark, but it is not yet a process-forgetting benchmark. Endpoint replay
appears to genuinely preserve old Task A behavior under the tested stresses.
```

### Next Decision

Do not add FR-flow to this exact setting yet. The current result does not
provide a failure for FR-flow to rescue.

Recommended next options:

```text
1. Close this TRM branch as an appendix/negative-control result for now.
2. If continuing TRM, increase stress in one controlled way:
   - stronger Task B pressure after lambda=0.3,
   - smaller replay memory with explicit weighted replay loss,
   - larger/harder Maze evaluation,
   - or task-specific frozen heads to remove shared-head confounds.
3. Keep the main paper evidence on the BF -> Dijkstra result, where
   endpoint-matched hard old-task stress failure is already clear.
```

## 25. Notes

- Keep entries short and factual.
- Include commands used for verification when possible.
- Mention any skipped tests and the reason they were skipped.
- Before launching GPU jobs, check this note: real training / experiment jobs
  should request at least `24:00:00`. Use shorter allocations only for explicit
  smoke tests or quick evaluations, and state that reason before launching.
