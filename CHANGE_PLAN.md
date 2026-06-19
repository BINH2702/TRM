# Change Plan: Process Forgetting in TRM

This file tracks the plan for modifying this repo to study process forgetting in
iterative reasoners such as TRM, and records what has changed over time.

## Current Goal

Implement experiments and diagnostics for **process forgetting**: a failure mode
where an iterative reasoner keeps old endpoint accuracy but loses the old
free-running solution process that moves latent states toward correct answers.

The target method is **Free-Running Reasoning Flow Consolidation (FR-RFC)**.
During continual learning, FR-RFC should preserve the old model's reasoning-flow
distribution under the current model's own rollout, not only match final
answers or teacher-forced hidden states.

## Research Idea Summary

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

## Main Method

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

## Diagnostics To Add

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

## Plan

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

## Key Risks

- Latent-coordinate matching is not fully representation invariant.
- Free-running teacher/student rollouts add compute cost.
- Full training runs may be expensive on this repo.
- Process metrics must be implemented without breaking normal `pretrain.py`.

## Implementation Notes

- Prefer optional flags and return keys instead of changing default model output.
- Keep trace tensors detached unless a loss explicitly needs gradients.
- Start with low-dimensional projections for state and velocity traces.
- Use cached teacher traces where possible to reduce training cost.
- Keep teacher-forced velocity matching as a baseline, not the main method.

## Change Log

| Date | Files Changed | Summary | Verification |
| --- | --- | --- | --- |
| 2026-06-19 | `CHANGE_PLAN.md` | Added this plan and change tracking file. | Not run; documentation-only change. |
| 2026-06-19 | `CHANGE_PLAN.md` | Summarized the process-forgetting proposal and converted the file into an implementation plan for FR-RFC. | Not run; documentation-only change. |

## Notes

- Keep entries short and factual.
- Include commands used for verification when possible.
- Mention any skipped tests and the reason they were skipped.
