#!/usr/bin/env bash
set -euo pipefail

# CPU-side helper: wait for the A-only checkpoint from a Maze gate, cancel the
# original serial gate job, then submit the sequential A->B branch as a GPU job.

RUN_ID="${RUN_ID:?Set RUN_ID, e.g. maze_cl_gate_pathlen_YYYYMMDD_HHMMSS.}"
A_JOB_ID="${A_JOB_ID:-}"

BASE="${BASE:-/mnt/data/binhnt6/trm_runs}"
SPLIT_ROOT="${SPLIT_ROOT:-/mnt/data/binhnt6/trm_data/cl_splits/${RUN_ID}}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${BASE}/checkpoints/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${BASE}/logs}"
WANDB_ROOT="${WANDB_ROOT:-${BASE}/wandb}"

EPOCHS_B="${EPOCHS_B:-2500}"
EVAL_INTERVAL="${EVAL_INTERVAL:-2500}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"

mkdir -p "${LOG_ROOT}"

echo "Waiting for A-only checkpoint under ${CHECKPOINT_ROOT}/a_only"
while true; do
  A_CKPT="$(
    find "${CHECKPOINT_ROOT}/a_only" -maxdepth 1 -type f -name 'step_*' ! -name '*all_preds*' 2>/dev/null \
      | sort -V \
      | tail -1
  )"

  if [[ -n "${A_CKPT}" ]]; then
    break
  fi

  sleep 60
done

echo "Found A-only checkpoint: ${A_CKPT}"

if [[ -n "${A_JOB_ID}" ]]; then
  echo "Canceling original serial job ${A_JOB_ID}"
  scancel "${A_JOB_ID}" || true
fi

SEQ_JOB_ID="$(
  sbatch --parsable \
    --partition=mig \
    --gres=gpu:1 \
    --ntasks=1 \
    --cpus-per-task=4 \
    --mem=40G \
    --time=12:00:00 \
    --job-name=maze_seq_parallel \
    --output="${LOG_ROOT}/maze_seq_parallel_%j.out" \
    --error="${LOG_ROOT}/maze_seq_parallel_%j.err" \
    --wrap="cd /home/binhnt6/TinyRecursiveModels && export DISABLE_COMPILE=1 WANDB_MODE=offline WANDB_DIR=${WANDB_ROOT} && python pretrain.py arch=trm data_paths='[${SPLIT_ROOT}/task_b]' evaluators='[]' epochs=${EPOCHS_B} eval_interval=${EVAL_INTERVAL} global_batch_size=${GLOBAL_BATCH_SIZE} lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 +run_name=${RUN_ID}_sequential_b_parallel +checkpoint_path=${CHECKPOINT_ROOT}/sequential_b_parallel +load_checkpoint=${A_CKPT} checkpoint_every_eval=True ema=False arch.mlp_t=False arch.L_layers=2 lr_warmup_steps=0 lr_min_ratio=1.0"
)"

echo "Submitted sequential branch job: ${SEQ_JOB_ID}"
