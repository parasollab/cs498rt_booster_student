#!/usr/bin/env bash
# Course training job body. Submit it through ./scripts/train.sh, which adds
# --account / --partition / --gpus-per-node / --mem / --time / --output from
# the locked settings. Submitting this file with plain `sbatch` will fail on
# Delta (no account) — that is intentional.
#
#SBATCH --job-name=course-train
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source scripts/cluster.env

TASK_ID="${1:?Usage: ./scripts/train.sh <Task-Id> [train args...]}"
shift

RUN_ROOT="${COURSE_WORK_DIR}/${USER}/runs"
mkdir -p "${RUN_ROOT}"

echo "== $(date) job ${SLURM_JOB_ID} on $(hostname) (account ${SLURM_JOB_ACCOUNT:-?}, partition ${SLURM_JOB_PARTITION:-?}) =="
echo "== task: ${TASK_ID}  extra args: $*"
nvidia-smi || true

# Keep all bulky artifacts off $HOME (W&B files go next to the runs).
export WANDB_DIR="${RUN_ROOT}"
# Log to the course W&B team + project. rsl_rl reads WANDB_USERNAME as the
# entity; the project name is fixed in the PPO config (hw0-booster).
export WANDB_USERNAME="${COURSE_WANDB_ENTITY}"
export WANDB_RUN_GROUP="${USER}"          # groups your runs on wandb.ai by NetID
# Jobs are non-interactive, so a missing login would abort the run: fall back
# to offline logging (upload later with `uv run wandb sync <run dir>`).
if [[ -z "${WANDB_API_KEY:-}" ]] && ! grep -qs "api.wandb.ai" "${HOME}/.netrc"; then
  export WANDB_MODE=offline
  echo "== W&B: not logged in -> WANDB_MODE=offline (run 'uv run wandb login' once on a login node)"
fi

# --log-root is mjlab's flag for where experiment logs/checkpoints go
# (the default would be ./logs/rsl_rl inside the repo, i.e. on $HOME — bad).
uv run train "${TASK_ID}" \
  --log-root "${RUN_ROOT}" \
  "$@"

echo "== $(date) done =="
