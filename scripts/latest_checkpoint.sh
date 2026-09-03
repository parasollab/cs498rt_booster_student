#!/usr/bin/env bash
# Print the most recently written model_<iter>.pt under a log root and the
# `uv run play` command that views it. Works on the cluster and locally.
#
#   ./scripts/latest_checkpoint.sh                      # logs/rsl_rl (local default)
#   ./scripts/latest_checkpoint.sh $COURSE_WORK_DIR/$USER/runs   # cluster
#   ./scripts/latest_checkpoint.sh logs/rsl_rl hw0_cartpole       # one experiment only
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="${1:-logs/rsl_rl}"
EXP="${2:-}"
DIR="${ROOT}${EXP:+/${EXP}}"

if [[ ! -d "${DIR}" ]]; then
  echo "No such directory: ${DIR}" >&2
  exit 1
fi
# shellcheck disable=SC2012
# W&B mirrors checkpoints into <log_root>/wandb/<run>/files/; skip those copies.
CKPT="$(find "${DIR}" -path '*/wandb/*' -prune -o -name 'model_*.pt' -print0 2>/dev/null | xargs -0 ls -t 2>/dev/null | head -1 || true)"
if [[ -z "${CKPT}" ]]; then
  echo "No model_*.pt under ${DIR} yet (checkpoints are written every save_interval iterations)." >&2
  exit 1
fi

RUN_DIR="$(dirname "${CKPT}")"
echo "Latest checkpoint: ${CKPT}"
echo "Run directory:     ${RUN_DIR}   (params/env.yaml + params/agent.yaml are in here)"
echo
echo "View it (pick the task id this run was trained on):"
echo "  uv run play <Task-Id> --checkpoint-file ${CKPT} --log-root ${ROOT}"
