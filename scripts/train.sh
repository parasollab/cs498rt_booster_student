#!/usr/bin/env bash
# Submit a training job on Delta. THIS IS THE ONLY SANCTIONED WAY TO TRAIN.
#
#   ./scripts/train.sh <Task-Id> [extra args passed to `uv run train`]
#   e.g. ./scripts/train.sh Course-Cartpole-Swingup --env.scene.num-envs 4096
#
# It sources the locked course settings and hands account, partition, GPU,
# memory, wall-time and the log path to sbatch on the command line, so those
# values live in exactly one place (the assignment env, e.g. scripts/hw0.env). Everything after the
# task id goes to `uv run train` verbatim. Training runs on Delta only.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/cluster.env

TASK_ID="${1:?Usage: ./scripts/train.sh <Task-Id> [train args...]}"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch not found: this is not a Delta login node." >&2
  exit 1
fi

OUT_DIR="${COURSE_WORK_DIR}/${USER}"
mkdir -p "${OUT_DIR}/runs"

echo "== settings from ${COURSE_ENV_SOURCE} (version ${COURSE_ENV_VERSION})"
echo "== account=${COURSE_ACCOUNT} partition=${COURSE_PARTITION_BATCH} gpus=${COURSE_GPUS_PER_JOB}x${COURSE_GPU}" \
     "cpus=${COURSE_CPUS_PER_JOB} mem=${COURSE_MEM} time=${COURSE_TRAIN_TIME}"
echo "== task=${TASK_ID} extra args: ${*:2}"

JOB_ID="$(sbatch --parsable \
  --account="${COURSE_ACCOUNT}" \
  --partition="${COURSE_PARTITION_BATCH}" \
  --gpus-per-node="${COURSE_GPUS_PER_JOB}" \
  --cpus-per-task="${COURSE_CPUS_PER_JOB}" \
  --mem="${COURSE_MEM}" \
  --time="${COURSE_TRAIN_TIME}" \
  --output="${OUT_DIR}/slurm-%j.out" \
  scripts/train.sbatch "$@")"

echo "== submitted job ${JOB_ID}"
echo "   watch the queue:   squeue --me"
echo "   follow the log:    tail -f ${OUT_DIR}/slurm-${JOB_ID}.out"
echo "   your runs:         ${OUT_DIR}/runs/"
