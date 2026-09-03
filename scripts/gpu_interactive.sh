#!/usr/bin/env bash
# Start a short interactive GPU session on Delta and print the viser tunnel command.
# Usage: ./scripts/gpu_interactive.sh [minutes]   (default from the assignment env, max 60)
#
# Delta's interactive partitions allow 1 running job per user and 1 hour max.
# Use this to debug on a real GPU; training goes through ./scripts/train.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/cluster.env

MINUTES="${1:-}"
if [[ -z "${MINUTES}" ]]; then
  TIME="${COURSE_INTERACTIVE_TIME}"
else
  if (( MINUTES > 60 )); then
    echo "Interactive sessions are capped at 60 min on ${COURSE_PARTITION_INTERACTIVE}." >&2
    echo "Longer work is a batch job: ./scripts/train.sh" >&2
    exit 1
  fi
  TIME="$(printf '00:%02d:00' "${MINUTES}")"
fi

echo "Requesting ${COURSE_GPUS_PER_JOB}x${COURSE_GPU} for ${TIME} on ${COURSE_PARTITION_INTERACTIVE} (account ${COURSE_ACCOUNT})..."
echo "(When the session starts, the exact viser tunnel command — with the compute node"
echo " already filled in — is printed. Nothing to copy from this screen.)"
echo
# The srun payload is DOUBLE-quoted on purpose: ${COURSE_LOGIN_HOST} is baked in
# at submit time, while \$(hostname) and \${USER} expand on the compute node, so
# the printed command is copy-paste ready (real node, real cluster username).
exec salloc \
  --account="${COURSE_ACCOUNT}" \
  --partition="${COURSE_PARTITION_INTERACTIVE}" \
  --nodes=1 --ntasks=1 --cpus-per-task="${COURSE_CPUS_PER_JOB}" \
  --gpus-per-node="${COURSE_GPUS_PER_JOB}" \
  --mem="${COURSE_MEM}" \
  --time="${TIME}" \
  srun --pty bash -c "
NODE=\$(hostname)
echo \"=== GPU session on \${NODE} ===\"
nvidia-smi -L
echo
echo \"--- Viewing viser from your laptop -------------------------------------\"
echo \"Open a NEW terminal ON YOUR LAPTOP (not on Delta) and paste exactly:\"
echo
echo \"  ssh -J \${USER}@${COURSE_LOGIN_HOST} -L 8080:localhost:8080 \${USER}@\${NODE}\"
echo
echo \"  1. Approve Duo if prompted (it can prompt twice: login hop, then node).\"
echo \"  2. Leave that laptop terminal open — it holds the tunnel.\"
echo \"  3. Start viser HERE (in this session), then browse http://localhost:8080\"
echo \"     on your laptop.\"
echo \"  If 8080 is already taken on your laptop, change only the FIRST number:\"
echo \"    -L 9000:localhost:8080   then browse http://localhost:9000\"
echo \"  The tunnel dies when this job (\${SLURM_JOB_ID:-?}) ends or hits its time limit.\"
echo \"------------------------------------------------------------------------\"
exec bash"
