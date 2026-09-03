#!/usr/bin/env bash
# Get (back) onto the compute node of one of your RUNNING jobs, to look at the
# GPU (nvidia-smi), your processes, or the log. Delta allows ssh into a node
# while you have a job running on it; `exit` leaves the job running.
#
#   ./scripts/gpu_attach.sh            # your only running job
#   ./scripts/gpu_attach.sh <jobid>    # a specific job
#
# This is for monitoring. Training is submitted with ./scripts/train.sh, never
# started by hand on a node: processes you start here belong to the job and
# die with it, and they are not what the accounting / W&B verification see.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/cluster.env

JOB="${1:-}"
if [[ -z "${JOB}" ]]; then
  RUNNING="$(squeue --me --states=RUNNING --noheader --format='%i %j %N %M' || true)"
  if [[ -z "${RUNNING}" ]]; then
    echo "You have no running job (a queued job has no node yet):"
    squeue --me
    echo
    echo "Start one:  ./scripts/gpu_interactive.sh   or   ./scripts/train.sh <Task-Id>"
    exit 1
  fi
  if (( $(echo "${RUNNING}" | wc -l) > 1 )); then
    echo "Several running jobs; pass the job id you want:"
    echo "  JOBID NAME NODE TIME"
    echo "${RUNNING}" | sed 's/^/  /'
    exit 1
  fi
  JOB="$(echo "${RUNNING}" | awk '{print $1}')"
fi

STATE="$(squeue -j "${JOB}" --noheader --format='%T' 2>/dev/null | head -1 || true)"
NODE="$(squeue -j "${JOB}" --noheader --format='%N' 2>/dev/null | head -1 || true)"
if [[ "${STATE}" != "RUNNING" || -z "${NODE}" || "${NODE}" == "(null)" ]]; then
  echo "Job ${JOB} is not running (state: ${STATE:-unknown}); nodes are reachable only while your job runs there." >&2
  squeue -j "${JOB}" 2>/dev/null || true
  exit 1
fi

echo "Job ${JOB} is running on ${NODE}. Connecting; 'exit' leaves the job running."
echo "Useful there:  nvidia-smi   |   top -u \$USER   |   tail -f ${COURSE_WORK_DIR}/${USER}/slurm-${JOB}.out"
echo "viser from your laptop:  ssh -J \$USER@${COURSE_LOGIN_HOST} \$USER@${NODE} -L 8080:localhost:8080"
echo
exec ssh -o StrictHostKeyChecking=accept-new "${NODE}"
