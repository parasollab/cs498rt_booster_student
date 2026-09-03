#!/usr/bin/env bash
# How many GPU-hours have YOU used, and how is the course allocation doing?
#
#   ./scripts/my_usage.sh            # since COURSE_TERM_START (assignment env)
#   ./scripts/my_usage.sh 2026-10-01 # since a specific date
#
# Works on any Delta login node. Read-only: only queries Slurm accounting.
# Delta charges service units (SU): 1 GPU-hour on an A40 costs 0.5 SU in
# the batch partition and 1.0 SU in the interactive one (Delta docs).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/cluster.env

START="${1:-${COURSE_TERM_START}}"

echo "=============================================================="
echo " Your GPU usage on account ${COURSE_ACCOUNT} since ${START}"
echo "=============================================================="

# Per-job history for this user (top-level jobs only, -X).
# AllocTRES contains e.g. "cpu=8,gres/gpu=1,mem=16G"; multiply the gpu
# count by elapsed seconds to get GPU-hours.
sacct -X --user "$USER" --account "$COURSE_ACCOUNT" \
      --starttime "$START" --parsable2 --noheader \
      --format=JobID,JobName%20,Partition,State,ElapsedRaw,AllocTRES |
awk -F'|' '
  {
    gpus = 0
    if (match($6, /gres\/gpu[^,=]*=[0-9]+/)) {
      tres = substr($6, RSTART, RLENGTH)
      sub(/.*=/, "", tres)
      gpus = tres + 0
    }
    gpu_h = gpus * $5 / 3600.0
    total += gpu_h
    n += 1
    printf "  %-12s %-20s %-22s %-12s %6.2f GPU-h\n", $1, $2, $3, $4, gpu_h
  }
  END {
    if (n == 0) print "  (no jobs found in this window)"
    printf "\n  YOUR TOTAL: %.2f GPU-hours across %d job(s)\n", total, n
  }'

echo
echo "=============================================================="
echo " Whole-course usage by user (shared allocation!)"
echo "=============================================================="
sreport -t Hours -nP cluster AccountUtilizationByUser \
        account="$COURSE_ACCOUNT" start="$START" end=now 2>/dev/null |
awk -F'|' 'NF { printf "  %-16s %8s hours\n", ($3 == "" ? "(account total)" : $3), $5 }' ||
  echo "  (sreport unavailable — ask staff)"

echo
# Delta's own tools: `accounts` (balance) and `jobcharge` (SU charges).
if command -v accounts >/dev/null 2>&1; then
  echo "=== Allocation balance (accounts) ==="
  accounts 2>/dev/null | grep -iE "Project|${COURSE_ACCOUNT}" || accounts
fi
JOBCHARGE="$(command -v jobcharge 2>/dev/null || true)"
[[ -z "$JOBCHARGE" && -x /sw/user/scripts/jobcharge ]] && JOBCHARGE=/sw/user/scripts/jobcharge
if [[ -n "$JOBCHARGE" ]]; then
  echo
  echo "=== Your SU charges (jobcharge) ==="
  "$JOBCHARGE" -a "$COURSE_ACCOUNT" -u "$USER" -s "$START" 2>/dev/null || true
fi

echo
echo "Reminder: the allocation is shared by the whole class. HW0 needs well"
echo "under 1 GPU-hour per student. If your total is far above that, come to"
echo "office hours BEFORE submitting more jobs."
