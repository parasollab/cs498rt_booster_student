#!/usr/bin/env bash
# Create your editable work files from the staff-shipped starters.
# Copies hwN/<name>_starter.py -> hwN/<name>.py for the CURRENT assignment,
# NEVER overwriting an existing work file — safe to re-run after every pull.
#
# Why this exists: staff updates only ever touch *_starter.py (and docs/
# scripts), and your answers live in files that upstream does not contain,
# so `git pull upstream main` can never conflict with your work.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/cluster.env
HW="${1:-${COURSE_ASSIGNMENT}}"
made=0
shopt -s nullglob
for starter in "${HW}"/*_starter.py; do
  work="${starter%_starter.py}.py"
  if [[ -e "${work}" ]]; then
    if ! cmp -s "${starter}" "${work}"; then
      echo "kept   ${work} (your edits; starter differs — diff with:"
      echo "         diff ${starter} ${work})"
    else
      echo "kept   ${work} (identical to starter)"
    fi
  else
    cp "${starter}" "${work}"
    echo "created ${work}  <- ${starter}"
    made=1
  fi
done
if (( made )); then
  echo "Now open the created file(s) and fill the TODO blanks."
else
  echo "Nothing to create for ${HW}."
fi
