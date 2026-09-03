# HW0 — What you answer and upload on Gradescope (60 pts)

Everything on this page is graded **on Gradescope**; the writeup and run
artifacts go to the course submission page (see `README.md` Step 7). You
can answer/submit in any order, any time before the deadline, and resubmit
freely.

## Part 1 — multiple choice (autograded, 5 pts each, 20 pts)

- **Q1.** What is the **4-letter project code** of the course allocation on
  Delta (you see it in `accounts` and in your `/projects/...` and
  `/work/hdd/...` paths)?
  `bign` · `bing` · `bbgn` · `dlta`
- **Q2.** Checkpoints of your run land in
  `$COURSE_WORK_DIR/$USER/runs/<X>/<timestamp>/`. What is `<X>`?
  `hw0_cartpole` · `cartpole_hw0` · `hw0-booster` · `Course-Cartpole-Swingup`
- **Q3.** Course batch training jobs run on which Slurm partition?
  `gpuA40x4` · `gpuA40x4-interactive` · `gpuA100x4` · `cpu`
- **Q4.** Which command shows your **queued/running jobs** on Delta?
  `squeue --me` · `sinfo -s` · `scancel <jobid>` · `tail -f slurm-<jobid>.out`

## Part 2 — your W&B run URL (20 pts)

Paste the **URL of the training run you want graded**, from your browser's
address bar on the run page:

    https://wandb.ai/cs498rt-26fall/hw0-booster/runs/<run-id>

(Anything after `?` is fine to include.) We verify, from W&B's server-side
record: Delta compute node, course account + batch partition, A40, your
venv under `/u/<netid>/`, and the untouched PPO config. Train through
`./scripts/train.sh`, logged in to W&B, and all of that is automatic.

## Part 3 — file upload: `hw0/cartpole_env_cfg.py` (autograded, 20 pts)

Upload exactly that one file. **The autograder is
`scripts/check_hw0.py` — the same command you run locally. No hidden
tests**: 4 points per blank, each scored by the same named checks you see
in your terminal, plus a 0-point diagnostics section with the full output.
If it passes on your machine with `uv run python scripts/check_hw0.py`, it
passes on Gradescope. Keep the public function names and signatures.
