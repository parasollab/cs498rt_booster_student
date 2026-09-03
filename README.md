# CS 498 — Robotics Team Project: course repository

This repository is everything you need for the programming assignments:
environment configs built on [mjlab](https://github.com/mujocolab/mjlab)
(MuJoCo Warp + an Isaac-Lab-style manager API), Slurm job wrappers for
NCSA's **Delta** cluster, and per-assignment handouts.

**You are probably reading this on GitHub, on your laptop, with no cluster
access yet. That's the expected starting point.** All coursework runs on
Delta (`login.delta.ncsa.illinois.edu`; *not* DeltaAI). You will not install
anything on your own machine except an SSH client. Follow the numbered path
below in order.

---

## Start here

### Step 1 — Day 0 (do this today; the waiting is on our side)

Two accounts, both self-service, both needed before anything else:

1. **Be ready for Delta**: follow **[`docs/00_delta_setup.md`](docs/00_delta_setup.md), Part A** —
   create your NCSA identity (associated with your UIUC account),
   **complete the profile**, enroll NCSA Duo, and send us your NCSA
   username. Staff then add you to the course project; that hop can take a
   day — do not leave it until an assignment is due.
2. **Create a free [Weights & Biases](https://wandb.ai) account.** The
   course W&B projects are open — no invitation needed — and every graded
   training run is verified through its W&B record.

### Step 2 — Clone this repo ON THE CLUSTER and install

Once you can SSH in, follow **`docs/00_delta_setup.md`, Part B**. In short,
on a Delta login node (not your laptop):

```bash
git clone https://github.com/parasollab/cs498rt_booster_student.git ~/cs498
cd ~/cs498
curl -LsSf https://astral.sh/uv/install.sh | sh    # then restart your shell
uv sync
./scripts/init_hw.sh          # creates your editable work files from the starters
```

To version-control your answers, set up the private mirror from
"How this repo is used all semester" below (5 commands, once).

Every student builds their own environment, in their own `$HOME`. It is
identical for everyone because `uv sync` reads `pyproject.toml` + `uv.lock`
and installs exactly that. Do not `pip install` anything, do not create
conda envs, do not install mjlab by hand. ([uv docs](https://docs.astral.sh/uv/))

### Step 3 — Verify

Still on the login node:

```bash
uv run python scripts/check_login_env.py
```

Every line must say `[PASS]` (CUDA being *unavailable* here is a pass —
login nodes have no GPU; that's normal and explained in the docs).

### Step 4 — Learn the workflow

Read **[`docs/01_workflow.md`](docs/01_workflow.md)**: where code runs
(login node vs. interactive GPU vs. batch jobs), where files go, how the
locked course settings work, and how to see the 3D viewer (viser) in your
laptop's browser through an SSH tunnel. Ten minutes of reading that will
save you hours all semester. Keep **[`docs/04_delta_cheatsheet.md`](docs/04_delta_cheatsheet.md)**
open for everything you will do repeatedly: queues, logs, getting back onto a
GPU node, storage, W&B.

### Step 5 — Do HW0

Open **[`hw0/README.md`](hw0/README.md)**. It walks you through everything
step by step and doubles as your training for every later assignment.

---

## How this repo is used all semester

- **Staff ship starters, you edit copies.** Every editable file arrives as
  `hwN/<name>_starter.py`; `./scripts/init_hw.sh` creates your working copy
  (`hwN/<name>.py`) next to it, and staff **never ship the working name** —
  so a sync can never touch, let alone conflict with, your answers.
- **Track your work in your own private mirror (no branches, no forks):**

  ```bash
  git clone https://github.com/parasollab/cs498rt_booster_student.git ~/cs498 && cd ~/cs498
  git remote rename origin upstream
  # make an EMPTY private repo on your GitHub, then:
  git remote add origin git@github.com:<you>/<your-private-repo>.git
  git push -u origin main
  ./scripts/init_hw.sh          # creates your editable work files
  ```

  Work directly on `main`; commit and `git push` as you like. (A public
  GitHub *fork* would publish your solutions — don't.)
- **Syncing a release is one command and never conflicts:**

  ```bash
  git pull upstream main        # new handouts, starters, settings
  git push                      # mirror it to your repo
  ```

  Releases add `hwN/` folders, update `*_starter.py`, docs, scripts and the
  settings marker — never your working files. If a starter you already
  copied was fixed, `./scripts/init_hw.sh` tells you and prints the diff
  command; folding such a fix in is a deliberate, visible step, not a merge
  conflict.
- All commands run through `uv` from the repo root, on Delta.
- Cluster settings (account, partitions, paths, W&B team) are **locked by
  staff, per assignment**: every assignment ships its own
  `scripts/<assignment>.env` (HW0: `hw0.env`; master copies under
  `/projects/bign/cs498/` on Delta). `scripts/cluster.env` loads whichever
  assignment the one-line marker `scripts/CURRENT_ASSIGNMENT` names —
  staff bump the marker with each release, so **`git pull` is what
  switches your settings**. You never type an account name and never edit
  these files.

---

## Daily commands on Delta (the cheatsheet five)

Helper scripts in `scripts/`. Run them from the repo root; they only affect
your own jobs. `docs/04_delta_cheatsheet.md` is the long version.

```bash
./scripts/train.sh <Task-Id> [args]   # the ONLY way to submit training
./scripts/my_jobs.sh                  # are all my training jobs done?
./scripts/my_usage.sh                 # how many GPU-hours / SUs have I used?
./scripts/kill_my_jobs.sh             # cancel my jobs (asks before acting)
./scripts/gpu_attach.sh [jobid]       # ssh onto the node of my running job
```

- **`my_jobs.sh`** — what's running, what finished, what failed (with the
  log file to check). Run it before logging off.
- **`my_usage.sh`** — your GPU-hours and the whole class's. The allocation
  is shared, so check it before submitting extra runs.
- **`kill_my_jobs.sh`** — cancels your jobs after showing you the list and
  asking for confirmation. Use it the moment you spot a bad run — saved
  checkpoints are kept, so you lose nothing.

## Repo layout

```
docs/
  00_delta_setup.md        Part A: get cluster access · Part B: install this repo
  01_workflow.md           login node / interactive GPU / batch jobs, storage, viser tunnel
  03_cartpole_reference.md every number in the cartpole, what it does, where it is documented
  04_delta_cheatsheet.md   day to day on Delta: sessions, jobs, GPU nodes, storage, environment
scripts/
  # locked settings (per assignment)
  cluster.env              loads the CURRENT assignment's locked settings (master on Delta wins)
  hw0.env                  LOCKED HW0 settings (staff-owned; read, never edit)
  # Delta jobs — the daily five
  train.sh                 submit a training job with the locked settings (the ONLY way)
  my_jobs.sh               running / finished / failed, with the log file to check
  my_usage.sh              your GPU-hours and the class total
  kill_my_jobs.sh          cancel your jobs (shows the list, asks first)
  gpu_attach.sh            ssh onto the node of your running job (nvidia-smi, logs)
  # GPU sessions & jobs plumbing
  gpu_interactive.sh       interactive GPU session; prints your tunnel command
  train.sbatch             the job body train.sh submits
  # checks & assignment tools
  check_login_env.py       sanity checks for the login node (no GPU)
  check_gpu_env.py         sanity checks to run inside an interactive GPU job
  check_hw0.py             HW0: which blanks are still empty, does the env behave (CPU)
  hw0_checklist.py         HW0: the graded pipeline report
  latest_checkpoint.sh     newest model_<iter>.pt + the play command
  plot_rewards.py          reward-curve figure for the writeup
          the Python package with all course environments
  hw0/README.md              assignment 0 handout — the hw0/ folder is ALSO the
                           Python package with the annotated cartpole:
hw0/cartpole_env_cfg.py    ...the config with your five blanks
hw0/cartpole.xml           ...the MuJoCo model it wraps
```
