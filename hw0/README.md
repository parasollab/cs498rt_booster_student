# HW0 — Delta, mjlab, and the cartpole

**What you'll learn.** The five habits every later assignment assumes:

1. Keep a healthy `uv` environment in your own `$HOME` on Delta
2. Find out how the cluster is set up (account, partitions, storage, GPUs) with cluster commands
3. Evaluate and *look at* an environment in viser from the login node, with no GPU
4. Read an mjlab manager-based environment config well enough to fill its blanks
5. Submit one training job with the course wrapper, logged to Weights & Biases, and play back the checkpoint

**What you'll do.** Read the annotated cartpole config
(`hw0/cartpole_env_cfg.py`), fill its **five blanks**,
verify them on the login node, and train `Course-Cartpole-Swingup` **once**
on one A40. That is the whole assignment.

**What is graded: the pipeline, not the policy.** Grading lives in two
places. **Gradescope** (see `hw0/GRADESCOPE.md` for the exact list): the
multiple-choice quiz (course facts + cluster discovery), the **URL of your
W&B run**, and your **`hw0/cartpole_env_cfg.py` upload — autograded by the
same `scripts/check_hw0.py` you run locally, no hidden tests**. The
**course submission page**: the writeup (Q4–Q5) and run artifacts, with
`uv run python scripts/hw0_checklist.py` all `[PASS]` as your own
pre-flight. The checklist covers:
environment verified, cluster discovery answers matching the locked
settings, W&B set up, the five blanks passing the CPU checks, one training
run with checkpoints and a W&B record, the reward figure, and the writeup.
How well the trained cartpole swings up is shown as `[INFO]` and never
costs points; what you *observed* and *understood* is what the writeup
grades.

**How your run is verified.** Weights & Biases records, server-side and at
start-up, the compute node, the GPU, your Unix username, the path of your
Python environment and the Slurm job id, account and partition. Staff
compare that with the locked course settings. So: train through the course
scripts, logged in to W&B, from your own account — nothing else is needed.

**Time estimate.** Steps 1–5 in one or two sittings (about 3–4 h). Step 6
is a 5-minute job plus queue time; Step 7 is the writeup.

**All coursework runs on Delta this term.** There is no supported
local/Colab path — if your cluster access is still pending, do Day 0
(README Step 1) now and use the wait for Steps 4–5's reading; everything
except training also works on the login node.

---

## Assignment steps

| Step | What | Where | Checklist item |
|------|------|-------|----------------|
| 0 | Get onto Delta | laptop | — |
| 1 | Install, verify, log in to W&B | login node | environment, W&B login |
| 2 | Discover your cluster | login node | Gradescope quiz, cluster part |
| 3 | Drive the mjlab CLI + viser | login node | — |
| 4 | Read the cartpole config and its parameter sheet | login node + browser | — |
| 5 | Fill the five blanks, verify on CPU | login node | code checks; Gradescope quiz, course-facts part |
| 6 | Train once on an A40 | `train.sh` + login node | run + W&B record; Q4 |
| 7 | Play back, figure, writeup, checklist, submit | login node | figure, writeup, `READY TO SUBMIT`; Q5 |

**The checklist.** At any point, from the repo root:

```bash
uv run python scripts/hw0_checklist.py
```

It prints `[PASS]` / `[TODO]` / `[INFO]` lines with the exact next command for
every `[TODO]`, writes `hw0/hw0_checklist.md` + `.json`, and needs no GPU.
Submit the final one.

**Where to read.** Every step below has a *Read* line. The mjlab links are
pinned to the version you installed; the official cartpole tutorial is the
through-line for Steps 4–5:
<https://mujocolab.github.io/mjlab/v1.6.0/source/tutorials/cartpole.html>.
`docs/03_cartpole_reference.md` collects every link by step.

---

## Step 0 — Get onto Delta

*Read:* `docs/00_delta_setup.md` Part A; Delta
[login methods](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/login.html).

**Done when:** `ssh <you>@login.delta.ncsa.illinois.edu` gives you a prompt
on `dt-login0N`, and you have the `Host delta` block in your laptop's
`~/.ssh/config`.

## Step 1 — Install, verify, log in to W&B

*Read:* `docs/00_delta_setup.md` Part B; [uv sync](https://docs.astral.sh/uv/concepts/projects/sync/).

On a login node:

```bash
git clone https://github.com/parasollab/cs498rt_booster_student.git ~/cs498 && cd ~/cs498
curl -LsSf https://astral.sh/uv/install.sh | sh      # then restart your shell
uv sync                                              # ~3 GB, a few minutes, once
./scripts/init_hw.sh                                 # creates hw0/cartpole_env_cfg.py (your copy) from the starter
uv run python scripts/check_login_env.py
source scripts/cluster.env && mkdir -p "$COURSE_WORK_DIR/$USER/runs"
uv run wandb login                                   # your own free W&B account — projects are open, no invite
```

**Done when:** every line of the check is `[PASS]` (the line
`Course-* tasks registered: none yet` is expected), and
`uv run python scripts/hw0_checklist.py` shows the environment and W&B
sections as `[PASS]`.

## Step 2 — Discover your cluster (graded)

*Read:* Delta [data management](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/data_mgmt.html),
[running jobs → partitions](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/running_jobs.html),
[job accounting](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/job_accounting.html).

The course scripts load a locked settings file so you never type an account
or partition — for HW0 that is `scripts/hw0.env` (master copy
`/projects/bign/cs498/hw0.env` on Delta). Every assignment has its own
settings file, selected by the `scripts/CURRENT_ASSIGNMENT` marker, so a
`git pull` at release time switches everything — always follow the current
handout. But you must know where those values come from. Run these on
the login node and read them off:

```bash
hostname; echo $HOME                   # which login node; your home directory
lscpu | grep 'Model name'              # the CPU
nvidia-smi                             # is there a GPU here? (no — and that is expected)
accounts                               # your Slurm accounts and balances: which one is for GPUs?
id -Gn                                 # your Unix groups: which one is the course project?
quota                                  # your file systems, block and file quotas
sinfo -s                               # every partition, its node count and time limit
sinfo -p gpuA40x4 -o "%P %G %D %l"     # the A40 partitions: GPU type, node count, max time
scontrol show partition gpuA40x4 | grep -E "MaxTime|DefMemPerCPU|MaxNodes"
scontrol show partition gpuA40x4-interactive | grep -E "MaxTime|MaxNodes"
```

There is nothing to write into a file: answer the **cluster part of the
Gradescope quiz** (multiple choice, autograded) straight from this output.
You can also read the same values off the locked file —

```bash
source scripts/cluster.env && env | grep COURSE_    # the locked values, and which file they came from
```

— and that is acceptable by design: the point is that you know *which
command* tells you each value on any cluster.

**Done when:** the cluster part of the Gradescope quiz is answered.

## Step 3 — Drive the mjlab CLI and viser

*Read:* `docs/01_workflow.md`;
[training and playback](https://mujocolab.github.io/mjlab/v1.6.0/source/training/rsl_rl.html#training-and-playback),
[viser](https://mujocolab.github.io/mjlab/v1.6.0/source/viewers.html#viser-browser-based).

mjlab is heavily CLI-based. Four commands cover 90% of this course. Your
own tasks do not exist yet (Step 5), so use mjlab's built-in cartpole:

```bash
uv run list-envs                                  # every registered task id
uv run play Mjlab-Cartpole-Balance --agent zero
uv run play Mjlab-Cartpole-Balance --agent random
uv run demo                                       # mjlab's showcase demo
```

`play` serves the **viser** viewer *on the login node it runs on*. It asks
for port **8080**, but if another student's `play` got there first, viser
silently takes the next free port and prints the one it actually bound —
**always read the port from `play`'s own output**. Set up the tunnel by the
book: the subsection *"Port forwarding to viser"* in `docs/01_workflow.md`
has the exact commands, the three failure modes (wrong port, busy laptop
port, wrong login node) and a one-line `curl` health check. Short version:
run `play` in the same `ssh delta` session that carries the forward, make
the tunnel's second number match `play`'s printed port, then open
<http://localhost:8080>, orbit the camera, find the reward readout, watch
what `--agent random` does to the pole.

Remember this loop: **zero/random agent + viser on the login node** is how
you debug every environment you will ever build here. It exercises loading,
resets, observations and rewards — everything except learning — and costs
zero GPU-hours.

**Done when:** you can see the cartpole moving in your browser via the
tunnel, for both `--agent zero` and `--agent random`.

## Step 4 — Read the cartpole

*Read:* the official tutorial, top to bottom (20 minutes):
<https://mujocolab.github.io/mjlab/v1.6.0/source/tutorials/cartpole.html>;
then the [architecture overview](https://mujocolab.github.io/mjlab/v1.6.0/source/architecture_overview.html).

Now open `hw0/cartpole_env_cfg.py` (your editable copy — created from
`cartpole_env_cfg_starter.py` by `./scripts/init_hw.sh`) and `cartpole.xml`
side by side. The config is the tutorial's code with course annotations:
seven numbered sections, each starting with a *Read:* line, following the
same order as the tutorial:

1. entity (XML → `EntityCfg`, actuator, initial state)
2. observations (four terms, 5 numbers)
3. reward (a product of four factors in [0, 1])
4. events (reset jitter)
5. + 6. actions, terminations, assembly and timing
7. PPO (fixed)

**Optional but recommended:** keep `docs/03_cartpole_reference.md` next to
it — it explains every number in the XML, the config and the PPO block
(value, unit, effect, doc link). Nothing in it is submitted or graded; it
is the reference you will wish you had when your own tasks misbehave.

**Done when:** you can say, for each of the five blanks, which section of
the tutorial explains it.

## Step 5 — Fill the five blanks and verify on the login node

*Read:* the *Read:* lines above each `TODO(` in the file;
`docs/03_cartpole_reference.md` section 4 lists them all.

| Blank | What you write | Where the answer is |
|---|---|---|
| TODO(1) | the joint name the XML motor drives (a string) | `cartpole.xml` `<actuator>`; tutorial *The XML model* |
| TODO(2) | the hinge angle for "hanging straight down" (a float) | tutorial *Entity*, Swingup tab |
| TODO(3) | three lines: read the hinge angle, return (cos, sin) | tutorial *Observations* |
| TODO(4) | one expression: the `upright` reward factor | tutorial *Rewards*, the equation |
| TODO(5) | a small symmetric `(low, high)` reset range in radians | tutorial *Events* |

Work in order and check after each one:

```bash
uv run python scripts/check_hw0.py                                            # names what is still missing
uv run list-envs | grep Course                                                # both Course-Cartpole-* appear after (1), (2), (5)
uv run play Course-Cartpole-Swingup --agent zero --env.scene.num-envs 4       # (2): pole hangs straight down
uv run play Course-Cartpole-Balance --agent zero --env.scene.num-envs 4       # pole balances until jitter topples it
uv run play Course-Cartpole-Swingup --agent random --env.scene.num-envs 16    # (3), (4): reward ≈0 hanging, ≈1 up
```

**Optional:** the three no-GPU experiments in
`docs/03_cartpole_reference.md` section 3 (change one value, look in viser,
revert) are the fastest way to build intuition for the reward and the reset
machinery. Nothing from them is submitted — each ends in a self-check
question with the answer derivable from the doc.

**Gradescope** (full list in `hw0/GRADESCOPE.md`): the multiple-choice
quiz (cluster discovery from Step 2 + course facts you meet by the end of
Step 6), your W&B run URL, and the `cartpole_env_cfg.py` upload —
autograded by the same `scripts/check_hw0.py`, no hidden tests. Answer any
time before the deadline; none of it goes in the writeup.

Rules: no Python loops over environments; keep the public function names and
signatures; don't touch the PPO config or the sim timing; a diff against the
starter (`diff hw0/cartpole_env_cfg_starter.py hw0/cartpole_env_cfg.py`)
must show only your five blanks before you train.

**Done when:** `check_hw0.py` is all `[PASS]` and the three `play`
commands look right in viser.

## Step 6 — Train once on an A40

*Read:* `docs/01_workflow.md` Step 3; `docs/04_delta_cheatsheet.md` sections 2–3
(queues, logs, getting onto the GPU node); Delta
[running jobs](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/running_jobs.html);
[W&B quickstart](https://docs.wandb.ai/quickstart).

```bash
./scripts/train.sh Course-Cartpole-Swingup --env.scene.num-envs 4096
squeue --me                                          # queued / running?
tail -f $COURSE_WORK_DIR/$USER/slurm-<jobid>.out     # the live log
./scripts/my_jobs.sh                                 # everything at once
```

The wrapper submits `scripts/train.sbatch` with the locked account,
partition, one A40, 16 GB and a 20-minute cap; everything after the task id
goes to `uv run train` verbatim. Reading the log, in order: (1) the Slurm
job id, node, account, partition and `nvidia-smi` — the facts W&B also
records and Q4 asks about; (2) the W&B run link — open it and watch
`Episode_Reward/smooth_reward` live; (3) one block per iteration with
`Steps per second` and the reward terms. Checkpoints land under
`$COURSE_WORK_DIR/$USER/runs/hw0_cartpole/<timestamp>/` (`hw0_cartpole` is
the `experiment_name` from section 7 of the config) as `model_49.pt` …
`model_499.pt`.

Cartpole swingup visibly learns within the run:
`Episode_Reward/smooth_reward` climbs past 0.9 by iteration ~100 and the
500 iterations take about 5 minutes on the A40 (65.5 M env-steps ≈ 0.2 M/s). If it
does not learn, your problem is setup, not RL — check the log,
`./scripts/my_jobs.sh`, and office hours.

★ **Q4:** From the W&B run page (Overview → system metadata, and the
*Steps per second* curve or the log): which node and GPU ran your job, what
driver version did `nvidia-smi` print, and roughly how many env-steps per
second did you get?

**Done when:** the run appears at
<https://wandb.ai/cs498rt-26fall/hw0-booster> under your name and the
checklist's section 5 shows the run with checkpoints and a W&B record.

## Step 7 — Play it back, figure, writeup, checklist, submit

*Read:* [checkpoints and logging](https://mujocolab.github.io/mjlab/v1.6.0/source/training/rsl_rl.html#checkpoints-and-logging).

Back on the login node, no GPU:

```bash
./scripts/latest_checkpoint.sh $COURSE_WORK_DIR/$USER/runs          # prints the play command
uv run play Course-Cartpole-Swingup \
  --checkpoint-file $COURSE_WORK_DIR/$USER/runs/hw0_cartpole/<run>/model_499.pt \
  --log-root $COURSE_WORK_DIR/$USER/runs
# or, straight from W&B (run id from your run page):
uv run play Course-Cartpole-Swingup --wandb-run-path cs498rt-26fall/hw0-booster/<run-id>
uv run python scripts/plot_rewards.py --log-root $COURSE_WORK_DIR/$USER/runs --out hw0/reward_curves.png
```

★ **Q5:** Describe in two or three sentences what the trained policy does
in viser — how it swings up, how it holds the pole, what the cart does —
and one thing you would change in the reward or the config to make it
better, and why.

Write `hw0/writeup.md` (or .txt/.docx/.pdf), at most 2 pages, containing:

- answers to ★ Q4–Q5 (the multiple-choice quiz is answered on Gradescope, not here)
- the reward figure and the W&B run link (or run id) of your run
- the run directory name and checkpoint filename you want us to look at

Then run the checklist until it says `READY TO SUBMIT`:

```bash
uv run python scripts/hw0_checklist.py
```

### What to submit

**On Gradescope** (exact questions: `hw0/GRADESCOPE.md`):

1. the multiple-choice quiz (course facts + cluster discovery)
2. the **URL of the W&B run you want graded**
   (`https://wandb.ai/cs498rt-26fall/hw0-booster/runs/<run-id>`)
3. **`hw0/cartpole_env_cfg.py`** — autograded by the same
   `scripts/check_hw0.py` you run locally; **no hidden tests**

**On the course submission page** (not email, not a push to `main`):

4. your writeup (Q4–Q5, the reward figure, the run directory + checkpoint
   name you want us to look at)
5. `hw0/hw0_checklist.md` and `hw0/hw0_checklist.json` (the final run)
6. `hw0/reward_curves.png`
7. the run folder's `params/` and the checkpoint you named (`model_<iter>.pt`)
   — `scp` them from `$COURSE_WORK_DIR/$USER/runs/hw0_cartpole/<run>/`

We grade from the checklist, your cluster answers, the W&B record of your
run, the code checks and the writeup; we may run your checkpoint to see
what it does, but its quality does not change your grade.
