# 01 — The workflow (and how to see viser)

You will repeat this loop in every single assignment. Official references:
[Delta running jobs](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/running_jobs.html),
[mjlab: training and playback](https://mujocolab.github.io/mjlab/v1.6.0/source/training/rsl_rl.html#training-and-playback),
[mjlab: viewers](https://mujocolab.github.io/mjlab/v1.6.0/source/viewers.html#viser-browser-based).

Day-to-day operations (queues, logs, reattaching to a GPU node, storage,
W&B) are collected in `04_delta_cheatsheet.md`; this page is the model.

## The locked settings

**Every assignment has its own locked settings file** —
`scripts/<assignment>.env` in the repo, master copy
`/projects/bign/cs498/<assignment>.env` on Delta — because account,
partition, W&B project and paths can all differ between assignments.
Every script starts with `source scripts/cluster.env`, a tiny loader that
reads the one-line marker `scripts/CURRENT_ASSIGNMENT` (staff bump it with
each release) and loads that assignment's file — so **`git pull` at the
start of every assignment is what switches your settings**; follow each
handout. You never type an account or partition name, and you never edit
those files. To see the active assignment and its values:

```bash
source scripts/cluster.env && env | grep COURSE_
```

## Step 1 — Login node: edit, evaluate, visualize (no GPU needed)

mjlab requires an NVIDIA GPU **for training** only. Building an environment,
stepping it with a scripted agent, and viewing it in the browser all work on
the GPU-less login node:

```bash
uv run list-envs                                   # what tasks exist
uv run play Mjlab-Cartpole-Balance --agent zero    # mjlab's built-in cartpole, env sanity check
uv run play Mjlab-Cartpole-Balance --agent random
uv run play Course-Cartpole-Swingup --agent zero   # the course task, once HW0's blanks are filled
uv run demo                                        # mjlab's built-in demo
```

### Port forwarding to viser (set this up once, carefully)

`play` serves mjlab's built-in **viser** viewer over a websocket. It *asks*
for port **8080** — but if 8080 is taken on that machine (login nodes are
shared by the whole class), **viser silently takes the next free port and
prints the one it actually bound**. So the one rule that prevents almost
every "black page" report:

> **Read the port from `play`'s own startup output, and make your tunnel's
> SECOND number match it.**

**Case A — `play` on a login node** (this step). Run the tunnel and `play`
in the **same** SSH session, so both land on the same `dt-login0N`
(`login.delta…` round-robins across four):

```bash
ssh -L 8080:localhost:8080 <user>@login.delta.ncsa.illinois.edu
# ...then, inside that same session:
cd ~/cs498 && uv run play Mjlab-Cartpole-Balance --agent zero
```

then open <http://localhost:8080>. (With the `Host delta` block from
`00_delta_setup.md` A.3, plain `ssh -L 8080:localhost:8080 delta` does this.
If you must use two sessions, SSH both to the same explicit `dt-login0N`.)

**Case B — `play` inside a GPU session.** Don't build the command yourself:
`./scripts/gpu_interactive.sh` prints a copy-paste-ready tunnel command
(with the compute node and your username filled in) the moment the session
starts. It hops through a login node with `-J`, because compute nodes are
not reachable from outside.

**If the page doesn't load — the three failure modes, in order of likelihood:**

1. **viser didn't get 8080 on Delta** (another student got there first).
   Look at `play`'s output: it names the real port, e.g. `:8081`. Re-open
   your tunnel with the second number changed:
   `ssh -L 8080:localhost:8081 …` — the browser stays on
   <http://localhost:8080>.
2. **8080 is already taken on YOUR laptop.** ssh then prints a one-line
   warning — `bind [127.0.0.1]:8080: Address already in use` — **but still
   logs you in**, so it is easy to miss: you get a working shell and a dead
   tunnel. Check before (macOS/Linux: `lsof -i :8080` · Windows:
   `netstat -ano | findstr :8080`), or just change the FIRST number:
   `ssh -L 9000:localhost:8080 …` and browse <http://localhost:9000>.
3. **Tunnel and `play` are on different login nodes** (round-robin).
   Symptom: instant "connection refused"/empty page. Fix: same session
   (Case A), or pin the same `dt-login0N` in both.

Quick health check, on your **laptop**, once `play` is running:

```bash
curl -sI http://localhost:8080 | head -1     # expect: HTTP/1.1 200 OK
```

No output → the tunnel is dead (modes 2/3). A 200 but a black page → wrong
remote port (mode 1).

This "edit → play → look at it in viser" loop on the login node is your
primary debugging tool. It costs zero GPU-hours.

Etiquette: login nodes are shared. One `play` with a handful of envs is
fine; never run training there.

## Step 2 — Interactive GPU session (debugging only)

Use this when you need a real GPU in the loop, for example to confirm your
env compiles with thousands of parallel instances, or to run
`scripts/check_gpu_env.py`. Always start it through the wrapper:

```bash
./scripts/gpu_interactive.sh          # 1 A40, 30 min (max 60), course account
uv run python scripts/check_gpu_env.py
```

**Exercise (do this once, in your first GPU session):** prove to yourself
that you can wire a compute node to your laptop before you ever need it
under deadline pressure. In the GPU session:

```bash
uv run demo          # mjlab's built-in demo — serves viser on the node
```

then, on your laptop, paste the tunnel command the wrapper printed when the
session started, and open <http://localhost:8080>. You should see the demo
scene render live from the compute node. If you see it: you understand the
whole chain (laptop → login node → compute node → viser). If you don't,
work through the three failure modes in "Port forwarding to viser" above —
that triage is the skill this exercise teaches.

The wrapper prints a copy-paste-ready tunnel command (your username and the
compute node already filled in) as soon as the session starts — use that,
don't compose it by hand. Compute nodes are not reachable from outside, so
it hops through a login node with `-J`; Delta allows `ssh` into a node while
your job runs on it. If the viewer doesn't load, the same three failure
modes (and the `curl` health check) from "Port forwarding to viser" in
Step 1 apply here unchanged.

Delta's interactive partitions are deliberately capped: 1 hour, 1 running
job per user, and they cost twice the batch rate. They are for
*explore/debug*, not for training runs.

## Step 3 — Batch training runs

> **A task must be registered before you train it.** An assignment's
> `Course-*` tasks appear in `uv run list-envs` only once that assignment's
> code blanks pass its check script (for HW0:
> `uv run python scripts/check_hw0.py`, zero GPU-hours on the login node).
> Submitting a job for an unregistered task just burns queue time on an
> immediate import error — do the assignment's code steps first; the
> handout walks you through them.

All training goes through the course wrapper, which passes the locked
account, partition, one A40, 16 GB of memory, the wall-time cap and the log
path to `sbatch`:

```bash
./scripts/train.sh <Task-Id> [args...]     # e.g. --env.scene.num-envs 4096
```

Everything after the task id is passed to `uv run train` verbatim. The
wrapper prints the job id and the monitoring commands:

```bash
squeue --me                                          # queued / running?
tail -f $COURSE_WORK_DIR/$USER/slurm-<jobid>.out     # the live log
./scripts/my_jobs.sh                                 # everything at once
```

The job starts by printing its Slurm job id, node, account and partition
and `nvidia-smi`; then the W&B run link; then one block per iteration with
`Steps per second` and the reward terms. Checkpoints and logs land under
`$COURSE_WORK_DIR/$USER/runs/<experiment>/<timestamp>/`, where
`<experiment>` is the assignment's `experiment_name` from its PPO config.

View a trained policy afterwards — back on the login node, no GPU:

```bash
./scripts/latest_checkpoint.sh $COURSE_WORK_DIR/$USER/runs      # prints the play command
uv run play <Task-Id> \
  --checkpoint-file $COURSE_WORK_DIR/$USER/runs/<experiment>/<run>/model_<iter>.pt \
  --log-root $COURSE_WORK_DIR/$USER/runs
# or, straight from W&B (team/project/run-id from the run page):
uv run play <Task-Id> --wandb-run-path cs498rt-26fall/<project>/<run-id>
```

The concrete, fully-worked HW0 training run — exact task id, what each log
block means, expected reward trajectory and timings — is in
`hw0/README.md` Steps 6–7.

## Weights & Biases

Every run logs to the course team **`cs498rt-26fall`**, into the
**per-assignment project** set by that assignment's PPO runner config
(`wandb_project` — HW0 uses `hw0-booster`, so HW0 runs appear at
<https://wandb.ai/cs498rt-26fall/hw0-booster>; later assignments have their
own project). A run records reward curves, the config, and — recorded by
W&B itself, not by you — the compute node, GPU, your Unix username, the
Python path of your venv and the Slurm job id, account and partition.
**That record is how each assignment verifies your run.** If a job ran
without a W&B login it logs offline; upload it later with
`uv run wandb sync <run dir>`.

## Budget

Delta charges *service units* per GPU-hour, per project: **one runaway job
spends everyone's hours.** An A40-hour costs 0.5 SU in the batch partition
and 1.0 SU interactive; a full HW0 run is about 5 minutes on one A40.

```bash
./scripts/my_usage.sh     # your GPU-hours, the class total, `accounts` balance, `jobcharge`
accounts                  # Delta's balance tool
```

Rule of thumb: if you haven't watched it behave under `--agent random` in
viser on the login node, it is not ready for `train.sh`.
