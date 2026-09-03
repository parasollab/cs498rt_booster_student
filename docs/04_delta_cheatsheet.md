# 04 — Delta day to day: sessions, jobs, GPU nodes, storage, environment

The operations you will repeat all semester, each with the command and what
to expect. The daily five, all from the repo root:

```bash
./scripts/train.sh <Task-Id> [args]   # the ONLY way to submit training
./scripts/my_jobs.sh                  # are all my training jobs done?
./scripts/my_usage.sh                 # how many GPU-hours / SUs have I used?
./scripts/kill_my_jobs.sh             # cancel my jobs (asks before acting)
./scripts/gpu_attach.sh [jobid]       # ssh onto the node of my running job
```

They read the locked settings of whichever assignment
`scripts/CURRENT_ASSIGNMENT` names (every assignment has its own
`scripts/<assignment>.env`; staff bump the marker with each release, so
`git pull` switches the settings). Everything runs from the repo root on a login node unless it says
otherwise. Official references: Delta
[login](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/login.html),
[running jobs](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/running_jobs.html),
[data management](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/data_mgmt.html),
[job accounting](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/job_accounting.html);
Slurm [squeue](https://slurm.schedmd.com/squeue.html), [scancel](https://slurm.schedmd.com/scancel.html),
[sacct](https://slurm.schedmd.com/sacct.html).

## The one-screen version

| I want to… | Command |
|---|---|
| log in | `ssh <you>@login.delta.ncsa.illinois.edu` (or `ssh delta` with the config block from `00_delta_setup.md`) |
| keep my work alive across disconnects | `tmux new -s work` … `Ctrl-b d` to detach … later `ssh <you>@dt-login0N…` then `tmux attach -t work` |
| see which login node I am on | `hostname` |
| submit training | `./scripts/train.sh <Task-Id> [train args]` |
| see my queued / running jobs | `squeue --me` or `./scripts/my_jobs.sh` |
| know why my job is still waiting | `squeue --me --start` and `scontrol show job <jobid> \| grep -E "Reason\|StartTime"` |
| follow a job's log | `tail -f $COURSE_WORK_DIR/$USER/slurm-<jobid>.out` |
| cancel a job | `scancel <jobid>` or `./scripts/kill_my_jobs.sh` |
| get onto the GPU node of my running job | `./scripts/gpu_attach.sh [jobid]` |
| start a short interactive GPU shell | `./scripts/gpu_interactive.sh [minutes]` |
| see finished jobs and exit codes | `sacct -X --user $USER --starttime today` or `./scripts/my_jobs.sh 2d` |
| see what I have spent | `./scripts/my_usage.sh`, `accounts`, `jobcharge -a $COURSE_ACCOUNT -u $USER` |
| find my newest checkpoint and the play command | `./scripts/latest_checkpoint.sh $COURSE_WORK_DIR/$USER/runs` |
| check my disk quotas | `quota` |
| see the locked course settings | `source scripts/cluster.env && env \| grep COURSE_` |
| update the repo and the environment | `git pull upstream main` (or `git pull` in a plain clone), then `uv sync`; re-run `./scripts/init_hw.sh` after a release | 
| upload an offline W&B run | `uv run wandb sync $COURSE_WORK_DIR/$USER/runs/wandb/offline-run-*` |

## 1. Sessions and connections

**Login nodes.** `login.delta.ncsa.illinois.edu` round-robins to `dt-login01`
… `dt-login04`. Your prompt and `hostname` show which one you got. Anything
that lives in a shell (a `tmux` session, a running `play`, an interactive
GPU allocation) lives on *that* node, so to get back to it you must connect
to the same node by name:

```bash
ssh <you>@dt-login03.delta.ncsa.illinois.edu
```

**tmux keeps things alive.** SSH sessions drop (laptop sleep, Wi-Fi change,
Duo timeouts). Anything started directly in an SSH shell dies with it,
including an interactive GPU session. Run long-lived things inside `tmux`:

```bash
tmux new -s work          # start a session named "work" (note the login node!)
# ... work ...
# Ctrl-b then d           # detach; everything keeps running
tmux ls                   # what sessions exist on this login node
tmux attach -t work       # come back
```

Batch jobs submitted with `train.sh` do **not** need tmux: Slurm runs them
regardless of your session.

**Port forwarding for viser.** `play` serves on port 8080 of the login node
where it runs. Forward it from your laptop, in the same SSH connection or
one to the same `dt-login0N`:

```bash
ssh -L 8080:localhost:8080 <you>@dt-login03.delta.ncsa.illinois.edu
```

then open <http://localhost:8080>. (Full walkthrough with the three failure
modes: *"Port forwarding to viser"* in `docs/01_workflow.md`. For a
**compute node** in a GPU session, don't build the command yourself —
`./scripts/gpu_interactive.sh` prints the ready-made `-J` jump command.)

**Blank page or connection refused?**
Run `hostname` in the terminal where `play` runs and in the one that
carries the forward. If they differ, the forward points at the wrong login
node: use one terminal for both, or connect the second one with
`ssh dt-login0N` (the per-node block in `00_delta_setup.md` A.3). Test the
tunnel from the laptop with `curl -s localhost:8080 | head -3`. If SSH
printed `bind: Address already in use` at connect time, the local port is
taken: forward a different one, `-L 8081:localhost:8080`, and open
`localhost:8081`. **If port 8080 was taken on the login node** (often
another student's `play` — login nodes are shared), viser silently binds
the next free port and prints it: read the actual port from `play`'s
output and point the tunnel's SECOND number at it,
`-L 8080:localhost:<that port>`. If the squatter is an older `play` of
your own: `pkill -u $USER -f "bin/play"`.

**Password + Duo every time.** SSH keys are disabled on Delta for regular
users. Use the `Host delta` block in `~/.ssh/config` to save typing, and
`tmux` to avoid logging in more often than needed.

**Copy files to and from your laptop.** From the laptop:

```bash
scp delta:cs498/hw0/reward_curves.png .
rsync -av "delta:/work/hdd/bign/$USER/runs/hw0_cartpole/<run>/" ./run/    # a whole run
scp ./writeup.md delta:cs498/hw0/
```

For anything large, Delta's Globus endpoint is in the data-management docs.

## 2. Jobs

**Submit.** Always through the wrapper; it adds the locked account,
partition, GPU, memory, wall-time and log path:

```bash
./scripts/train.sh Course-Cartpole-Swingup --env.scene.num-envs 4096
```

It prints `submitted job <id>`. Plain `sbatch scripts/train.sbatch` fails on
purpose (no account).

**Queue.** `squeue --me` shows state `PD` (pending), `R` (running), `CG`
(completing). The `NODELIST(REASON)` column is the node, or why it waits:

| Reason | Meaning | What to do |
|---|---|---|
| `Priority` | others are ahead of you | wait; `squeue --me --start` estimates the start |
| `Resources` | no free A40 right now | wait |
| `QOSGrpBillingMinutes` / `AssocGrpBillingMinutes` | the course allocation is out of balance | tell staff |
| `MaxGRESPerAccount` | the account's GPU cap is reached | wait for classmates' jobs |
| `QOSMaxJobsPerUserLimit` | interactive partition: 1 running / 2 queued per user | finish or cancel the other one |
| `ReqNodeNotAvail` / `Reservation` | maintenance window | check Delta's status page |

`./scripts/my_jobs.sh` prints active jobs, recent history and a verdict.

**Log.** Each job writes `$COURSE_WORK_DIR/$USER/slurm-<jobid>.out`. It
starts with the node, account, partition and `nvidia-smi`, then the W&B run
link, then one block per iteration. `tail -f` it; `grep "Steps per second"`
for Q4; `grep -i -E "error|traceback"` when something looks wrong.

**Cancel.** `scancel <jobid>`, or `./scripts/kill_my_jobs.sh` to cancel all
of yours after confirmation. Saved checkpoints are kept. Cancel a job you
know is bad immediately; the class pays for the wall-time.

**History and exit codes.** After a job ends, `squeue` no longer shows it:

```bash
sacct -X --user $USER --starttime today --format=JobID,JobName%20,Partition,State,Elapsed,ExitCode
./scripts/my_jobs.sh 2d          # the same for the last two days, with log paths for failures
```

`State=FAILED` with a non-zero exit code: read the end of the log.
`TIMEOUT`: the wall-time cap hit; the run's checkpoints are still there.
`OUT_OF_MEMORY`: ask staff before changing `--mem`.

**Cost.** `./scripts/my_usage.sh` totals your GPU-hours and shows the class
usage; `accounts` shows the allocation balance; `jobcharge -a
bign-delta-gpu -u $USER` lists your charges in service units.

## 3. Getting (back) onto a GPU node

There are three situations.

**A. You started an interactive session inside tmux and got disconnected.**
The allocation is still alive. Reconnect to the same login node and
reattach:

```bash
ssh <you>@dt-login02.delta.ncsa.illinois.edu    # the node where you ran tmux
tmux attach -t work                              # your srun shell is still there
```

**B. You started an interactive session *without* tmux and got
disconnected.** The shell died and Slurm cancelled the allocation with it.
`squeue --me` shows nothing. Start a new one, this time inside tmux:

```bash
tmux new -s gpu
./scripts/gpu_interactive.sh 45
```

Delta's interactive partition allows one running interactive job per user
and one hour at most; queued interactive requests beyond two are refused.

**C. A batch job is running and you want to look at its GPU.** Delta allows
`ssh` into a compute node while you have a job running on it. The helper
finds the node of your running job and connects:

```bash
./scripts/gpu_attach.sh              # your (only) running job
./scripts/gpu_attach.sh 1234567      # a specific job id
```

Equivalent by hand: `squeue --me` → the node in `NODELIST` → `ssh gpuaNNN`.
On the node:

```bash
nvidia-smi                                    # is the GPU busy? memory used?
top -u $USER                                  # your processes
tail -f $COURSE_WORK_DIR/$USER/slurm-<id>.out
```

Typing `exit` leaves the job running. Processes you start there belong to
the job and die when it ends, so use the node for monitoring, not for
starting new training (that goes through `train.sh`, which is what the
accounting and the W&B verification are based on). The same GPU usage is
also on the run's W&B page under *System*.

**Viser from a compute node.** Compute nodes are not reachable from your
laptop directly; hop through a login node:

```bash
ssh -J <you>@login.delta.ncsa.illinois.edu <you>@gpuaNNN -L 8080:localhost:8080
```

`gpu_interactive.sh` and `gpu_attach.sh` print this command with the right
node name.

## 4. Runs, checkpoints, resuming

Runs live under `$COURSE_WORK_DIR/$USER/runs/<experiment>/<timestamp>/`
with `model_<iter>.pt`, `params/env.yaml`, `params/agent.yaml`, the event
file and a `git/` diff of your repo at launch. W&B's local files are next to
them under `runs/wandb/`.

```bash
./scripts/latest_checkpoint.sh $COURSE_WORK_DIR/$USER/runs              # newest checkpoint + play command
./scripts/latest_checkpoint.sh $COURSE_WORK_DIR/$USER/runs hw0_cartpole # one experiment only
ls $COURSE_WORK_DIR/$USER/runs/hw0_cartpole/                             # all runs of that experiment
```

**Play a checkpoint** (login node, no GPU):

```bash
uv run play Course-Cartpole-Swingup --checkpoint-file <run>/model_499.pt --log-root $COURSE_WORK_DIR/$USER/runs
uv run play Course-Cartpole-Swingup --wandb-run-path cs498rt-26fall/hw0-booster/<run-id>   # straight from W&B
uv run play Course-Cartpole-Swingup --checkpoint-file <...> --num-envs 4 --video True    # record instead of viewing
```

**Resume a run that timed out or was cancelled** (a new job continues from
the last checkpoint; only for assignments that allow it):

```bash
./scripts/train.sh Course-Cartpole-Swingup --env.scene.num-envs 4096 \
  --agent.resume True --agent.load-run <run directory name> --agent.load-checkpoint model_249.pt
```

**Reward curves** for a writeup, from the event files, no W&B needed:

```bash
uv run python scripts/plot_rewards.py --log-root $COURSE_WORK_DIR/$USER/runs --out hw0/reward_curves.png
```

## 5. Weights & Biases

```bash
uv run wandb login                                              # once; key from https://wandb.ai/authorize
uv run wandb sync $COURSE_WORK_DIR/$USER/runs/wandb/offline-run-<...>   # upload a run made while logged out
```

Runs land in <https://wandb.ai/cs498rt-26fall/hw0-booster>; the run link is
printed at the top of every job log. The *Overview* tab shows the host,
GPU, your username and the Slurm job the run came from; *System* shows GPU
utilisation over time. If a run is missing there, check the log for
`WANDB_MODE=offline` (you were not logged in when the job started) and sync
the folder.

## 6. Storage

```bash
quota                                  # every file system you can write to, with block and file limits
du -sh ~/.cache/uv ~/.cache/warp ~/cs498/.venv 2>/dev/null   # the usual big things in HOME
du -sh $COURSE_WORK_DIR/$USER/runs/*/* | sort -h | tail       # biggest runs
```

- Keep the repo and the venv in `$HOME` (100 GB, 750k files). Runs go to
  `/work/hdd/bign/<you>`; the course scripts do this for you. Never write
  training output into the repo.
- Free space: delete old run directories you have already copied or
  synced; `uv cache clean` (the next `uv sync` re-downloads); W&B offline
  folders after syncing.
- **Deleted something in HOME?** Daily snapshots: `ls ~/.snapshot/`, then
  copy the file back from `~/.snapshot/<snapshot>/…`. `/work` has no
  snapshots.
- Do not `chmod` or share your HOME or run directories; staff read your
  runs through W&B and your submission.

## 7. Keeping the environment healthy

```bash
git status                              # on your own branch, only your five blanks changed?
git pull                                # new handouts and starter code (releases only add files)
uv sync                                 # after every pull; a no-op when nothing changed
uv run python scripts/check_login_env.py
```

- `uv: command not found` after a fresh login: `source ~/.bashrc`, or add
  `~/.local/bin` to `PATH`.
- `uv sync` wants to change `uv.lock`: you edited `pyproject.toml`; `git
  checkout pyproject.toml uv.lock` and sync again.
- Warp kernels are compiled on first use and cached under `~/.cache/warp`;
  the first `play` or job after an update is slower. If kernel compilation
  fails after a driver or module change, delete that cache.
- Never `pip install`, never `conda`, never `module load` a Python: the
  venv brings its own Python, torch and CUDA libraries; only the NVIDIA
  driver comes from the node.
- A job or `play` that worked yesterday and fails today with an import
  error usually means a half-finished `git pull` or `uv sync`; run both again.

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Permission denied (publickey,…)` at login | SSH keys are disabled; your client tried only a key | `ssh -o PreferredAuthentications=keyboard-interactive,password …` |
| Duo prompt never arrives | notification blocked or phone offline | enter the six-digit code from the Duo app instead of `1` |
| `sbatch: error: invalid account` | you bypassed `train.sh` | use `./scripts/train.sh` |
| job pending with `QOSGrpBillingMinutes` | allocation balance exhausted | tell staff; nothing you can do |
| `CUDA driver version is insufficient` in the job log | node driver older than the CUDA build in the venv | tell staff (the lock file must change); do not reinstall torch yourself |
| `torch.cuda.is_available()` is False in a job | no GPU was allocated | submit through `train.sh`; check `nvidia-smi` at the top of the log |
| `WANDB_MODE=offline` in the log | not logged in when the job started | `uv run wandb login`, then `uv run wandb sync <offline run dir>` |
| `wandb: WARNING start_method is deprecated` | harmless | ignore |
| `play` shows nothing in the browser | tunnel to the wrong login node, or port in use | tunnel to the `dt-login0N` where `play` runs; try `-L 8081:localhost:8080` |
| `Disk quota exceeded` | HOME full (often `~/.cache`) or file count hit | `quota`; clean caches and old runs; runs belong on `/work` |
| interactive session vanished after a disconnect | it was not inside tmux | start it again inside `tmux` |
| `squeue` shows my job on a node but `ssh` to it is refused | the job has not started yet, or it ended | `squeue --me`; only running jobs allow node access |
