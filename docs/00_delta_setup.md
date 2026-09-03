# 00 — Getting onto Delta and installing the course environment

This guide assumes **nothing**: no NCSA account, no cluster experience.
Part A gets you from your laptop to a shell on a Delta login node.
Part B installs the course environment there. Do them in order.

Official references, worth a bookmark (the handout will send you back to
them):

- [Delta login methods](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/login.html)
- [Delta data management](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/data_mgmt.html) (file systems, quotas)
- [Delta running jobs](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/running_jobs.html) (partitions, sbatch, srun)
- [Delta job accounting](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/job_accounting.html) (`accounts`, service units)

We use **Delta**, not DeltaAI. They share `/work` and `/projects`, but Delta
is x86_64 with A40/A100 GPUs while DeltaAI is ARM; software built on one
does not run on the other. Everything in this course is built and run on
Delta.

---

# Part A — Get access to Delta

## A.1 Accounts and the course allocation

Getting onto Delta is a **handshake**: you set up your NCSA identity
(steps 1–3, do them today — they are all self-service), then **staff add
you** to the course project (step 4 — you cannot request or join it
yourself). Reference: [NCSA Group/Project Member
Management](https://docs.ncsa.illinois.edu/en/latest/account-mgmt/group-mgmt.html).

1. **Create/activate your NCSA identity** at
   <https://identity.ncsa.illinois.edu>. Sign in with your **University of
   Illinois** credentials (CILogon → "University of Illinois at
   Urbana-Champaign") so your NCSA account is associated with your UIUC
   account — your NCSA username is normally your NetID. Set your NCSA
   (Kerberos) password there; it is **separate from your UIUC password**.
2. **Complete your profile** on that same portal — institution, school,
   department, etc. Do not skip this: staff cannot add an
   incomplete/unfindable profile to the project, and this is the most
   common reason someone is "still not added" days later.
3. **Enroll in NCSA Duo** at <https://duo.security.ncsa.illinois.edu> —
   NCSA Duo is **separate from UIUC Duo**; enrolling your phone again for
   NCSA is expected, and without it SSH will never let you in.
4. **Tell us your NCSA username** (the course announcement says where) and
   wait: a **project leader (course staff) adds you** to the course project
   **`bign`**. Membership can take a few hours to propagate to Delta after
   we add you.
5. **Create a free Weights & Biases account** at <https://wandb.ai> (any
   sign-up method). The course W&B projects are open — no invitation
   needed; you will `wandb login` on the cluster in Part B.6.

What you end up with: an **NCSA username** (usually your NetID), an NCSA
password, **NCSA Duo** enrolled, and membership in **`bign`** — visible in
the `accounts` command and as `/projects/bign` once you can log in. If
`accounts` doesn't show `bign-delta-gpu` a day after staff confirmed the
add, check step 2 first, then ask on the course forum.

## A.2 What Delta looks like

- Login: `ssh <username>@login.delta.ncsa.illinois.edu`. That name
  round-robins to four login nodes, `dt-login01` … `dt-login04`. Your prompt
  shows which one you got.
- Authentication is password + Duo push. **SSH keys are disabled on Delta**
  for regular users; you will type the password and approve Duo every time
  (a `tmux` session on a specific `dt-loginNN` keeps work alive; see the
  login docs).
- Login nodes are shared, x86_64 (AMD EPYC "Milan"), and have **no GPU**.
  They are for editing, building, submitting jobs, and light evaluation.
- GPUs live on compute nodes and are reached only through Slurm (Part B of
  `01_workflow.md`). The course uses the 4×A40 nodes.
- Open OnDemand (a browser desktop / VS Code / Jupyter on Delta) is an
  alternative to SSH for a quick "does my account work?" check; the link is
  in the login docs.

## A.3 First SSH connection

From your laptop's terminal (macOS/Linux: built in; Windows: Windows
Terminal + OpenSSH, or WSL):

```bash
ssh <your_ncsa_username>@login.delta.ncsa.illinois.edu
```

Enter your NCSA password, approve the Duo push (type `1`), and you should
land in a shell whose prompt looks like `<you>@dt-login0X`. Every future
instruction that says "on the login node" means *here*.

Add this to `~/.ssh/config` **on your laptop** — you'll need the port
forwarding constantly for the 3D viewer (see `01_workflow.md`):

```
Host delta
    HostName login.delta.ncsa.illinois.edu
    User <your_ncsa_username>
    LocalForward 8080 localhost:8080

Host dt-login01 dt-login02 dt-login03 dt-login04
    HostName %h.delta.ncsa.illinois.edu
    User <your_ncsa_username>
    LocalForward 8080 localhost:8080
```

After that, `ssh delta` both logs you in and forwards the viewer port, and
`ssh dt-login03` does the same to one specific login node. You need the
second form whenever you use two terminals: `delta` round-robins over four
nodes, and the forward reaches port 8080 only on the node *that* session
landed on. Run `hostname` where `play` runs, and connect to that node.

## A.4 Know where your files live

| Location | Quota (per Delta docs; check with `quota`) | Use for | Do NOT use for |
|---|---|---|---|
| `$HOME` = `/u/<you>` | 100 GB, 750k files; daily snapshots; never purged | this repo, your `uv` environment, dotfiles | job outputs (slow, small) |
| `/projects/bign` | 1000 GB, 1.7M files; shared by the whole project | staff files: the locked settings, shared data | your own files |
| `/work/hdd/bign` | 1000 GB, 2.55M files; the "scratch" file system for job I/O | **your training runs**: `/work/hdd/bign/<you>/runs` | — |
| `/work/nvme/bign` | 500 GB; fast small-file I/O | not needed in this course | — |
| `/tmp` on a compute node | local SSD, wiped after the job | scratch inside a job | anything you want to keep |

You don't need to memorize this: the course scripts put job outputs in the
right place automatically. HW0 Step 2 has you read these numbers off the
cluster yourself.

---

# Part B — Install the course environment (on the login node)

Everything below happens **on the login node**, in the SSH session from
A.3. Nothing is installed on your laptop.

## B.1 Clone the repo

```bash
git clone https://github.com/parasollab/cs498rt_booster_student.git ~/cs498
cd ~/cs498
```

## B.2 Install uv

`uv` is the Python package/environment manager the course standardizes on
([docs](https://docs.astral.sh/uv/)):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then restart your shell (or `source ~/.bashrc`), and confirm `uv --version`
prints something.

## B.3 Sync the environment

```bash
uv sync
```

This reads `pyproject.toml` + `uv.lock` and builds, in `.venv/` inside the
repo, the exact environment the course and the grader use — about 3 GB of
downloads, a few minutes ([what `uv sync` does](https://docs.astral.sh/uv/concepts/projects/sync/)).
It includes PyTorch and Warp with their own CUDA libraries; only the NVIDIA
*driver* comes from the compute node.

Rules that keep everyone's environment identical:

- Never `pip install` into this project, never conda, never edit
  `pyproject.toml` or `uv.lock` unless an assignment explicitly says to.
- If `uv sync` proposes changing `uv.lock`, you edited something you shouldn't.
- The environment lives in `.venv/` inside the repo on your `$HOME`, which
  every compute node also mounts, so the same `uv run ...` commands work on
  login *and* GPU nodes. You sync once, here, and never on a GPU node.

## B.4 Verify

```bash
uv run python scripts/check_login_env.py
```

Expected: every line `[PASS]`, including
`CUDA unavailable (expected on a login node)`. Common first-run issues:

- `uv: command not found` → B.2's PATH step; restart your shell.
- mjlab import errors → you're not in `~/cs498`, or sync didn't finish;
  rerun `uv sync` and read its last lines.
- `Course-* tasks registered: none yet` is **normal** until HW0 Step 5.

## B.5 The locked course settings

Account, partitions, paths and the W&B team are fixed by staff and loaded
by every script. Look at them once:

```bash
source scripts/cluster.env
env | grep COURSE_
echo "$COURSE_ENV_SOURCE"        # on Delta: /projects/bign/cs498/hw0.env (per-assignment)
```

Then create your run directory on the scratch file system:

```bash
mkdir -p "$COURSE_WORK_DIR/$USER/runs"
```

## B.6 Weights & Biases

Training runs are logged to the course W&B team **`cs498rt-26fall`** —
that record is how we verify your run. The team's per-assignment projects
are **open**: anyone logged in to W&B can submit runs, so there is **no
invitation to wait for**. You need exactly two things: a free account at
<https://wandb.ai> (Day 0 — see A.1), and, once, on the login node:

```bash
uv run wandb login          # paste the key from https://wandb.ai/authorize
```

The key is stored in `~/.netrc`, which batch jobs read.
([W&B quickstart](https://docs.wandb.ai/quickstart))

## B.7 Where to go next

Read `docs/01_workflow.md` (how to actually *use* the cluster and see the
3D viewer in your browser), then start `hw0/README.md`.
