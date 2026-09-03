"""HW0 pipeline checklist — your progress report, and what we grade.

  uv run python scripts/hw0_checklist.py                                       # laptop / login node
  uv run python scripts/hw0_checklist.py --log-root $COURSE_WORK_DIR/$USER/runs   # on Delta (default there)
  uv run python scripts/hw0_checklist.py --no-rollout                          # skip the checkpoint smoke test
  uv run python scripts/hw0_checklist.py --online                              # also ask wandb.ai who you are

Writes hw0/hw0_checklist.md (+ .json), prints it, and exits 0 when nothing
required is left. Every line is one of

  [PASS]  done
  [TODO]  not done yet — the line tells you the command to run next
  [INFO]  information only; it never affects your grade (for example how well
          the checkpoint balances — HW0 grades the pipeline, not the policy)

Nothing here needs a GPU: the checkpoint smoke test steps 16 environments on
the CPU for five seconds. Run it whenever you like; submit the final one.
"""

from __future__ import annotations

import argparse
import json
import math
import netrc
import os
import platform
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PASS, TODO, INFO = "PASS", "TODO", "INFO"

# The one graded training run: (experiment_name, variant) -> (task id, what it is)
EXPECTED_RUN = (("hw0_cartpole", "Swingup"), ("Course-Cartpole-Swingup", "Step 6: cartpole swingup"))
MAX_ITERATIONS = 500

# Step 2's cluster-discovery answers are a Gradescope quiz since 2026-09
# (autograded there); the checklist no longer compares a local answers file.


class Report:
  def __init__(self):
    self.sections: dict[str, list[dict]] = {}
    self.current = None

  def section(self, name: str):
    self.current = name
    self.sections.setdefault(name, [])

  def add(self, status: str, text: str, required: bool = True, **extra):
    self.sections[self.current].append({"status": status, "text": text, "required": required, **extra})

  @property
  def required_todo(self) -> list[str]:
    return [i["text"] for items in self.sections.values() for i in items if i["status"] == TODO and i["required"]]

  def markdown(self, header: str) -> str:
    box = {PASS: "[x]", TODO: "[ ]", INFO: "(i)"}
    out = ["# HW0 pipeline checklist", "", header, ""]
    for name, items in self.sections.items():
      out.append(f"## {name}")
      for i in items:
        out.append(f"- {box[i['status']]} **{i['status']}** {i['text']}")
      out.append("")
    todo = self.required_todo
    out.append("## Summary")
    out.append(f"- {'READY TO SUBMIT' if not todo else f'{len(todo)} required item(s) left'}")
    for t in todo:
      out.append(f"  - {t}")
    return "\n".join(out) + "\n"


def sh(cmd: list[str], **kw) -> str:
  try:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=60, **kw).stdout.strip()
  except Exception:  # noqa: BLE001
    return ""


def cluster_env(key: str) -> str | None:
  """Value of a COURSE_* setting as scripts/cluster.env resolves it (master copy on Delta, else repo copy)."""
  if key in os.environ:
    return os.environ[key]
  f = REPO / "scripts" / "cluster.env"
  return (sh(["bash", "-c", f"source '{f}' && printf %s \"${key}\""]) or None) if f.exists() else None


def on_delta() -> bool:
  host = socket.gethostname()
  if re.match(r"^(dt-login|gpu[a-z]|cn)\d+", host) or ".delta.ncsa.illinois.edu" in socket.getfqdn():
    return True
  d = cluster_env("COURSE_PROJECT_DIR")
  return bool(d and Path(d).is_dir())


def read_env_file(path: Path) -> dict[str, str]:
  out = {}
  if not path.exists():
    return out
  for line in path.read_text(errors="ignore").splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
      continue
    k, v = line.split("=", 1)
    v = v.split("#", 1)[0].strip() if not v.strip().startswith(('"', "'")) else v.strip()
    if v[:1] in ('"', "'"):
      q = v[0]
      v = v[1:].split(q, 1)[0]
    out[k.strip().removeprefix("export ").strip()] = v.strip()
  return out


def load_params_yaml(text: str):
  """mjlab dumps configs with !!python/* tags; read them without unsafe_load."""
  import yaml

  class Loader(yaml.SafeLoader):
    pass

  def any_python(loader, suffix, node):
    if isinstance(node, yaml.SequenceNode):
      return loader.construct_sequence(node, deep=True)
    if isinstance(node, yaml.MappingNode):
      return loader.construct_mapping(node, deep=True)
    return f"python/{suffix}:{loader.construct_scalar(node)}"

  Loader.add_constructor("tag:yaml.org,2002:python/tuple",
                         lambda loader, node: tuple(loader.construct_sequence(node, deep=True)))
  Loader.add_multi_constructor("tag:yaml.org,2002:python/", any_python)
  return yaml.load(text, Loader=Loader)


# --------------------------------------------------------------- sections
def section_environment(r: Report) -> bool:
  r.section("1. Environment (Step 1)")
  res = subprocess.run([sys.executable, str(REPO / "scripts" / "check_login_env.py")], capture_output=True, text=True)
  n = 0
  for line in res.stdout.splitlines():
    if line.startswith("[PASS]"):
      n += 1
    elif line.startswith("[FAIL]"):
      r.add(TODO, f"{line[7:]} — fix this first (docs/00_delta_setup.md, Part B)")
  if n:
    r.add(PASS, f"check_login_env.py: {n} checks pass (python {platform.python_version()}, {platform.machine()})")
  if res.returncode != 0 and n == 0:
    r.add(TODO, "check_login_env.py could not run: `uv sync`, then try again")
  cluster = on_delta()
  r.add(INFO, f"running on {socket.gethostname()} ({'Delta' if cluster else 'not Delta — fine for local work'}); "
              f"settings from {cluster_env('COURSE_ENV_SOURCE') or 'scripts/hw0.env'}", required=False)
  return cluster


def section_cluster_answers(r: Report, cluster: bool):
  r.section("2. Cluster discovery (Step 2 — answered on Gradescope)")
  legacy = REPO / "hw0" / "my_cluster.env"
  if legacy.exists():
    r.add(INFO, "hw0/my_cluster.env is no longer used — the Step 2 answers are the "
                "Gradescope quiz's cluster part (autograded there); you can delete the file",
          required=False)
  else:
    r.add(INFO, "answered on Gradescope (multiple choice) — nothing checked locally",
          required=False)


def section_wandb(r: Report, online: bool):
  r.section("3. Weights & Biases (the proof of your run)")
  try:
    import wandb

    v = wandb.__version__
    ok = tuple(int(x) for x in v.split(".")[:2]) < (0, 29)
    r.add(PASS if ok else TODO, f"wandb {v} installed" + ("" if ok else " — must be < 0.29 (rsl_rl crashes on 0.29): run `uv sync`"))
  except Exception as e:  # noqa: BLE001
    r.add(TODO, f"wandb not importable ({e!r}): run `uv sync`")
    return
  logged_in = bool(os.environ.get("WANDB_API_KEY"))
  try:
    auth = netrc.netrc().authenticators("api.wandb.ai")
    logged_in = logged_in or auth is not None
  except (FileNotFoundError, netrc.NetrcParseError):
    pass
  if logged_in:
    r.add(PASS, "logged in (API key in ~/.netrc or WANDB_API_KEY): training runs upload to wandb.ai")
  else:
    r.add(TODO, "not logged in: run `uv run wandb login` once (paste the key from https://wandb.ai/authorize). "
                "Until then runs are logged offline and must be uploaded with `uv run wandb sync`.")
  entity = cluster_env("COURSE_WANDB_ENTITY") or ""
  project = cluster_env("COURSE_WANDB_PROJECT") or "?"
  try:
    from hw0.cartpole_env_cfg import cartpole_ppo_runner_cfg

    cfg_project = cartpole_ppo_runner_cfg().wandb_project
  except Exception:  # noqa: BLE001
    cfg_project = "?"
  if cfg_project != project:
    r.add(TODO, f"PPO config logs to project `{cfg_project}` but the course project is `{project}` — restore wandb_project")
  r.add(INFO, f"runs go to https://wandb.ai/{entity}/{project} (you must be a member of the team `{entity}`)", required=False)
  if online and logged_in:
    try:
      api = wandb.Api(timeout=20)
      me = api.viewer
      teams = [t for t in (getattr(me, "teams", None) or [])]
      in_team = (entity in teams) if teams else None
      r.add(PASS, f"wandb.ai reachable; you are `{me.username}`" + ("" if in_team is None else
            (f", member of `{entity}`" if in_team else f" — NOT a member of `{entity}`: accept the team invite")))
    except Exception as e:  # noqa: BLE001
      r.add(INFO, f"could not reach wandb.ai right now ({type(e).__name__}); not a problem offline", required=False)


def section_code(r: Report) -> bool:
  r.section("4. The five blanks (Step 5)")
  res = subprocess.run([sys.executable, str(REPO / "scripts" / "check_hw0.py")], capture_output=True, text=True, cwd=REPO)
  lines = [ln for ln in res.stdout.splitlines() if ln.startswith(("[PASS]", "[FAIL]", "[TODO]"))]
  n_pass = sum(ln.startswith("[PASS]") for ln in lines)
  for ln in lines:
    if ln.startswith("[TODO]"):
      r.add(TODO, ln[7:] + " (hw0/cartpole_env_cfg.py)")
    elif ln.startswith("[FAIL]"):
      r.add(TODO, f"check failed: {ln[7:]}")
  if n_pass and res.returncode == 0:
    r.add(PASS, f"check_hw0.py: all {n_pass} checks pass")
  elif n_pass:
    r.add(INFO, f"check_hw0.py: {n_pass} checks pass so far", required=False)
  elif not lines:
    r.add(TODO, "check_hw0.py could not run: " + (res.stderr.strip().splitlines() or ["?"])[-1][:160])
  return res.returncode == 0


def find_runs(log_root: Path) -> list[dict]:
  runs = []
  for exp in sorted(p for p in log_root.iterdir() if p.is_dir() and p.name != "wandb"):
    for run in sorted(p for p in exp.iterdir() if p.is_dir()):
      env_yaml = run / "params" / "env.yaml"
      if not env_yaml.exists():
        continue
      variant, num_envs = None, None
      try:
        cfg = load_params_yaml(env_yaml.read_text())
        num_envs = cfg["scene"]["num_envs"]
        h1 = next(iter(cfg["scene"]["entities"].values()))["init_state"]["joint_pos"].get("hinge_1", 0.0)
        variant = "Swingup" if abs(float(h1) - math.pi) < 0.5 else "Balance"
      except Exception:  # noqa: BLE001
        pass
      ckpts = sorted(run.glob("model_*.pt"), key=lambda p: int(re.findall(r"\d+", p.name)[0]))
      events = list(run.glob("events.out.tfevents.*"))
      runs.append({"dir": run, "experiment": exp.name, "variant": variant, "num_envs": num_envs,
                   "checkpoints": ckpts, "events": bool(events)})
  return runs


def wandb_runs(*roots: Path) -> dict[str, dict]:
  """Map training run-dir name -> {'mode': online|offline, 'id': ..., 'dir': ...} from W&B run folders.

  rsl_rl's writer calls wandb.save on every checkpoint, which leaves
  files/model_<iter>.pt symlinks pointing into the training run directory —
  present for online AND offline runs. Online runs also have files/config.yaml
  with the log_dir; used as a fallback."""
  out = {}
  for root in roots:
    wdir = root / "wandb"
    for d in sorted(wdir.glob("*run-*")) if wdir.is_dir() else []:
      log_dir = None
      for link in (d / "files").glob("model_*.pt"):
        try:
          log_dir = Path(os.readlink(link)).parent
          break
        except OSError:
          continue
      if log_dir is None:
        cfg = d / "files" / "config.yaml"
        if cfg.exists():
          m = re.search(r"log_dir:\s*\n\s*(?:desc: .*\n\s*)?value:\s*(.+)", cfg.read_text(errors="ignore"))
          log_dir = Path(m.group(1).strip()) if m else None
      if log_dir is None:
        continue
      out[log_dir.name] = {"mode": "offline" if d.name.startswith("offline") else "online",
                           "id": d.name.split("-")[-1], "dir": d}
  return out


def section_runs(r: Report, log_root: Path, rollout: bool, code_ok: bool):
  r.section("5. The training run (Step 6)")
  (exp, variant), (task, what) = EXPECTED_RUN
  if not log_root.is_dir():
    r.add(TODO, f"no runs yet under {log_root}: train with `./scripts/train.sh {task}`")
    return
  runs = find_runs(log_root)
  wb = wandb_runs(log_root, REPO)
  matches = [x for x in runs if x["experiment"] == exp and x["variant"] == variant]
  others = [x for x in runs if x not in matches]
  if others:
    r.add(INFO, "other runs found (not graded): " + ", ".join(f"{x['experiment']}/{x['dir'].name} ({x['variant']})" for x in others),
          required=False)
  if not matches:
    r.add(TODO, f"{what}: no run found — `./scripts/train.sh {task}`")
    return
  run = matches[-1]
  if not run["checkpoints"]:
    r.add(TODO, f"{what}: run {run['dir'].name} has no model_*.pt yet — still running, or it crashed (check the slurm log)")
    return
  last_it = int(re.findall(r"\d+", run["checkpoints"][-1].name)[0]) + 1
  r.add(PASS, f"{what}: run `{exp}/{run['dir'].name}` ({run['num_envs']} envs), {len(run['checkpoints'])} checkpoints, "
              f"last iteration {last_it}, {'event file present' if run['events'] else 'NO event file'}")
  if last_it < MAX_ITERATIONS:
    r.add(INFO, f"  stopped at iteration {last_it} of {MAX_ITERATIONS} (fine for the pipeline; say so in the writeup)", required=False)
  w = wb.get(run["dir"].name)
  if w and w["mode"] == "online":
    r.add(PASS, f"  logged to W&B online, run id `{w['id']}` — put the run link in your writeup")
  elif w:
    r.add(PASS, f"  logged to W&B offline, run id `{w['id']}` — upload with `uv run wandb sync {w['dir']}` after `uv run wandb login`, "
                "then put the run link in your writeup")
  else:
    r.add(TODO, "  no W&B record found for this run: log in (`uv run wandb login`) and train again through the course script "
                "(it sets WANDB_DIR to the log root)")
  if rollout and code_ok:
    try:
      m = smoke_rollout(task, run["checkpoints"][-1])
      r.add(PASS, f"  checkpoint `{run['checkpoints'][-1].name}` loads and runs on CPU (16 envs, 5 s)")
      r.add(INFO, f"  policy quality (not graded): {m}", required=False)
    except Exception as e:  # noqa: BLE001
      r.add(TODO, f"  checkpoint `{run['checkpoints'][-1].name}` does not load into the current config "
                  f"({type(e).__name__}: {str(e)[:120]}) — if you changed observations after training, train again")
  elif rollout:
    r.add(INFO, "  checkpoint smoke test skipped until the code checks pass", required=False)


def smoke_rollout(task: str, ckpt: Path, num_envs: int = 16, seconds: float = 5.0) -> str:
  from dataclasses import asdict

  import torch

  import mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  cfg = load_env_cfg(task, play=True)
  cfg.scene.num_envs = num_envs
  cfg.seed = 0
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  wrapped = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(task).clip_actions)
  runner = MjlabOnPolicyRunner(wrapped, asdict(load_rl_cfg(task)), device="cpu")
  runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location="cpu")
  policy = runner.get_inference_policy(device="cpu")
  asset = env.scene["cartpole"]
  h1 = list(asset.joint_names).index("hinge_1")
  obs = wrapped.get_observations()
  obs = obs[0] if isinstance(obs, tuple) else obs
  up_last = 0.0
  first_up = [None] * num_envs
  steps = int(seconds / env.step_dt)
  for s in range(steps):
    with torch.no_grad():
      obs, _, _, _ = wrapped.step(policy(obs))
    up = torch.cos(asset.data.joint_pos[:, h1]) > math.cos(0.3)
    for i in torch.nonzero(up).flatten().tolist():
      first_up[i] = first_up[i] or (s + 1) * env.step_dt
    up_last = up.float().mean().item()
  env.close()
  reached = [t for t in first_up if t]
  return (f"{up_last * 100:.0f}% of envs upright at t={seconds:.0f}s; "
          f"{len(reached)}/{num_envs} reached upright" + (f", median after {sorted(reached)[len(reached) // 2]:.1f}s" if reached else ""))


def section_deliverables(r: Report, log_root: Path):
  r.section("6. Writeup and figure (Step 7)")
  fig = REPO / "hw0" / "reward_curves.png"
  if fig.exists():
    r.add(PASS, f"reward-curve figure: {fig.relative_to(REPO)}")
  else:
    r.add(TODO, f"no figure yet: `uv run python scripts/plot_rewards.py --log-root {log_root} --out hw0/reward_curves.png`")
  writeups = [p for p in (REPO / "hw0").glob("writeup.*") if p.suffix in (".md", ".txt", ".docx", ".doc", ".pdf")]
  if writeups:
    r.add(PASS, f"writeup: {writeups[0].relative_to(REPO)}")
  else:
    r.add(TODO, "no writeup yet: hw0/writeup.md (or .txt/.docx/.pdf), at most 2 pages — see hw0/README.md Step 7")


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--log-root", default=None, help="where `train` wrote runs (default: $COURSE_WORK_DIR/$USER/runs if it exists, else logs/rsl_rl)")
  ap.add_argument("--no-rollout", action="store_true")
  ap.add_argument("--online", action="store_true", help="also verify the W&B login and team membership against wandb.ai")
  ap.add_argument("--out", default=str(REPO / "hw0" / "hw0_checklist.md"))
  args = ap.parse_args()
  if args.log_root:
    log_root = Path(args.log_root)
  else:
    work = cluster_env("COURSE_WORK_DIR")
    user = os.environ.get("USER") or os.environ.get("USERNAME") or ""
    cluster_runs = Path(work) / user / "runs" if work and user else None
    log_root = cluster_runs if cluster_runs and cluster_runs.is_dir() else REPO / "logs" / "rsl_rl"

  r = Report()
  cluster = section_environment(r)
  section_cluster_answers(r, cluster)
  section_wandb(r, args.online)
  code_ok = section_code(r)
  section_runs(r, log_root, not args.no_rollout, code_ok)
  section_deliverables(r, log_root)

  header = (f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} on `{socket.gethostname()}` "
            f"({platform.machine()}), runs from `{log_root}`. Re-run: `uv run python scripts/hw0_checklist.py`.")
  md = r.markdown(header)
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(md, encoding="utf-8")
  out.with_suffix(".json").write_text(json.dumps({
    "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "hostname": socket.gethostname(), "arch": platform.machine(),
    "log_root": str(log_root), "course_env_version": cluster_env("COURSE_ENV_VERSION"),
    "sections": {k: [{kk: (str(vv) if isinstance(vv, Path) else vv) for kk, vv in i.items()} for i in v]
                 for k, v in r.sections.items()},
    "required_todo": r.required_todo, "ready": not r.required_todo,
  }, indent=1), encoding="utf-8")
  print(md)
  print(f"(written to {out} and {out.with_suffix('.json')})")
  return 0 if not r.required_todo else 1


if __name__ == "__main__":
  raise SystemExit(main())
