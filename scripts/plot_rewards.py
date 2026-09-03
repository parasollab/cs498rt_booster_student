"""Make the reward-curve figure for your writeup from your run directories.

  uv run python scripts/plot_rewards.py                                    # logs/rsl_rl -> reward_curves.png
  uv run python scripts/plot_rewards.py --log-root $COURSE_WORK_DIR/$USER/runs --out hw0/reward_curves.png
  uv run python scripts/plot_rewards.py --experiment hw0_cartpole          # only the cartpole runs
  uv run python scripts/plot_rewards.py --runs 2026-09-02_05-40-56          # pick runs by directory name
  uv run python scripts/plot_rewards.py --tag Episode_Reward/smooth_reward --tag Train/mean_reward --smooth 20

Works offline. Every run — W&B online, W&B offline (`WANDB_MODE=offline`), or
the plain TensorBoard logger — writes the same `events.out.tfevents.*` file
into `<log_root>/<experiment>/<run>/`, and that is what this reads. It also
writes a CSV next to the PNG with the final value and the first iteration at
which each curve reached 0.9, so you can quote numbers instead of eyeballing.

Copy the PNG into your writeup (`![reward curves](reward_curves.png)`), or
`scp` it from the cluster:  scp delta:cs498/hw0/reward_curves.png .
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path

DEFAULT_TAG = "Episode_Reward/smooth_reward"  # per-step reward, mean over finished episodes


def run_label(run_dir: Path) -> str:
  """'<experiment>/<run> · Swingup · 4096 envs' from params/env.yaml (best effort)."""
  label = f"{run_dir.parent.name}/{run_dir.name}"
  env_yaml = run_dir / "params" / "env.yaml"
  if not env_yaml.exists():
    return label
  text = env_yaml.read_text(errors="ignore")
  variant, num_envs = None, None
  try:
    import yaml

    cfg = yaml.safe_load(text)
    num_envs = cfg["scene"]["num_envs"]
    entity = next(iter(cfg["scene"]["entities"].values()))
    hinge1 = entity["init_state"]["joint_pos"].get("hinge_1")
    if hinge1 is not None:
      variant = "Swingup" if abs(float(hinge1) - math.pi) < 0.5 else "Balance"
  except Exception:  # noqa: BLE001  (fall back to regexes)
    m = re.search(r"^\s*num_envs:\s*(\d+)", text, re.M)
    num_envs = int(m.group(1)) if m else None
    m = re.search(r"hinge_1:\s*([-\d.eE+]+)", text)
    if m:
      variant = "Swingup" if abs(float(m.group(1)) - math.pi) < 0.5 else "Balance"
  extras = [x for x in (variant, f"{num_envs} envs" if num_envs else None) if x]
  return label + (" · " + " · ".join(extras) if extras else "")


def load_scalars(run_dir: Path, tags: list[str]) -> dict[str, tuple[list[int], list[float]]]:
  from tensorboard.backend.event_processing import event_accumulator

  ea = event_accumulator.EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
  ea.Reload()
  available = set(ea.Tags()["scalars"])
  out = {}
  for tag in tags:
    if tag in available:
      pts = ea.Scalars(tag)
      out[tag] = ([p.step for p in pts], [p.value for p in pts])
  return out


def smooth(y: list[float], k: int) -> list[float]:
  if k <= 1 or len(y) < k:
    return y
  out, acc = [], 0.0
  for i, v in enumerate(y):
    acc += v
    if i >= k:
      acc -= y[i - k]
    out.append(acc / min(i + 1, k))
  return out


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--log-root", default="logs/rsl_rl", help="where train wrote runs (--log-root of `uv run train`)")
  ap.add_argument("--experiment", help="only this experiment directory, e.g. hw0_cartpole")
  ap.add_argument("--runs", nargs="*", help="only these run directory names (timestamps)")
  ap.add_argument("--tag", action="append", help=f"scalar tag(s) to plot (default {DEFAULT_TAG}); repeatable")
  ap.add_argument("--smooth", type=int, default=1, help="moving-average window in iterations (default 1 = raw)")
  ap.add_argument("--out", default="reward_curves.png")
  ap.add_argument("--title", default=None)
  args = ap.parse_args()
  tags = args.tag or [DEFAULT_TAG]

  root = Path(args.log_root)
  if not root.is_dir():
    print(f"log root not found: {root}", file=sys.stderr)
    return 1
  run_dirs = []
  for exp in sorted(root.iterdir()):
    if not exp.is_dir() or exp.name == "wandb":
      continue
    if args.experiment and exp.name != args.experiment:
      continue
    for run in sorted(exp.iterdir()):
      if run.is_dir() and any(run.glob("events.out.tfevents.*")):
        if args.runs and run.name not in args.runs:
          continue
        run_dirs.append(run)
  if not run_dirs:
    print(f"no runs with event files under {root}", file=sys.stderr)
    return 1

  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  fig, axes = plt.subplots(len(tags), 1, figsize=(8.5, 4.2 * len(tags)), dpi=130, squeeze=False)
  summary = []
  for run in run_dirs:
    data = load_scalars(run, tags)
    label = run_label(run)
    for ax, tag in zip(axes[:, 0], tags):
      if tag not in data:
        continue
      x, y = data[tag]
      ys = smooth(y, args.smooth)
      final = ys[-1]
      hit = next((xi for xi, yi in zip(x, ys) if yi >= 0.9), None) if tag == DEFAULT_TAG else None
      ax.plot(x, ys, linewidth=1.3, label=f"{label} (final {final:.2f})")
      summary.append({"run": f"{run.parent.name}/{run.name}", "label": label, "tag": tag,
                      "iterations": x[-1], "final": f"{final:.3f}", "first_iter_ge_0.9": hit,
                      "max": f"{max(ys):.3f}"})
  for ax, tag in zip(axes[:, 0], tags):
    ax.set_xlabel("PPO iteration")
    ax.set_ylabel(tag)
    if tag == DEFAULT_TAG:
      ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
  axes[0, 0].set_title(args.title or f"HW0 training curves — {root}" + (f" (smooth {args.smooth})" if args.smooth > 1 else ""))
  fig.tight_layout()
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out)

  csv_path = out.with_suffix(".csv")
  with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    w.writeheader()
    w.writerows(summary)
  print(f"wrote {out} and {csv_path}\n")
  print(f"{'run':62s} {'iters':>6s} {'final':>7s} {'>=0.9 at':>9s}  tag")
  for s in summary:
    print(f"{s['label'][:62]:62s} {s['iterations']:6d} {s['final']:>7s} {str(s['first_iter_ge_0.9'] or '-'):>9s}  {s['tag']}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
