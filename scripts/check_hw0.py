"""HW0 self-check for the five cartpole blanks (also the base of our autograder).

  uv run python scripts/check_hw0.py             # CPU, 16 envs — works on the login node
  uv run python scripts/check_hw0.py --num-envs 64

Imports your `hw0.cartpole_env_cfg` by the public names the
handout fixes and checks what you would otherwise eyeball in viser: which
blanks are still empty, that both Course-Cartpole-* tasks register, the
fixed PPO config, observation/action shapes, the initial states, your
(cos, sin) observation, the reward's value with the pole held up / hanging
down, the reset jitter, and that a 16-env batch steps without error under
random actions. Every line must be [PASS] before you train.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

PASS, FAIL, TODO = "[PASS]", "[FAIL]", "[TODO]"
failures: list[str] = []
todos: list[str] = []
results: list[dict] = []  # structured mirror of every check/todo (--json)


def check(name: str, ok: bool, detail: str = "") -> None:
  print(f"{PASS if ok else FAIL} {name}" + (f" — {detail}" if detail else ""))
  results.append({"kind": "check", "name": name, "ok": bool(ok), "detail": detail})
  if not ok:
    failures.append(name)


def todo(text: str) -> None:
  print(f"{TODO} {text}")
  results.append({"kind": "todo", "name": text, "ok": False, "detail": ""})
  todos.append(text)


EXPECTED_PPO = {
  "experiment_name": "hw0_cartpole",
  "wandb_project": "hw0-booster",
  "save_interval": 50,
  "num_steps_per_env": 32,
  "max_iterations": 500,
}
EXPECTED_ALGO = {
  "clip_param": 0.2, "entropy_coef": 0.01, "num_learning_epochs": 5, "num_mini_batches": 4,
  "learning_rate": 1.0e-3, "schedule": "adaptive", "gamma": 0.99, "lam": 0.95,
  "desired_kl": 0.01, "max_grad_norm": 1.0,
}


def main() -> int:
  num_envs = 16
  if "--num-envs" in sys.argv:
    num_envs = int(sys.argv[sys.argv.index("--num-envs") + 1])

  import warnings

  import torch

  with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # the "not registered yet" warning is reported below instead
    import mjlab.tasks  # noqa: F401  (registers Course-* via the entry point)
    try:
      import hw0.cartpole_env_cfg as m
    except ModuleNotFoundError:
      todo("hw0 not initialized: run `./scripts/init_hw.sh` once — it creates "
           "hw0/cartpole_env_cfg.py from the starter; your answers go there")
      return 1
    except Exception as e:  # noqa: BLE001
      check("import hw0.cartpole_env_cfg", False, repr(e))
      return 1
  from mjlab.tasks.registry import list_tasks, load_env_cfg

  # 1. Public API present.
  for fn in ("cartpole_balance_env_cfg", "cartpole_swingup_env_cfg", "cartpole_ppo_runner_cfg",
             "cartpole_smooth_reward", "pole_angle_cos_sin"):
    check(f"public function {fn}", callable(getattr(m, fn, None)))
  if failures:
    return 1

  # 2. The five blanks.
  src = Path(m.__file__).read_text(errors="ignore")
  value_blank = False
  if m._ACTUATED_JOINT is None:
    todo("TODO(1) not done: set _ACTUATED_JOINT (the joint the XML motor drives)")
    value_blank = True
  else:
    check("TODO(1): _ACTUATED_JOINT names the joint the motor drives", m._ACTUATED_JOINT == "slider",
          f"got {m._ACTUATED_JOINT!r}" + ("" if m._ACTUATED_JOINT == "slider" else
          " — hint: 'slide' is the MOTOR's name; look at its joint=\"...\" attribute"))
  if m._SWINGUP_HINGE_ANGLE is None:
    todo("TODO(2) not done: set _SWINGUP_HINGE_ANGLE (radians, pole hanging straight down)")
    value_blank = True
  else:
    ok = abs(abs(float(m._SWINGUP_HINGE_ANGLE)) - math.pi) < 0.05
    check("TODO(2): _SWINGUP_HINGE_ANGLE is straight down (pi radians)", ok, f"got {m._SWINGUP_HINGE_ANGLE}")
  if m._HINGE_RESET_RANGE is None:
    todo("TODO(5) not done: set _HINGE_RESET_RANGE ((low, high) in radians)")
    value_blank = True
  else:
    try:
      lo, hi = (float(x) for x in m._HINGE_RESET_RANGE)
      ok = abs(lo + hi) < 1e-9 and 0.01 <= hi <= 0.2
    except Exception:  # noqa: BLE001
      lo = hi = float("nan")
      ok = False
    check("TODO(5): _HINGE_RESET_RANGE is a small symmetric (low, high), 0.01..0.2 rad", ok, f"got {m._HINGE_RESET_RANGE}")
  # TODO(3)/(4) live between "# your code here" / "# your code ends" markers;
  # the fallback raise below TODO(3)'s block and the `upright = None` line
  # above TODO(4)'s block stay in the file BY DESIGN, so completion means
  # "the marker block contains code", not "the sentinel strings are gone".
  # Files without markers (older forks) fall back to the sentinel strings.
  def _marker_blocks_filled(source: str) -> list[bool]:
    filled, inside, has_code = [], False, False
    for line in source.splitlines():
      t = line.strip()
      if t == "# your code here":
        inside, has_code = True, False
      elif t == "# your code ends":
        if inside:
          filled.append(has_code)
        inside = False
      elif inside and t and not t.startswith("#"):
        has_code = True
    return filled

  blocks = _marker_blocks_filled(src)
  if len(blocks) >= 2:
    if not blocks[0]:
      todo("TODO(3) not done: write the body of pole_angle_cos_sin between its markers")
    if not blocks[1]:
      todo("TODO(4) not done: write `upright` between its markers in cartpole_smooth_reward")
  else:  # marker-less file (older layout): the original sentinel checks
    if 'raise NotImplementedError("TODO(3)' in src:
      todo("TODO(3) not done: write the body of pole_angle_cos_sin")
    if "upright = None  # TODO(4)" in src:
      todo("TODO(4) not done: write `upright` in cartpole_smooth_reward")
  if value_blank:
    print()
    print("The Course-Cartpole-* tasks are not registered until TODO(1), (2) and (5) are filled.")
    print(f"{len(todos)} blank(s) left: fill them and run this again.")
    return 1

  # 3. Registration.
  tasks = list_tasks()
  check("both tasks registered (uv run list-envs)",
        {"Course-Cartpole-Balance", "Course-Cartpole-Swingup"} <= set(tasks),
        ", ".join(t for t in tasks if t.startswith("Course-")) or "no Course-* task found")
  if failures:
    return 1

  # 4. PPO config untouched.
  c = m.cartpole_ppo_runner_cfg()
  bad = [k for k, v in EXPECTED_PPO.items() if getattr(c, k, None) != v]
  bad += [f"algorithm.{k}" for k, v in EXPECTED_ALGO.items() if getattr(c.algorithm, k, None) != v]
  bad += [] if (tuple(c.actor.hidden_dims) == (64, 64) and tuple(c.critic.hidden_dims) == (64, 64)) else ["hidden_dims"]
  check("PPO config is the fixed course config", not bad, "changed: " + ", ".join(bad) if bad else "")

  # 5. Environment structure and the function blanks, per task.
  from mjlab.envs import ManagerBasedRlEnv

  for task, swing in (("Course-Cartpole-Balance", False), ("Course-Cartpole-Swingup", True)):
    cfg = load_env_cfg(task, play=True)
    check(f"{task}: scene entity named 'cartpole'", "cartpole" in cfg.scene.entities)
    check(f"{task}: time_out termination present, no failure termination",
          len(cfg.terminations) == 1 and all(getattr(t, "time_out", False) for t in cfg.terminations.values()))
    check(f"{task}: reward term uses cartpole_smooth_reward",
          any(r.func is m.cartpole_smooth_reward for r in cfg.rewards.values()))
    check(f"{task}: physics 0.01 s x decimation 5, episode 50 s (untouched)",
          cfg.sim.mujoco.timestep == 0.01 and cfg.decimation == 5 and (swing or cfg.episode_length_s > 0))

    cfg.scene.num_envs = num_envs
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
      obs, _ = env.reset()
    except NotImplementedError as e:
      todo(f"{task}: {e}")
      env.close()
      continue
    asset = env.scene["cartpole"]
    names = list(asset.joint_names)
    h1, s0 = names.index("hinge_1"), names.index("slider")
    jp = asset.data.joint_pos
    check(f"{task}: actor obs dim 5 (cart 1 + cos/sin 2 + cart vel 1 + pole vel 1)",
          tuple(obs["actor"].shape) == (num_envs, 5), str(tuple(obs["actor"].shape)))
    check(f"{task}: action dim 1", tuple(env.action_space.shape) == (num_envs, 1))
    want = math.pi if swing else 0.0
    check(f"{task}: hinge_1 starts at {want:.2f} (+- reset jitter)", (jp[:, h1] - want).abs().max().item() < 0.3,
          f"mean {jp[:, h1].mean().item():.3f}")
    hi = float(m._HINGE_RESET_RANGE[1])
    check(f"{task}: TODO(5) reset jitter stays within +-{hi:.3f} rad of the start",
          (jp[:, h1] - want).abs().max().item() <= hi + 1e-3,
          f"max offset {(jp[:, h1] - want).abs().max().item():.4f}")

    # TODO(3): through the observation manager (columns 1:3 of the actor obs)
    # and by calling the function directly with a resolved SceneEntityCfg.
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    hinge = SceneEntityCfg("cartpole", joint_names=("hinge_1",))
    hinge.resolve(env.scene)
    cs = m.pole_angle_cos_sin(env, hinge)
    check(f"{task}: TODO(3) pole_angle_cos_sin returns [num_envs, 2]", tuple(cs.shape) == (num_envs, 2), str(tuple(cs.shape)))
    ok3 = (tuple(cs.shape) == (num_envs, 2)
           and ((cs[:, 0] ** 2 + cs[:, 1] ** 2) - 1).abs().max().item() < 1e-4
           and (cs[:, 0] - torch.cos(jp[:, h1])).abs().max().item() < 1e-4
           and (cs[:, 1] - torch.sin(jp[:, h1])).abs().max().item() < 1e-4)
    check(f"{task}: TODO(3) cos^2 + sin^2 = 1 and (cos, sin) match the hinge angle", ok3)
    check(f"{task}: TODO(3) the actor observation carries (cos, sin) in columns 1:3",
          (obs["actor"][:, 1] - torch.cos(jp[:, h1])).abs().max().item() < 1e-4
          and (obs["actor"][:, 2] - torch.sin(jp[:, h1])).abs().max().item() < 1e-4)

    # Random rollout: TODO(4) is exercised here.
    act_dim = env.action_space.shape[-1]
    rewards = []
    try:
      for _ in range(100):
        a = 2 * torch.rand((num_envs, act_dim)) - 1
        _, rew, *_ = env.step(a)
        rewards.append(rew)
    except NotImplementedError as e:
      todo(f"{task}: {e}")
      env.close()
      continue
    r = torch.stack(rewards) / env.step_dt  # undo mjlab's dt scaling
    check(f"{task}: 100 random steps with {num_envs} envs ran; reward in [0, 1]",
          torch.isfinite(r).all().item() and r.min().item() >= -1e-6 and r.max().item() <= 1 + 1e-6,
          f"min {r.min().item():.3f} max {r.max().item():.3f}")

    ids = torch.arange(num_envs)

    def forced(q1: float) -> float:
      q = torch.zeros((num_envs, len(names)))
      q[:, h1] = q1
      q[:, s0] = 0.0
      env.reset()
      asset.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=ids)
      _, rew, *_ = env.step(torch.zeros((num_envs, act_dim)))
      return (rew / env.step_dt).mean().item()

    up, down, side = forced(0.0), forced(math.pi), forced(math.pi / 2)
    check(f"{task}: TODO(4) reward ~1 with the pole held up (cart centred, zero action)", up > 0.9, f"{up:.3f}")
    check(f"{task}: TODO(4) reward ~0 with the pole hanging down", down < 0.1, f"{down:.3f}")
    check(f"{task}: TODO(4) reward about 0.5 with the pole horizontal", 0.3 < side < 0.7, f"{side:.3f}")
    env.close()

  print()
  if todos:
    print(f"{len(todos)} blank(s) left: " + "; ".join(todos))
  if failures:
    print(f"{len(failures)} CHECK(S) FAILED: " + ", ".join(failures))
    return 1
  if todos:
    return 1
  print("ALL CHECKS PASSED — go watch both tasks in viser, then train (hw0/README.md Step 6).")
  return 0


def main_with_json() -> int:
  """--json PATH: also dump every check as structured JSON. This is the
  Gradescope autograder's input — the SAME checks students run locally
  (verifier == grader; no hidden tests)."""
  json_path = None
  if "--json" in sys.argv:
    i = sys.argv.index("--json")
    json_path = sys.argv[i + 1]
    del sys.argv[i:i + 2]
  code = main()
  if json_path:
    import json as _json

    pathlib_path = Path(json_path)
    pathlib_path.parent.mkdir(parents=True, exist_ok=True)
    pathlib_path.write_text(_json.dumps(
      {"exit_code": code, "results": results,
       "n_failures": len(failures), "n_todos": len(todos)}, indent=1))
    print(f"[json] {json_path}")
  return code


if __name__ == "__main__":
  raise SystemExit(main_with_json())
