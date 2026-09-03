"""Login-node environment checks. Run: uv run python scripts/check_login_env.py

Everything here must pass on a Delta login node (dt-login01..04): no GPU, no
display, x86_64. CUDA being unavailable is EXPECTED there and treated as a
pass. The same script runs on a laptop; the notes tell you what differs.
"""

from __future__ import annotations

import platform
import re
import socket
import sys

PASS, FAIL, WARN, INFO = "[PASS]", "[FAIL]", "[warn]", "[info]"
failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
  print(f"{PASS if ok else FAIL} {name}" + (f" — {detail}" if detail else ""))
  if not ok:
    failures.append(name)


def on_delta() -> bool:
  host = socket.gethostname()
  return bool(re.match(r"^(dt-login|gpu[a-z]|cn)\d+", host)) or ".delta.ncsa.illinois.edu" in socket.getfqdn()


def main() -> int:
  # 1. Platform.
  arch = platform.machine()
  host = socket.gethostname()
  check("python >= 3.13", sys.version_info >= (3, 13), platform.python_version())
  if on_delta():
    print(f"{INFO} running on Delta ({host}, {arch})")
  else:
    print(f"{INFO} running on {host} ({arch}) — not Delta; fine for local work")
  if arch != "x86_64":
    print(f"{WARN} architecture is {arch}, not x86_64 — fine on an ARM laptop / Mac, "
          "wrong if you think you're on Delta")

  # 2. Torch imports and does CPU work; CUDA absence is expected on a login node.
  try:
    import torch

    x = torch.randn(64, 64, requires_grad=True)
    (x @ x).sum().backward()
    check("torch import + CPU autograd", x.grad is not None, torch.__version__)
    if torch.cuda.is_available():
      print(f"{WARN} CUDA available — you're probably NOT on a login node (fine on your own GPU)")
    else:
      print(f"{PASS} CUDA unavailable (expected on a login node: Delta login nodes have no GPU)")
  except Exception as e:  # noqa: BLE001
    check("torch import + CPU autograd", False, repr(e))

  # 3. Plain MuJoCo physics stepping on CPU.
  try:
    import mujoco

    model = mujoco.MjModel.from_xml_string(
      "<mujoco><worldbody><body><joint type='free'/>"
      "<geom size='0.1'/></body></worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    for _ in range(100):
      mujoco.mj_step(model, data)
    check("mujoco CPU stepping (100 steps)", True, mujoco.__version__)
  except Exception as e:  # noqa: BLE001
    check("mujoco CPU stepping (100 steps)", False, repr(e))

  # 4. mjlab imports, the course package is found through the entry point, and
  #    mjlab's built-in cartpole tasks are registered. The Course-* tasks appear
  #    only after HW0 Step 5 (the blanks), so they are reported, not required.
  try:
    import warnings

    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      import mjlab.tasks  # noqa: F401  (populates registry, incl. entry points)
      import hw0  # noqa: F401
    from mjlab.tasks.registry import list_tasks

    tasks = list_tasks()
    check("mjlab import + course package found (entry point 'mjlab.tasks')", True, f"{len(tasks)} tasks")
    check("mjlab's built-in Mjlab-Cartpole-* tasks registered",
          {"Mjlab-Cartpole-Balance", "Mjlab-Cartpole-Swingup"} <= set(tasks))
    course = [t for t in tasks if t.startswith("Course-")]
    print(f"{INFO} Course-* tasks registered: " + (", ".join(course) if course else
          "none yet (expected until you fill the blanks in HW0 Step 5)"))
  except Exception as e:  # noqa: BLE001
    check("mjlab import / task registry", False, repr(e))

  # 5. viser importable (the viewer used by `uv run play` / `demo`).
  try:
    import viser  # noqa: F401

    check("viser import", True)
  except Exception as e:  # noqa: BLE001
    check("viser import", False, repr(e))

  # 6. wandb: the course logger; must be < 0.29 (see pyproject.toml).
  try:
    import wandb

    v = tuple(int(p) for p in wandb.__version__.split(".")[:2])
    check("wandb import, version < 0.29", v < (0, 29), wandb.__version__)
  except Exception as e:  # noqa: BLE001
    check("wandb import, version < 0.29", False, repr(e))

  print()
  if failures:
    print(f"{len(failures)} CHECK(S) FAILED: " + ", ".join(failures))
    return 1
  print("ALL CHECKS PASSED — environment is ready.")
  print("Next: `uv run play Mjlab-Cartpole-Balance --agent zero` and open the")
  print("viser URL through your SSH tunnel (docs/01_workflow.md).")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
