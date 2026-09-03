"""HW0 task registration (the `hw0` package IS the assignment folder).

mjlab discovers this package through the `mjlab.tasks` entry point declared
in pyproject.toml and imports it, which runs the registration below. Each
assignment is its own top-level package (hw0, hw1, ...) with its own entry
point — releasing a new assignment adds a folder and one pyproject line.

Staff ship cartpole_env_cfg_starter.py; your editable copy,
cartpole_env_cfg.py, is created by `./scripts/init_hw.sh` and is NEVER
shipped by staff — that is why `git pull` can never conflict with your work.

Task ids are namespaced `Course-...` so they can't collide with mjlab's
built-in `Mjlab-...` tasks (mjlab raises if an id is registered twice).

Registration happens at import time, so an unfinished (or not yet created)
work file must not crash `uv run list-envs` for everything else: both cases
degrade to a warning, and the two Course-Cartpole-* tasks simply do not
appear until the blanks are filled. mjlab's own Mjlab-Cartpole-Balance /
-Swingup are always there.
"""

import warnings

from mjlab.tasks.registry import register_mjlab_task

try:
  from hw0.cartpole_env_cfg import (
    cartpole_balance_env_cfg,
    cartpole_ppo_runner_cfg,
    cartpole_swingup_env_cfg,
  )
except ImportError:
  warnings.warn(
    "hw0 is not initialized: run `./scripts/init_hw.sh` once to create "
    "hw0/cartpole_env_cfg.py from the starter, then fill its blanks.",
    stacklevel=1,
  )
else:
  try:
    register_mjlab_task(
      task_id="Course-Cartpole-Balance",
      env_cfg=cartpole_balance_env_cfg(),
      play_env_cfg=cartpole_balance_env_cfg(play=True),
      rl_cfg=cartpole_ppo_runner_cfg(),
    )
    register_mjlab_task(
      task_id="Course-Cartpole-Swingup",
      env_cfg=cartpole_swingup_env_cfg(),
      play_env_cfg=cartpole_swingup_env_cfg(play=True),
      rl_cfg=cartpole_ppo_runner_cfg(),
    )
  except NotImplementedError as e:
    warnings.warn(
      f"Course-Cartpole-* not registered yet: {e}. "
      "Run `uv run python scripts/check_hw0.py` to see what is missing.",
      stacklevel=1,
    )
