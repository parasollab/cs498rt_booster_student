"""HW0 — the cartpole, annotated, with five blanks for you to fill.

READ THIS FILE TOP TO BOTTOM. It is the only environment you need to
understand for HW0, and every mjlab environment you will build this semester,
up to the humanoid, is made of exactly the same pieces:

    cartpole.xml   (MuJoCo model: bodies, joints, geoms, one motor)
        |
        v
    1. EntityCfg       wrap the XML, name the actuator, set the start state
    2. observations    what the policy sees        -> tensor [num_envs, 5]
    3. actions         what the policy controls    -> tensor [num_envs, 1]
    4. events          what happens at every reset (start-state jitter)
    5. rewards         what training maximises     (one shaped term in [0, 1])
    6. terminations    when an episode ends        (time limit only)
        |
        v
    ManagerBasedRlEnvCfg (+ sim timing)  --register_mjlab_task-->  uv run train / play
    7. PPO config      the learning algorithm's settings (fixed course-wide)

This file is mjlab's own cartpole tutorial with course annotations, so the
official tutorial is the step-by-step explanation of every section below.
Keep it open in a browser tab:

    https://mujocolab.github.io/mjlab/v1.6.0/source/tutorials/cartpole.html

Every section starts with a "Read:" line: the tutorial section for that
piece, plus the concept page that documents the objects it uses. Every
number in this file (masses, ranges, margins, timesteps, PPO settings) is
explained one by one in docs/03_cartpole_reference.md, which also has the
no-GPU experiments the writeup asks about.

THE FIVE BLANKS. Search for "TODO(" — there are five. Each one says what to
write, the shape of the answer, where it is explained, and how to check it
on the login node without a GPU:

    uv run python scripts/check_hw0.py     # names the blank that is still missing

Until TODO(1), (2) and (5) are filled, the two course tasks
(Course-Cartpole-Balance, Course-Cartpole-Swingup) are not registered and
`uv run list-envs` shows only mjlab's built-in Mjlab-Cartpole-* tasks.
TODO(3) and (4) are checked when the environment runs.

Rules: keep the public names and signatures (the check script and the grader
import them); never write a Python loop over environments (everything is one
batched torch op); do not change the PPO config or the simulation timing.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import torch

from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import (
  joint_pos_rel,
  joint_vel_rel,
  reset_joints_by_offset,
  time_out,
)
from mjlab.envs.mdp.actions import JointEffortActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# Official documentation, pinned to the mjlab version the course installs
# (v1.6.0 — a different version's docs may not match this file).
# Every "Read:" line below is a FULL clickable URL; these two are the roots.
# NOTE: the docs landing page is the root below — the deeper source/ path has
# no index page of its own (pasting it alone gives a 404).
_MJLAB_DOCS = "https://mujocolab.github.io/mjlab/v1.6.0/"
_MUJOCO_XML = "https://mujoco.readthedocs.io/en/stable/XMLreference.html"

# ===========================================================================
# 1. The entity: wrap the MuJoCo model so mjlab can batch it on the GPU.
#
# Read:  cartpole.xml — the file sitting NEXT TO this one; open it first
#        https://mujocolab.github.io/mjlab/v1.6.0/source/tutorials/cartpole.html#the-xml-model
#        https://mujocolab.github.io/mjlab/v1.6.0/source/tutorials/cartpole.html#entity-wrapping-the-xml
#        https://mujocolab.github.io/mjlab/v1.6.0/source/entity/index.html  (spec_fn, init_state, articulation)
#        https://mujocolab.github.io/mjlab/v1.6.0/source/actuators.html#xml-actuators
#        MuJoCo XML reference (one page, use the anchors):
#          https://mujoco.readthedocs.io/en/stable/XMLreference.html#body
#          …#body-joint   …#body-geom   …#actuator-motor
#
# cartpole.xml defines the physical system. Open it next to this file:
#   * body "cart"   : a 1 kg box on a rail. Its joint "slider" is a SLIDE
#                     joint along x, limited to +-1.8 m (the rail).
#   * body "pole_1" : a 0.1 kg, 1 m capsule. Its joint "hinge_1" is a HINGE
#                     about y with NO limits: angle 0 = pole straight UP,
#                     pi = hanging straight down, and it can spin forever.
#   * <motor name="slide" joint="slider" gear="10" ctrlrange="-1 1">:
#                     the only actuator. It pushes the CART (not the pole):
#                     force = gear * ctrl, so at most 10 N. Inside MuJoCo the
#                     control is clamped to [-1, 1] (ctrllimited).
#   * <option timestep="0.01">: physics step of 10 ms (see section 6).
#
# Two names matter everywhere below:
#   - the ENTITY name, "cartpole": the key we give the scene in section 6;
#   - the JOINT names, "slider" and "hinge_1": how observation, event and
#     reward terms say WHICH joint they read. SceneEntityCfg(entity, joint
#     names) is that pointer, and it resolves names to tensor column indices
#     (asset_cfg.joint_ids) once, at start-up.
# ===========================================================================

_CARTPOLE_XML: Path = Path(__file__).parent / "cartpole.xml"
_CART_CFG = SceneEntityCfg("cartpole", joint_names=("slider",))
_HINGE_CFG = SceneEntityCfg("cartpole", joint_names=("hinge_1",))


def _get_spec() -> mujoco.MjSpec:
  # mjlab consumes an MjSpec (MuJoCo's editable model format), not a path,
  # so entities can be procedurally modified before compilation.
  return mujoco.MjSpec.from_file(str(_CARTPOLE_XML))


# ---------------------------------------------------------------------------
# TODO(1): which joint does the motor drive?
#
# XmlActuatorCfg tells mjlab "use the actuator already defined in the XML,
# as is". target_names_expr is a tuple of regexes matched against the names
# of the JOINTS that XML actuators act on, not the actuator's own name.
# Look at the <actuator> block of cartpole.xml: the motor is called "slide"
# and its joint="..." attribute names the joint it drives. Put that joint
# name here (a string).
#
# Read:  cartpole.xml, in this folder — the <actuator> block IS the answer;
#        find it and read the joint="..." attribute with your own eyes.
#        https://mujocolab.github.io/mjlab/v1.6.0/source/actuators.html#xml-actuators
#        https://mujocolab.github.io/mjlab/v1.6.0/source/tutorials/cartpole.html#the-xml-model
#        https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator-motor
# Check: uv run python scripts/check_hw0.py   (then TODO(2), then `play`)
# ---------------------------------------------------------------------------
_ACTUATED_JOINT: str | None = None


# ---------------------------------------------------------------------------
# Initial state. Two variants of the same physical system differ ONLY in where
# the pole starts. joint_pos maps joint name -> value (radians for hinges,
# metres for slides); regexes over joint names are allowed (".*" = all).
#
# Balance: pole straight up (hinge_1 = 0). The policy only has to keep it there.
# ---------------------------------------------------------------------------
_BALANCE_INIT = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.0),
  joint_pos={"slider": 0.0, "hinge_1": 0.0},
  joint_vel={".*": 0.0},
)

# ---------------------------------------------------------------------------
# TODO(2): the swing-up start.
#
# In Swingup the pole starts hanging straight DOWN and the policy must swing
# it up and balance it. hinge_1 is measured from the upright position, in
# radians (see section 1: 0 = up). Which angle is "straight down"? Write it
# as a float; math.pi is available.
#
# Read:  https://mujocolab.github.io/mjlab/v1.6.0/source/tutorials/cartpole.html#entity-wrapping-the-xml  (the Swingup tab)
#        https://mujocolab.github.io/mjlab/v1.6.0/source/entity/index.html#init-state
# Check: uv run play Course-Cartpole-Swingup --agent zero --env.scene.num-envs 4
#        -> in viser the pole must hang straight down (small jitter is normal)
# ---------------------------------------------------------------------------
_SWINGUP_HINGE_ANGLE: float | None = None


def _get_cartpole_cfg(swing_up: bool = False) -> EntityCfg:
  if _ACTUATED_JOINT is None:
    raise NotImplementedError("TODO(1): set _ACTUATED_JOINT in cartpole_env_cfg.py")
  if _SWINGUP_HINGE_ANGLE is None:
    raise NotImplementedError("TODO(2): set _SWINGUP_HINGE_ANGLE in cartpole_env_cfg.py")
  # Which actuators from the XML does this entity expose? Names are matched
  # against the joints those actuators drive.
  articulation = EntityArticulationInfoCfg(
    actuators=(XmlActuatorCfg(target_names_expr=(_ACTUATED_JOINT,)),),
  )
  swingup_init = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    joint_pos={"slider": 0.0, "hinge_1": _SWINGUP_HINGE_ANGLE},
    joint_vel={".*": 0.0},
  )
  return EntityCfg(
    spec_fn=_get_spec,
    articulation=articulation,
    init_state=swingup_init if swing_up else _BALANCE_INIT,
  )


# ===========================================================================
# 2. Observations: what the policy sees.
#
# Read:  https://mujocolab.github.io/mjlab/v1.6.0/source/tutorials/cartpole.html#observations-what-the-agent-sees
#        https://mujocolab.github.io/mjlab/v1.6.0/source/observations.html  (groups, built-in and custom functions)
#
# An observation TERM is a plain function  env -> tensor [num_envs, dim].
# The observation manager calls every term each control step and
# concatenates the results, in dictionary order, into one vector per
# environment. mjlab ships the common ones in mjlab.envs.mdp: joint_pos_rel
# and joint_vel_rel read joint positions/velocities relative to the
# entity's default state. You write your own only when the built-ins are
# not enough — like here, for the pole angle.
#
# The cartpole has two moving parts, so its physical state is fully described
# by two positions and two velocities. The policy gets:
#
#     term        dim   what it answers
#     cart_pos     1    where is the cart on the rail?            [m]
#     pole_angle   2    which way does the pole point? (cos, sin)  [-]
#     cart_vel     1    how fast is the cart moving?              [m/s]
#     pole_vel     1    how fast is the pole rotating?            [rad/s]
#                 ---
#                  5    = the actor observation dimension
#
# ALL data in mjlab is batched: tensors have a leading num_envs dimension
# because thousands of environments run in parallel on the GPU. Every
# function you write takes and returns tensors shaped [num_envs, ...].
# ===========================================================================


def pole_angle_cos_sin(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _HINGE_CFG,
) -> torch.Tensor:
  """Cosine and sine of the pole hinge angle. Shape: [num_envs, 2].

  TODO(3): write the body of this function.

  Why not the raw angle? hinge_1 has no joint limits, so as the pole spins
  the raw value keeps growing (2*pi, 4*pi, ...) although the pole is in the
  same place. (cos, sin) is the same for the same physical angle no matter
  how many turns happened, and it is continuous everywhere — no jump from
  +pi to -pi. Do the same for every unlimited angle you will ever observe.

  Steps (each is one line):
    1. asset = env.scene[asset_cfg.name]           the Entity named "cartpole"
    2. angle = asset.data.joint_pos[:, asset_cfg.joint_ids]
                                                    -> tensor [num_envs, 1]
    3. return a tensor of cos and sin of the angle
                                                    -> tensor [num_envs, 2]

  Read:  https://mujocolab.github.io/mjlab/v1.6.0/source/tutorials/cartpole.html#observations-what-the-agent-sees
         https://mujocolab.github.io/mjlab/v1.6.0/source/observations.html#writing-custom-observation-functions
         https://mujocolab.github.io/mjlab/v1.6.0/source/entity/entity_data.html   (asset.data.joint_pos and friends)
         For step 3, find the right torch functions yourself in the PyTorch
         docs — knowing where to look them up is part of the exercise.
  Check: uv run python scripts/check_hw0.py   (obs dim 5; cos^2 + sin^2 = 1)
  """
  # your code here

  # your code ends
  raise NotImplementedError("TODO(3): implement pole_angle_cos_sin in cartpole_env_cfg.py")


# ===========================================================================
# 3. Reward: the training signal.
#
# Read:  https://mujocolab.github.io/mjlab/v1.6.0/source/tutorials/cartpole.html#rewards-the-training-signal
#        https://mujocolab.github.io/mjlab/v1.6.0/source/rewards.html  (reward scaling by dt, custom reward functions)
#        dm_control's original: https://github.com/google-deepmind/dm_control/blob/main/dm_control/suite/cartpole.py
#
# A reward TERM returns one number per environment, tensor [num_envs]. The
# reward manager multiplies each term by its weight and by the control dt
# and sums the terms. We use ONE term: dm_control's classic smooth cartpole
# reward, a PRODUCT of four shaped factors, each in [0, 1]:
#
#     upright         (cos(theta) + 1) / 2     1 when up, 0 when hanging
#     centered        (1 + g(x)) / 2           g: Gaussian bump, 1 at x = 0,
#                                              0.1 at |x| = 2 m (margin)
#     small_control   (4 + q(u)) / 5           q: inverted parabola, 1 at u = 0,
#                                              0 for |u| >= 1 (margin)
#     small_velocity  (1 + g(theta_dot)) / 2   g with margin 5 rad/s
#
# A product acts like a soft AND: the reward is only high when the pole is
# up AND the cart is centered AND the control is small AND the pole is slow,
# so the policy cannot trade one factor for another. The "(1 + g)/2" and
# "(4 + q)/5" shapes keep each factor above 0.5 or 0.8 even when it is
# violated, so no single factor can zero out the reward except "upright".
#
# Everything is written with torch ops over the whole batch: one tensor
# expression services all num_envs environments at once.
# ===========================================================================

_GAUSSIAN_SCALE = math.sqrt(-2 * math.log(0.1))  # value_at_margin = 0.1
_QUADRATIC_SCALE = math.sqrt(1 - 0.1)


def _gaussian_tolerance(x: torch.Tensor, margin: float) -> torch.Tensor:
  """1 at x=0, decaying to 0.1 at |x| = margin (never reaches 0)."""
  if margin == 0:
    return (x == 0).float()
  scaled = x / margin * _GAUSSIAN_SCALE
  return torch.exp(-0.5 * scaled**2)


def _quadratic_tolerance(x: torch.Tensor, margin: float) -> torch.Tensor:
  """1 at x=0, hitting exactly 0 for |x| >= margin."""
  if margin == 0:
    return (x == 0).float()
  scaled = x / margin * _QUADRATIC_SCALE
  return torch.clamp(1 - scaled**2, min=0.0)


def cartpole_smooth_reward(
  env: ManagerBasedRlEnv,
  cart_cfg: SceneEntityCfg = _CART_CFG,
  hinge_cfg: SceneEntityCfg = _HINGE_CFG,
) -> torch.Tensor:
  """upright * centered * small_control * small_velocity, all in [0, 1]."""
  asset: Entity = env.scene[cart_cfg.name]

  # [num_envs, 1] -> [num_envs]: reward terms return one value per env.
  hinge_angle = asset.data.joint_pos[:, hinge_cfg.joint_ids].squeeze(-1)

  # -------------------------------------------------------------------------
  # TODO(4): the "upright" factor.
  #
  # It must be 1.0 when the pole points straight up (hinge_angle = 0), 0.0
  # when it hangs straight down (hinge_angle = pi), and smooth in between.
  # The exact formula (theta = hinge_angle, in radians):
  #
  #                  cos(theta) + 1
  #     upright  =  ----------------          in [0, 1] for every theta
  #                        2
  #
  # Your job is to translate that equation into ONE torch expression over the
  # batched tensor `hinge_angle` (shape [num_envs]) — remember every torch op
  # works elementwise on the whole batch at once.
  #
  # Read:  https://mujocolab.github.io/mjlab/v1.6.0/source/tutorials/cartpole.html#rewards-the-training-signal
  # Check: uv run python scripts/check_hw0.py
  #        (reward ~1 with the pole held up, ~0 hanging; always in [0, 1])
  # -------------------------------------------------------------------------
  upright = None  # TODO(4): replace inside the markers below
  # your code here

  # your code ends
  if upright is None:
    raise NotImplementedError("TODO(4): write `upright` in cartpole_smooth_reward")

  cart_pos = asset.data.joint_pos[:, cart_cfg.joint_ids].squeeze(-1)
  centered = (1 + _gaussian_tolerance(cart_pos, margin=2.0)) / 2

  # The last action the policy produced, [num_envs, 1] -> [num_envs].
  control = env.action_manager.action.squeeze(-1)
  small_control = (4 + _quadratic_tolerance(control, margin=1.0)) / 5

  hinge_vel = asset.data.joint_vel[:, hinge_cfg.joint_ids].squeeze(-1)
  small_velocity = (1 + _gaussian_tolerance(hinge_vel, margin=5.0)) / 2

  return upright * centered * small_control * small_velocity


# ===========================================================================
# 4. Events: what happens at reset.
#
# Read:  https://mujocolab.github.io/mjlab/v1.6.0/source/tutorials/cartpole.html#events-resetting-the-state
#        https://mujocolab.github.io/mjlab/v1.6.0/source/events.html  (lifecycle modes, built-in event functions)
#
# Events run at a lifecycle point given by mode=: "startup" (once),
# "reset" (every episode reset, per environment) or "interval" (every N
# seconds). reset_joints_by_offset adds a uniform random offset, drawn from
# position_range / velocity_range, to the entity's init_state joints named
# by asset_cfg. That small randomisation is what stops the policy from
# memorising one trajectory, and it is why the reward you see at t = 0 with
# the zero agent is not exactly 1.0.
# ===========================================================================

# ---------------------------------------------------------------------------
# TODO(5): the pole's reset jitter.
#
# The tutorial resets the hinge with a small symmetric offset, about 2
# degrees either way, so every episode starts slightly different but still
# "up" (Balance) or still "down" (Swingup). Give the (low, high) tuple in
# radians. Convert 2 degrees yourself (math.radians(2) is about 0.035; the
# tutorial rounds to 0.034), or pick any symmetric value in [0.01, 0.2].
#
# Read:  https://mujocolab.github.io/mjlab/v1.6.0/source/tutorials/cartpole.html#events-resetting-the-state
#        https://mujocolab.github.io/mjlab/v1.6.0/source/events.html#built-in-event-functions   (reset_joints_by_offset)
# Check: uv run python scripts/check_hw0.py
#        optionally docs/03_cartpole_reference.md experiment E1 (make it huge, look, revert)
# ---------------------------------------------------------------------------
_HINGE_RESET_RANGE: tuple[float, float] | None = None


# ===========================================================================
# 5. + 6. Assemble the environment: scene, actions, terminations, timing.
#
# Read:  https://mujocolab.github.io/mjlab/v1.6.0/source/tutorials/cartpole.html#actions-what-the-agent-does
#        https://mujocolab.github.io/mjlab/v1.6.0/source/tutorials/cartpole.html#terminations-when-to-stop
#        https://mujocolab.github.io/mjlab/v1.6.0/source/tutorials/cartpole.html#snapping-everything-together
#        https://mujocolab.github.io/mjlab/v1.6.0/source/actions.html   and   https://mujocolab.github.io/mjlab/v1.6.0/source/terminations.html
#        https://mujocolab.github.io/mjlab/v1.6.0/source/environment_config.html#timing-decimation-timestep-and-episode-length
# ===========================================================================


def _make_env_cfg(swing_up: bool = False) -> ManagerBasedRlEnvCfg:
  entity_cfg = _get_cartpole_cfg(swing_up=swing_up)  # raises for TODO(1)/(2) first
  if _HINGE_RESET_RANGE is None:
    raise NotImplementedError("TODO(5): set _HINGE_RESET_RANGE in cartpole_env_cfg.py")
  cart_cfg = SceneEntityCfg("cartpole", joint_names=("slider",))
  hinge_cfg = SceneEntityCfg("cartpole", joint_names=("hinge_1",))

  # Observations: named terms, concatenated in this order. "actor" is what
  # the policy sees (enable_corruption=True lets per-term noise apply during
  # training; no term defines noise here, so nothing changes yet); "critic"
  # is the value function's view and may see privileged, clean data. Here
  # the two groups hold the same four terms.
  actor_terms = {
    "cart_pos": ObservationTermCfg(func=joint_pos_rel, params={"asset_cfg": cart_cfg}),
    "pole_angle": ObservationTermCfg(
      func=pole_angle_cos_sin, params={"asset_cfg": hinge_cfg}
    ),
    "cart_vel": ObservationTermCfg(func=joint_vel_rel, params={"asset_cfg": cart_cfg}),
    "pole_vel": ObservationTermCfg(func=joint_vel_rel, params={"asset_cfg": hinge_cfg}),
  }
  observations = {
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=True),
    "critic": ObservationGroupCfg({**actor_terms}),
  }

  # Actions: one scalar effort on the cart's actuator. The policy outputs
  # roughly [-1, 1]; scale maps that to actuator units (1.0 = unchanged),
  # and the XML actuator clamps to ctrlrange and multiplies by gear (10 N).
  actions: dict[str, ActionTermCfg] = {
    "effort": JointEffortActionCfg(
      entity_name="cartpole",
      actuator_names=("slider",),
      scale=1.0,
    ),
  }

  # Reset events: the cart gets +-0.1 m of start jitter in Balance and none
  # in Swingup; the pole gets your TODO(5) range in both. Velocities get
  # +-0.01 (m/s or rad/s).
  slider_range = (-0.1, 0.1) if not swing_up else (0.0, 0.0)
  events = {
    "reset_slider": EventTermCfg(
      func=reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": slider_range,
        "velocity_range": (-0.01, 0.01),
        "asset_cfg": SceneEntityCfg("cartpole", joint_names=("slider",)),
      },
    ),
    "reset_hinge": EventTermCfg(
      func=reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": _HINGE_RESET_RANGE,
        "velocity_range": (-0.01, 0.01),
        "asset_cfg": SceneEntityCfg("cartpole", joint_names=("hinge_1",)),
      },
    ),
  }

  rewards = {
    "smooth_reward": RewardTermCfg(
      func=cartpole_smooth_reward,
      weight=1.0,
      params={"cart_cfg": cart_cfg, "hinge_cfg": hinge_cfg},
    ),
  }

  # Terminations: the cartpole has no failure state (a fallen pole is just a
  # low reward, and in Swingup it is the STARTING state), so the only
  # termination is the time limit. time_out=True marks it as a truncation
  # ("the clock ran out") rather than a failure: PPO then bootstraps the value
  # function past the episode boundary instead of treating it as the end.
  terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"cartpole": entity_cfg},  # the ENTITY name used by every SceneEntityCfg
      num_envs=1,  # overridden at train time: --env.scene.num-envs 4096
      env_spacing=4.0,  # metres between the parallel copies (viser grid)
    ),
    observations=observations,
    actions=actions,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="cartpole",
      body_name="cart",
      distance=4.0,
      elevation=-15.0,
      azimuth=0.0,
    ),
    sim=SimulationCfg(
      # Contacts disabled: nothing in this scene needs them, and it's faster.
      mujoco=MujocoCfg(timestep=0.01, disableflags=("contact",)),
    ),
    # Timing chain: physics 0.01 s x decimation 5 = control dt 0.05 s (20 Hz);
    # episode_length_s 50 / 0.05 = 1000 policy steps per episode.
    decimation=5,
    episode_length_s=50.0,
  )


def cartpole_balance_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = _make_env_cfg(swing_up=False)
  if play:
    # In play mode: run forever, no observation noise.
    cfg.episode_length_s = 1e10
    cfg.observations["actor"].enable_corruption = False
  return cfg


def cartpole_swingup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = _make_env_cfg(swing_up=True)
  if play:
    cfg.episode_length_s = 1e10
    cfg.observations["actor"].enable_corruption = False
  return cfg


# ===========================================================================
# 7. The RL side: a small PPO config (rsl_rl). FIXED for the course: everyone
# trains with the same settings, so runs are comparable and the grader can
# verify them. You will reuse this shape all semester; only sizes and
# horizons change as tasks get harder.
#
# Read:  https://mujocolab.github.io/mjlab/v1.6.0/source/tutorials/cartpole.html#registration-and-training
#        https://mujocolab.github.io/mjlab/v1.6.0/source/training/rsl_rl.html  (task registry, configuration, checkpoints and logging)
#        https://github.com/leggedrobotics/rsl_rl
#
# How the numbers connect (docs/03_cartpole_reference.md has the full table):
#   * num_steps_per_env=32 with 4096 envs = 131,072 samples per iteration;
#     500 iterations = 65.5 M environment steps in about 5 minutes on an A40.
#   * save_interval=50 -> model_49.pt, model_99.pt, ... model_499.pt.
#   * experiment_name is the run directory:  <log_root>/hw0_cartpole/<timestamp>/
#   * wandb_project is where the run appears; the team ("entity") comes from
#     the assignment env (scripts/hw0.env) through the WANDB_USERNAME variable.
# ===========================================================================


def cartpole_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(64, 64),  # two hidden layers of 64 units: plenty for 5 inputs
      activation="elu",
      obs_normalization=False,
      distribution_cfg={
        "class_name": "GaussianDistribution",  # action = mean + std * noise
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(64, 64),
      activation="elu",
      obs_normalization=False,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,  # PPO's trust region on the policy ratio
      entropy_coef=0.01,  # small bonus for exploration
      num_learning_epochs=5,  # passes over each batch of samples
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",  # lr adapts to keep the KL step near desired_kl
      gamma=0.99,  # discount: horizon of ~100 steps = 5 s
      lam=0.95,  # GAE lambda
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="hw0_cartpole",
    wandb_project="hw0-booster",  # HW0 W&B project; the team/entity comes from hw0.env
    save_interval=50,
    num_steps_per_env=32,
    max_iterations=500,
  )
