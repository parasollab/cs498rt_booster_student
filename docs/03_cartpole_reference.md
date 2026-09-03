# 03 — The cartpole, number by number (OPTIONAL)

**This whole document is optional reference material** — nothing in it is
submitted or graded (HW0's Q1–Q3 are a separate multiple-choice quiz on
Gradescope). It is the deep dive: every value in the XML, the environment
config and the PPO config, what it means, what changes if you touch it, and
where it is documented. Then three optional no-GPU experiments with
self-check questions, and the reading list per step.

Documentation roots used below (pinned to the mjlab version the course installs):

- **mjlab docs** `https://mujocolab.github.io/mjlab/v1.6.0/source/` — the cartpole tutorial is
  [`tutorials/cartpole.html`](https://mujocolab.github.io/mjlab/v1.6.0/source/tutorials/cartpole.html)
- **MuJoCo XML reference** `https://mujoco.readthedocs.io/en/stable/XMLreference.html`
- **dm_control cartpole** (the origin of the reward)
  [`suite/cartpole.py`](https://github.com/google-deepmind/dm_control/blob/main/dm_control/suite/cartpole.py)
- **rsl_rl** (the PPO implementation) <https://github.com/leggedrobotics/rsl_rl>

## 1. The physical system — `cartpole.xml`

| Element | Attribute | Value | Meaning · effect of changing it | Read |
|---|---|---|---|---|
| `<option>` | `timestep` | 0.01 s | physics integration step (10 ms). Smaller = more accurate and slower; larger risks instability | XML ref: *option* |
| `<default class="pole">` `<joint>` | `type`, `axis` | hinge, (0 1 0) | pole rotates about the y axis, i.e. in the x–z plane of the rail | XML ref: *joint* |
| | `damping` | 2e-6 | almost frictionless hinge: the pole keeps swinging for a long time | |
| `<default class="pole">` `<geom>` | `type`, `fromto`, `size` | capsule, 0→1 m along z, radius 0.045 m | a 1 m pole. Longer pole = slower swing, easier to balance, harder to swing up | XML ref: *geom* |
| | `mass` | 0.1 kg | light pole. Heavier pole needs more force to swing up | |
| `<body name="cart">` | `pos` | (0, 0, 1) | rail height 1 m above the floor (cosmetic) | XML ref: *body* |
| `<joint name="slider">` | `type`, `axis` | slide, (1 0 0) | the cart translates along x | |
| | `range`, `limited` | ±1.8 m, true | the rail ends; the "centered" reward factor keeps the policy away from them | |
| | `solreflimit` | (0.08, 1) | softness of the rail end-stop | XML ref: *solver parameters* |
| | `damping` | 5e-4 | tiny rolling friction on the cart | |
| `<geom name="cart">` | `size`, `mass` | box 0.2×0.15×0.1 m half-sizes, 1 kg | the cart is 10× heavier than the pole | |
| `<body name="pole_1">` `<joint name="hinge_1">` | (no `range`) | unlimited | angle 0 = up, π = down, and it can turn forever: the reason for the (cos, sin) observation | |
| `<motor name="slide">` | `joint` | slider | the one actuator drives the CART joint (the answer to TODO(1)) | XML ref: *motor* |
| | `gear` | 10 | force = gear × ctrl → **max 10 N** | |
| | `ctrlrange`, `ctrllimited` | [−1, 1], true | MuJoCo clamps the control internally, so any policy output is safe | |

Rails, floor, camera and light are decoration. No contacts are needed
between anything, which is why the config disables contact computation.

## 2. The environment config — `cartpole_env_cfg.py`

### Entity and initial state (section 1)

| Name | Value | Meaning · effect | Read |
|---|---|---|---|
| `XmlActuatorCfg(target_names_expr=("slider",))` | TODO(1) | "use the XML motor as is", matched by the joint it drives | mjlab *actuators* → XML actuators |
| `_BALANCE_INIT.joint_pos["hinge_1"]` | 0.0 rad | Balance starts upright | mjlab *entity* → init_state |
| swingup `joint_pos["hinge_1"]` | TODO(2) = π rad | Swingup starts hanging | tutorial → *Entity* (Swingup tab) |
| `joint_pos["slider"]` | 0.0 m | cart in the middle of the rail | |
| `joint_vel={".*": 0.0}` | all joints | at rest; `".*"` is a regex over joint names | |
| `SceneEntityCfg("cartpole", joint_names=(...))` | entity + joint names | the pointer every term uses; resolved to tensor column indices at start-up | mjlab *scene* |

### Observations (section 2)

| Term | Function | Dim | Unit | Notes | Read |
|---|---|---|---|---|---|
| `cart_pos` | `joint_pos_rel` | 1 | m | position relative to the default state | mjlab *observations* → built-in functions |
| `pole_angle` | `pole_angle_cos_sin` | 2 | – | TODO(3): (cos θ, sin θ); continuous and turn-count-free | tutorial → *Observations* |
| `cart_vel` | `joint_vel_rel` | 1 | m/s | | |
| `pole_vel` | `joint_vel_rel` | 1 | rad/s | | |
| total | | **5** | | the actor network input size | |
| `enable_corruption=True` (actor) | | | | lets per-term noise apply during training; no term defines noise here, so nothing changes yet — the tutorial's "Next steps" shows how to add it | mjlab *observations* → processing pipeline |
| `critic` group | same 4 terms | 5 | | could see privileged, clean data (asymmetric actor-critic) | mjlab *observations* → asymmetric actor-critic |

### Actions (section 5/6)

| Name | Value | Meaning · effect | Read |
|---|---|---|---|
| `JointEffortActionCfg` | actuator `slider` | policy output → actuator effort target | mjlab *actions* |
| `scale` | 1.0 | multiplies the policy output before it reaches the actuator; the XML then clamps to [−1, 1] and multiplies by gear 10. Halving `scale` halves the usable force | |
| action dim | 1 | one number per environment per step | |

### Events (section 4)

| Event | Parameter | Value | Meaning · effect | Read |
|---|---|---|---|---|
| `reset_slider` | `position_range` | ±0.1 m (Balance), 0 (Swingup) | cart start jitter | mjlab *events* → built-in functions |
| | `velocity_range` | ±0.01 m/s | | |
| `reset_hinge` | `position_range` | TODO(5), tutorial: ±0.034 rad ≈ ±2° | pole start jitter; why the zero-agent reward at t = 0 is not exactly 1 | tutorial → *Events* |
| | `velocity_range` | ±0.01 rad/s | | |
| both | `mode` | `"reset"` | run per environment at every episode reset; other modes: `"startup"`, `"interval"` | mjlab *events* → lifecycle modes |

Offsets are added to the entity's initial state, so in Swingup the pole
starts near π, not near 0.

### Reward (section 3)

| Factor | Formula | Margin | Value when violated | Read |
|---|---|---|---|---|
| `upright` | (cos θ + 1)/2 — TODO(4) | – | 0 when hanging | tutorial → *Rewards* |
| `centered` | (1 + g(x))/2 | 2.0 m | ≥ 0.55 (g bottoms out at 0.1 at the margin, then keeps decaying) | dm_control `rewards.tolerance` |
| `small_control` | (4 + q(u))/5 | 1.0 | ≥ 0.8 (q is exactly 0 beyond the margin) | |
| `small_velocity` | (1 + g(θ̇))/2 | 5 rad/s | ≥ 0.55 | |
| `g(x)` | exp(−½ (x/margin · s)²), s = √(−2 ln 0.1) | | Gaussian, equals 0.1 at |x| = margin | |
| `q(x)` | max(0, 1 − (x/margin · s')²), s' = √0.9 | | inverted parabola, 0.1 at the margin, 0 beyond | |
| `RewardTermCfg.weight` | 1.0 | | the manager multiplies each term by its weight **and by the control dt (0.05 s)**, so a per-step reward of 1 shows up as 0.05 in `env.step` and as ≈1 in the logged `Episode_Reward/smooth_reward` (mjlab reports per-step means) | mjlab *rewards* → reward scaling by dt |

### Terminations, scene, timing (section 5/6)

| Name | Value | Meaning · effect | Read |
|---|---|---|---|
| `time_out` (`time_out=True`) | only termination | truncation, not failure: PPO bootstraps the value past the boundary. There is no "pole fell" termination because in Swingup a fallen pole is the *start* state | tutorial → *Terminations*; mjlab *terminations* |
| `episode_length_s` | 50 s (train), 1e10 (play) | 50 / 0.05 = 1000 policy steps per episode | mjlab *environment_config* → timing |
| `MujocoCfg.timestep` | 0.01 s | must match the XML's intent | |
| `decimation` | 5 | physics steps per policy step → control dt 0.05 s = **20 Hz** | |
| `disableflags=("contact",)` | | skip contact computation: nothing collides | |
| `SceneCfg.num_envs` | 1 (CLI: 4096) | parallel copies; `--env.scene.num-envs` | mjlab *scene* |
| `env_spacing` | 4.0 m | distance between copies in the viewer grid | |
| `terrain_type` | plane | a ground plane (cosmetic here) | mjlab *terrain* |
| `ViewerConfig` | cart body, distance 4 m, elevation −15° | where the viser camera starts | mjlab *viewers* |

### PPO (section 7) — fixed course-wide

| Name | Value | Meaning · effect | Read |
|---|---|---|---|
| actor / critic `hidden_dims` | (64, 64), `elu` | two small MLPs; plenty for 5 inputs | mjlab *training/rsl_rl* → configuration |
| `obs_normalization` | False | raw observations (all are O(1) here) | |
| `GaussianDistribution`, `init_std` 1.0, `scalar` | | action = mean + std · noise; one learned std | |
| `clip_param` | 0.2 | PPO's trust region on the probability ratio | rsl_rl |
| `entropy_coef` | 0.01 | small exploration bonus | |
| `num_learning_epochs` / `num_mini_batches` | 5 / 4 | passes over each batch / minibatches per pass | |
| `learning_rate`, `schedule`, `desired_kl` | 1e-3, adaptive, 0.01 | the lr is adapted so the KL step stays near 0.01 | |
| `gamma`, `lam` | 0.99, 0.95 | discount (horizon ≈ 100 steps = 5 s) and GAE λ | |
| `max_grad_norm` | 1.0 | gradient clipping | |
| `num_steps_per_env` | 32 | steps collected per env per iteration → **4096 × 32 = 131,072 samples/iteration**, 32,768 per minibatch | |
| `max_iterations` | 500 | 65.5 M environment steps; about 5 min on an A40 | |
| `save_interval` | 50 | checkpoints `model_49.pt` … `model_499.pt` | mjlab *training/rsl_rl* → checkpoints and logging |
| `experiment_name` | `hw0_cartpole` | run directory `<log_root>/hw0_cartpole/<timestamp>/` | |
| `wandb_project` | `hw0-booster` | HW0's W&B project; the team comes from the assignment env (`hw0.env`) | |

## 3. Experiments without a GPU (optional — nothing to submit)

Each one: change one value, look in viser, write down what you saw and
why, **revert the change** (`git diff` must be empty except your five
blanks before you train).

**E1 — reset jitter.** Set `_HINGE_RESET_RANGE = (-1.0, 1.0)` and run
`uv run play Course-Cartpole-Balance --agent zero --env.scene.num-envs 8`.
The eight poles start at very different angles and fall in different
directions. Now put the small range back and look at the reward readout
at t = 0. *Self-check: with the normal range and the zero agent in Balance, what is
the reward at the very first step, and why is it not exactly 1.0? Which
factors of the product are below 1?*

**E2 — motor strength.** In `cartpole.xml` change the motor's
`gear="10"` to `gear="2"` and run
`uv run play Course-Cartpole-Swingup --agent random`. The cart barely
moves. Revert. *Self-check: what is the maximum force on the cart in newtons, and
which three settings between the policy output and that force could each
change it (name the file and attribute/field for each)?*

**E3 — reward factors.** Set `_SWINGUP_HINGE_ANGLE` to `math.pi / 2`
(horizontal) and run `uv run play Course-Cartpole-Swingup --agent zero
--env.scene.num-envs 4`. Read the reward at t = 0, then watch it as the pole
falls. Revert. *Self-check: compute the expected reward at t = 0 by hand from the
four factors (cart centred, zero action, pole at rest) and compare with the
readout. Then explain why the product form makes it impossible to earn
reward by, say, keeping the cart perfectly centred while the pole hangs.*

Optional, ungraded, from the tutorial's *Next steps*: add
`UniformNoiseCfg` to an observation term and train again; try domain
randomization of the pole mass.

## 4. Reading list by step

| HW0 step / blank | Read first |
|---|---|
| Step 0–2, cluster | Delta [login](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/login.html), [data management](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/data_mgmt.html), [running jobs](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/running_jobs.html), [job accounting](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/job_accounting.html) |
| Step 1, install | [uv sync](https://docs.astral.sh/uv/concepts/projects/sync/), [mjlab installation](https://mujocolab.github.io/mjlab/v1.6.0/source/installation.html) (method 1) |
| Step 3, CLI and viser | [training and playback](https://mujocolab.github.io/mjlab/v1.6.0/source/training/rsl_rl.html#training-and-playback), [viser](https://mujocolab.github.io/mjlab/v1.6.0/source/viewers.html#viser-browser-based) — note: the docs page named *Commands* is about RL command terms, not the CLI |
| Step 4, the big picture | [architecture overview](https://mujocolab.github.io/mjlab/v1.6.0/source/architecture_overview.html), [environment configuration](https://mujocolab.github.io/mjlab/v1.6.0/source/environment_config.html) |
| TODO(1) actuator | tutorial [*The XML model*](https://mujocolab.github.io/mjlab/v1.6.0/source/tutorials/cartpole.html#the-xml-model), XML ref [*motor*](https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator-motor), [XML actuators](https://mujocolab.github.io/mjlab/v1.6.0/source/actuators.html#xml-actuators) |
| TODO(2) initial state | tutorial [*Entity*](https://mujocolab.github.io/mjlab/v1.6.0/source/tutorials/cartpole.html#entity-wrapping-the-xml), [entity → init_state](https://mujocolab.github.io/mjlab/v1.6.0/source/entity/index.html#init-state) |
| TODO(3) observation | tutorial [*Observations*](https://mujocolab.github.io/mjlab/v1.6.0/source/tutorials/cartpole.html#observations-what-the-agent-sees), [custom observation functions](https://mujocolab.github.io/mjlab/v1.6.0/source/observations.html#writing-custom-observation-functions), [entity data](https://mujocolab.github.io/mjlab/v1.6.0/source/entity/entity_data.html) |
| TODO(4) reward | tutorial [*Rewards*](https://mujocolab.github.io/mjlab/v1.6.0/source/tutorials/cartpole.html#rewards-the-training-signal), [rewards](https://mujocolab.github.io/mjlab/v1.6.0/source/rewards.html), [dm_control cartpole](https://github.com/google-deepmind/dm_control/blob/main/dm_control/suite/cartpole.py) |
| TODO(5) reset event | tutorial [*Events*](https://mujocolab.github.io/mjlab/v1.6.0/source/tutorials/cartpole.html#events-resetting-the-state), [events](https://mujocolab.github.io/mjlab/v1.6.0/source/events.html) |
| Assembly, timing | tutorial [*Snapping everything together*](https://mujocolab.github.io/mjlab/v1.6.0/source/tutorials/cartpole.html#snapping-everything-together), [timing](https://mujocolab.github.io/mjlab/v1.6.0/source/environment_config.html#timing-decimation-timestep-and-episode-length), [actions](https://mujocolab.github.io/mjlab/v1.6.0/source/actions.html), [terminations](https://mujocolab.github.io/mjlab/v1.6.0/source/terminations.html) |
| Registration, PPO | tutorial [*Registration and training*](https://mujocolab.github.io/mjlab/v1.6.0/source/tutorials/cartpole.html#registration-and-training), [task registry](https://mujocolab.github.io/mjlab/v1.6.0/source/training/rsl_rl.html#task-registry), [configuration](https://mujocolab.github.io/mjlab/v1.6.0/source/training/rsl_rl.html#configuration), `pyproject.toml` entry point |
| Step 6, training | Delta [running jobs](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/running_jobs.html), [sbatch](https://slurm.schedmd.com/sbatch.html), [W&B quickstart](https://docs.wandb.ai/quickstart) |
| Step 7, checkpoints and curves | [checkpoints and logging](https://mujocolab.github.io/mjlab/v1.6.0/source/training/rsl_rl.html#checkpoints-and-logging), [rsl_rl](https://github.com/leggedrobotics/rsl_rl) |
