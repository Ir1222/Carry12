# Clean CarryBox Perturbation Evaluator

This experiment evaluates the current clean velocity-command CarryBox policy:

```text
approach box -> pick up -> carry -> velocity-command locomotion
```

It does not modify:

- `legged_gym/legged_gym/envs/g1/carrybox.py`
- `legged_gym/legged_gym/envs/g1/carrybox_config.py`
- `experiments/carrybox_perturb_debug/`

The evaluator subclass inherits directly from:

```python
from legged_gym.envs.g1.carrybox import LeggedRobot as CarryBoxBase
```

It does not inherit from or import the PI, boxperturb, resume, or old debug env
lineages.


# CarryBox Clean Perturb Evaluation

## Controlled Evaluation Mode

Formal perturbation evaluation may use either a single trial or a sweep. Both use
the same controlled evaluator configuration. Use:

```text
--sweep
```

for batch evaluation across the requested condition lists.

For formal force evaluation, do **not** use:

```text
--parity_mode play_baseline
--parity_debug
--checkpoint_parity
```

The desired controlled evaluation environment is:

```text
num_envs = 1
episode_length_s = 30
vx = 0.6 m/s
vy = 0.0 m/s
yaw_rate = 0.0 rad/s

box random_size = False
box random_density = False
```

Training uses dynamic carry-command resampling with stop, forward, and turning
segments. Controlled evaluation disables that resampling and explicitly sets:

```text
vx = 0.6 m/s
vy = 0
yaw_rate = 0
```

---

## No Force Version
```bash
python3 experiments/carrybox_clean_perturb/evaluate.py --resume_path legged_gym/logs/Ampstage1_UpAndWalk/Aug29_15-54-01_stage1_UpAndWalk/model_19000.pt --no_force
```

## Single No-Force Baseline

For comparing the same controlled evaluation environment without external force.

```bash
python3 experiments/carrybox_clean_perturb/evaluate.py --resume_path legged_gym/logs/Ampstage1_UpAndWalk/Aug13_14-39-11_stage1_UpAndWalk/model_9500.pt --no_force --hold_durations=5.0 --seeds=2 --sweep --verbose

```

```
python3 experiments/carrybox_clean_perturb/evaluate.py --resume_path legged_gym/logs/Ampstage1_UpAndWalk/Aug13_14-39-11_stage1_UpAndWalk/model_9500.pt --profile smooth_hold --directions=+box_x --betas=2.0 --hold_durations=8.0 --seeds=2 --sweep --verbose
```

The force parameters are still required to define the singleton sweep condition, but no external force is applied because:

```text
--no_force
```

disables force scheduling.

This should be used as the direct no-force control for the corresponding force trial above.

---

## Single Force Trial

For visually inspecting one robot under one force condition in the Isaac Gym viewer.

Use `--sweep` with only one value for each sweep parameter.

```bash
python experiments/carrybox_clean_perturb/evaluate.py \
  --resume_path legged_gym/logs/Ampstage1_UpAndWalk/Aug13_14-39-11_stage1_UpAndWalk/model_9500.pt \
  --profile smooth_hold \
  --sweep \
  --directions=+box_x \
  --betas=2.0 \
  --seeds=2 \
  --hold_durations=5.0 \
  --verbose
```

This gives:

```text
num_envs = 1
episode_length_s = 30
command = (0.6, 0.0, 0.0)

direction = +box_x
beta = 2.0
seed = 2
hold_duration = 5.0 s
```

No CSV is generated because `--save_csv` is not specified.

---

## Full Sweep

Fixed evaluation environment, suitable for systematic force testing across directions, magnitudes, seeds, and durations.

```bash
python experiments/carrybox_clean_perturb/evaluate.py \
  --resume_path legged_gym/logs/Ampstage1_UpAndWalk/Aug13_14-39-11_stage1_UpAndWalk/model_9500.pt \
  --profile smooth_hold \
  --sweep \
  --directions=+box_x,-box_x,+box_y,-box_y \
  --betas=0.1,0.3,0.5 \
  --seeds=1,2,3 \
  --hold_durations=0.5,1.0,2.0 \
  --save_csv
```

This performs the Cartesian product of:

```text
4 directions
× 3 beta values
× 3 seeds
× 3 hold durations
```

while keeping the evaluator environment definition fixed.

---

## Parity Mode

Parity is a **diagnostic mode**, not the environment that should be used for formal force evaluation.

Its purpose is to check whether the evaluator remains behaviorally consistent with the original clean `play.py` / CarryBox policy execution.

Explicit play-baseline parity is invoked with:

```bash
python experiments/carrybox_clean_perturb/evaluate.py \
  --resume_path legged_gym/logs/Ampstage1_UpAndWalk/Aug13_14-39-11_stage1_UpAndWalk/model_9500.pt \
  --parity_mode play_baseline
```

The parity baseline uses a play-style command:

```text
vx = 0.8 m/s
vy = 0.0 m/s
yaw_rate = 0.0 rad/s
```

and uses:

```text
episode_length_s = 10
play-style environment configuration
```

This mode is useful for debugging questions such as:

```text
Does the checkpoint still behave correctly?

Does actor-only inference reproduce the original policy behavior?

Did changes to evaluate.py alter the nominal CarryBox rollout?
```

It should **not** be used for controlled external-force experiments.

---

## Rule of Thumb

For all formal force experiments in the current repo:

```text
Always include --sweep.
```

### One condition

```text
--sweep
--directions=<one direction>
--betas=<one beta>
--seeds=<one seed>
--hold_durations=<one duration>
```

### Multiple conditions

```text
--sweep
--directions=<multiple directions>
--betas=<multiple betas>
--seeds=<multiple seeds>
--hold_durations=<multiple durations>
```

### No-force control

Use the **same singleton sweep configuration** and add:

```text
--no_force
```

This keeps the policy, seed, command, box configuration, episode horizon, and evaluation environment consistent, while changing only whether the external perturbation is applied.

## Phase A Audit

Clean `carrybox.py` findings:

- Actor task observation is built in `compute_task_observations()` as:
  `box_pos_local(3) + box_rot_6d_local(6) + box_size(3) + self.carry_policy_commands[:, :3](3)`.
- Current task observation dimension is 15.
- Actor observation dimension is `(108 proprio + 15 task) * 6 history = 738`.
- Commands are sampled in `_resample_carry_commands()`, called from `_reset_task()`.
- Training samples stop, forward (`vx in [0.2, 0.8]`), and turning
  (`|yaw| in [0.1, 0.4]`) segments through `_resample_carry_commands()`.
- The physics loop in `step()` runs `cfg.control.decimation` substeps and calls
  `gym.simulate()` once per substep.
- The box actor is created before the robot actor; the evaluator resolves the box
  rigid-body handle with `gym.get_actor_rigid_body_handle(..., box_handle, 0)`
  instead of relying on a hard-coded index.
- The clean base already acquires `net_contact_force_tensor` and exposes it as
  `self.contact_forces`, but this tensor does not identify the other contact body.
- Clean carry-stage logic is `is_stage_carry = lifted_from_platform OR moved_from_start`.
- Hand collision bodies are available through `hand_colli_indices`; the evaluator
  resolves left/right contact proxy bodies by name and cross-checks them against
  those base indices.

Old `experiments/carrybox_perturb_debug/` classification:

- Safe to reuse conceptually: `force_profiles.py` profile formulas,
  deterministic trial condition generation, CSV writer shape, substep force
  insertion pattern, frozen world-direction commitment.
- Needs adaptation: old `evaluate.py` state machine and metrics, because it
  patched `env.commands` inside policy stepping and included old task metadata.
- Must not reuse: old env inheritance, PI interaction privileged observations,
  goal/relocation fields, `success_buf` relocation semantics, long-range goal
  setup, object-to-goal metrics, and old debug fixed-scene goal utilities.

## Usage

No-force nominal regression:

```bash
python experiments/carrybox_clean_perturb/evaluate.py \
  --resume_path legged_gym/logs/Ampstage1_UpAndWalk/Aug13_14-39-11_stage1_UpAndWalk/model_9500.pt \
  --no_force
```

`--no_force` is intentionally a play-parity nominal regression. It uses the
evaluation subclass but preserves the known-good play.py reset/config surface:
`num_envs <= 10`, `episode_length_s = 10`, box `random_size/random_density`
unchanged, `reset_mode = default`, and play.py-style command injection
`vx = 0.8` immediately before each policy call. It reuses the runner-reset
observation for the single nominal trial instead of adding an extra trial reset.
Single perturbation trials use the same nominal parity surface before the force
event is scheduled, so the perturbation evaluator does not fail before
confirmed carry.

Play-equivalent parity baseline from inside this experiment:

```bash
python experiments/carrybox_clean_perturb/evaluate.py \
  --resume_path legged_gym/logs/Ampstage1_UpAndWalk/Aug13_14-39-11_stage1_UpAndWalk/model_9500.pt \
  --parity_mode play_baseline \
  --parity_debug
```

No-force evaluator with actor-history, initial-state, first-20-step, and
checkpoint-loader parity diagnostics:

```bash
python experiments/carrybox_clean_perturb/evaluate.py \
  --resume_path legged_gym/logs/Ampstage1_UpAndWalk/Aug13_14-39-11_stage1_UpAndWalk/model_9500.pt \
  --no_force \
  --parity_debug \
  --checkpoint_parity
```

For independent evaluator trials, the experiment now clears evaluator state,
actor observation history, and AMP observation history before calling the
controlled trial reset. The clean training task and policy code are unchanged.

Pulse:

```bash
python experiments/carrybox_clean_perturb/evaluate.py \
  --resume_path legged_gym/logs/Ampstage1_UpAndWalk/Aug13_14-39-11_stage1_UpAndWalk/model_9500.pt \
  --profile half_sine \
  --direction=-box_y \
  --beta 0.5 \
  --pulse_duration 0.10
```

Sustained force:

```bash
python experiments/carrybox_clean_perturb/evaluate.py \
  --resume_path legged_gym/logs/Ampstage1_UpAndWalk/Aug13_14-39-11_stage1_UpAndWalk/model_9500.pt \
  --profile smooth_hold \
  --direction=-box_y \
  --beta 0.5 \
  --ramp_up 0.15 \
  --hold_duration 1.0 \
  --ramp_down 0.15
```

CSV output:

```bash
python experiments/carrybox_clean_perturb/evaluate.py \
  --resume_path legged_gym/logs/Ampstage1_UpAndWalk/Aug13_14-39-11_stage1_UpAndWalk/model_9500.pt \
  --profile smooth_hold \
  --direction=-box_y \
  --beta 0.5 \
  --save_csv \
  --output_dir experiments/carrybox_clean_perturb/results/single
```

Without `--output_dir`, CSV v2 is written below the checkpoint result directory
as `metrics_v2/summary.csv` with per-trial traces in `metrics_v2/traces/`.
The v2 outcome columns are `humanoid_failure`, `box_failure`, `timeout`,
`carry_achieved`, and `force_scheduled`; the old `physical_failure` column is
not emitted. Response-only metrics such as `contact_loss` and recovery are NaN
when no force event was scheduled. Existing CSV files with a different header
are rejected instead of being appended with misaligned columns.

Sweep:

fix env setting, suitable for full testing

```bash
python experiments/carrybox_clean_perturb/evaluate.py \
  --resume_path legged_gym/logs/Ampstage1_UpAndWalk/Aug13_14-39-11_stage1_UpAndWalk/model_9500.pt \
  --profile smooth_hold \
  --sweep \
  --directions=+box_x,-box_x,+box_y,-box_y \
  --betas=0.1,0.3,0.5 \
  --seeds=1,2,3 \
  --hold_durations=0.5,1.0,2.0 \
  --save_csv
```

for single robot walk testing, without csv output

```bash
python3 experiments/carrybox_clean_perturb/evaluate.py --resume_path legged_gym/logs/Ampstage1_UpAndWalk/Aug13_14-39-11_stage1_UpAndWalk/model_9500.pt --profile smooth_hold --directions=+box_x --betas=2.0 --hold_durations=5.0 --seeds=2 --sweep --verbose
```
For explicit play parity diagnostics (`--parity_mode play_baseline`), the command
is injected like `play.py`:

```text
vx = 0.8 m/s
vy = 0.0 m/s
yaw_rate = 0.0 rad/s
```

Normal single-trial and sweep evaluation both keep the independent controlled
setup: `num_envs = 1`, `episode_length_s = 30`, and fixed command `vx = 0.6`.

## Confirmed Carry

Confirmed carry is evaluation-only and never enters actor observations, task
observations, rewards, AMP observations, policy networks, or training logic.

The raw condition is:

```text
is_stage_carry
AND left_hand_contact_proxy
AND right_hand_contact_proxy
AND ||box_linear_velocity - robot_linear_velocity|| < 1.0 m/s
AND ||box_angular_velocity|| < 3.0 rad/s
```

`left_hand_contact_proxy` and `right_hand_contact_proxy` are thresholded net
contact forces on the left/right hand collision bodies. They are named as
proxies because `net_contact_force_tensor` does not expose the other body in the
contact pair. The force threshold is 1.0 N.

The raw condition must hold for 10 consecutive policy steps before
`confirmed_carry_buf` becomes true.

Recovery succeeds after force removal once the same raw confirmed-carry
condition is regained for 10 consecutive policy steps. `recovery_time` is
measured from the first policy step after force completion to that recovery
confirmation.

## Force Semantics

The force is applied at the box center of mass with:

```python
gym.apply_rigid_body_force_tensors(..., space=gymapi.CoordinateSpace.GLOBAL_SPACE)
```

For `+box_x`, `-box_x`, `+box_y`, and `-box_y`, the local box direction is
transformed to world space once at event commitment using the box orientation at
that instant. The resulting `direction_world` is frozen for the full event.

The force magnitude is:

```text
F_peak = beta * m_box * 9.81
```

One generator is used for both profiles:

```text
force_world = profile_scale(t) * F_peak * direction_world
```

The env subclass applies this force immediately before every `gym.simulate()`
call inside the policy decimation loop. Duration and impulse are measured in
physics steps using `sim_params.dt`.

## Validation Status

Static checks performed:

- Python compilation passed for the new evaluator files.
- Import/inheritance audit found no references in this new tree to:
  `carrybox_PI`, `carrybox_config_PI`, `carrybox_boxperturb`,
  `carrybox_boxperturb_config`, `carrybox_resume_config`, or
  `experiments/carrybox_perturb_debug`.

Runtime simulator checks are blocked in the current shell because all visible
Python interpreters fail on `import isaacgym`.
