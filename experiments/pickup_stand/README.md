# Scratch PhysHSI Approach → Pickup → Stand Hold

Branch: `research/pickup-stand` in `Ir1222/Carry12`. One Actor is trained from
random initialization with AMP + PPO. No teacher, specialist, distillation,
CarryWith locomotion, deceleration or walking-to-stop phase is present.

## Comparison with original PhysHSI

The comparison uses the locally available upstream commit
[`07bf22bdaf9c23c1d40366334a744c1816b6d42b`](https://github.com/InternRobotics/PhysHSI/tree/07bf22bdaf9c23c1d40366334a744c1816b6d42b).
`audit.py` reads the actual upstream Git blobs, checks the original functions and
records their blob hashes and source in `audit.json`; it does not assume the
velocity-tracking branch is original PhysHSI.

| Component | Original PhysHSI | This task |
|---|---|---|
| walk_task | World-frame speed toward box, fixed target_speed_loco=.85, position/facing | Same far-box formula and configurable .85 target; only APPROACH |
| Near box | Approach terms saturate | Speed/position saturate, facing stays active until aligned |
| carryup_task | Palm-midpoint alignment and box-height reward, plus goal exceptions | Same alignment/lifting formula without relocation/putdown goal exceptions |
| AMP | 600D, .25 coefficient, runner's .5 AMP prediction scale | Same feature representation/scales, stage filtering for loco/pickUp |
| Reset | .8 hybrid RSI, skill-weighted motions/default reset | Explicit default/loco approach + stratified pickup RSI distribution |
| Relocation/putdown/carryWith | Part of original broader task | Excluded from task, resets, goals and AMP experts |
| Training initialization | AMP + PPO | Random Actor, Critic, discriminator and fresh optimizer |

The runtime inherits the current G1 simulator/PD/asset implementation for reuse,
but defines its own objective and PPO configuration. Carry velocity tracking,
heading hold, coupled velocity reward and carry contact reward scales are all
explicitly zero. Command sampling and curriculum callbacks are disabled/overridden.
The inherited hand-placement/lifting implementation is equivalent to the original
PhysHSI pickup component after removing its relocation/goal shortcuts.

## Phases and observations

* APPROACH rewards motion **toward the box** at `rewards.target_speed_loco=.85`
  m/s and facing it. Speed is a task reward parameter, not a sampled velocity
  command. Distance <.7 m and facing error ≤.50 rad for .10 s enter PICKUP.
* PICKUP starts with reaching/grasping, before bilateral contact is mandatory.
  It rewards inherited hand placement/lifting, bilateral grasp, standing up and
  continuous loaded balance. It has no forward-motion objective.
* HOLD requires verified pickup completion, targets loaded standing and low
  base/box motion, and penalizes lost support/unnecessary foot motion. Small
  corrective actions are allowed. Stable hold for `success_hold_seconds=3.0`
  marks success and ends the episode by default, with a terminal bonus of 5.

Actor shape remains 738 = six histories of 123; Critic remains 126 and actions
remain 29 in the original G1 joint order. The last three values of **each** Actor
history and of the Critic task observation are now `[APPROACH,PICKUP,HOLD]`
one-hot. Both normal observations, terminal Critic observations and reset history
seeding use these semantics. Box/proprioception noise behavior is retained.

All internal velocity command buffers remain zero in every phase. They no longer
drive observations or reward. Phase semantics are tagged
`pickup_stand_phase_onehot_v2` in saved checkpoints; the same numerical input
shape does **not** make legacy command-conditioned weights semantically compatible.

PD gains, action scale .25, torque/joint limits, 29D ordering, box physics and
the 5 ms simulation / 20 ms policy period are retained. Existing `carrybox` and
`carrybox_locomotion` environment/configuration files and robot assets are unchanged.

## Foot support fix and completion diagnostics

The old detector checked `ankle_pitch_link`. The URDF's shoe collision meshes are
on **left/right_ankle_roll_link**, connected through separate revolute joints with
`collapse_fixed_joints=False`. The roll mesh is about .208 m long and .0756 m wide;
the pitch mesh is a small ankle component. A supported roll link need not produce
any force on the pitch link. That explains a concrete false-negative mechanism;
it does not prove that every baseline episode satisfies the remaining checks.

The task resolves support rows explicitly with
`find_actor_rigid_body_index(..., gymapi.DOMAIN_ENV)`, checking the three-body
platform/target-platform/box prefix of the per-environment tensor. Each foot must
have **positive world-Z force >10 N**, separately. Norms, absolute negative forces,
or support from only one foot do not qualify. Pitch-link force is logged alongside
roll-link force to expose the previous error. Support/slip/footstep rewards use
the roll bodies. Original pitch-link end effectors remain in the AMP observation,
preserving its 600D motion convention.

The original completion thresholds were **not relaxed**. Every predicate is
logged as `completion/<name>`; underlying measurements use `completion_value/`:

| Predicate | Threshold |
|---|---|
| base_height | pelvis ≥.70 m above terrain |
| base_upright | torso tilt ≤.25 rad |
| box_height_min / box_height_max | box center [.67,1.05] m |
| box_upright | box tilt ≤.25 rad |
| support_clearance | box bottom ≥.08 m above ground and source-platform top |
| bilateral_contact | >1 N on both rubber hands, palms within .06 m of opposing box Y surfaces |
| both_feet_support | roll-link vertical force >10 N on each side |
| relative_speed | box/root relative linear speed ≤.20 m/s |
| base_vertical_speed / box_vertical_speed | each ≤.15 m/s in magnitude |

All checks must persist for .20 s to enter HOLD. Horizontal stillness is not a
prerequisite that creates a hidden moving-carry stage. Stable HOLD additionally
requires base/box linear speed ≤.12 m/s, angular speed ≤.20 rad/s, and each support
body speed ≤.08 m/s. The .50 s stability confirmation and configurable success
duration use continuous, reset-safe timers. Failed completion conditions for
>.50 s during HOLD terminate a failed hold. Original fall/timeout checks remain.

The `loaded_balance` reward supplies continuous feedback **during late PICKUP**,
even if a foot or stand-up predicate has not passed: lift progress × bilateral
grasp × a weighted mixture of base/box stillness, standing posture, per-foot load,
foot motion and box levelness. The soft `pickup_standup` term also operates before
HOLD. Hard success checks therefore no longer gate all balance feedback.

Net contact force plus palm geometry is still a proxy for collision-pair sensing.
Shoe force is measured at the body; slip diagnostics use its body origin, not
individual contact-point velocities. These limitations and contact-noise behavior
need Isaac Gym validation; visual standing alone is never labeled validated.

## Rewards, AMP and resets

Task rates: walk 1, carryup 1, pickup grasp .5, pickup stand-up .75, late-pickup
loaded balance 1; HOLD base stillness .75, box stillness .75, contact 1, box pose
.75, posture .75, feet .5, extra action-rate cost -.02 and footstep cost -.25.
Original PhysHSI regularizers are listed explicitly. Terminal success is a bonus
of 5, not a rate. No arm pose lock, joint freeze or zero-joint-velocity target is
introduced. A soft leg-posture term leaves the arms free to support the load.

Only the original eleven loco and three pickUp clips are loaded. Dynamic experts
are sampled from loco in APPROACH and pickUp in PICKUP. The 600D representation
is unchanged. HOLD and reset/mixed-stage histories never enter discriminator loss
or normalization. Dynamic reward is .75 task + .25 AMP; HOLD/masked histories
receive **1.0 task**. All-HOLD minibatches do not update discriminator weights.

Pickup reference tails remain moving (see `audit.json`); no last-frame static
expert or HOLD state is fabricated. Robot/box RSI uses matching clip/time IDs,
preserves reference velocities, corrects root-Z differentiation, and clamps AMP
interpolation to each clip. Parent terminal action/contact histories are cleared;
reset end-effector geometry is computed with URDF FK before policy inference.

`pickup_stand.reset_weights` columns are:

```
[validated_hold, early_pickup_RSI, mid_pickup_RSI, late_pickup_RSI, full_approach]
default: [0, .20, .20, .20, .40]
```

Early pickup spans [start,contact], mid/late split the post-contact interval.
**Every pickUp RSI state starts in PICKUP**, so its reaching segment uses pickup
AMP too. Full approach starts unloaded: by default half from loco RSI and half
from the default robot pose, controlled by `approach_rsi_probability=.50`.
This gives 60% pickup RSI, 20% loco approach RSI and 20% default approach coverage.

Weights must be finite, nonnegative, have positive sum and contain five entries;
they are normalized explicitly. Nonzero HOLD weight without a bank is an error.
There is no redistribution/fallback. A bank is optional with zero default weight.
Version-2 banks must come from a phase-v2 policy's continuously validated simulated
HOLD and retain actual velocities, force/contact evidence, size and mass.
Version-1 banks based on the previous detector are rejected. Dimension/mass match
tolerances remain .015 m/.15 kg. Revalidation after reset is still required.

## Scratch initialization and evaluation isolation

The dedicated entrypoint and the registered task's `scratch_only` runner guard
reject `--finetune_path`, `--resume`, policy-init paths and legacy checkpoint flags
for training. No Actor, Critic, AMP or optimizer state is imported. Before the
first update, `.initialization.json` records empty optimizer state, iteration zero,
parameter SHA256 hashes, `checkpoint_loaded=false`, reset weights and reward scales.

`--checkpoint PATH` is accepted only by evaluate/collect. It requires matching
phase-v2 task/observation/joint/architecture metadata and approach-speed semantics.
The old `model_33999.pt` remains unchanged for independent legacy evaluation; it
cannot be relabeled as a phase-conditioned scratch policy by matching shapes.

## Ubuntu commands (repository root, existing physhsi environment)

CPU checks and reproducible reference/mesh audit:

```bash
python -m pytest experiments/pickup_stand/tests -q
python experiments/pickup_stand/audit.py
```

Validate actual Gym support indices/forces first. This uses an unloaded default
PD stance with perturbations disabled and **no policy/checkpoint**. It requires
every environment to demonstrate ≥.20 s of simultaneous two-foot support and low
base speed, and saves raw force traces. A pass is not proof of loaded HOLD.

```bash
python legged_gym/legged_gym/scripts/pickup_stand.py --mode contact-check --steps 500 --num_envs 16 --headless --seed 1 --output outputs/pickup_stand/contact_probe.json
```

Random-policy simulator smoke, then two bounded PPO training iterations:

```bash
python legged_gym/legged_gym/scripts/pickup_stand.py --mode smoke --steps 100 --num_envs 16 --headless --seed 1 --output outputs/pickup_stand/scratch_smoke.json
python legged_gym/legged_gym/scripts/pickup_stand.py --mode train-smoke --smoke-iterations 2 --num_envs 32 --headless --seed 1 --output outputs/pickup_stand/train_smoke.json
```

`train-smoke` is bounded to ≤10 iterations and ≤128 environments, with 32 steps
per rollout and two PPO epochs/minibatches. It records parameter updates and checks
finite parameters. It loads no checkpoint. Do not infer policy success from this
short test; a random scratch actor is expected to fail initially.

Only after reviewing probe/smoke results, explicitly start longer scratch training:

```bash
python legged_gym/legged_gym/scripts/pickup_stand.py --mode train --num_envs 4096 --headless --seed 1 --approach-speed 0.85 --hold-seconds 3 --reset-weights 0 .2 .2 .2 .4 --max_iterations 20000 --output outputs/pickup_stand/scratch_train.json
```

Logs/checkpoints go to `legged_gym/logs/carrybox_pickup_stand_scratch/`. Replace
`SCRATCH_CHECKPOINT` with an actual saved phase-v2 checkpoint for evaluation:

```bash
python legged_gym/legged_gym/scripts/pickup_stand.py --mode evaluate --checkpoint SCRATCH_CHECKPOINT --reset-suite approach --steps 3000 --num_envs 256 --headless --seed 1 --output outputs/pickup_stand/approach_eval.json
python legged_gym/legged_gym/scripts/pickup_stand.py --mode evaluate --checkpoint SCRATCH_CHECKPOINT --reset-suite pickup --steps 3000 --num_envs 256 --headless --seed 1 --output outputs/pickup_stand/pickup_eval.json
```

Reports include per-predicate PICKUP failure counts, per-phase predicate pass
rates/numeric values, pitch/roll forces, base/box linear/angular speeds, contact,
height/orientation error, support-body slip, footstep counts, hold duration and
success, approach/pickup quality, reset-category statistics and censored episodes.
`--baseline-metrics PATH` compares matched phase-v2 evaluations only. HOLD-bank
resets are excluded from pickup-success and completion-to-stability denominators.

Optional bank collection after scratch training:

```bash
python legged_gym/legged_gym/scripts/pickup_stand.py --mode collect --checkpoint SCRATCH_CHECKPOINT --reset-suite pickup --steps 3000 --num_envs 256 --headless --seed 1 --output outputs/pickup_stand/validated.json
```

This writes `validated.pt` only if real simulated stable states exist. To use it,
explicitly set `--hold-bank PATH --reset-weights .1 .2 .2 .2 .3`; all sampled box
sizes/masses need matching bank coverage. Bank-based resets do not load policy
parameters. Pure scratch training needs no bank.

Independent old-policy comparison uses its **original task/command semantics**:

```bash
python legged_gym/legged_gym/scripts/play.py --task carrybox --resume --resume_path shared_artifacts/2026-10-19/carry_track/model_33999.pt --num_envs 16 --headless
```

That legacy play run uses the existing carrybox command behavior; its rewards are
not interchangeable with the new phase-task rewards. It never initializes scratch
training. Legacy checkpoint shape auditing remains available separately with
`audit.py --checkpoint PATH --output outputs/pickup_stand/legacy_audit.json`.

## Validation and unresolved issues

* CPU tests: **55 passed**, including original-reward numerical comparison,
  collision-mesh/index audit, every predicate, reset distributions, phase semantics,
  command disabling, AMP filtering/mixing and checkpoint guards.
* Startup regression executes the real CarryBox constructor up to the Gym
  boundary. The new task owns its common configuration parser and accepts zero
  carry command ranges; it never calls the parent's moving-carry-only positive
  speed check. Timing, rewards, command-range buffers and terrain settings are
  still parsed. The original CarryBox positive-speed validation is unchanged.
* A two-iteration CPU training smoke runs the actual HIMOnPolicyRunner/HIMPPO and
  full 738D Actor, 126D Critic and 600D discriminator with a synthetic VecEnv.
  `torch.load` is forbidden during that test; all three networks update from
  random weights and Adam starts empty. Finite losses/parameters and new checkpoint
  metadata round-trip are checked. This does **not** execute PhysX.
* Syntax compilation and `git diff --check` pass. Existing CarryBox/Locomotion
  environment/config files, URDF and old checkpoint are unchanged.
* Isaac Gym contact probe was attempted locally and stopped at import with
  `ModuleNotFoundError: isaacgym`. The current Windows host has CPU PyTorch 2.6;
  no Ubuntu SSH connection was supplied. **Gym contact validation and Gym training
  smoke remain unrun**; run the commands above on Ubuntu before long training.
* Thresholds remain physically motivated but need simulator evidence, especially
  per-foot force noise and palm geometry tolerance. Moving reference tails cannot
  establish a static HOLD initial state. Current reference box orientation is
  heading-derived; root angular derivatives use MotionLib's original approximation.
  Reset gyro seeds root twist; Gym refreshes the actual torso twist on the first step.

No long training was started and no trained success rate is claimed. See
`validation.json` for the explicit run/blocked status.

## Exact files changed in this revision

Modified:

* `legged_gym/legged_gym/envs/g1/carrybox_pickup_stand.py`
* `legged_gym/legged_gym/envs/g1/carrybox_pickup_stand_config.py`
* `legged_gym/legged_gym/envs/g1/pickup_stand_logic.py`
* `legged_gym/legged_gym/scripts/pickup_stand.py`
* `legged_gym/legged_gym/utils/task_registry.py` — opt-in scratch load guard
* `rsl_rl/rsl_rl/runners/him_on_policy_runner.py` — optional new-task save metadata
* `rsl_rl/rsl_rl/algorithms/him_ppo.py` — opt-in set_to_none gradients so HOLD
  batches do not move an inactive discriminator via old Adam momentum/weight decay
* `experiments/pickup_stand/audit.py`
* `experiments/pickup_stand/audit.json`
* `experiments/pickup_stand/tests/test_pickup_stand.py`
* `experiments/pickup_stand/README.md`

New:

* `legged_gym/legged_gym/utils/pickup_stand_experiment.py`
* `experiments/pickup_stand/validation.json`
