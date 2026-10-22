# carrybox_pickup_stand

Branch: `research/pickup-stand`, based on `velocity-tracking`.

The independent task is **APPROACH → PICKUP → HOLD**. It has no loaded walking,
carryWith, moving-carry or walk-to-stop phase. Existing carrybox/carrybox_locomotion
environment and configuration files are unchanged.

## Interface and control

The inherited actor receives six histories of 123 values (738 total): angular
velocity, gravity, 29 joint positions, 29 joint velocities, five end effectors,
29 previous actions, and the 15-value box/task observation. The critic receives
126 values. The task does not append a phase indicator. Its commands, box geometry
and proprioception supply the existing policy interface. Actions remain the same
29 joint targets, action scale 0.25, P control, original joint order/gains/limits,
5 ms simulation and 20 ms policy period. Box mass/size mixture, gravity,
friction, collision and platform physics are inherited without changes.

`audit.json` records a strict load of the supplied Full AMP checkpoint into the
actual ActorCritic and AMP implementations: actor 738→512→256→256→29, critic
126→512→256→256→1, discriminator 600→512→256→1. A finite actor forward pass
succeeded. Current URDF movable joints match joint_id.txt and default joint order.
The old checkpoint has no joint-order metadata: historical provenance cannot be
proved from the weights alone. The runtime rejects a different current joint order.

## Reference and reset audit

`carrybox.yaml` has pickUp/putDown/carryWith/loco; `carrybox_no_relocation.yaml`
still contains carryWith. The new YAML selects the original three pickup and
eleven approach loco clips, with original weights. Neither carryWith nor putDown
is loaded as an expert.

The final frames look upright and loaded, but do not describe static two-foot
standing. Measured at 60 Hz, with box Z corrected exactly as MotionLib resets it:

| Motion | Final base height m | Final root speed m/s | Final root angular speed rad/s | Final box speed m/s | Final joint velocity RMS rad/s | Foot height difference m |
|---|---:|---:|---:|---:|---:|---:|
| pickup1 | .712 | .523 | 1.170 | .466 | 2.499 | .046 |
| pickup2 | .731 | .383 | 3.762 | 1.204 | .822 | .049 |
| pickup3 | .739 | .059 | 1.557 | .792 | .519 | .140 |

No clip passes even the kinematic stationary-candidate screen during its last
0.5 s. Reference poses contain no measured contact forces, so a kinematic pass
would still not establish physical validity. No reference frame is labeled HOLD,
no velocities are zeroed to fabricate a static state, and no static AMP expert
is synthesized. See `audit.py`/`audit.json` for the complete inspection.

MotionLib hardcodes a pickUp contact_index allocation, but has no mandatory
carryWith load dependency. Its putDown goal accessor is never called here. The
parent box reset loops over pickUp/carryWith with guards; the absent skill is
safe. This task bypasses the parent's skill sampler and excludes its default RSI
root-velocity randomization. Robot, box, platform and phase share exact motion/time
IDs. The root's corrected Z derivative and box linear/yaw derivatives are retained;
the supported, pre-grasp box remains stationary. All selected skipped RSI ranges
are empty. No putDown reset/goal is used. MotionLib's original RPY-based root angular
velocity approximation and heading-only box orientation remain reference limitations.

The task adapter clamps interpolation inside each clip, including the final ceil
index (the original history helper can leak into the next clip). Dynamic experts
use fixed physical 50 Hz history spacing; time-warp augmentation is omitted to
avoid changing time spacing without recomputing derivative features.

## Transitions and thresholds

Distances are metres, angles radians, velocities m/s and rad/s. Thresholds are
configuration values in `pickup_stand`, intended for simulator validation/tuning.

* APPROACH keeps the inherited heading/box approach reward and samples forward
  speed in [.2,.8]. Bilateral box contact within .7 m for .10 s enters PICKUP.
* PICKUP stops task movement commands immediately on grasp; the inherited hand
  placement, lifting and contact rewards remain active. It never requests loaded
  forward motion while waiting to stand up.
* Completion requires .20 s of uninterrupted evidence: pelvis height ≥.70 above
  terrain, torso tilt ≤.25, box center height [.67,1.05], box tilt ≤.25, box bottom
  ≥.08 above both ground and source platform top, bilateral contact, both feet
  vertical forces >10 N, box/root relative speed ≤.20, and root/box vertical
  speeds ≤.15. These reject a lifted box during a crouch, a tossed box, support
  from the platform, unilateral grasp, and an unfinished upward swing. XY speed
  is deliberately not a completion gate; a completed lift immediately enters HOLD
  even if residual horizontal momentum still needs balance correction.
* HOLD is irreversible until reset. All command components remain zero, velocity
  tracking gates are active with zero targets, and the approach reward is zero.
  Failed physical completion conditions for >.50 s terminate a dropped/unloaded
  hold. Original fall/timeout conditions still apply.

Bilateral contact requires >1 N net force on each rubber hand, each palm within
.06 m of opposing box-local Y faces and near the box surface. Force alone could
come from the platform/floor, so geometry and opposite sides are required too.
This remains a geometric proxy for a collision-pair contact sensor; verify its
6 cm allowance against actual hand collision geometry in Gym.

Stable HOLD additionally requires full base/box linear speeds ≤.12, angular
speeds ≤.20, and each foot speed ≤.08 for .50 s. Successful hold duration is the
longest continuous interval satisfying these checks; success requires 3 s.
Pickup-to-stable delay is measured from completion to the **start** of the first
validated .50 s stable interval. Missing stable holds are reported as missing,
not a zero-second success. HOLD-reset episodes are excluded from pickup success.

## Rewards and AMP

Inherited pickup/contact rewards and regularizers stay active. HOLD adds seven
small terms: base horizontal/full angular stillness (.75), box linear/angular
stillness (.75), bilateral contact (1), box height/orientation retention (.75),
upright standing with a [.70,.88] pelvis-height dead zone and soft leg deviation
(.75), two-foot contact/low foot speed (.5), and action-rate cost (-.02).
Arms are free to find a loaded posture. There is no joint freeze, zero-joint-speed
target, hard-coded action or displacement-based carry phase.

The 600-value AMP observation is unchanged. Optional rollout stage metadata is
stored separately. Expert minibatches are sampled from loco for APPROACH and
pickUp for PICKUP. HOLD, initial histories and mixed-stage histories are excluded
from both discriminator loss and normalization. At least ten consecutive policy
steps in one dynamic phase are required after a reset/transition. Stage metadata
and diagnostics are copied **before** autoreset. PPO still learns from all rows.

Dynamic reward is .75 task + .25 AMP (including the inherited .5 AMP reward scale).
HOLD/invalid-history reward is **1.0 task**, with no AMP contribution. All-HOLD
minibatches produce zero discriminator loss and leave discriminator weights
unchanged. Legacy tasks retain their original minibatch format, expert sampling
and reward mixing unless they explicitly emit `amp_stage`.

## Reset coverage and curriculum

Reset columns are `[validated HOLD, early pickup, mid pickup, late pickup, full
approach]`. Early spans [0,contact], mid spans the first half after contact and late
spans the second half. A reference reset's phase is APPROACH before grasp and
PICKUP afterward. Reference history is not falsely treated as static HOLD.

| Stage | Default reset weights |
|---|---|
| hold | .65, .05, .05, .20, .05 |
| pickup | .20, .20, .25, .25, .10 |
| full | .10, .10, .10, .15, .55 |

Stage advancement is explicit, after evaluation. Nonzero early/mid/approach
coverage, retained pickup AMP and a fixed 1e-4 learning rate protect pickup skills;
they cannot guarantee absence of forgetting, so evaluate before promotion.

There is **no validated HOLD bank available in this checkout**. By default the
missing bank's weight is visibly reassigned to late dynamic pickup (warning),
allowing a hold-focused bootstrap from the Full AMP actor. To require genuine
loaded-standing reset coverage use `--hold-bank PATH --require-hold-bank`.
The bank collector accepts only simulated states that pass all physical checks
continuously for .50 s. It stores actual root/box/joint velocities, contact
evidence, dimensions, mass and joint-order metadata. Empty collection fails; it
never writes fabricated states. Loading rejects moving/invalid states or mismatched
joint order. Sampling requires box dimensions within .015 m and mass within .15 kg;
insufficient coverage fails explicitly. Use matching seed/env count for initial
collection and training, or collect sufficient coverage of the mass/size mixture.
Robot randomization and subsequent contact evolution require online revalidation
after any bank reset; a saved row does not count as a new successful hold.

Reset histories are cleared per environment, seeded from reset robot/box state,
and end-effector geometry comes from URDF forward kinematics. Initial observation
rotation uses the reset torso orientation; torso gyro/linear velocity is seeded
from root velocity (waist joint twist is refreshed by Gym on the first step).
The first reset does not execute an unloaded zero action before policy inference.
Terminal actions/contact histories cannot leak across autoresets.

## Commands (run from repository root in the Isaac Gym training environment)

Offline audit and CPU tests:

```bash
python experiments/pickup_stand/audit.py --checkpoint shared_artifacts/2026-10-19/carry_track/model_33999.pt
python -m pytest experiments/pickup_stand/tests -q
```

Bounded integration smoke and baseline evaluation (no optimization):

```bash
python legged_gym/legged_gym/scripts/pickup_stand.py --mode smoke --steps 100 --num_envs 16 --headless --seed 1 --finetune_path shared_artifacts/2026-10-19/carry_track/model_33999.pt --output outputs/pickup_stand/smoke.json
python legged_gym/legged_gym/scripts/pickup_stand.py --mode evaluate --steps 3000 --num_envs 256 --headless --seed 1 --reset-suite approach --finetune_path shared_artifacts/2026-10-19/carry_track/model_33999.pt --output outputs/pickup_stand/baseline.json
```

Optional bootstrap when collection from the original Actor yields no stable holds:

```bash
python legged_gym/legged_gym/scripts/pickup_stand.py --mode train --stage hold --num_envs 4096 --headless --seed 1 --max_iterations 2000 --finetune_path shared_artifacts/2026-10-19/carry_track/model_33999.pt
```

After selecting a checkpoint, replace `HOLD_CHECKPOINT`, `PICKUP_CHECKPOINT` and
`FULL_CHECKPOINT` with actual saved paths. Collection writes validated.pt next to
validated.json. Collection defaults to full approach; use `--reset-suite pickup`
for more frequent pickup-to-hold attempts.

```bash
python legged_gym/legged_gym/scripts/pickup_stand.py --mode collect --stage hold --reset-suite pickup --steps 3000 --num_envs 4096 --headless --seed 1 --finetune_path HOLD_CHECKPOINT --output outputs/pickup_stand/validated.json
python legged_gym/legged_gym/scripts/pickup_stand.py --mode train --stage hold --hold-bank outputs/pickup_stand/validated.pt --require-hold-bank --num_envs 4096 --headless --seed 1 --max_iterations 2000 --finetune_path HOLD_CHECKPOINT
python legged_gym/legged_gym/scripts/pickup_stand.py --mode train --stage pickup --hold-bank outputs/pickup_stand/validated.pt --require-hold-bank --num_envs 4096 --headless --seed 1 --max_iterations 3000 --finetune_path HOLD_CHECKPOINT
python legged_gym/legged_gym/scripts/pickup_stand.py --mode train --stage full --hold-bank outputs/pickup_stand/validated.pt --require-hold-bank --num_envs 4096 --headless --seed 1 --max_iterations 5000 --finetune_path PICKUP_CHECKPOINT
python legged_gym/legged_gym/scripts/pickup_stand.py --mode evaluate --stage full --steps 3000 --num_envs 256 --headless --seed 1 --reset-suite approach --finetune_path FULL_CHECKPOINT --baseline-metrics outputs/pickup_stand/baseline.json --output outputs/pickup_stand/full_eval.json
```

Also evaluate `--reset-suite pickup` and `--reset-suite hold --hold-bank PATH`
against their own matched baselines. Promote hold→pickup after reliable ≥3 s
holds and pickup→full after pickup success stays within 5 percentage points of its
baseline while hold succeeds. These are suggested evaluation gates, not claims
about the untrained implementation. Each finetune strictly loads Actor/Critic and
AMP, starts a fresh optimizer and resets iteration to zero. No teacher,
distillation or locomotion specialist is involved.

Reports contain pickup and approach success, completion-to-stability delay,
continuous hold duration, base/box linear/angular speeds, bilateral contact,
height/orientation deviations, two-foot contact and slip. Full per-episode data,
reset-category breakdown and censored partial episodes are saved. Baseline
comparisons require matching seed, env count, horizon and reset suite; deltas
include approach/pickup quality and hold success.

## Validation and remaining work

CPU audit, strict checkpoint loading, and **30 tensor/runtime-harness tests pass**.
Tests exercise the production MotionLib, stage sampler, environment methods and
actual HIMPPO optimization with legacy, mixed, and all-HOLD batches. The Gym
boundary is replaced by a harness; these tests do not establish physical stability.
The local Windows interpreter has CPU PyTorch 2.6 and no Isaac Gym. Consequently
GPU integration smoke, pickup success, hold duration and regression metrics are
**not measured**, and no expensive training was started. The numerical transition
and reward tolerances still need physical validation on the original Isaac Gym
stack. Box orientations are heading-derived, not independently measured in the
reference files; the saved HOLD bank must come from actual simulated motion.

The existing carry-preservation suite was also checked: 17 of 18 tests passed;
one evaluator-history test cannot import `isaacgym`. Compilation and
`git diff --check` pass. The local interpreter lacks pytest, so the test run loaded
the installed pure-Python pytest from `D:/Anaconda3/Lib/site-packages` after
initializing the local PyTorch optimizer; no packages or global environments were
modified. Standard `python -m pytest` applies in an environment with pytest installed.

## Exact files changed

New:

* `legged_gym/legged_gym/envs/g1/carrybox_pickup_stand.py`
* `legged_gym/legged_gym/envs/g1/carrybox_pickup_stand_config.py`
* `legged_gym/legged_gym/envs/g1/pickup_stand_logic.py`
* `legged_gym/legged_gym/envs/motionlib/pickup_stand_sampling.py`
* `legged_gym/resources/config/carrybox_pickup_stand.yaml`
* `legged_gym/legged_gym/scripts/pickup_stand.py`
* `experiments/pickup_stand/audit.py`
* `experiments/pickup_stand/audit.json`
* `experiments/pickup_stand/tests/test_pickup_stand.py`
* `experiments/pickup_stand/README.md`

Modified (registration and optional stage metadata/reward path only):

* `legged_gym/legged_gym/envs/__init__.py`
* `rsl_rl/rsl_rl/storage/him_rollout_storage.py`
* `rsl_rl/rsl_rl/algorithms/him_ppo.py`
* `rsl_rl/rsl_rl/modules/amp.py`
* `rsl_rl/rsl_rl/runners/him_on_policy_runner.py`
