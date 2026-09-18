# Physical carry constraints for loaded locomotion

This replaces the previous CarryWith target-matching design. The policy tracks arbitrary
velocity commands while satisfying broad physical carry constraints. CarryWith is an
**offline scale audit and coherent RSI source only**. The runtime reward and evaluation
trace do not load a calibration file, motion ID, motion time, phase, pose, ray, mean,
or quaternion from the demonstrations.

## Three modules and reward semantics

1. **Stable bilateral grasp.** Express each palm in the current box frame. Left belongs
   to +Y and right to -Y, using the actual randomized dimensions. Full surface reward
   applies within 30 mm of the correct face and within the x/z face rectangle plus a
   10 mm edge allowance. Tangential placement has no preferred point or direction.
   Differentiate each palm's box-local position and constrain x/z sliding separately.
   Existing filtered bilateral contact remains a separate physical reward.
2. **Upper-body guardrail.** The 14 named arm joints have broad intervals. There is no
   preferred center, no waist/leg supervision, and no target arm pose. Joint-limit,
   balance, torque and locomotion regularizers retain their existing roles.
3. **Box-body coupling.** Constrain the box center to a broad workspace in `torso_link`
   coordinates. Differentiate this local position to constrain relative sliding.
   This derivative cancels rigid translation/rotation, including yaw with an offset
   box. There is no world-COM velocity subtraction and no orientation target.

For interval constraints, `e = relu(lower - value) + relu(value - upper)`.
For tolerances, `e = relu(abs(error) - tolerance)`; hand slip uses the local x/z speed
norm before applying its tolerance. Each reward is:

```text
r = 1 / (1 + sum((e / softness)^2))
```

The surface term sums over both hands and all three constraint components. Arm
violations sum over joints so one badly twisted joint is not diluted by 13 valid
joints. This is a continuously differentiable dead-zone reward: exactly 1 throughout
the feasible region, zero gradient there, and a smooth decrease outside. `1-r` is
the corresponding bounded penalty. Softness determines only how rapidly a violation
reduces reward; it is not a statistical sigma or a demonstration confidence bound.

| Reward | Scale (before existing dt multiplication) |
|---|---:|
| carry_lin_vel_tracking | **3.0, unchanged** |
| carry_yaw_vel_tracking | **2.5, unchanged** |
| carry_hand_box_surface | 1.5 |
| carry_hand_slip | 0.5 |
| carry_bilateral_contact | 0.5, unchanged |
| carry_arm_range | 0.2 |
| carry_relative_position | 0.75 |
| carry_relative_velocity | 0.5 |
| carry_relative_orientation | **0.0**, implementation removed |

Tracking maxima sum to 5.5, versus 3.95 for these physical carry rewards. Surface
grip is the strongest individual carry reward. Existing tilt/uprightness, joint,
action, torque, feet and zero-command terms are unchanged.

## Offline evidence and deliberate engineering allowances

The audit was run **before editing final config values**, using the existing MotionLib
and URDF FK analysis on all 773 frames (256/194/323), sampled at 60 Hz, with causal
50 Hz local differences that never cross clip boundaries. Reproduce:

```bash
python -m experiments.carry_preservation.analyze_constraints
```

`constraint_statistics.json` records source hashes, pooled and per-motion min/max,
P5/P95, other robust summaries, and histograms. It never generates training settings.
The previous `statistics.json`, `reference_summary.md`, calibration JSON, and
`carry_preservation.py` are **historical offline artifacts**, not runtime dependencies.
The old analyzer still reproduces that historical audit.

The files contain no measured box sizes or contact forces. Box center is the stored
palm midpoint and orientation is synthesized from pelvis heading. Thus nominal-box
hand errors cannot be interpreted as measured contact tolerance. Two clips have
frozen arms; four wrist pitch/yaw channels are zero in every frame. Those zeros do
not justify fixing wrist joints.

| Quantity | Observed CarryWith range/scale | Final zero-violation range | Broadening and rationale |
|---|---|---|---|
| Left signed Y-side error, nominal box | -0.0704 .. -0.0472 m | -0.030 .. +0.030 m around **actual physical face** | Band width 0.060 m versus 0.0232 m observed variation (2.6x), re-anchored to physical geometry. Allows virtual palm/contact-patch offsets and 10 mm simulator contact offset. Does not reinterpret deep nominal-box penetration as valid. |
| Right signed Y-side error, nominal box | -0.0716 .. -0.0469 m | -0.030 .. +0.030 m | 2.4x observed variation; same physical reasoning. |
| Left tangential x/z position | x -0.0555 .. 0.0232; z -0.0092 .. 0.0114 m | x +/- (half_x + .010); z +/- (half_z + .010) | Entire physical face, with 10 mm edge tolerance. Nominal widths .370/.320 m, about 4.7x/15.5x observed spans; no ray or point target. |
| Right tangential x/z position | x -0.0189 .. 0.0587; z -0.0111 .. 0.0093 m | same size-aware rectangle | Nominal widths about 4.8x/15.7x observed spans. Measured nominal overflow is zero; 10 mm is an engineering edge allowance, not a percentile. |
| Left/right tangential hand speed | P95 .2109/.2098; maxima .2707/.2713 m/s | 0 .. .35 m/s each | About 1.66x P95, 1.29x max. Chosen as a round contact-motion allowance, leaving room for ordinary oscillation and moderate repositioning. Persistent faster sliding loses reward every step. |
| Torso-relative box x | .3266 .. .3637 m; P5/P95 .3376/.3579 | .18 .. .53 m | Adds .1466/.1663 m on the lower/upper sides; 9.4x full observed span. Allows substantial reach/load changes. |
| Torso-relative box y | -.0318 .. .0366 m; P5/P95 -.0194/.0124 | -.20 .. .20 m | Adds .1682/.1634 m; 5.9x full span. Symmetric freedom for lateral commands, turns and balance. |
| Torso-relative box z | .0252 .. .0824 m; P5/P95 .0338/.0639 | -.15 .. .25 m | Adds .1752/.1676 m; 7.0x full span. Allows vertical load and arm-height adjustment. |
| Local box vx | -.1573 .. .1326; absolute P95 .0810 m/s | -.35 .. .35 m/s | 2.23x absolute observed max / 4.32x absolute P95. Faster sustained drift must still encounter the position workspace. |
| Local box vy | -.2949 .. .3202; absolute P95 .1407 m/s | -.45 .. .45 m/s | 1.41x max / 3.20x P95. Larger lateral allowance for contact oscillation during turns. |
| Local box vz | -.3066 .. .2666; absolute P95 .1554 m/s | -.45 .. .45 m/s | 1.47x max / 2.90x P95. Larger vertical allowance for loaded gait oscillation. |

The side band is deliberately **not** widened to swallow the 5-7 cm synthetic
nominal-box mismatch. Raw reference palms still violate the simulated side band;
RSI is unchanged, and no offline palm retargeting is applied online.

Post-dead-zone softness values are 0.040 m (surface), 0.35 m/s (hand slip),
0.35 rad (arms), 0.10 m per box-position axis, and [.35,.45,.45] m/s for local box
velocity. A single violation equal to its softness gives reward 0.5. These are
engineering penalty slopes, not additional feasible bounds or quantities fitted
to reference residuals. The surface slope is intentionally strongest; arm and
workspace penalties allow gradual recovery instead of a hard cliff.

## Arm intervals, per joint

All values are radians. Added margins below are measured against the **full observed
min/max**, not just P5/P95. Shoulder pitch/yaw allow reach and turning; roll ranges
retain side-aware but broad lateral adjustment; elbows allow extension/flexion;
wrist roll has wide contact-orientation freedom; the frozen wrist pitch/yaw
channels each receive +/-0.60 rad. All intervals are inside the URDF hard limits.

| Joint | Observed min .. max | Observed P5 .. P95 | Final interval | Added lower / upper margin |
|---|---|---|---|---|
| left_shoulder_pitch_joint | -0.2354 .. -0.0237 | -0.1837 .. -0.0616 | -0.90 .. 0.55 | 0.665 / 0.574 |
| left_shoulder_roll_joint | 0.0263 .. 0.1872 | 0.0379 .. 0.1206 | -0.25 .. 0.80 | 0.276 / 0.613 |
| left_shoulder_yaw_joint | -0.1540 .. 0.0109 | -0.1331 .. -0.0324 | -0.85 .. 0.70 | 0.696 / 0.689 |
| left_elbow_joint | 0.0738 .. 0.2331 | 0.1189 .. 0.1994 | -0.25 .. 1.15 | 0.324 / 0.917 |
| left_wrist_roll_joint | -0.3320 .. -0.1662 | -0.2824 .. -0.2098 | -1.10 .. 0.55 | 0.768 / 0.716 |
| left_wrist_pitch_joint | 0.0000 .. 0.0000 | 0.0000 .. 0.0000 | -0.60 .. 0.60 | 0.600 / 0.600 |
| left_wrist_yaw_joint | 0.0000 .. 0.0000 | 0.0000 .. 0.0000 | -0.60 .. 0.60 | 0.600 / 0.600 |
| right_shoulder_pitch_joint | -0.5235 .. -0.2952 | -0.4881 .. -0.3633 | -1.10 .. 0.25 | 0.576 / 0.545 |
| right_shoulder_roll_joint | -0.3113 .. -0.2151 | -0.2978 .. -0.2562 | -1.00 .. 0.25 | 0.689 / 0.465 |
| right_shoulder_yaw_joint | 0.1681 .. 0.3399 | 0.2042 .. 0.2791 | -0.55 .. 1.00 | 0.718 / 0.660 |
| right_elbow_joint | 0.3991 .. 0.5802 | 0.4207 .. 0.5071 | -0.25 .. 1.40 | 0.649 / 0.820 |
| right_wrist_roll_joint | 0.2042 .. 0.2936 | 0.2326 .. 0.2716 | -0.55 .. 1.10 | 0.754 / 0.806 |
| right_wrist_pitch_joint | 0.0000 .. 0.0000 | 0.0000 .. 0.0000 | -0.60 .. 0.60 | 0.600 / 0.600 |
| right_wrist_yaw_joint | 0.0000 .. 0.0000 | 0.0000 .. 0.0000 | -0.60 .. 0.60 | 0.600 / 0.600 |

All nonconstant joint ranges occur in clip 1. Clips 2 and 3 have the same constant
14-vector (rounded):
`[-.1258,.0788,-.0899,.1583,-.2450,0,0,-.4299,-.2777,.2429,.4647,.2493,0,0]`.
Their per-motion intervals have zero width; clip 1's min/max equal the pooled
min/max in the table. The JSON retains full precision and each motion separately.

## Per-motion dynamic/workspace audit

Vectors are xyz; hand speeds are left/right. Positions in m, velocities in m/s.

| Motion | Torso box min .. max | Absolute local box velocity maxima | Hand tangential speed maxima |
|---|---|---|---|
| carrywith1.pt | [0.3266, -0.0318, 0.0252] .. [0.3637, 0.0366, 0.0824] | [0.1573, 0.3202, 0.3066] | [0.2553, 0.2541] |
| carrywith2.pt | [0.3464, -0.0040, 0.0481] .. [0.3486, -0.0030, 0.0488] | [0.0096, 0.0063, 0.0034] | [0.2311, 0.2115] |
| carrywith3.pt | [0.3468, -0.0042, 0.0483] .. [0.3490, -0.0027, 0.0489] | [0.0125, 0.0082, 0.0031] | [0.2707, 0.2713] |

These allowances are engineering starting points, not a guarantee of simulator
contact success or a measured noise envelope. Slow sub-tolerance drift is possible
inside the workspace; position/face bounds stop it accumulating without limit.
Large bounded penalties saturate, so track contact loss and boundary clearance when
evaluating training. No long PPO run was performed.

## History, diagnostics, and preserved behavior

`compute_reward` samples local positions once per policy step, after contact
filtering/termination checks and before parent reset, regardless of reward scales.
Reward accessors are pure: they never mutate history. Both local derivatives share
a per-environment validity mask. Reset invalidates only affected environments;
the first fresh post-reset sample initializes history, earns **zero temporal reward**,
and is excluded from temporal diagnostics. It is not reported as a successful zero
velocity. Subsequent samples use the actual `dt`, with no demonstration-dt assertion.

Training logs raw left/right side errors and their dead-zone violations, face
overflow and its violation, raw left/right tangential speeds and their violations,
maximum arm violation and minimum signed arm-boundary clearance, torso-region
violation and signed clearance, raw torso-relative xyz positions, local xyz
velocities/norm and velocity violation, and bilateral contact rate. Clearance is
positive inside, zero at a boundary, negative outside; it exposes approach to a
boundary despite zero penalty. The existing runner retains `TrackingRMSE/vx`,
`TrackingRMSE/vy`, and `TrackingRMSE/yaw_rate`.

Episode metrics count actual samples (not randomized episode ages), include terminal
samples, and clear stale episode dictionaries. No-valid-temporal-sample episode
aggregates are NaN. Evaluation reads the same cached physical metrics, excludes
invalid temporal samples, and performs no reference-based diagnostic calculation.
Old direction/orientation/reference-error CSV columns are removed; new physical
columns replace them. Historical CSV reductions leave missing new fields unavailable.
The pre-existing evaluator continues to skip terminal reset frames.

Unchanged: actor/critic observations and dimensions (738/126), architecture, 29-DoF
PD action space, control timing, PPO/optimizer/learning rate, command mixture/ranges,
coherent robot+box CarryWith RSI, reset distribution, termination logic, box size/mass
randomization, and general locomotion regularizers. The inherited unused AMP
observation return is still discarded by the existing specialist runner; no AMP,
discriminator, distillation, or online phase supervision is added.

Removed from runtime configuration and code:

- `carry_hand_direction_left`, `carry_hand_direction_right`,
  `carry_hand_direction_sigma`, box-to-palm target rays and direction error.
- `carry_hand_normal_sigma` and the continuous zero-centered surface Gaussian.
- `carry_arm_target`, `carry_arm_sigma`, and `carry_arm_pose`.
- `carry_box_relative_position_target`, `carry_box_relative_position_sigma`.
- `carry_box_relative_orientation_target`, `carry_box_relative_orientation_sigma`,
  quaternion target matching and orientation reward implementation.
- `carry_box_relative_velocity_sigma` and its Gaussian.
- `carry_reference_policy_dt` and its calibration-width lock.

Historical offline artifacts retain their original target/sigma names only to
reproduce the earlier audit; neither training nor evaluation imports them.

## Validation

CPU runtime tests execute the actual specialist methods, current config and parent
reward dispatcher, with simulator initialization/reset stubbed and project
TorchScript quaternion operations. They cover flat valid regions/zero gradients,
wrong side/same side/box-center hands, side/edge violation, tangential slip, individual
arm violations, box workspace/drift, all box size corners, no waist/leg coupling,
partial resets, terminal metrics, invalid first samples, disabled reward scales,
pure reward access, 4096 finite float32 environments, batch independence, full 3D
rigid motion and quaternion signs, and all three recorded motion clips. AST checks
protect unrelated config and reset/termination/command methods against a070bb2.

The suites passed: 22 carry/offline tests (including 9 new runtime-semantic tests),
6 evaluation metric tests, and 2 command-suite tests; 1 historical CUDA test skipped.
Syntax compilation, Python 3.8 AST parsing, import checks for simulator-independent
modules, and `git diff --check` also pass. Full simulator imports/dynamics cannot be
verified on this Windows host: CPU PyTorch 2.6.0 is present, Isaac Gym/CUDA are absent.
No policy-performance improvement is claimed without simulator evaluation.

```bash
python -m unittest discover -s experiments/carry_preservation/tests -v
python -m unittest discover -s experiments/carrybox_locomotion_eval/tests -v
python -c "from experiments.carrybox_locomotion_eval.tests import test_command_suite as t; t.test_default_suite_counts_ranges_and_metadata(); t.test_mode_filter_restarts_trial_ids()"
```

On a configured Isaac Gym machine, only a short smoke run is needed for this change:

```bash
python legged_gym/legged_gym/scripts/train_carry_locomotion.py \
  --task carrybox_locomotion --init_policy_path <pretrained_checkpoint> \
  --num_envs 64 --max_iterations 2 --headless
```
