# Carry preservation for the locomotion specialist

The specialist now rewards bilateral grasp geometry, arm posture, torso-relative
box pose and relative translation. Linear/yaw command tracking remains at 3.0/2.5.
All calibration uses the repository's CarryWith dataset. No discriminator,
adversarial reward, teacher, reference-phase tracking or new termination rule was
added. Actual reduction of policy drift still needs an Isaac Gym training run.

## Reference findings

The audit follows both `carrybox_locomotion.yaml` and `carrybox_no_relocation.yaml`.
They identify the same three CarryWith files: 256, 194 and 323 frames, respectively,
at 60 Hz (773 frames total). The analyzer loads them through the existing
MotionLib and reconstructs the current URDF's torso, palms and rubber-hand links.
Both raw stored palms and FK palms are reported; their disagreement is a few
millimetres. MotionLib's common reset-height correction is applied consistently
to robot and object, so it cancels from relative geometry.

`statistics.json` contains mean, median, standard deviation, MAD, 5/25/50/75/95
percentiles, min/max, 12-bin histograms and covariance for each vector quantity,
both pooled and per motion. It also contains phase thirds, correlations with
phase/vx/vy/yaw rate, and per-motion reward distributions. Covariance matrices and
histograms are diagnostic; the online objective uses simple diagonal kernels.

| Candidate | Measured evidence | Decision |
|---|---|---|
| Left/right hand in box coordinates | FK palm mean positions are approximately `[-.0168, .1161, .0028]` and `[.0218, -.1157, -.0027]` m. Every frame has left `+Y`, right `-Y`. Standard deviations are about `[.0205, .0043, .0044]` / `[.0204, .0049, .0044]` m. | Explicit signed side faces, never nearest arbitrary surface. |
| Fixed tangential hand target | Hand X correlates with torso–box relative yaw with absolute correlation about 0.985–1.000. Histograms cover a continuous rotation-dependent range rather than independent grasp modes. | Preserve the hand ray expressed in the torso frame; its box-frame projection changes with current relative rotation. |
| Hand midpoint | Stored box center equals stored palm midpoint to within `1.2e-7` m. FK midpoint differs by a few millimetres. | Diagnostic-only; adding a second centering reward would mostly duplicate the bilateral geometry constraint. |
| Arm joints | Clips 2/3 have constant arms; about 67% of pooled frames equal the median. Clip 1's nonconstant arm-joint standard deviations span roughly 0.019–0.055 rad. Four wrist pitch/yaw channels are always zero. | Static 14-joint prior; no data-supported alternate arm modes. Equal weighting with a 0.15 rad floor, including the zero channels, avoids interpreting frozen data as certainty. No waist/leg targets. |
| Root/base vs pelvis | They are the same frame in this URDF. | Do not present them as independent candidates. |
| Pelvis-relative box position | Pooled standard deviation `[.00875, .06012, .02055]` m. | Too much lateral variation for a precise carrying relation. Keep this frame only for existing actor observations and command tracking. |
| Torso-relative box position | Pooled standard deviation `[.00516, .00842, .007995]` m; per-motion centers closely agree. X/Z correlation is about 0.85, but engineering floors dominate their narrow measured spreads. | Use a static torso-relative center and diagonal widths. Full covariance adds little after flooring. |
| Relative orientation | Torso-relative rotation-vector standard deviation is approximately `[.0279, .0386, .1711]` rad. Per-clip angular deviation from the clip mean averages 0.137/0.117/0.190 rad; pelvis values are lower, 0.045/0.050/0.073 rad. | The lower pelvis variance is partly constructed by MotionLib. Use a weak torso-relative orientation prior with broad yaw tolerance, not tight pelvis locking. |
| World linear velocity difference | Torso–box difference norm means are 0.168/0.164/0.341 m/s; clip 3 P95 is about 1.013 m/s. | Reject as a reward: a valid rotating offset naturally has different COM/origin velocity. |
| Rigid-body compensated velocity | Torso-frame norm means are 0.130/0.0048/0.0051 m/s. Pelvis compensation still gives means 0.306/0.249/0.342 m/s. | Supports torso attachment rather than pelvis attachment. |
| Derivative of local position | Matches compensated torso velocity to below 0.002 m/s component RMS in every clip. | Use a causal 50 Hz difference online. It avoids assumptions about the simulator's link-origin versus COM linear-velocity convention. |
| Relative angular velocity / full transform derivative | Torso-relative angular-speed means are 0.724/0.682/0.939 rad/s, with P95 up to 1.884 rad/s. The angular derivative of the relative rotation agrees with this. | Analyze and report, but do not add a second motion reward based on synthetic heading dynamics. Position derivative plus the weak orientation prior is the smaller useful set. |
| Box-size dependence | No dimensions are stored, so correlation with reference box size cannot be estimated. | Use actual randomized dimensions for side surfaces and face boundaries; do not invent empirical size correlations for the torso center. |

The dataset contains **synthetic object information**: box center is the hand
midpoint and box orientation is reconstructed as pelvis heading. It does not
establish measured contact geometry. Palm half-separation is about 0.116 m,
whereas nominal simulated half-width is 0.175 m. Blindly normalizing those palms
by nominal half-extents would reward points inside the simulated box.

Surface-retargeted validation moves each reference palm along its measured
box-center-to-palm ray until it meets the designated current box side. It is a
geometry test only: no IK, collision feasibility or dynamics success is implied,
and runtime resets remain unchanged. Raw and retargeted scores are separate in
`reference_summary.md`; raw grasp scores remain approximately 0.014–0.017 rather
than being hidden by a broad tolerance.

## Final objective

Let `T` be `torso_link`, `B` the box, `h` a palm, and `b` the current box
half-extents. Quaternions use XYZW throughout.

For each hand, define

```text
p_h^B = R_B^T (p_h - p_B)
d_h = s_h * p_h,y^B - b_y                 s_left=+1, s_right=-1
o_h = max(abs(p_h,[x,z]^B) - b_[x,z], 0)
u_h^T = normalize(R_T^T (p_h - p_B))
theta_h = atan2(norm(u_h^T cross mu_h), dot(u_h^T, mu_h))
E_h = (d_h/sigma_n,h)^2 + sum((o_h/sigma_n,h)^2)
      + (theta_h/sigma_angle,h)^2
r_grasp = exp(-(E_left + E_right)/4)
```

The grasp reward is the geometric mean of the two hand kernels. A hand at the
box center gets a finite maximal direction error. Face overflow prevents
rewarding a point on the infinite side plane outside the actual box face.
The ray/side intersection scales naturally with box width; X/Z bounds use the
actual depth/height. The learned directions preserve rotation-dependent
tangential placement without a motion index or phase.

For other target matching, `K(e,sigma)=exp(-0.5*sum((e/sigma)^2))`:

| Reward key | Frame/error and target | Width | Configured scale |
|---|---|---|---:|
| `carry_hand_box_surface` | Formula above; torso rays `[-.18807,.98184,-.02482]` / `[.22922,-.97291,.03018]` | Normal 0.020 m; direction 0.13025/0.11122 rad | 1.5 |
| `carry_arm_pose` | Fourteen named arm joints; equal-motion average of per-motion medians; `exp(-0.5*mean((error/sigma)^2))` | 0.15 rad per joint | 0.2 |
| `carry_relative_position` | `R_T^T(p_B-p_T) - [.347501,-.003911,.049451]` m | `[.030000,.038154,.034662]` m | 0.75 |
| `carry_relative_orientation` | `Log(q_target^-1 * (q_T^-1*q_B))`; target `[-.016304,-.055857,.022885,.998043]` | `[.090123,.116653,.473898]` rad, in the nominal box rotation's tangent basis | 0.25 |
| `carry_relative_velocity` | `(p_B^T(t)-p_B^T(t-1))/0.02`; target zero | `[.175188,.314361,.305326]` m/s | 0.5 |
| `carry_bilateral_contact` | Existing filtered contact on both rubber-hand links | Existing 1 N threshold | 0.5 |

Scale values are before the parent's existing multiplication by policy `dt`.
Preservation/contact maxima sum to 3.7; linear/yaw tracking maxima remain 5.5.
There is no contact-based gate on command tracking and no reference global
velocity target. Existing physical regularizers, including world box tilt,
remain active. The legacy `carry_contact_task` is disabled so its hidden
uncompensated velocity and saturated distance terms cannot conflict with yaw.

Calibration widths are `max(1.5 * maximum_per_motion_P95(abs residual), floor)`.
Centers give equal influence to each clip. Normal widths use deviations from
each clip's palm-side median, not the mismatch between synthetic grasp span and
the nominal physical box. Direction centers are normalized equal-clip mean
rays. Quaternion means use equal total weight per clip and quaternion-log
residuals, with shortest-arc/sign-invariant errors.

Floors are engineering allowances, not claimed empirical confidence bounds:
20 mm side distance accommodates the virtual palm/contact-patch convention,
10 mm simulator contact offset and millimetric FK discrepancy; 30 mm center
position and 0.15 rad arms permit size/balance adjustments beyond the frozen
reference clips; 5 degrees prevents synthetic orientation/ray data from imposing
near-rigid targets; 0.10 m/s prevents a nearly zero derivative distribution from
rewarding excessive stiffness. The observed dynamic clip determines the actual
motion widths. Position stays in metres because humanoid reach does not scale
with object dimensions, and this dataset has no evidence for a different rule.

## Implementation and diagnostics

The standalone PyTorch helper has no Isaac Gym dependency. It loads named
calibration tensors once, uses only batched device operations per step, and
returns bounded rewards plus raw errors. The specialist computes it once per
physics step before reward accumulation/reset. The first sample after reset has
no derivative; its motion reward is zero and its motion diagnostic is excluded
from aggregation. History never crosses episode boundaries.

Torso/palm and arm indices are resolved by name. The existing `upper_body_index`
still denotes pelvis. Actor/critic dimensions and semantics remain **738/126**;
actions remain 29 PD joint-position targets at 50 Hz. The actor, PPO, optimizer,
command sampling, domain randomization and all termination checks are unchanged.
The existing runner still discards the parent's unused AMP observation output;
this change adds no AMP training path or discriminator state.

Episode diagnostics include terminal samples and use independently counted
samples, not the runner's randomly initialized episode ages. Stale episode
dictionaries are removed on steps without a new reset. No-valid-sample aggregates
are NaN, rather than a fabricated successful zero. Evaluation excludes invalid
temporal samples, retains all old CSV fields and adds means/P95s for the same
errors to trial and mode summaries. Failed trials contribute observed integrity
metrics; unavailable pre-measure data remains NaN. Existing evaluation behavior
still skips the terminal reset frame; terminal preservation errors are retained
by training episode diagnostics, not fabricated as post-reset evaluation rows.

Watch these TensorBoard tags together:

- `TrackingRMSE/vx`, `TrackingRMSE/vy`, `TrackingRMSE/yaw_rate`: command tracking.
- `Episode/carry/left_hand_side_error_m` and `right_hand_side_error_m`: side-normal drift.
- `Episode/carry/left_hand_direction_error_rad` and `right_hand_direction_error_rad`: tangential/grasp-direction drift.
- `Episode/carry/hand_midpoint_error_m`: asymmetric displacement from box center.
- `Episode/carry/box_relative_position_error_m`, `box_relative_orientation_error_rad`: torso–box pose drift.
- `Episode/carry/box_relative_motion_error_mps`: box lag/sliding, including during turns.
- `Episode/carry/arm_reference_error_rad`: arm-joint RMS drift.
- Existing reward, episode-length and evaluation completion/contact metrics: detect survival bias.

Use the new relative-motion error for yaw interpretation. The retained legacy
CSV world-velocity difference is not a rigid-carry error. Absolute geodesic
orientation error is deliberately less sensitive in the reward's yaw direction;
inspect it alongside tracking and grasp directions rather than interpreting it
as an isotropic reward penalty.

## Validation and remaining runtime checks

CPU tests cover reference calibration, actual helper outputs on every clip,
bilateral swaps/same-side hands/single-hand loss, face overflow, zero-length hand
rays, arm drift, box translation/rotation/sliding, quaternion signs, batch
independence, all size-range corners, and rigid transforms across all five
command families. Integration tests exercise the specialist's actual Python
methods against a bookkeeping-only base to check caching, partial resets,
terminal samples and stale logging. These are not simulated dynamics tests.

Validation result on this checkout: **22 passed, 1 skipped** (CUDA parity).
The standalone helper imports successfully, all nine implementation/analysis
Python files pass Python 3.8 syntax and compilation checks, and `git diff --check`
passes. A 4096-environment float32 kernel batch produces finite metrics and
bounded rewards. Calibration reproduction includes source-hash checks, with
text line endings normalized for portability between Windows and Linux.

Observed offline means: arm >=0.9753, relative position >=0.8273, orientation
>=0.8104 and motion >=0.8667 across all three clips. Surface-retargeted grasp
means are 0.9471/0.9989/0.9992. The asymmetric size-corner checks also pass.
Rigid-turn local motion error is numerically zero while the world linear-velocity
difference is nonzero, as expected for an offset attachment.

This Windows runtime has CPU PyTorch and **no Isaac Gym or CUDA**. Full simulator
imports, CUDA parity, contact feasibility and training improvement cannot be
claimed here. No local early/later specialist checkpoint pair is available; the
bundled carrybox checkpoint's actor shape is compatible, but it is not a
substitute for the requested drift comparison.

Reproduce analysis and tests from the repository root:

```bash
python -m experiments.carry_preservation.analyze
python -m pytest experiments/carry_preservation/tests experiments/carrybox_locomotion_eval/tests -q
```

On the Isaac Gym training machine, first run a short import/rollout/PPO smoke test:

```bash
python legged_gym/legged_gym/scripts/train_carry_locomotion.py \
    --task carrybox_locomotion \
    --init_policy_path <pretrained_carrybox_checkpoint> \
    --num_envs 64 --max_iterations 2 --headless
```

Then start the intended fresh finetuning run:

```bash
python legged_gym/legged_gym/scripts/train_carry_locomotion.py \
    --task carrybox_locomotion \
    --init_policy_path <pretrained_carrybox_checkpoint> \
    --headless
```

Evaluate the early and later specialists with identical seeds/commands/reset
frames, repeating for motion IDs 0/1/2 and checking all five command families:

```bash
python experiments/carrybox_locomotion_eval/evaluator.py \
    --resume_path <early_or_later_specialist_checkpoint> \
    --suite train_distribution --seed 1 --carry_motion_id 0 \
    --carry_phase 0.5 --warmup 0.20 --duration 10.0 --save_csv --headless
```

Acceptance is improved preservation through later checkpoints while tracking
and completion remain good in each command family. No policy-performance claim
follows merely from the passing geometry tests. The three short synthetic-object
clips provide limited evidence about physical contact tolerances at speed.

## Exact files changed by this implementation

Modified:

- `legged_gym/legged_gym/envs/g1/carrybox_locomotion.py`
- `legged_gym/legged_gym/envs/g1/carrybox_locomotion_config.py`
- `experiments/carrybox_locomotion_eval/evaluation/trial.py`
- `experiments/carrybox_locomotion_eval/evaluation/metrics.py`
- `experiments/carrybox_locomotion_eval/tests/test_metrics.py`

Added:

- `legged_gym/legged_gym/carry_preservation.py`
- `legged_gym/resources/config/carry_preservation.json`
- `experiments/carry_preservation/__init__.py`
- `experiments/carry_preservation/analyze.py`
- `experiments/carry_preservation/tests/__init__.py`
- `experiments/carry_preservation/tests/test_preservation.py`
- `analysis/carry_preservation/REPORT.md`
- `analysis/carry_preservation/reference_summary.md`
- `analysis/carry_preservation/statistics.json`

The pre-existing edit to `experiments/carrybox_locomotion_eval/README.md` is
preserved and is not part of this implementation. Parent CarryBox, MotionLib,
training entrypoint, runner, PPO and the reference motion files are unchanged.
