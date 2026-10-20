# Experiment 0: G1 contact and shuffling diagnostics

This is evaluation only. No training command, optimizer, new reward, gait clock,
PD change, action-adapter change or checkpoint write is introduced.

## Run

From the repository root in the **Isaac Gym / CUDA Python environment**:

```powershell
python experiments/teacher_warmup/diagnose_g1_gait.py --preflight_only
```

The runner searches all repository `model_*.pt` files, including ignored logs.
It recognizes A5 using the saved `handoff_contract`, requires positive training
iteration, and searches for the posture reference by the exact saved SHA-256.
Among A5 candidates it chooses the most recently modified file and records the
complete inventory. Modification time establishes local recency, not training
quality; provide the intended checkpoint explicitly when copying artifacts
changes timestamps. Original locomotion checkpoints are never substitutes.

Supply the actual A5 artifacts to execute the experiment:

```powershell
python experiments/teacher_warmup/diagnose_g1_gait.py `
  --student_checkpoint $env:A5_CHECKPOINT `
  --posture_reference $env:A5_POSTURE_REFERENCE
```

`A5_CHECKPOINT` and `A5_POSTURE_REFERENCE` must point to existing artifacts from
the successful A5 training run. On Linux the equivalent command uses
`--student_checkpoint "$A5_CHECKPOINT" --posture_reference "$A5_POSTURE_REFERENCE"`.
The runner sets child-process PYTHONPATH, probes Isaac Gym before torch in a
fresh interpreter, runs **1 environment / 3 episodes / seed 1 / 30 s carry**,
then verifies finite outputs, phase ownership, complete episodes, distinct
environment-domain body indices and all output files before starting **two
4-environment / 10-episode / 45 s carry evaluations with seeds 1 and 2**.

Every run creates a new `results/gait_diagnostics_vx08_TIMESTAMP/` directory.
`commands.json` records the exact evaluator argument arrays; stage logs are
retained. `smoke_seed1/`, `eval_seed1/` and `eval_seed2/` each contain:

- `manifest.json`, original `summary.json`, full `steps.csv`;
- `gait_summary.json`, `gait_diagnostics.md`;
- `contact_timeline.png`, `swing_clearance.png`, `pitch_roll_contacts.png`;
- `leg_limiter_activation.png`.

On failed preflight the root manifest and report state the exact blockers.
The root CSV is **headers only**, gait metrics are null, and physical plots
are not generated. Synthetic tests never populate experiment rollout files.

The existing evaluator also accepts `--gait_diagnostics` directly. Use
`--ablation A5 --command_suite fixed --fixed_command 0.8 0 0`, the actual
Teacher/A5/reference paths, and a fresh `--output` path. The diagnostic config
is selected automatically unless `--handoff_config` is given. Custom A5
settings require the original run's overrides plus the explicit diagnostic
vx=0.8 configuration; mismatches fail before the first simulation step.
Do not use `--collect_teacher`.

## Contracts and unchanged behavior

`config_gait_diagnostics_vx08.json` explicitly fixes Teacher warmup vx=0.8,
Student command [0.8,0,0], suite=fixed, ablation=A5 and adapter=rate_limited.
Evaluation carry duration, Teacher warmup vx, fixed command and command suite
may deliberately differ from checkpoint training conditions. Diagnostics still
require evaluation warmup vx=0.8, fixed command [0.8,0,0] and suite=fixed. Any
differences are printed and recorded in `evaluation_handoff_overrides` alongside
`checkpoint_handoff`; they are not changes to the policy weights. All remaining
saved handoff settings must match;
Teacher hash, reference hash, 738D/29D actor, joint order, action scale, dt,
and the entire reward/posture configuration must also match.

Only two optional observer hooks are added to the handoff environment. One
copies raw/desired/previous/applied actions after the existing adapter. One
copies forces and body state during reward snapshot construction **before
automatic reset**. Neither hook writes simulator state, calls a policy,
uses randomness, modifies actions, or participates in reward computation.
Original CarryBox/Locomotion classes, training entrypoint, rewards and indices
remain unchanged. CPU tests compare enabled/disabled real supervisor methods;
physical trajectory equivalence still requires an installed simulator.

## Measurements

For each foot and each ankle link, the trace contains XYZ net force, force
magnitude, raw contact (`norm > 1 N`), and filtered contact (current OR
previous raw sample). Combined physical-foot contact is the OR of the two
filtered sources. These are **net-body contact proxies**, not ground-pair
sensors. Raw and filtered disagreement, pitch-only, roll-only and legacy-vs-
combined disagreement are reported separately.

The robot is created after two platforms and the box. The observer explicitly
requests **DOMAIN_ENV** indices for the environment-shaped tensors, validates
four distinct foot bodies across environments and records original feet indices
for comparison. It does not globally change `asset.foot_name` or any reward index.

Clearance is the minimum world Z over the active ankle-roll collision STL
vertices, with collision scale/origin and the full runtime link quaternion
applied. Plane terrain is required; raw ankle Z is never called sole clearance.
This is an approximation of the collision envelope, not validated PhysX sole
distance: cooking, shape inflation, contact offsets, mesh quality and contact
partners can affect the residual. Contact-phase clearance distributions allow
checking the offset against measured contacts when simulation becomes available.

The trace logs roll-link XYZ velocity, horizontal speed, and the velocity of a
fixed representative local sole point (`v + omega cross rotated_offset`).
Slip distance integrates that point's horizontal speed during filtered contact;
foot rolling can contribute. Near-ground motion uses mesh clearance <2 cm and
horizontal sole speed >0.1 m/s. It is labeled **possible scuff**, not proven drag.
Valid swing samples require no combined contact for at least 0.04 s. The report
includes time-weighted P10/P50/P90 clearance and separate complete-swing peak
clearance distributions. Duration bouts touching reporting boundaries are
censored rather than treated as complete swings or stances.

All 12 leg joints retain original raw/applied action and PD-target columns,
plus previous applied action, hard clipping and rate-limit flags. Activation
ratios are reported overall, per joint, and during each foot's stance, swing
and liftoff. Teacher rows do not contribute Student limiter activations.

## Phase and denominator rules

Use **pre-step `control_phase`**, not the supervisor's post-step phase. Windows
are Teacher warmup, Student transition, then Student carry at [0,1), [1,3),
[3,15) and >=15 s measured since the first Student-controlled action. The
transition occupies its own disjoint window; with a 1 s transition, carry
[0,1) may legitimately be empty. An absent segment is null, never zero gait.

Contact filter/event history resets independently per environment at episode
or Teacher/Student ownership changes. An initial contact is not a touchdown.
There is no event-history reset just because transition becomes carry.
Simultaneous left/right touchdowns are counted separately and excluded from
alternation pairs. Reporting-window edges censor duration bouts.

Every physical approach/pickup attempt, including failed pickup and early
termination, stays in success denominators. `all_episodes` exposes every phase.
The primary `successful_carry` cohort uses Student frames from completed carry
episodes and excludes terminal rows. Compare both cohorts to see survivor bias.
Tracking RMSE, bilateral contact (legacy and explicit physical body indices),
box tilt, relative motion, posture error, survival, coarse termination reason
and the original termination predicate flags are preserved. Predicate flags
can overlap; their sum is not a count of unique failed episodes.

## Tests and next decision

```powershell
python -m pytest experiments/teacher_warmup/tests -q
```

New tests exercise actual mesh parsing/rotation, environment-domain indexing,
force thresholds/debounce, asynchronous history/reset boundaries, captured
terminal snapshots, action/observation/reward transparency, unchanged RNG,
strict artifact contracts, failure denominators, censoring, phase windows,
and all four plot exports using explicitly synthetic temporary fixtures.

Prioritize validating the contact source before choosing a gait reward change.
If physical traces show contact-phase dragging, the smallest candidate is to
evaluate the existing slip term with a validated contact source. If swings are
low with little slip/limiting, a small swing-only clearance term is a candidate.
If rate limiting clusters at liftoff, diagnose target bandwidth before choosing
a reward adjustment. Experiment 0 implements none of these interventions.
