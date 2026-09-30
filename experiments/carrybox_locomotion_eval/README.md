# CarryBox locomotion evaluation

This is the carry-only, no-force benchmark for Actors trained with
`carrybox_locomotion`. It subclasses the specialist environment, starts from a
fixed `carryWith` reference frame, and does not use the full-task pickup/carry
state machine in `carrybox_clean_perturb`.

## Run

```bash
python3 experiments/carrybox_locomotion_eval/evaluator.py --resume_path <BEST_CURRENT_VELOCITY_TRACKING_CHECKPOINT> --suite train_distribution --seed 1 --carry_motion_id 0 --carry_phase 0.5 --warmup 0.20 --duration 3.0 --save_csv
```

Add `--headless` for headless execution. Use `--mode stand|vx|vy|yaw|mixed`
to run one command family. Replacing only `--resume_path` enables a direct
comparison with another Actor under the same contact gate and command manifest.
The default `--protocol constant` retains the 52 fixed-command cases. The
following runs both fixed commands and four five-segment step sequences across
all three carryWith motions, three phases, and three reproducibility seeds:

```bash
python3 experiments/carrybox_locomotion_eval/evaluator.py --resume_path <checkpoint> --suite train_distribution --protocol both --mode all --seeds 1 2 3 --carry_motion_ids 0 1 2 --carry_phases 0.25 0.5 0.75 --stable_hold 0.20 --settle_timeout 2.0 --warmup 0.20 --duration 3.0 --save_csv --headless
```

This is 1,404 fixed-command reset attempts plus 108 step-sequence attempts.
`--protocol step --mode stand` is invalid; `--protocol both --mode stand` runs
only the constant stand case. Singular and plural versions of each context
argument are mutually exclusive. The simulator validates motion IDs against the
loaded carryWith dataset before rollout. Seeds are deterministic repeats with
randomization disabled; they are not different randomized physics scenarios.

Run the five command families separately with the same checkpoint and seed:

```bash
python3 experiments/carrybox_locomotion_eval/evaluator.py --resume_path <checkpoint> --mode stand --output_dir experiments/carrybox_locomotion_eval/results/<label>/stand --save_csv --headless
python3 experiments/carrybox_locomotion_eval/evaluator.py --resume_path <checkpoint> --mode vx --output_dir experiments/carrybox_locomotion_eval/results/<label>/vx --save_csv --headless
python3 experiments/carrybox_locomotion_eval/evaluator.py --resume_path <checkpoint> --mode vy --output_dir experiments/carrybox_locomotion_eval/results/<label>/vy --save_csv --headless
python3 experiments/carrybox_locomotion_eval/evaluator.py --resume_path <checkpoint> --mode yaw --output_dir experiments/carrybox_locomotion_eval/results/<label>/yaw --save_csv --headless
python3 experiments/carrybox_locomotion_eval/evaluator.py --resume_path <checkpoint> --mode mixed --output_dir experiments/carrybox_locomotion_eval/results/<label>/mixed --save_csv --headless
```

The causal training variants are registered as tasks without duplicating the
environment:

- `carrybox_locomotion_ablation_a`: expanded commands, lower-body constraints off.
- `carrybox_locomotion_ablation_b`: legacy command overrides on the expanded
  base config, lower-body constraints off.
- `carrybox_locomotion`: full mixed loaded-locomotion command distribution,
  current lower-body constraints on.

Primary planar velocity is `base_lin_vel_yaw[:, :2]` (pelvis-yaw frame), and
primary yaw rate is `base_yaw_rate_world` (pelvis world-z angular velocity).
Box planar velocity is rotated into the same pelvis-yaw frame. Legacy body
velocity is trace-only diagnostic data.

Every trial follows `RESET_CARRY_REFERENCE -> SETTLE -> WARMUP -> MEASURE -> END`.
During SETTLE the loaded Actor controls the robot with zero velocity command.
The contact gate requires both raw rubber-hand net forces above the task's
threshold, palms near the correct box sides, limited hand slip, limited
torso-relative box motion, and a healthy box pose for a continuous 0.20 s
(10 policy steps at the current 0.02 s dt). It times out after 2 s by default.
These forces are *contact proxies*, not filtered hand-box pair forces. The gate
does not change the reference pose or fix an initial hand-box bias. Failed
resets and timeouts are recorded as initialization failures without retries.
The mandatory zero-action reset step is excluded from gate timing. Target
commands start only after qualification. `--warmup` remains the duration under
the target command before measurement. Once tracking starts, subsequent contact
loss remains a tracking outcome. Step segments switch without reset or another
gate; the last segment requests a full stop.

The subclass disables only random command resampling, preserves the native
yaw-reference integration, and checks raw, policy, and observation commands
after every policy step. Actor history is cleared before the normal mandatory
zero-action reset step appends its one fresh frame. At command onset only the
newest observation frame's command values change; preceding settled history
and action history stay intact.

For each axis, `error = actual - command`,
`MAE = mean(abs(error))`, and `RMSE = sqrt(mean(error**2))`. The normalized
vector error is:

```text
sqrt((vx_error/1.2)^2 + (vy_error/0.5)^2 + (yaw_error/0.7)^2)
```

The scales are the maximum absolute values of the full specialist ranges.
With `--save_csv`, a unique run directory under `results/<checkpoint_label>/`
contains `run_metadata.json`, `command_manifest.csv`, `summary.csv` (one row
per reset), `segment_summary.csv` (one row per commanded segment),
`mode_summary.csv`, and `traces/*.csv`. An explicitly supplied `--output_dir`
must not already exist. The metadata records checkpoint hash, Git revision,
frames, dt, command ranges, and all readiness thresholds. Trace phases include
SETTLE, WARMUP, and MEASURE; a terminated step sequence's later segments are
marked `not_reached`. No tracking value is fabricated for an initialization
failure. `ready_*` checks remain diagnostic after onset, while `stable_steps`
counts only the SETTLE gate and is zero outside that phase.

After a run, generate PNG/SVG figures and a Markdown report without Isaac Gym:

```bash
python3 experiments/carrybox_locomotion_eval/analyze.py --runs <run_directory>
python3 experiments/carrybox_locomotion_eval/analyze.py --runs <checkpoint_A_run> <checkpoint_B_run>
```

The analysis places individual-run figures in `run_N/` and overlays checkpoints
only on matching planned contexts in `matched_group_N/`. Incompatible dt, frame,
command range, or gate settings remain in separate individual-run figures.
Plotting requires Matplotlib in the analysis Python environment; Isaac Gym is
not required for this offline step.

The pure-axis benchmark values stay outside the `0.10` moving-command
deadzone:

```text
vx  = [-0.60, -0.35, -0.15, 0.15, 0.40, 0.80, 1.20]
vy  = [-0.50, -0.30, -0.15, 0.15, 0.30, 0.50]
yaw = [-0.70, -0.45, -0.20, 0.20, 0.45, 0.70]
```

Mixed evaluation includes eight full-range corner stress cases and 24 Halton
interior cases. The manifest and summary identify them with `case_type`.

`mode_summary.csv` separates task survival from tracking accuracy by protocol:

- `initialization_success_rate`, `conditional_completion_rate` (among initialized
  attempts), and `end_to_end_completion_rate` have distinct denominators. A step
  sequence's single initialization is counted once.
- `completed_only_*` is the distribution of per-trial tracking metrics
  conditioned on successful segment completion.
- `all_observed_*` uses every finite per-trial metric produced by valid MEASURE
  samples, including samples collected before a failed trial terminates. A
  failure before MEASURE remains NaN for tracking; no numeric penalty or
  fabricated sample is inserted.
- Carry-integrity and survival fields always include incomplete trials.

Step responses retain the first 0.20 s in traces for dynamics while the
existing error metrics use the later measurement window. An axis is settled
only when all remaining observed samples are within
`max(0.05, 0.05 * abs(command_change))` in that axis's units, with at least
0.20 s remaining. Overshoot is measured in the direction of the command
change. Early termination censors settling time; observed-prefix overshoot is
explicitly labeled. Per-segment P95 distributions are not pooled-sample P95s.
`train_distribution` is deterministic training-*range* coverage, not an
unbiased estimate of performance under the training sampler.

Mode-specific interpretation is: `stand` measures drift/stillness; `vx`
measures forward tracking plus lateral/yaw leakage; `vy` measures lateral
tracking plus forward/yaw leakage; `yaw` measures yaw tracking plus translation
leakage; and `mixed` measures coupled three-DoF tracking.

Lower-body trace columns include signed left/right hip roll/yaw error, feet and
knee width plus interval violation, signed left/right foot-yaw error, all three
waist joints, and torso-relative-to-pelvis yaw/roll/pitch. `segment_summary.csv` reduces
these to hip roll/yaw RMS and absolute P95, feet/knee width mean/P95/violation
rate, foot-yaw RMS/P95, and waist/torso-pelvis RMS. These are also aggregated by
command family in `mode_summary.csv`.
