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
comparison with the original Actor.

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

Every trial follows `RESET_CARRY_REFERENCE -> WARMUP -> MEASURE -> END`.
The subclass disables only random command resampling, preserves the native
yaw-reference integration, and checks raw, policy, and observation commands
after every policy step. Actor history is cleared before the normal mandatory
zero-action reset step appends its one fresh frame.

For each axis, `error = actual - command`,
`MAE = mean(abs(error))`, and `RMSE = sqrt(mean(error**2))`. The normalized
vector error is:

```text
sqrt((vx_error/1.2)^2 + (vy_error/0.5)^2 + (yaw_error/0.7)^2)
```

The scales are the maximum absolute values of the full specialist ranges.
With `--save_csv`, results go to `results/<checkpoint_label>/` as
`command_manifest.csv`, `summary.csv`, `mode_summary.csv`, and `traces/*.csv`.

The pure-axis benchmark values stay outside the `0.10` moving-command
deadzone:

```text
vx  = [-0.60, -0.35, -0.15, 0.15, 0.40, 0.80, 1.20]
vy  = [-0.50, -0.30, -0.15, 0.15, 0.30, 0.50]
yaw = [-0.70, -0.45, -0.20, 0.20, 0.45, 0.70]
```

Mixed evaluation includes eight full-range corner stress cases and 24 Halton
interior cases. The manifest and summary identify them with `case_type`.

`mode_summary.csv` separates task survival from tracking accuracy:

- `completion_rate` is the fraction of command trials that survive through the
  complete measurement window.
- `completed_only_*` is the distribution of per-trial tracking metrics
  conditioned on successful completion.
- `all_observed_*` uses every finite per-trial metric produced by valid MEASURE
  samples, including samples collected before a failed trial terminates. A
  failure before MEASURE remains NaN for tracking; no numeric penalty or
  fabricated sample is inserted.
- Carry-integrity and survival fields always include incomplete trials.

Mode-specific interpretation is: `stand` measures drift/stillness; `vx`
measures forward tracking plus lateral/yaw leakage; `vy` measures lateral
tracking plus forward/yaw leakage; `yaw` measures yaw tracking plus translation
leakage; and `mixed` measures coupled three-DoF tracking.

Lower-body trace columns include signed left/right hip roll/yaw error, feet and
knee width plus interval violation, signed left/right foot-yaw error, all three
waist joints, and torso-relative-to-pelvis yaw/roll/pitch. `summary.csv` reduces
these to hip roll/yaw RMS and absolute P95, feet/knee width mean/P95/violation
rate, foot-yaw RMS/P95, and waist/torso-pelvis RMS. These are also aggregated by
command family in `mode_summary.csv`.
