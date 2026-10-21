# B2 gait refinement: implementation and execution record

## Current status (2026-10-20, Asia/Shanghai)

Implementation and CPU checks were performed in the Windows checkout on
`research/teacher-warmup-handoff`. No shared-machine SSH connection was used.
No Isaac Gym evaluation or training was executed. No trained B2 checkpoint,
learning curve, carrying comparison, or checkpoint recommendation is available.
The numerical Experiment 0 findings supplied in the task are accepted as prior
evidence; the full investigation was not repeated.

The exact Teacher exists locally:

`shared_artifacts/2026-10-19/carry_track/model_33999.pt`

SHA256: `1a7adf066916d19f45d92ac5b564fbd0a21c0401a3b91fe1d5df0111e45c4ded`.

These exact inputs are absent locally:

- `legged_gym/logs/carrybox_teacher_warmup/20261019_235711_985014_A5_fixed/model_5000.pt`
- `experiments/teacher_warmup/results/lab_20261019_233908/full_teacher_upper.json`

The real Student hash, weights, stored Teacher/adapter contract, and original
A5 warmup/fixed velocity (0.4 versus 0.8) therefore remain **unverified**.
No checkpoint was substituted or modified. Windows Python is CPU PyTorch and
has no Isaac Gym. The README's target environment is `physhsi`, Python 3.8,
CUDA PyTorch and Isaac Gym Preview 4.

The machine-readable local audit is
`results/B2_local_preflight_20261020/report.json`. It explicitly records
`simulation_executed=false` and `training_executed=false`.

## Modified and added files

Modified:

1. `legged_gym/legged_gym/envs/g1/carrybox_teacher_warmup.py`
2. `legged_gym/legged_gym/envs/g1/carrybox_teacher_warmup_config.py`
3. `legged_gym/legged_gym/scripts/train_teacher_warmup_loco.py`
4. `legged_gym/legged_gym/scripts/eval_teacher_warmup_loco.py`
5. `legged_gym/legged_gym/scripts/gait_diagnostics.py`
6. `legged_gym/legged_gym/scripts/analyze_gait_diagnostics.py`
7. `experiments/teacher_warmup/tests/test_gait_diagnostics.py` (startup precision regression checks)

Added:

- `experiments/teacher_warmup/config_B2_gait_vx08.json`
- `experiments/teacher_warmup/b2_preflight.py`
- `experiments/teacher_warmup/compare_b2.py`
- `experiments/teacher_warmup/tests/test_b2_gait.py`
- `experiments/teacher_warmup/tests/test_b2_analysis.py`
- This document and `results/B2_local_preflight_20261020/` audit artifacts.

`carrybox.py`, `carrybox_locomotion.py`, their configs, PPO/runner algorithms,
the action adapter, `config_4090.json`, and pretrained checkpoints are unchanged.
No GitHub push was performed.

## Contact, geometry, timing and rewards

Body names are resolved with `get_actor_rigid_body_names`, followed by
`get_actor_rigid_body_index(..., DOMAIN_ENV)` for every environment. Only
`left_ankle_roll_link` and `right_ankle_roll_link` supply new gait contacts.
All environment-domain indices must agree and be valid. Values 10/18 are never
hardcoded. `self.feet_indices` and every existing posture geometry remain intact.

Raw contact is `norm(net_contact_force) > 1 N`. Filtered contact is current OR
previous raw contact. New state updates once at `compute_reward`, after refreshed
physics tensors and before automatic reset. It uses pre-step `control_mask` and
`control_command`, even if the supervisor has just changed the post-step phase.
Reset and ownership changes clear history. A touchdown requires a filtered
noncontact-to-contact edge with positive preceding airborne duration and an
observed preceding stance. Initial/censored flight cannot earn air-time.
No period, symmetry, double-support penalty or gait clock is added.

Horizontal speed is ankle-roll body XY velocity magnitude. Clearance is minimum
world Z over rotated representative collision points, above the existing global
plane at Z=0. It is never ankle-origin Z. Startup reuses the existing diagnostic
URDF parser (mesh scale and collision origin included). Greedy support-point
selection consumes no RNG and runs only once on CPU. Training transforms at most
128 points per foot with vectorized GPU tensors; the hot path has no NumPy,
`.cpu()`, `.numpy()` or scalar `.item()` calls.

Measured offline approximation errors relative to the complete mesh, on an
independent set of 8192 full-sphere support directions per foot:

| Foot | Full vertices | Training points | Mean error | P95 error | Max error |
|---|---:|---:|---:|---:|---:|
| Left | 6543 | 128 | 0.102 mm | 0.326 mm | 0.833 mm |
| Right | 6547 | 128 | 0.111 mm | 0.355 mm | 0.721 mm |

Startup rejects sampled error exceeding 1 mm. These are sampled geometric
errors, not a formal all-orientation bound or measured PhysX cooking/contact
offset errors. Small selected-point clearance overestimation is possible.
Physical evaluation additionally records approximation-minus-full-mesh error
for each foot. The full mesh remains evaluation-only.

Let `v` be horizontal speed, `h` geometric clearance, `c` filtered contact,
`s=1-c`, `t` preceding swing duration, `d` valid first-touchdown event, and
`clip(x)=clamp(x,0,1)`. All four methods return `[num_envs]` and apply
`control_mask AND (norm(command_xy)>0.1 OR abs(command_yaw)>0.1)`.

| Method | Unweighted value | B2 scale |
|---|---|---:|
| `_reward_gait_foot_slip` | mean over feet of `c * clip((v-0.05)/0.30)` | -0.08 |
| `_reward_gait_swing_clearance` | mean of `s * clip((v-0.05)/0.30) * clip(h/0.04)` | +0.15 |
| `_reward_gait_foot_scuff` | mean of `s * clip(v/0.30) * exp(-max(h,0)/0.02)` | -0.08 |
| `_reward_gait_air_time` | `min(sum(d * clip((t-0.06)/0.20)),1)` | +0.05 |

Values are nonnegative and bounded by 1. No clearance bonus above 4 cm.
Air-time is paid once on valid touchdown, never in stand or continuous flight.
All new scales default to zero. Only the B2 JSON disables `feet_air_time`,
`feet_slip`, and `feet_clearance`; legacy implementations remain untouched.
Other safety, hand-box, tracking and posture rewards retain their A5 values.

The existing reward pipeline multiplies scales by dt once. At 50 Hz, the
theoretical combined B2 contribution lies within [-0.0032,+0.0040] per step.
This bound is **not measured reward performance**. Inspect actual raw/weighted
contributions against tracking and hand-contact terms in the physical smoke.

## Preserved contracts and validation

The isolated task still performs default physical reset, frozen Full Teacher
Approach/Pickup, original readiness/handoff transitions, 2-second imitation,
and student-valid PPO. Actor/critic/action dimensions are 738/160/29 at 50 Hz.
Action scale is 0.25; PD and rate adapter implementations are unchanged.
Rates remain arms=2, waist=1.5, legs=12 rad/s. Imitation coefficient is 0.5,
with weights 2/2/0.25. Upper-body prior remains 0.5; carry horizon is 45 seconds.

The explicit B2 JSON fixes both warmup and Student vx at 0.8, fixed suite, A5,
rate-limited adapter, 512 environments, 100 rollout steps, learning rate 5e-5,
5 epochs, 4 minibatches, min_valid_samples=2048, seed=1 and save interval=100.
It inherits the original PPO architecture and uses the original adaptive LR
schedule. Initial maximum iterations is deliberately 300.

Before B2 training, the entrypoint requires the exact three input files,
audits the trained A5 architecture [738,512,256,256,29], iteration 5000,
Teacher/reference hashes, full adapter/handoff contract, original environment,
PPO and policy contracts, and all unrelated reward values. It prints A5's saved
warmup/fixed commands as provenance, without adopting them. Startup prints the
effective 0.8 commands and runtime dimensions. After `load_init_policy`, Actor
and std are checked tensor-for-tensor against A5 before an optimizer update;
the optimizer must have no state. The original fresh Critic behavior is retained.

All gait parameters, compressed points, geometric validation results, body
indices, execution settings and A5 initialization path/hash are stored through
the existing environment checkpoint contract and run `config.json`.

CPU tests exercise actual environment methods and reward registration, exact
formulas, bounds, finite output, pre-step masks, stand/yaw cases, touchdown
filtering, episode/ownership resets, censored flight, geometry rotation and
compression, disabled-gait compatibility, fail-closed provenance/config checks,
and synthetic analysis/curve export. Existing supervisor/runner tests also run.
Synthetic fixtures are confined to tests and are not physical evidence.
Physical disabled/enabled trajectory equivalence, Actor initialization against
the real A5 file, and Teacher pickup stability remain pending on Linux.

Startup compatibility fix: the config's Python `.005 * 4` is `.02`, whereas
float32 SimParams can produce `.019999999552965164` in the saved runtime
contract. The contract check and runtime 50-Hz guard now accept at most 1 ns
absolute representation error, with zero relative tolerance. They still reject
real timestep differences and never rewrite simulation settings or checkpoints.
Contract failures now report checkpoint/requested values. The pre-simulator
environment comparison also reconstructs the `push_interval` snapshot field
that the original `_parse_cfg` adds after initialization; it does not modify
the actual configuration. Regression tests cover both precision directions,
invalid periods and realistic post-initialization A5 metadata.

## Commands prepared for the target machine (not executed here)

Run from the repository root in the existing README environment. These commands
do not install packages, connect machines, or replace the training pipeline.

```bash
cd /home/han/pyx/Carry12
conda activate physhsi
python -m pytest experiments/teacher_warmup/tests -q
python experiments/teacher_warmup/b2_preflight.py

TEACHER=shared_artifacts/2026-10-19/carry_track/model_33999.pt
A5=legged_gym/logs/carrybox_teacher_warmup/20261019_235711_985014_A5_fixed/model_5000.pt
REF=experiments/teacher_warmup/results/lab_20261019_233908/full_teacher_upper.json
CFG=experiments/teacher_warmup/config_B2_gait_vx08.json
COMMON=(--task carrybox_teacher_warmup --teacher_checkpoint "$TEACHER"
        --posture_reference "$REF" --ablation A5 --command_suite fixed
        --fixed_command 0.8 0 0 --rl_device cuda:0 --sim_device cuda:0 --pipeline gpu --headless)
```

First run the short real-physics reward smoke, with A5 weights and the new
rewards enabled. The explicit smoke flag permits exactly the requested reward
delta while still checking the original A5 handoff/hash/adapter contract:

```bash
python legged_gym/legged_gym/scripts/eval_teacher_warmup_loco.py \
  "${COMMON[@]}" --student_checkpoint "$A5" --handoff_config "$CFG" \
  --gait_diagnostics --gait_contact_source roll --b2_reward_smoke \
  --num_envs 1 --seed 1 --episodes 3 --carry_seconds 45 \
  --output experiments/teacher_warmup/results/B2_reward_smoke_seed1
```

Inspect successful real pickup/handoff, finite reward CSV columns, hand-box
preservation, and geometric residuals. Resolve failures before the main run.
All new reward components plus existing reward raw/weighted values are in
`steps.csv`. Full-mesh diagnostic quantities are captured before auto-reset.

Prepare a comparable original A5 baseline (old rewards, corrected diagnostic
contact source). This is performance evaluation, not another Experiment 0
investigation:

```bash
python legged_gym/legged_gym/scripts/eval_teacher_warmup_loco.py \
  "${COMMON[@]}" --student_checkpoint "$A5" \
  --handoff_config experiments/teacher_warmup/config_gait_diagnostics_vx08.json \
  --gait_diagnostics --gait_contact_source roll --num_envs 1 --seed 1 \
  --episodes 100 --carry_seconds 45 \
  --output experiments/teacher_warmup/results/A5_baseline_vx08_seed1_100
```

The exact **initial B2 training command to execute after validation** is:

```bash
python legged_gym/legged_gym/scripts/train_teacher_warmup_loco.py \
  "${COMMON[@]}" --student_checkpoint "$A5" --handoff_config "$CFG" \
  --num_envs 512 --seed 1 --max_iterations 300
```

No `--resume` is used for A5 initialization. The new directory will be printed
as `[Run] .../legged_gym/logs/carrybox_teacher_warmup/<timestamp>_A5_B2_gait_fixed_vx08`.
There is **no actual B2 training directory yet**. Set `B2_RUN` to the directory
actually printed, rather than guessing a timestamp.

After 300 iterations, evaluate the real checkpoint with the same seed, horizon,
reset distribution, Teacher and diagnostic source:

```bash
B2_RUN=/absolute/path/printed/by/the/training/command
python legged_gym/legged_gym/scripts/eval_teacher_warmup_loco.py \
  "${COMMON[@]}" --student_checkpoint "$B2_RUN/model_300.pt" --handoff_config "$CFG" \
  --gait_diagnostics --gait_contact_source roll --num_envs 1 --seed 1 \
  --episodes 100 --carry_seconds 45 \
  --output experiments/teacher_warmup/results/B2_300_vx08_seed1_100

python experiments/teacher_warmup/compare_b2.py \
  --baseline experiments/teacher_warmup/results/A5_baseline_vx08_seed1_100 \
  --candidate experiments/teacher_warmup/results/B2_300_vx08_seed1_100 \
  --run "$B2_RUN" --output experiments/teacher_warmup/results/B2_comparison_300_seed1
```

Intermediate checks may use 20 episodes and `compare_b2.py --allow_small_sample`,
with a separate 20-episode baseline under the same conditions. Final comparisons
require at least 100 episodes per checkpoint. An independent seed 2 replication
is useful; use seed 2 for both baseline and candidate, and separate output names.

If carrying/tracking are preserved, resume **only the newly trained B2** optimizer
to reach 500 additional B2 iterations:

```bash
python legged_gym/legged_gym/scripts/train_teacher_warmup_loco.py \
  "${COMMON[@]}" --handoff_config "$CFG" --resume \
  --resume_path "$B2_RUN/model_300.pt" --num_envs 512 --seed 1 --max_iterations 200
```

The existing runner writes a fresh timestamped segment directory on resume;
record it as `B2_RUN_500`. It restores B2 optimizer/RNG/iteration, while physical
episodes restart with Teacher warmup as the existing runner specifies. Evaluate
`$B2_RUN_500/model_500.pt` with the same command pattern. To reach 1000, resume
that B2 checkpoint with `--max_iterations 500`, recording the new segment path.
Only if 300/500/1000 checks show no material regressions, extend from 1000 by
500 or 1000 iterations to 1500 or 2000. Do not automatically start 5000 more.

`model_1000.pt` always means 1000 additional B2 iterations after A5 model_5000;
it is not named model_6000. Keep every segment and its TensorBoard logs.

## Outputs and selection

The existing runner records `Physics/reward_raw/<term>` and
`Physics/reward_weighted/<term>` for all enabled nontermination terms, as well as
tracking, episode outcomes and valid-sample diagnostics. The comparison tool
exports actual TensorBoard scalar CSV and PNG curves when `--run` is supplied.
For segmented training export/inspect each segment's logs separately.

Each physical evaluation produces `manifest.json`, `steps.csv`, existing carry
analysis, `gait_summary.json`, gait report and plots. These cover end-to-end and
conditional completion, handoff eligibility, vx/yaw RMSE, bilateral contact,
tilt, relative velocity, upper-body posture, bilateral swing duration and
peak-clearance P50, noncontact scuff rate, stance foot speed, duty, step frequency
and limiter activation. The comparison script rejects unequal Teacher,
reference, seed, horizon, number of environments, physical reset config,
handoff and diagnostic configuration.

It reports per-episode metric differences with bootstrap intervals and Wilson
intervals for completion/eligibility rates. Small differences whose intervals
cross zero remain uncertain; sampled frames are not treated as independent
episodes. Inspect both successful-carry and all-episode cohorts to avoid
survivor bias. These intervals do not correct for checkpoint selection or
multiple comparisons. Do not choose a checkpoint from higher gait rewards
alone. Completion/grasp, velocity tracking, then gait determine the choice.

**Recommendation is pending real training and evaluation.** No real performance
table, learning curve, success rate, or preferred B2 checkpoint is asserted here.


## Complete carry-only velocity benchmark (2026-10-21)

The fixed-vx=0.8 handoff/gait evaluation is not the complete velocity benchmark.
Use `experiments/carrybox_locomotion_eval/evaluator.py` for stand, vx, vy, yaw,
mixed commands and step responses. A5/B2 checkpoints now use their saved
rate limiter and inferred Actor architecture in this evaluator. The original
locomotion Actor inference path remains unchanged. Control metadata records the
adapter and that this suite uses fixed carry-reference resets without a Teacher.

The original full grid is `--protocol both --mode all --seeds 1 2 3
--carry_motion_ids 0 1 2 --carry_phases 0.25 0.5 0.75`: 1404 constant
command attempts plus 108 five-segment sequences (1512 total attempts).
Ranges are vx=[-0.6,1.2] m/s, vy=[-0.5,0.5] m/s, yaw=[-0.7,0.7] rad/s.
It checks 32 mixed combinations (8 corners and 24 interior cases), as well
as single-axis coverage and stopping/reversal/transient responses.

```bash
VELOCITY_OUT="/home/han/pyx/Carry12/experiments/carrybox_locomotion_eval/results/B2_4900_velocity_$(date +%Y%m%d_%H%M%S)"
python /home/han/pyx/Carry12/experiments/carrybox_locomotion_eval/evaluator.py \
  --resume_path "$B2_RUN" --suite train_distribution --protocol both --mode all \
  --seeds 1 2 3 --carry_motion_ids 0 1 2 --carry_phases 0.25 0.5 0.75 \
  --stable_hold 0.20 --settle_timeout 2.0 --warmup 0.20 --duration 3.0 \
  --save_csv --output_dir "$VELOCITY_OUT" \
  --rl_device cuda:0 --sim_device cuda:0 --pipeline gpu --headless
python /home/han/pyx/Carry12/experiments/carrybox_locomotion_eval/analyze.py \
  --runs "$VELOCITY_OUT"
```

Here `B2_RUN` is the full path to the B2 model_4900.pt file, not a directory.
The full command grid matches the earlier mixed locomotion benchmark. Its
initialization, horizon and conditional success denominators differ from the
Teacher warmup task, so report carry-only velocity coverage separately from
end-to-end pickup/handoff success. A fixed-vx B2 fine-tune may lose mixed-command
abilities; testing those abilities does not mean B2 trained on those commands.
Both evaluations are needed for the user's combined assessment. A protocol
that starts each of these 1512 attempts with physical Teacher pickup has not
been implemented by this carry-only compatibility change.
