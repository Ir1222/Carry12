# B2 -> M1 Full Mixed locomotion

Implemented on `mixed-loco`, created from `research/teacher-warmup-handoff` at
`d757e739dd9cbaa87305aab9f1bff679ddb2ed5e`. The starting working tree was clean.
Changes remain uncommitted; nothing was pushed. Existing result/checkpoint files,
A5/B2 JSONs, verification scripts, reward implementations and PPO/storage code are unchanged.
The two locomotion implementation/config files match `velocity-tracking` exactly.

## Control contract

`carrybox_teacher_warmup_mixed` inherits the existing Warmup environment, which
inherits CarryLocomotion. It reuses the actual original sampler, reward functions,
termination supervisor, teacher pipeline, A5 action adapter and masked PPO.

Teacher: real approach/pickup with `[0.8,0,0]`. On supervisor eligibility the next
(first student-owned) action sees `[0.8,0,0]`, and target and applied command are
both initialized to that value. Only completed student-owned physics steps accrue
the hold timer, including HANDOFF_TRANSITION. Release requires STUDENT_CARRY and
at least `mixed_command_hold_s` (default 2.0s). A longer transition delays release.

After release, targets follow the inherited 10/10/10/10/60 stand/vx/vy/yaw/mixed
sampler and original dead zones, with vx `[-0.6,1.2]`, vy `[-0.5,0.5]` and yaw
`[-0.7,0.7]`. Targets resample 4-6 seconds from sampling, including time spent
ramping. Applied commands independently slew at `[0.6,0.5,0.6] * env.dt` per step.
Hold and slew settings are Python parameters; the default engineering values
have not been optimized or physically validated.

Heading updates preserve student raw vy/yaw. Next commands are prepared only
after reward/snapshot construction and episode resets, before the next actor
observation. Velocity rewards, gait gates and physical-step diagnostics use the
command seen by the actor for that action. Resets clear only the selected envs'
manager state. No teleportation, blending, AMP, gait clock or extra PPO loss was added.

## M1 production settings

| Setting | M1 default |
| --- | --- |
| `max_iterations` | 10000 (CLI `--max_iterations` supported; resume means additional iterations) |
| `num_envs` | 4096 (CLI `--num_envs` supported, including resume) |
| `num_steps_per_env` | 100 |
| `save_interval` | 500 |
| `learning_rate` | 5e-5 |
| `min_valid_samples` | 2048 |

M1 validates its settings independently. Environment count must be a positive
integer, with no fixed-size restriction. Original B2 retains its exact 512-env,
100-step rollout and 100-iteration checkpoint validation. M1 source selection
never hardcodes a B2 checkpoint path. TensorBoard records every iteration;
periodic saves run every 500 iterations, and the existing independent final
checkpoint save remains active at a non-multiple of 500.

M1 resume preserves all semantic contracts and rollout length while allowing
CLI changes to environment count and execution metadata (including migrating an
older M1 save cadence). Original B2 still compares its complete saved contract.

## Files and purpose

| File (relative to repository root) | Change |
| --- | --- |
| `legged_gym/legged_gym/envs/g1/carrybox_teacher_warmup_mixed.py` | New command-only specialization, hold/release, independent timers, slew and physical-step metadata. |
| `legged_gym/legged_gym/envs/g1/carrybox_teacher_warmup_mixed_config.py` | Independent M1 Python configuration, exact B2 scales and A5/PPO inheritance. |
| `legged_gym/legged_gym/envs/__init__.py` | Register the independent task. |
| `legged_gym/legged_gym/scripts/train_teacher_warmup_loco.py` | Initialization alias/conflict checks, task defaults, separate M1 validation, effective environment/provenance snapshots; legacy B2 validator retained. |
| `legged_gym/legged_gym/scripts/mixed_initialization.py` | Verify trained B2 lineage, hashes, actor/std, 160D critic input, joints, adapter, PPO/reward/environment contracts and M1 resume settings. |
| `rsl_rl/rsl_rl/runners/teacher_warmup_on_policy_runner.py` | Existing runner gains Train/Mixed metrics, checkpoint provenance and physical snapshot tracking; same PPO/storage. |
| `rsl_rl/rsl_rl/mixed_command_metrics.py` | Shared mode/ramp/settled accumulator, finite-sample counts and omission of unavailable statistics. |
| `legged_gym/legged_gym/scripts/eval_teacher_warmup_loco.py` | M1 and B2 comparison through the M1 command harness; named evaluation suites, override manifests and additional traces. |
| `legged_gym/legged_gym/scripts/analyze_teacher_warmup.py` | Add successful-handoff rate and student-controlled physical duration. |
| `experiments/teacher_warmup/compare_mixed.py` | Reject comparisons with different/missing conditions; write B2/M1 metric deltas. |
| `experiments/teacher_warmup/tests/test_mixed_loco.py` | Deterministic CPU command, config, contract, runner, TensorBoard, evaluation and resume tests. |
| `rsl_rl/rsl_rl/m1_performance.py` | Opt-in synchronized real CUDA rollout/update timings, memory peaks, coverage checks and JSON/JSONL reports. |
| `experiments/teacher_warmup/m1_smoke.py` | Fresh-process GPU smoke matrix for 1024/2048/4096 envs, hardware probes, device-wide memory polling, OOM/timeout reporting. |
| `experiments/teacher_warmup/tests/test_m1_production.py` | Production defaults, CLI overrides, independent B2 validation, resume metadata, per-iteration TensorBoard vs 500-step saves and smoke plumbing. |
| `outputs/M1_production_smoke_probe_20261021_v2/report.json` | Executed hardware probe: all three sizes explicitly unmeasured because Isaac Gym/CUDA are unavailable. |
| `experiments/teacher_warmup/M1_MIXED.md` | This implementation/run/validation report. |

## Commands

Run in a configured Isaac Gym/CUDA Python environment with this repository's
`legged_gym` and `rsl_rl` packages installed. The commands below use PowerShell
and this checkout's absolute paths. On a Linux simulator host, set `$Repo`'s
shell-equivalent to its checkout path and use the same Python flags.

Set these variables to real files. The B2 path and iteration are deliberately
user-selected. No B2 model is bundled or invented by this implementation.

```powershell
$Repo = 'D:/AAAProject/0806/PhysHSI'
$Scripts = "$Repo/legged_gym/legged_gym/scripts"
$Teacher = "$Repo/shared_artifacts/2026-10-19/carry_track/model_33999.pt"
$Reference = "$Repo/experiments/teacher_warmup/results/lab_20261019_233908/full_teacher_upper.json"
$B2 = '/absolute/path/to/selected/trained_B2/model_N.pt'
$M1 = '/absolute/path/to/selected/M1/model_N.pt'
```

Before production training, run the memory/performance matrix using the selected
real B2 checkpoint and its matching teacher/reference:

```powershell
python "$Repo/experiments/teacher_warmup/m1_smoke.py" --teacher_checkpoint "$Teacher" --posture_reference "$Reference" --init_policy_path "$B2" --num_envs 1024 2048 4096 --iterations 40 --seed 1 --rl_device cuda:0 --output "$Repo/outputs/M1_gpu_smoke"
```

Each size launches a fresh Python process with the real M1 entrypoint, 100-step
rollouts and save interval 500. Only run length is shortened; teacher pickup,
handoff, hold, mixed sampling and PPO masking are preserved. Smoke runs still
write TensorBoard every iteration and save a final checkpoint in a unique run
directory; they do not auto-launch production or select a different env count.
The matrix reports failure/incomplete coverage instead of passing a teacher-only
run. A case requires a finite actual PPO update and physical mixed-command
coverage. This verifies exercising the pipeline, not successful Mixed gait quality.

Reports contain per-iteration rollout seconds, PPO update seconds, end-to-end
iteration throughput, Student sample counts and coverage, Torch allocated/reserved
peaks, and sampled device used/free memory. CUDA synchronization is enabled only
for these timing runs. Torch peaks include environment and runner initialization;
`nvidia-smi` also samples device-wide memory during initialization and training,
including PhysX and any concurrent processes. Device samples can miss transient
peaks; they are not process-private memory. Reported timings include smoke
instrumentation overhead; final checkpoint time is separate from iteration FPS.

`report.json` aggregates the three sizes and references their logs, JSON/JSONL
measurements and unique training run directories. Partial rows survive worker
failure; OOM and timeout are reported without fabricated GPU statistics. If
40 iterations do not reach mixed Student PPO, the result is incomplete; increase
`--iterations` explicitly and use a new output directory. No phase budget is shortened.

A read-only availability probe can run without checkpoint paths:

```powershell
python "$Repo/experiments/teacher_warmup/m1_smoke.py" --probe_only --output "$Repo/outputs/M1_gpu_probe"
```

New training run (no hand-authored JSON required):

```powershell
python "$Scripts/train_teacher_warmup_loco.py" --task carrybox_teacher_warmup_mixed --teacher_checkpoint "$Teacher" --posture_reference "$Reference" --init_policy_path "$B2" --num_envs 4096 --seed 1 --max_iterations 10000 --run_name M1_mixed --rl_device cuda:0 --headless
```

Actor and std load exactly; Critic and Adam start fresh. `--student_checkpoint`
is a supported alias. Passing both aliases is rejected. B2 provenance must
include the existing A5 -> B2 initialization audit, positive training iteration,
rollout timesteps and nonempty trained Adam state. A5, carry-only, incompatible
or unaudited legacy checkpoints fail before simulator creation. After simulator
creation, calibrated collision geometry must also match the source checkpoint.

Resume (no initialization argument):

```powershell
python "$Scripts/train_teacher_warmup_loco.py" --task carrybox_teacher_warmup_mixed --teacher_checkpoint "$Teacher" --posture_reference "$Reference" --resume --resume_path "$M1" --num_envs 4096 --seed 1 --max_iterations 1000 --run_name M1_resume --rl_device cuda:0 --headless
```

`max_iterations` is the number of additional iterations, preserving the existing
runner convention. Restore the same semantic configuration and rollout length
as the saved run; environment count can change through CLI; Actor/Critic, Adam and RNG restore, while simulator episodes
restart with real teacher pickup. An early valid M1 save with no Student PPO
updates can resume. Checkpoint contracts exclude initialization filenames but
retain command-manager parameters. Every new run gets a unique output directory,
`config.json` with effective configuration/provenance, TensorBoard events and
checkpoints containing contracts and initialization metadata.

Fixed-0.8 regression evaluation:

```powershell
python "$Scripts/eval_teacher_warmup_loco.py" --task carrybox_teacher_warmup_mixed --teacher_checkpoint "$Teacher" --posture_reference "$Reference" --student_checkpoint "$M1" --eval_suite fixed08 --episodes 20 --carry_seconds 45 --num_envs 1 --seed 1 --rl_device cuda:0 --headless --output "$Repo/outputs/M1_fixed08_eval"
```

Full Mixed evaluation:

```powershell
python "$Scripts/eval_teacher_warmup_loco.py" --task carrybox_teacher_warmup_mixed --teacher_checkpoint "$Teacher" --posture_reference "$Reference" --student_checkpoint "$M1" --eval_suite mixed --episodes 20 --carry_seconds 45 --num_envs 1 --seed 1 --rl_device cuda:0 --headless --output "$Repo/outputs/M1_mixed_eval"
```

Other suites: `stand`, `vx_forward` (+0.8), `vx_backward` (-0.4), `vy_left` (+0.3),
`vy_right` (-0.3), `yaw_left` (+0.4), `yaw_right` (-0.4). Fixed suites keep the
initial 0.8 hold then slew to the explicit target, preserving trained weights
and adapter settings; overrides are recorded. `--command_suite fixed
--fixed_command vx vy yaw` also supports explicit custom evaluation commands.
Do not use legacy `--gait_diagnostics` for M1; its B2 gait diagnostics are already
included. The old flag remains available for original A5/B2 usage.

For an identical-condition B2 baseline, use the same evaluation command,
`--student_checkpoint "$B2"`, and a new output directory. The M1 harness gives
B2 and M1 the same hold/slew/sampling protocol, retaining each source's adapter.
For example:

```powershell
python "$Scripts/eval_teacher_warmup_loco.py" --task carrybox_teacher_warmup_mixed --teacher_checkpoint "$Teacher" --posture_reference "$Reference" --student_checkpoint "$B2" --eval_suite mixed --episodes 20 --carry_seconds 45 --num_envs 1 --seed 1 --rl_device cuda:0 --headless --output "$Repo/outputs/B2_mixed_eval"
python "$Repo/experiments/teacher_warmup/compare_mixed.py" --baseline "$Repo/outputs/B2_mixed_eval" --candidate "$Repo/outputs/M1_mixed_eval" --output "$Repo/outputs/B2_vs_M1_mixed.json"
```

Output paths must be new. `summary.json` reports pickup, eligibility, successful
handoff, completion, carry duration, physical-applied-command RMSE, bilateral
contact, tilt and relative motion. `mixed_summary.json` adds per-target-mode,
all/ramping/settled applied and raw-target RMSE, hand-box slip/relative speed and
B2 gait metrics. `steps.csv` retains raw physical commands/targets, modes,
ramping/hold flags, joint traces, gait values and relative box positions.
Absent modes, no-stance slip, no-swing clearance and no-touchdown air-time samples
are omitted/blank, never assigned zero error.

TensorBoard:

```powershell
tensorboard --logdir "$Repo/legged_gym/logs/carrybox_teacher_warmup_mixed" --port 6006
```

Groups: `Loss/*`, `Handoff/*`, `Phase/*`, `Physics/*`, `Episode/*`,
`TrackingRMSE/*`, `Perf/*`, `Train/*`, `Mixed/*`.
`Train/mean_reward` is mean reward per Student-controlled physical step;
`Train/mean_episode_length` is completed episode length in policy steps (rolling
last 100 episodes); `Train/valid_student_samples` counts valid rollout samples.
Mean metrics appear once their sample populations exist.
`Mixed/hold_ratio` and `Mixed/ramp_ratio` use Student-controlled steps.
Mode names `stand`, `vx`, `vy`, `yaw`, `mixed` label sampled targets, with separate
`all`, `ramping` and `settled` subgroups. `target_samples` counts new targets;
`steps` counts physical steps. A yaw target during vx braking contributes to
`yaw/ramping`, never `yaw/settled`. Touchdown air time uses completed eligible
flights; instantaneous air time remains available separately.

## Validation and remaining limitations

CPU suite:

```powershell
python -m pytest "$Repo/experiments/teacher_warmup/tests" -q --basetemp "$Repo/.pytest_cache/mixed_loco_validation"
```

Final result (2026-10-21): **102 passed in 27.95s**, including all 61 existing
regression tests and 41 M1 cases. Both entrypoints' `--help`, syntax
compilation and `git diff --check` passed. Git comparison confirmed the original
A5/B2 JSONs, Warmup environment/config, masked PPO/storage and legacy verification
code were unchanged; locomotion files still match `velocity-tracking`.
Full effective Python config snapshots were also checked for JSON serialization.

Final Git status: branch `mixed-loco`; 5 modified tracked files and 11 new files
(the table above), uncommitted. No pre-existing user changes were present.

The workspace cache basetemp avoids the Windows sandbox's external temporary
folder rename restriction during atomic checkpoint saves. It does not change
production checkpoint saving. Unit tests include real masked PPO updates,
Actor/std exact transfer, fresh Critic/Adam, M1 save/resume and TensorBoard event
reads, using explicitly synthetic state/transition fixtures. These are code
validation, not contact simulation.

Production smoke availability probe: `outputs/M1_production_smoke_probe_20261021_v2/report.json`.
All three cases (1024/2048/4096) are `not_executed`, with null memory/FPS;
exit status 2 denotes blocked availability. Python 3.11, PyTorch 2.6.0+cpu, CUDA unavailable,
`isaacgym` not installed. No physical smoke test, expensive training, B2/M1
physics comparison or gait-success claim was performed. No trained B2
checkpoint was available locally to audit beyond synthetic checkpoint fixtures.

On an Isaac Gym/CUDA host, a minimal physical smoke uses the evaluation command
with `--episodes 2 --num_envs 2 --carry_seconds 30` and a new output directory.
Inspect the trace for actual pickup/eligibility, first Student command at 0.8,
100 Student-controlled held steps at dt=0.02, successful transition before
release, and bounded command changes after release. Then run all suites on B2
and selected M1 checkpoints with identical seeds/settings and compare manifests.
No automatic curriculum/reward adjustment is triggered by results.

Remaining limits: slew settings are initial engineering values; physical pickup,
Mixed tracking and gait quality still require Isaac Gym evaluation. Missing or
incompatible source contracts are rejected rather than guessed. Resume restores
policy/optimizer/RNG, not a simulator snapshot. Full Mixed episodes can consume
random draws at different physical times as policies fail or hand off at
different times; equal seeds/configuration provide the same evaluation protocol,
not guaranteed identical per-step target sequences across different policies.
