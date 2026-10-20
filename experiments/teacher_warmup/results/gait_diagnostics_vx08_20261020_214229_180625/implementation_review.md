# Experiment 0 implementation and local findings

The diagnostic implementation is complete and CPU-tested. Physical evaluation
is blocked: the local repository has no trained A5 checkpoint or exact posture
reference, and the inspected Python environments lack Isaac Gym. Default and
gym38 PyTorch builds are CPU-only. No policy training or fine-tuning was run.

## Files changed and reasons

Paths are relative to repository root `D:/AAAProject/0806/PhysHSI`.

| File | Reason |
|---|---|
| `legged_gym/legged_gym/envs/g1/carrybox_teacher_warmup.py` | Four optional hook lines: copy action-adapter inputs/outputs and pre-reset physics state. |
| `legged_gym/legged_gym/scripts/eval_teacher_warmup_loco.py` | Opt-in flag, vx=0.8/A5 guards, artifact-contract checks, extra CSV columns, manifest metadata and diagnostic analysis. |
| `legged_gym/legged_gym/scripts/gait_diagnostics.py` | Independent contact histories, explicit DOMAIN_ENV body lookup, STL-derived rotated foot geometry, action/limiter capture, finite-value guards. |
| `legged_gym/legged_gym/scripts/analyze_gait_diagnostics.py` | Streaming support, swing/stance, slip, clearance, disagreement, per-joint limiter and preservation analysis; four plots and Markdown report. |
| `experiments/teacher_warmup/config_gait_diagnostics_vx08.json` | Separate fixed-command A5 configuration with warmup_vx=0.8 and existing rate-limited adapter. |
| `experiments/teacher_warmup/diagnose_g1_gait.py` | CPU artifact/environment preflight; gated 3-episode smoke then 20 seeded evaluation episodes; unique output directories and command/log capture. |
| `experiments/teacher_warmup/tests/test_gait_diagnostics.py` | Eleven CPU diagnostic tests, including real method transparency and synthetic report/plot integration. |
| `experiments/teacher_warmup/GAIT_DIAGNOSTICS.md` | Commands, measurement definitions, phase/censoring rules and limitations. |
| This results directory | Actual preflight inventory, null gait results, CSV schema, prepared commands and verification record. |

Original CarryBox/Locomotion classes/configs, Teacher checkpoint, action adapter,
observation dimensions, action dimensions, PD gains, PPO and reward terms were
not changed. Earlier local preflight output is retained in the separate
`gait_diagnostics_vx08_20261020_213517_946017` directory.

## Quantitative static findings

| Quantity | Left | Right |
|---|---:|---:|
| Pitch collision mesh X span | 0.036000 m | 0.036000 m |
| Roll collision mesh X span | 0.208207 m | 0.208207 m |
| Roll collision mesh local minimum Z | -0.035409 m | -0.035409 m |
| Pitch collision mesh local minimum Z | -0.024558 m | -0.024558 m |
| Roll-joint origin Z relative to pitch link | -0.017558 m | -0.017558 m |

At zero ankle-roll rotation, the roll mesh reaches approximately 0.028409 m
below the pitch mesh in the pitch-link frame. Thus pitch-origin height is not
sole clearance, and a roll-link contact comparison is necessary. This does not
establish PhysX contact fidelity or the cause of the observed gait.

The baseline leg target limit is 12 rad/s. At the repository's 0.02 s policy
timestep and action scale 0.25, that is a maximum normalized action change of
0.96 per policy step, or a PD-target change of 0.24 rad. Its actual activation
in A5 remains unmeasured; the actual checkpoint contract is unavailable.

Nine model files were read successfully. All inspected actors have 738 input
dimensions and 29 output dimensions, but none has a saved A5 handoff contract.
The Full Teacher SHA-256 is
`1a7adf066916d19f45d92ac5b564fbd0a21c0401a3b91fe1d5df0111e45c4ded`.
The original locomotion checkpoint was inspected, never substituted for A5.

## Answers to the requested eight questions

| Question | Confirmed evidence / unresolved part |
|---|---|
| Does pitch reflect real foot-ground contact? | Existing reward indices select pitch; its mesh is smaller and shallower. Actual contact agreement is unverified. |
| Is roll better, or combine? | Roll contains the broad foot collision mesh. Diagnostic OR-combination is implemented; selecting a reward source requires physical comparison. |
| Proper alternating swing? | Unverified: zero physical episodes. Event histories and alternation calculations pass synthetic tests. |
| Double/single/no-contact fractions? | Unavailable, represented by null gait metrics. No fabricated zero ratios. |
| Dragging versus low-clearance swing? | Unverified. Rotated-mesh clearance, contact-phase sole-point speed/slip proxy and possible-scuff counters are implemented. |
| Limiting during liftoff/swing? | Unverified. All 12 joints have raw/applied/previous actions, clipping and limiting flags; phase-conditioned summaries are implemented. |
| Handoff-only or steady-state issue? | Unverified. Teacher, transition and four carry windows are separated using pre-step ownership. |
| Tracking/grasp/posture consistent with visuals? | Unverified. Existing measurements are retained, and physical hand-body contact is logged separately. |

## Verification actually executed

- Before implementation: `python -m pytest experiments/teacher_warmup/tests -q` — 27 passed.
- Final: same command — **38 passed in 14.91 s**.
- `git diff --check` — passed.
- `python -m py_compile` for the evaluator, both new diagnostic modules and the runner — passed.
- `python experiments/teacher_warmup/diagnose_g1_gait.py --preflight_only` — exit 2 with four explicit blockers.
- Isaac Gym import probes in the default, gym38, tf and dts209tc environments found no Isaac Gym.

The new tests prove CPU collector transparency and statistical/export behavior,
including finite guards, independent histories, complete-bout censoring and
failure denominators. They do not establish simulator trajectory equivalence.
No simulator smoke or 20-episode evaluation was executed. Physical plots were
not produced; four plot exports were tested with synthetic temporary fixtures.

## Prioritized Experiment 1 recommendation

1. Obtain the actual successful A5 checkpoint and its exact posture reference,
   and run the prepared smoke/evaluation in an Isaac Gym/CUDA environment.
2. Validate pitch/roll/combined contact signals and contact-phase mesh-clearance
   residuals. If the current reward source misses foot support, isolate a
   contact-source correction before introducing another gait objective.
3. If dragging dominates with reliable contact, the smallest reward candidate
   is the existing slip term using the validated source. If low valid-swing
   clearance dominates with little slip/limiting, test one small swing-only
   clearance term. If limiting dominates at liftoff, diagnose target bandwidth
   before selecting a reward change.

No gait reward change is yet supported as necessary by measured gait data.
Experiment 0 implements none of these future interventions.

## Executable continuation command

From the repository root, with `A5_CHECKPOINT` and `A5_POSTURE_REFERENCE` set to
the exact existing artifacts and the Isaac Gym/CUDA Python environment active:

```powershell
python experiments/teacher_warmup/diagnose_g1_gait.py `
  --student_checkpoint $env:A5_CHECKPOINT `
  --posture_reference $env:A5_POSTURE_REFERENCE
```

The runner supplies all remaining exact arguments and creates fresh outputs.
`commands.json` records the direct evaluator argument arrays; `/path/to/...`
entries explicitly mark the two missing artifacts, not valid local paths.
