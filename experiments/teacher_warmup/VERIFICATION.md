# Verification — 2026-10-19

Base commit: `aac6877` (`velocity-tracking`). Working branch: `research/teacher-warmup-handoff`.

## Milestone status

| Milestone | Implementation and checks | Result |
|---|---|---|
| 1 — audit | Read actual Full/Loco checkpoint tensors, source observation/control/reset/loading paths; record SHA-256 and joint order | Both actors 738→512→256→256→29; both original critics 126D; positive 29D std; Full contains AMP state |
| 2 — supervisor/adapter | Async phases, strict/loose readiness, separate budgets, raw Student 3D commands, preserved handoff state, per-group PD target rates, selected-env reset | CPU semantic tests pass; simulator setup/physics is not exercised by these tests |
| 3 — masked PPO | Explicit valid mask, next-state bootstrap, done/Teacher boundaries, final rollout sample, empty-buffer skip, raw sample likelihood | CPU GAE/minibatch/density/update tests pass |
| 4 — objectives | Deterministic teacher mean, detached labels, 29D weights, temporal mask, actual backward; persistent pose reward; teacher-only successful-episode statistics | Gradient, persistent prior, provenance and data-filter tests pass; no real posture reference has been generated |
| 5 — integration/evaluation | Independent runner/entrypoints; actor+std initialization, fresh critic/Adam, save/resume and contract checks; A0–A5 CSV/streaming analysis and reference builder | Two synthetic vector envs × 8 slots × 3 iterations, save, resume and 1 extra iteration pass; both CLI help commands pass |
| 6 — verification/runbook | New tests, regression subset, static checks, small physical smoke attempt, 4090/Ablation/Viewer/resume commands | 26 new tests pass; 7 existing evaluation tests pass; all 18 changed Python files compile; new Python files pass Pyflakes; physical smoke blocked by missing Isaac Gym |

## Executed CPU checks

```text
python -m unittest discover -s experiments/teacher_warmup/tests -q
Ran 26 tests — OK

python -m unittest discover -s experiments/carrybox_locomotion_eval/tests -q
Ran 7 tests — OK
```

The new suite imports and executes the real Torch PPO/runner/storage implementation. For supervisor tests it executes the actual source method bodies while replacing simulator setup/stepping seams with explicit fixtures. It does not simulate rigid-body contacts. The reference-builder test uses generated test-fixture data only in a temporary test directory; that data is not a real Teacher reference and is not shipped as a training prior.

Runner smoke: 2 synthetic environments, 8 simulation-independent slots per environment per iteration, 3 real PPO/Adam iterations, checkpoint save, load into a new runner, then a fourth iteration. Effective Student ratio is 0.625 in this fixture. This verifies control/data plumbing and optimization, and provides no evidence of locomotion, contact stability or physical reward improvement.

The suite also loads both real historical checkpoints, checks the inferred networks and positive stds, and runs frozen Actor forward passes on 738D tensors. Its small runner smoke uses a deliberately small MLP fixture to exercise width inference independently of the current config. No original checkpoint is written.

Static checks:

- Python `compile(..., mode='exec')` on every changed/untracked `.py`: 18 files pass.
- `python -m pyflakes` on all 13 new Python files: pass, no diagnostics.
- `git diff --check`: pass.
- `train_teacher_warmup_loco.py --help` and `eval_teacher_warmup_loco.py --help`: pass without Isaac Gym.
- Protected Full/Loco envs/configs, original train scripts and original handoff evaluator compare exactly equal to `velocity-tracking` using `git diff --exit-code`.

## Existing regression failures

```text
python -m unittest discover -s experiments/carry_preservation/tests -q
Ran 18 tests: 16 pass, 1 failure, 1 error
```

- `test_batched_finite_and_evaluator_history`: `ModuleNotFoundError: No module named 'isaacgym'` inside the existing evaluator helper.
- `test_carry_posture_reward_implementations_are_unchanged`: expected AST hash map differs from the actual existing reward-method ASTs on this host. The environment file checked by that test and its evaluator source are unchanged from `velocity-tracking`; no hash assertions were weakened or rewritten to make the test pass. Root cause of the existing hash mismatch is not established here.

## Physical smoke attempt and unavailable checks

Host: Windows, Python 3.11, PyTorch `2.6.0+cpu`, `torch.cuda.is_available() == False`, Isaac Gym is not installed.

The real training entrypoint was attempted with the two actual checkpoint paths, `--ablation A2 --num_envs 4 --max_iterations 3 --seed 1 --headless`. It stops before constructing a simulator:

```text
ModuleNotFoundError: No module named 'isaacgym'
  train_teacher_warmup_loco.py, parse_args, __import__('isaacgym')
```

Consequently no Isaac Gym pickup/handoff rollout, trained fine-tuning checkpoint, Viewer evaluation, physical success-rate CSV, 4090 memory/throughput measurement, or real Teacher posture statistics are claimed. The executable commands to perform those checks in the existing Linux/CUDA Isaac Gym environment are in `README.md`. The physical contact dynamics, rate settings and reward coefficients remain research hypotheses to validate there.
