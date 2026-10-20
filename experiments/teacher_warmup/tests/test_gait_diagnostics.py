"""Synthetic CPU fixtures only; these tests are not physical gait evidence."""
import copy
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "legged_gym/legged_gym/scripts"))
sys.path.insert(0, str(ROOT / "rsl_rl"))
from gait_diagnostics import (GaitDiagnostics, foot_geometry, rotate, validate_a5_contract,
                              evaluation_handoff_overrides, window, WINDOWS)
from analyze_gait_diagnostics import Bucket, analyze_gait
from rsl_rl.teacher_handoff import adapt_action, joint_groups

NAMES = [line.split()[1] for line in (ROOT / "legged_gym/resources/config/joint_id.txt").read_text().splitlines()]
LEGS = [NAMES[i] for i in joint_groups(NAMES)["legs"]]
URDF = ROOT / "legged_gym/resources/robots/g1/urdf/g1_29dof.urdf"


class FakeGym:
    """Only body-name/index lookups; no physics is emulated."""
    names = [f"{s}_ankle_{c}_link" for s in ("left", "right") for c in ("pitch", "roll")] + ["left_rubber_hand", "right_rubber_hand"]

    def get_actor_rigid_body_names(self, env, actor):
        return self.names

    def get_actor_rigid_body_index(self, env, actor, index, domain):
        assert domain == 123
        return index + 3  # Two platforms and box precede robot.


def fixture():
    env = NS(num_envs=2, envs=[0, 1], actor_handles=[3, 3], gym=FakeGym(), dt=.02,
             dof_names=NAMES, groups=joint_groups(NAMES))
    env.cfg = NS(terrain=NS(mesh_type="plane"), asset=NS(hand_colli_name="rubber_hand"),
                 normalization=NS(clip_actions=1.), control=NS(action_scale=.25),
                 handoff=NS(), rewards=NS(hand_contact_threshold=1., box_drop_height=.15,
                 box_tilt_termination_deg=70., robot_box_max_distance=1., grasp_loss_grace_s=.3))
    env.feet_indices = torch.tensor([3, 5])
    env.contact_forces = torch.zeros(2, 9, 3)
    env.rigid_body_states = torch.zeros(2, 9, 13)
    env.rigid_body_states[:, :, 6] = 1.
    env.rigid_body_states[:, :, 2] = 1.
    env.contact_filt = torch.zeros(2, 2, dtype=torch.bool)
    env.first_contacts = env.contact_filt.clone()
    env.feet_air_time = torch.zeros(2, 2)
    env.root_states = torch.zeros(2, 13)
    env.root_states[:, 2] = .8
    env.box_states = env.root_states.clone()
    env._box_size = torch.full((2, 3), .3)
    env.env_origins = torch.zeros(2, 3)
    env.termination_contact_indices = torch.tensor([], dtype=torch.long)
    env.hip_yaw_indices = torch.tensor([], dtype=torch.long)
    env.roll = env.pitch = torch.zeros(2)
    env.head_index = 0
    env.projected_gravity_box = torch.tensor([[0., 0., -1.]]).repeat(2, 1)
    env.box_cfg = NS(box_termination=False)
    env.robot2object_dist = torch.full((2,), .4)
    env.grasp_loss_time = torch.zeros(2)
    env.control_mask = torch.tensor([True, False])
    env.last_step_snapshot = dict(tilt=torch.zeros(2))
    env.raw_student_action = torch.full((2, 29), 5.)
    env.actions = torch.zeros(2, 29)
    env.rates = torch.full((29,), 12.)
    diag = GaitDiagnostics(env, URDF, NS(DOMAIN_ENV=123))
    return env, diag


def capture(env, diag):
    desired = torch.where(env.control_mask[:, None], env.raw_student_action, torch.full_like(env.actions, .1))
    applied = adapt_action(desired, env.actions, env.rates, env.dt, .25, 1., env.control_mask, "rate_limited")
    diag.capture_action(env, desired, applied)
    diag.capture_state(env)
    return applied


def base_row(ep=0, phase=2, age=4., t=.02):
    row = dict(episode=ep, env=0, time_s=t, student_valid=int(phase != 0), phase=phase,
               control_phase=phase, handoff_relative_s=age if phase else -1., done=0,
               reason=0, pickup=1, eligible=int(phase != 0), bilateral=1, tilt=0., relative=0.,
               omega=0., posture_error=.01, gap_arms=.1, gap_waist=.1, gap_legs=.1)
    for axis in ("vx", "vy", "yaw"):
        row["command_" + axis] = row["actual_" + axis] = .8 if axis == "vx" else 0.
    return row


def test_collision_mesh_and_rotated_clearance():
    g = foot_geometry(URDF)
    roll = g["left_ankle_roll_link"]["vertices"]
    pitch = g["left_ankle_pitch_link"]["vertices"]
    assert roll[:, 2].min() == pytest.approx(-.035409145, abs=1e-7)
    assert np.ptp(roll[:, 0]) > .20 and np.ptp(pitch[:, 0]) < .04
    q = np.array([0., np.sin(np.pi/4), 0., np.cos(np.pi/4)])
    assert rotate(q, roll)[:, 2].min() == pytest.approx(-roll[:, 0].max())
    assert len(g) == 4  # Active mesh collision only; commented spheres ignored.
    assert np.allclose(rotate([0, 0, 0, 1], roll), roll)
    with pytest.raises(ValueError, match="quaternion"):
        rotate([0, 0, 0, 0], roll)


def test_body_indices_and_missing_body_fail_closed():
    env, diag = fixture()
    assert diag.indices["left_ankle_pitch_link"] == 3
    assert diag.indices["left_ankle_roll_link"] == 4
    env.gym.get_actor_rigid_body_names = lambda *a: []
    with pytest.raises(ValueError, match="Missing diagnostic body"):
        GaitDiagnostics(env, URDF, NS(DOMAIN_ENV=123))


def test_contact_filter_owner_reset_snapshot_and_limiter():
    env, diag = fixture()
    env.contact_forces[0, 4, 2] = 10.  # Roll-only contact.
    env.contact_forces[0, 3, 0] = 1.   # Strict >1N, not >=1.
    env.rigid_body_states[0, 4, 7] = .3
    before = {k: v.clone() for k, v in vars(env).items() if torch.is_tensor(v)}
    rng = torch.get_rng_state().clone()
    applied = capture(env, diag)
    assert torch.equal(rng, torch.get_rng_state())
    for key, value in before.items():
        assert torch.equal(value, getattr(env, key)), key
    assert applied[0, 0] == pytest.approx(.96)
    # Deliberately mutate live tensors to mimic auto-reset. Snapshot must survive.
    env.contact_forces.zero_()
    row = diag.row(base_row(), 0, env)
    assert row["left_contact"] == 1 and row["left_pitch_contact_raw"] == 0
    assert row["left_roll_contact_raw"] == 1 and row["left_touchdown"] == 0
    assert row["leg_limiter_fraction"] == 1.
    assert all(row["action_clipped:" + j] == 1 for j in LEGS)
    assert row["left_slip_step_m"] == pytest.approx(.006)
    assert diag.row(base_row(ep=1, phase=0), 1, env)["left_contact"] == 0
    capture(env, diag)
    assert diag.row(base_row(t=.04), 0, env)["left_contact"] == 1  # One-sample debounce.
    capture(env, diag)
    row = diag.row(base_row(t=.06), 0, env)
    assert row["left_liftoff"] == 1 and row["left_contact"] == 0
    # New episode beginning with contact is not a new touchdown.
    env.contact_forces[0, 4, 2] = 10.
    capture(env, diag)
    row = diag.row(base_row(ep=2), 0, env)
    assert row["left_touchdown"] == 0
    # Teacher/student owner change also restarts filter and event history.
    env.contact_forces.zero_()
    capture(env, diag)
    row = diag.row(base_row(ep=2, phase=0), 0, env)
    assert row["left_contact"] == 0 and row["left_liftoff"] == 0


def test_window_pre_step_ownership():
    row = base_row(phase=0)
    row["phase"] = 1  # Supervisor switches after this Teacher-controlled step.
    assert window(row) == "teacher_warmup"
    assert window(base_row(phase=1)) == "student_transition"
    for age, expected in ((.8, 2), (1., 3), (3., 4), (15., 5)):
        assert window(base_row(age=age)) == WINDOWS[expected]


def test_nonfinite_measurements_fail_the_smoke_gate():
    env, diag = fixture()
    env.contact_forces[0, 4, 2] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite body/contact"):
        capture(env, diag)
    env.contact_forces.zero_()
    env.raw_student_action[0, 0] = float("inf")
    with pytest.raises(ValueError, match="Nonfinite action"):
        capture(env, diag)


def test_strict_checkpoint_contract_rejects_original_and_mismatch():
    spec = dict(dims=[738, 512, 256, 256, 29])
    handoff = dict(ablation="A5", warmup_vx=.8, fixed_command=[.8, 0., 0.], command_suite="fixed", adapter="rate_limited",
                   target_rate_arms=2., target_rate_waist=1.5, target_rate_legs=12., joint_target_rates=None)
    contract = dict(teacher_sha256="teacher", reference_sha256="reference", handoff=handoff,
                    dof_names=NAMES, dt=.02, action_scale=.25, actor_obs=738, rewards=dict(scales={"test": 1}))
    ckpt = dict(iter=500, handoff_contract=contract)
    args = (spec, "teacher", "reference", handoff, NAMES, .25, .02, contract["rewards"])
    validate_a5_contract(ckpt, *args)
    with pytest.raises(ValueError, match="trained A5"):
        validate_a5_contract(dict(iter=19999), *args)
    broken = copy.deepcopy(ckpt)
    broken["handoff_contract"]["reference_sha256"] = "different"
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_a5_contract(broken, *args)
    changed = dict(handoff, warmup_vx=.4)
    with pytest.raises(ValueError, match="warmup_vx"):
        validate_a5_contract(ckpt, spec, "teacher", "reference", changed, NAMES, .25, .02)
    # The reported regression: training command metadata need not be the
    # requested vx=0.8 evaluation condition. Keep the saved checkpoint intact.
    trained_elsewhere = copy.deepcopy(ckpt)
    saved = trained_elsewhere["handoff_contract"]["handoff"]
    saved.update(fixed_command=[.4, 0., 0.], warmup_vx=.4, command_suite="mixed", student_timeout_s=45.)
    before = copy.deepcopy(trained_elsewhere)
    evaluated = dict(handoff, student_timeout_s=30.)
    validate_a5_contract(trained_elsewhere, spec, "teacher", "reference", evaluated, NAMES, .25, .02, contract["rewards"])
    overrides = evaluation_handoff_overrides(saved, evaluated)
    assert overrides["fixed_command"] == dict(checkpoint=[.4, 0., 0.], evaluation=[.8, 0., 0.])
    assert overrides["warmup_vx"] == dict(checkpoint=.4, evaluation=.8)
    assert set(overrides) == {"fixed_command", "warmup_vx", "command_suite", "student_timeout_s"}
    assert trained_elsewhere == before
    # Evaluation overrides cannot bypass adapter, rate or transition contracts.
    for key, value in (("adapter", "hard"), ("target_rate_legs", 6.), ("transition_duration_s", 2.)):
        mismatch = copy.deepcopy(trained_elsewhere)
        mismatch["handoff_contract"]["handoff"][key] = value
        with pytest.raises(ValueError, match=key):
            validate_a5_contract(mismatch, spec, "teacher", "reference", evaluated, NAMES, .25, .02, contract["rewards"])
    with pytest.raises(ValueError, match="reward/posture"):
        validate_a5_contract(trained_elsewhere, spec, "teacher", "reference", evaluated, NAMES, .25, .02, {"scales": {"test": 2}})


def trace_rows():
    env, diag = fixture()
    rows = []
    for ep in range(2):
        for i in range(30 if ep == 0 else 3):
            phase = 0 if ep == 1 or i < 3 else 1 if i < 6 else 2
            env.control_mask[0] = phase != 0
            env.contact_forces.zero_()
            if (i // 3) % 2 == 0:
                env.contact_forces[0, 4, 2] = 100.
            else:
                env.contact_forces[0, 6, 2] = 100.
            env.rigid_body_states[0, [4, 6], 2] = .09
            capture(env, diag)
            row = base_row(ep=ep, phase=phase, age=3. + i*.02, t=(i+1)*.02)
            row["pickup"] = int(ep == 0)
            row.update(diag.row(row, 0, env))
            if i == (29 if ep == 0 else 2):
                row.update(done=1, reason=4 if ep == 0 else 2)
            rows.append(row)
    return env, diag, rows


def test_streaming_analysis_failure_denominators_and_plots(tmp_path):
    env, diag, rows = trace_rows()
    manifest = dict(episodes=2, dt=.02, dof_names=NAMES, gait_diagnostics=diag.metadata(), synthetic_test_fixture=True)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with (tmp_path / "steps.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = analyze_gait(tmp_path)
    assert report["end_to_end_completion"] == .5
    assert report["pickup_success"] == .5
    assert report["handoff_conditional_completion"] == 1.
    primary = report["windows"]["successful_carry"]
    assert primary["teacher_warmup"] is None
    assert primary["student_0_1s"] is None
    steady = primary["student_3_15s"]
    assert steady["samples"] == 23  # Terminal frame excluded.
    assert steady["tracking_rmse"]["vx"] == 0.
    assert steady["double_support_ratio"] + steady["single_support_ratio"] + steady["no_contact_ratio"] == pytest.approx(1.)
    assert steady["feet"]["left"]["pitch_roll_comparison"]["raw"]["roll_only_ratio"] > 0
    assert steady["feet"]["left"]["swing_duration"]["count"] > 0
    for name in ("gait_summary.json", "gait_diagnostics.md", "contact_timeline.png", "swing_clearance.png", "pitch_roll_contacts.png", "leg_limiter_activation.png"):
        assert (tmp_path / name).stat().st_size > 100
    rows[-1]["done"] = 0
    with (tmp_path / "steps.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="Incomplete"):
        analyze_gait(tmp_path)


def test_bout_censoring_and_simultaneous_touchdowns():
    _, _, rows = trace_rows()
    bucket = Bucket(.02, LEGS)
    for i, row in enumerate(rows[:10]):
        row = dict(row, time_s=(i+1)*.02)
        row.update(left_touchdown=int(i in (2, 6)), right_touchdown=int(i == 2))
        bucket.add(row)
    report = bucket.report()
    assert report["simultaneous_touchdown_samples"] == 1
    assert report["touchdown_pair_count"] == 0


def test_no_simulator_mutation_calls_in_diagnostic_module():
    import ast
    import gait_diagnostics
    tree = ast.parse(Path(gait_diagnostics.__file__).read_text())
    calls = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert not any(n.startswith("set_") or n in ("simulate", "fetch_results", "rand", "randn") for n in calls)
    # Isaac Gym is commonly installed on Python 3.8. Keep the new entrypoints
    # syntactically compatible with that environment.
    for path in (ROOT / "experiments/teacher_warmup/diagnose_g1_gait.py",
                 ROOT / "legged_gym/legged_gym/scripts/analyze_gait_diagnostics.py"):
        ast.parse(path.read_text(), feature_version=(3, 8))


def test_actual_supervisor_action_hook_is_transparent(monkeypatch):
    import test_teacher_warmup as original
    _, diag = fixture()
    baseline, raw, baseline_result = original.SupervisorTests().step_fixture()
    make_fixture = original.supervisor_fixture

    def enabled_fixture():
        env = make_fixture()
        env.gait_diagnostics = diag
        return env

    monkeypatch.setattr(original, "supervisor_fixture", enabled_fixture)
    observed, observed_raw, result = original.SupervisorTests().step_fixture()
    for name in ("actions", "raw_student_action", "applied_action", "applied_pd_target", "obs_buf"):
        assert torch.equal(getattr(baseline, name), getattr(observed, name)), name
    assert torch.equal(raw, observed_raw)
    assert torch.equal(result[2], baseline_result[2])
    assert torch.equal(result[3], baseline_result[3])
    assert np.array_equal(diag.action["previous"], np.zeros((3, 29)))


def test_actual_pre_reset_reward_hook_is_transparent(monkeypatch):
    from test_teacher_warmup import load_env_class
    simple, diag = fixture()
    cls = load_env_class()
    env = cls.__new__(cls)
    env.__dict__.update(vars(simple))
    env.dof_pos = torch.zeros(2, 29)
    env.dof_vel = torch.zeros(2, 29)
    env.applied_action = torch.zeros(2, 29)
    env.joint_pos_target = torch.zeros(2, 29)
    env.phase = env.control_phase = torch.tensor([2, 0])
    env.handoff_age = env.phase_age = env.warmup_elapsed = torch.zeros(2)
    env.pickup_seen = env.handoff_seen = torch.ones(2, dtype=torch.bool)
    env.failure_reason = torch.zeros(2, dtype=torch.long)
    env.control_command = torch.zeros(2, 3)
    env.base_lin_vel_yaw = torch.zeros(2, 3)
    env.base_yaw_rate_world = torch.zeros(2)
    env.pose_reference = None
    env.readiness = lambda: (torch.ones(2, dtype=torch.bool), dict(lift=torch.ones(2), bilateral=torch.ones(2),
                            tilt=torch.zeros(2), relative=torch.zeros(2), omega=torch.zeros(2)))
    parent = cls.compute_reward.__globals__["CarryLocomotion"]
    monkeypatch.setattr(parent, "compute_reward", lambda self: setattr(self, "rew_buf", torch.tensor([2., 3.])))
    env.compute_reward()
    before = {k: v.clone() for k, v in env.last_step_snapshot.items()}
    rewards = env.rew_buf.clone()
    env.gait_diagnostics = diag
    env.contact_forces[0, 4, 2] = 9.
    rng = torch.get_rng_state().clone()
    env.compute_reward()
    assert torch.equal(env.rew_buf, rewards)
    assert torch.equal(torch.get_rng_state(), rng)
    for key, value in before.items():
        assert torch.equal(value, env.last_step_snapshot[key]), key
    env.contact_forces.zero_()
    assert diag.state["forces"][0, 4, 2] == 9.  # Snapshot persists through reset seam.
