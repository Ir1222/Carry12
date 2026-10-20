"""B2 CPU contracts and numerical checks. No synthetic result represents physics."""
import ast
import copy
import inspect
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from test_teacher_warmup import ROOT, load_env_class, load_configs, load_script
from test_gait_diagnostics import fixture as diagnostic_fixture, FakeGym, URDF
import gait_diagnostics
from gait_diagnostics import build_gait_geometry, foot_geometry, rotate

GAIT = ("gait_foot_slip", "gait_swing_clearance", "gait_foot_scuff", "gait_air_time")
CONFIG = ROOT / "experiments/teacher_warmup/config_B2_gait_vx08.json"


@pytest.fixture(scope="module")
def geometry():
    return build_gait_geometry(URDF)


@pytest.fixture
def env(monkeypatch, geometry):
    simple, _ = diagnostic_fixture()
    cfg, _, _ = load_configs()
    cls = load_env_class()
    # The existing AST loader deliberately strips module-level imports.
    cls._init_gait_buffers.__globals__["json"] = json
    env = cls.__new__(cls)
    env.__dict__.update(vars(simple))
    env.cfg.gait = cfg.gait
    env.cfg.asset.file = str(URDF)
    env.device = "cpu"
    env.dof_pos = torch.zeros(2, 29)
    env.control_mask[:] = True
    env.control_command = torch.tensor([[.8, 0., 0.], [.8, 0., 0.]])
    env.gait_enabled = True
    monkeypatch.setitem(sys.modules, "isaacgym", NS(gymapi=NS(DOMAIN_ENV=123)))
    monkeypatch.setitem(sys.modules, "legged_gym", NS(LEGGED_GYM_ROOT_DIR=str(ROOT / "legged_gym")))
    monkeypatch.setitem(sys.modules, "legged_gym.scripts.gait_diagnostics", gait_diagnostics)
    monkeypatch.setattr(gait_diagnostics, "build_gait_geometry", lambda *args: copy.deepcopy(geometry))
    before = env.feet_indices.clone()
    env._init_gait_buffers()
    assert env.gait_feet_indices.tolist() == [4, 6]  # Name lookup + DOMAIN_ENV offset.
    assert torch.equal(before, env.feet_indices)
    env.rigid_body_states[:, [4, 6], 7] = .35
    return env


def contact(env, left=True, right=True):
    env.contact_forces.zero_()
    env.contact_forces[:, 4, 2] = 10. if left else 0.
    env.contact_forces[:, 6, 2] = 10. if right else 0.
    env._update_gait_state()


def test_exact_formulas_shape_bounds_masks(env):
    # First env: left contact, right swing. Second env is Teacher-owned.
    env.control_mask[1] = False
    env.gait_horizontal_speed[:] = torch.tensor([.35, .35])
    env.gait_contact_filtered[:] = torch.tensor([True, False])
    env.gait_swing[:] = ~env.gait_contact_filtered
    env.gait_clearance[:] = torch.tensor([0., .04])
    env.gait_preceding_air_time[:] = .26
    env.gait_touchdown[:] = True
    expected = [.5, .5, .5 * math.exp(-2), 1.]
    for name, value in zip(GAIT, expected):
        result = getattr(env, "_reward_" + name)()
        assert result.shape == (2,)
        assert torch.isfinite(result).all()
        assert result.tolist() == pytest.approx([value, 0.])
    # All terms are gated by the pre-step command; yaw alone counts as moving.
    env.control_command.zero_()
    for name in GAIT:
        assert getattr(env, "_reward_" + name)().eq(0).all()
    env.control_command[0, 2] = .11
    assert env._reward_gait_air_time()[0] == 1
    env.control_command[0, 2] = .1
    assert env._reward_gait_air_time()[0] == 0
    env.control_command[0, 0] = .8
    env.gait_horizontal_speed.fill_(.05)
    assert env._reward_gait_foot_slip().eq(0).all()
    assert env._reward_gait_swing_clearance().eq(0).all()


def test_clearance_saturation_and_scuff_penetration(env):
    env.gait_swing[:] = True
    env.gait_horizontal_speed.fill_(30.)
    env.gait_clearance.fill_(.04)
    at_target = env._reward_gait_swing_clearance().clone()
    env.gait_clearance.fill_(1.)
    assert torch.equal(env._reward_gait_swing_clearance(), at_target)
    env.gait_clearance.fill_(-1.)
    assert env._reward_gait_swing_clearance().eq(0).all()
    assert env._reward_gait_foot_scuff().eq(1).all()


def test_contact_filter_touchdown_once_and_duration(env):
    contact(env)
    assert not env.gait_touchdown.any()  # No touchdown at history boundary.
    contact(env, False, True)
    assert env.gait_contact_filtered.all()  # Previous raw suppresses a single dropout.
    assert env.gait_air_time.eq(0).all()
    for _ in range(10):
        contact(env, False, True)
    assert env.gait_air_time[:, 0].tolist() == pytest.approx([.20, .20])
    assert not env.gait_touchdown.any()
    contact(env)
    assert env.gait_touchdown[:, 0].all() and not env.gait_touchdown[:, 1].any()
    assert env.gait_preceding_air_time[:, 0].tolist() == pytest.approx([.20, .20])
    assert env._reward_gait_air_time().tolist() == pytest.approx([.7, .7])
    contact(env)
    assert env._reward_gait_air_time().eq(0).all()
    # Single-step contact noise never produces an artificial swing/touchdown.
    contact(env, False, False)
    contact(env)
    assert not env.gait_touchdown.any()


def test_reset_ownership_and_censored_flight(env):
    # Starting in flight must not earn air-time on landing.
    for _ in range(20):
        contact(env, False, False)
    contact(env)
    assert not env.gait_touchdown.any()
    contact(env, False, False)
    for _ in range(10):
        contact(env, False, False)
    env.control_mask[0] = False
    contact(env)
    assert not env.gait_touchdown[0].any()
    env.control_mask[0] = True
    contact(env)
    assert not env.gait_touchdown[0].any()
    env._reset_gait_history(torch.tensor([1]))
    assert not env.gait_history_valid[1]
    assert env.gait_air_time[1].eq(0).all()
    contact(env)
    assert not env.gait_touchdown[1].any()
    env.control_command.zero_()
    for _ in range(15):
        contact(env, False, False)
    env.control_command[:, 0] = .8
    contact(env)
    assert env._reward_gait_air_time().eq(0).all()


def test_state_clearance_uses_rotated_collision_geometry(env, geometry):
    # Nonzero angular pose and height: never substitute ankle-origin Z.
    angle = .3
    env.rigid_body_states[:, [4, 6], 3:7] = torch.tensor([0., math.sin(angle/2), 0., math.cos(angle/2)])
    env.rigid_body_states[:, [4, 6], 2] = .09
    env._update_gait_state()
    full = foot_geometry(URDF)
    for side, i in (("left", 0), ("right", 1)):
        q = env.rigid_body_states[0, [4, 6][i], 3:7].numpy()
        exact = .09 + rotate(q, full[side + "_ankle_roll_link"]["vertices"])[:, 2].min()
        assert abs(float(env.gait_clearance[0, i]) - exact) < .001
        assert float(env.gait_clearance[0, i]) != pytest.approx(.09)


def test_geometry_independent_validation(geometry):
    for body in geometry["bodies"].values():
        assert body["point_count"] <= 128
        assert body["full_vertex_count"] > 6000
        assert body["error"]["max_m"] < .001
        assert len(body["mesh_sha256"][0]) == 64
    with pytest.raises(ValueError, match="validation failed"):
        build_gait_geometry(URDF, max_points=4)


def test_configuration_additive_and_actual_reward_registration(env):
    base, _, _ = load_configs()
    cfg, train, _ = load_configs()
    own = NS(teacher_checkpoint="teacher.pt", student_checkpoint="student.pt", ablation="A5",
             command_suite="fixed", fixed_command=[.4, 0., 0.], posture_reference="reference.json",
             handoff_config=str(CONFIG))
    load_script("train_teacher_warmup_loco").apply_configuration(cfg, train, own)
    assert cfg.handoff.warmup_vx == .8 and cfg.handoff.fixed_command == [.8, 0., 0.]
    assert train.runner.run_name == "A5_B2_gait_fixed_vx08"
    assert cfg.env.num_envs == 512 and train.seed == 1
    assert train.runner.save_interval == 100 and train.runner.num_steps_per_env == 100
    assert cfg.env.num_privileged_obs == 160
    assert cfg.control.action_scale == .25
    changed = {key for key in dir(base.rewards.scales) if not key.startswith("_")
               and getattr(base.rewards.scales, key) != getattr(cfg.rewards.scales, key)}
    assert changed == set(GAIT) | {"carry_upper_body_pose", "feet_air_time", "feet_slip"}
    for name in GAIT:
        assert getattr(base.rewards.scales, name) == 0
    env.reward_scales = {key: getattr(cfg.rewards.scales, key) for key in GAIT}
    env._prepare_reward_function()  # Calls the REAL parent's registration and dt scaling.
    assert set(env.reward_names) == set(GAIT)
    env.gait_contact_filtered[:] = True
    env.gait_horizontal_speed.fill_(.35)
    for name, function in zip(env.reward_names, env.reward_functions):
        result = function()
        assert result.shape == (2,)
        assert torch.equal(result * env.reward_scales[name], env.gait_reward_diagnostics["reward_weighted/" + name])
    assert env.reward_scales["gait_foot_slip"] == pytest.approx(-.08 * .02)


def test_json_rejects_unknown_sections_and_keys(tmp_path):
    script = load_script("train_teacher_warmup_loco")
    for bad in ({"gat": {}}, {"gait": {"target_heigt_m": .04}}, {"train": {"runner": {}}}, {"gait": []}):
        cfg, train, _ = load_configs()
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(bad))
        own = NS(teacher_checkpoint="x", ablation="A5", command_suite="fixed", fixed_command=[.8, 0, 0],
                 posture_reference="y", handoff_config=str(path))
        with pytest.raises(ValueError):
            script.apply_configuration(cfg, train, own)


def test_hot_path_has_no_cpu_or_numpy_and_legacy_rewards_untouched():
    cls = load_env_class()
    for name in ("_update_gait_state", "_gait_moving", *("_reward_" + key for key in GAIT)):
        source = inspect.getsource(getattr(cls, name))
        assert ".cpu(" not in source and ".numpy(" not in source and ".item(" not in source
    # Python 3.8 is the documented Isaac Gym environment.
    for name in ("carrybox_teacher_warmup.py", "carrybox_teacher_warmup_config.py"):
        ast.parse((ROOT / "legged_gym/legged_gym/envs/g1" / name).read_text(), feature_version=(3, 8))
    assert "_reward_feet_slip" not in cls.__dict__
    assert "_reward_feet_air_time" not in cls.__dict__


def test_b2_diagnostic_contact_source_is_explicit():
    simple, diag = diagnostic_fixture()
    roll = gait_diagnostics.GaitDiagnostics(simple, URDF, NS(DOMAIN_ENV=123), contact_source="roll")
    from test_gait_diagnostics import capture, base_row
    simple.contact_forces[:, 3, 2] = 10.  # Pitch only; corrected roll must remain false.
    capture(simple, diag)
    capture(simple, roll)
    assert diag.row(base_row(), 0, simple)["left_contact"] == 1
    assert roll.row(base_row(), 0, simple)["left_contact"] == 0
    assert roll.metadata()["settings"]["contact_source"] == "roll"


def config_dict(obj):
    if not hasattr(obj, "__dict__"):
        return obj
    return {key: [config_dict(v) for v in getattr(obj, key)] if isinstance(getattr(obj, key), list)
            else config_dict(getattr(obj, key)) for key in dir(obj) if not key.startswith("_")}


def test_b2_exact_provenance_guard_before_simulator(tmp_path, monkeypatch):
    from rsl_rl.teacher_handoff import file_sha256
    script = load_script("train_teacher_warmup_loco")
    monkeypatch.setattr(script, "__file__", str(tmp_path / "legged_gym/legged_gym/scripts/train_teacher_warmup_loco.py"))
    monkeypatch.setitem(sys.modules, "legged_gym.utils.helpers", NS(class_to_dict=config_dict))
    teacher = tmp_path / "shared_artifacts/2026-10-19/carry_track/model_33999.pt"
    student = tmp_path / "legged_gym/logs/carrybox_teacher_warmup/20261019_235711_985014_A5_fixed/model_5000.pt"
    reference = tmp_path / "experiments/teacher_warmup/results/lab_20261019_233908/full_teacher_upper.json"
    for path in (teacher, student, reference):
        path.parent.mkdir(parents=True, exist_ok=True)
    teacher.write_bytes(b"synthetic teacher hash fixture, not a real checkpoint")
    reference.write_text("{}")
    cfg, train, _ = load_configs()
    own = NS(teacher_checkpoint=str(teacher), student_checkpoint=str(student), posture_reference=str(reference),
             handoff_config=str(CONFIG), ablation="A5", command_suite="fixed", fixed_command=[.8, 0, 0])
    script.apply_configuration(cfg, train, own)
    args = NS(resume=False, rl_device="cuda:0")
    with pytest.raises(FileNotFoundError, match="exact B2 student"):
        script.validate_b2_initialization(cfg, train, own, args)
    joints = tmp_path / "legged_gym/resources/config/joint_id.txt"
    joints.parent.mkdir(parents=True)
    joints.write_bytes((ROOT / "legged_gym/resources/config/joint_id.txt").read_bytes())
    from test_teacher_warmup import NAMES
    rewards = config_dict(cfg.rewards)
    for key in GAIT:
        rewards["scales"].pop(key)
    rewards["scales"].update(feet_air_time=.2, feet_slip=-.2)
    handoff = config_dict(cfg.handoff)
    for key in ("teacher_checkpoint", "posture_reference"):
        handoff.pop(key)
    handoff.update(warmup_vx=.4, fixed_command=[.4, 0., 0.])
    # Reproduce Gym's float32 SimParams.dt -> Python policy period boundary.
    import ctypes
    saved = dict(handoff=handoff, teacher_sha256=file_sha256(teacher), reference_sha256=file_sha256(reference),
                 dof_names=NAMES, action_scale=.25, dt=ctypes.c_float(.005).value*4, actor_obs=738, critic_obs=160,
                 rewards=rewards, algorithm=config_dict(train.algorithm),
                 policy=dict(config_dict(train.policy), actor_hidden_dims=[512, 256, 256], activation="elu"),
                 environment={key: config_dict(getattr(cfg, key)) for key in
                              ("control", "normalization", "domain_rand", "asset", "commands", "dataset", "sim",
                               "noise", "terrain", "init_state")})
    # Real runner manifests are written after _parse_cfg adds this field.
    saved["environment"]["domain_rand"]["push_interval"] = math.ceil(
        cfg.domain_rand.push_interval_s / saved["dt"])
    state = {"std": torch.ones(29), "critic.0.weight": torch.zeros(256, 160)}
    dims = [738, 512, 256, 256, 29]
    for i, (a, b) in enumerate(zip(dims, dims[1:])):
        state[f"actor.{i*2}.weight"] = torch.zeros(b, a)
        state[f"actor.{i*2}.bias"] = torch.zeros(b)
    payload = dict(model_state_dict=state, handoff_contract=saved, iter=5000)
    torch.save(payload, student)
    before = file_sha256(student)
    result = script.validate_b2_initialization(cfg, train, own, args)
    assert result["student_sha256"] == before and result["original_warmup_vx"] == .4
    assert result["original_fixed_command"] == [.4, 0., 0.]
    assert file_sha256(student) == before
    cfg.handoff.warmup_vx = .4
    with pytest.raises(ValueError, match="warmup_vx"):
        script.validate_b2_initialization(cfg, train, own, args)
    cfg.handoff.warmup_vx = .8
    cfg.rewards.scales.carry_hand_slip = 50.
    with pytest.raises(ValueError, match="unrelated A5 reward"):
        script.validate_b2_initialization(cfg, train, own, args)
    cfg.rewards.scales.carry_hand_slip = .5
    saved["teacher_sha256"] = "wrong"
    torch.save(payload, student)
    with pytest.raises(ValueError, match="hash mismatch"):
        script.validate_b2_initialization(cfg, train, own, args)


def test_disabled_gait_has_no_effect_on_reward_registration(env):
    env.gait_enabled = False
    env.control_mask = torch.tensor([True, False])
    # Existing reward registration/masking executes the original callback once.
    calls = []
    env._reward_old_fixture = lambda: (calls.append(1) or torch.tensor([2., 3.]))
    env.reward_scales = {"old_fixture": 1., **dict.fromkeys(GAIT, 0.)}
    rng = torch.get_rng_state().clone()
    env._prepare_reward_function()
    assert env.reward_names == ["old_fixture"]
    assert env.reward_functions[0]().tolist() == [2., 0.]
    assert len(calls) == 1 and torch.equal(rng, torch.get_rng_state())
    assert env.reward_scales == {"old_fixture": .02}
