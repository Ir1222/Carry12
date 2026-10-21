"""Reproduce Isaac Gym SimParams float32 dt metadata without requiring Isaac Gym."""
import copy
import ctypes
import math
import sys
from types import SimpleNamespace as NS

import pytest
import torch

from test_mixed_loco import configs, checkpoint_fixture, as_dict
from test_teacher_warmup import load_script, SyntheticVecEnv
from rsl_rl.teacher_handoff import matching_control_dt, file_sha256
from rsl_rl.runners.teacher_warmup_on_policy_runner import TeacherWarmupOnPolicyRunner

CONTROL_DT_FLOAT32 = ctypes.c_float(.005).value * 4


@pytest.mark.parametrize("actual,expected", [(CONTROL_DT_FLOAT32, .02), (.02, CONTROL_DT_FLOAT32),
                                            (.02, .02), (.02 + .5e-9, .02)])
def test_timestep_comparison_only_accepts_representation_error(actual, expected):
    assert matching_control_dt(actual, expected)


@pytest.mark.parametrize("bad", [.01, .04, .02 + 2e-9, 0., -1., float("nan"), float("inf"),
                                 None, ".02", True])
def test_invalid_or_changed_timestep_rejected(bad):
    assert not matching_control_dt(bad, .02)
    assert not matching_control_dt(.02, bad)


@pytest.mark.parametrize("resume", [False, True])
def test_m1_contract_float32_dt_and_derived_push_interval(tmp_path, resume):
    payload, spec, expected, _ = checkpoint_fixture(tmp_path)
    saved = payload["handoff_contract"]
    saved["dt"] = CONTROL_DT_FLOAT32
    seconds = expected["environment"]["domain_rand"]["push_interval_s"]
    saved["environment"]["domain_rand"]["push_interval"] = math.ceil(seconds / CONTROL_DT_FLOAT32)
    assert saved["environment"]["domain_rand"]["push_interval"] != math.ceil(seconds / expected["dt"])
    if resume:
        saved["task"] = "carrybox_teacher_warmup_mixed"
        saved["handoff"] = copy.deepcopy(expected["handoff"])
    module = load_script("mixed_initialization")
    before = copy.deepcopy(expected)
    module.validate_m1_contract(payload, spec, expected, resume=resume)
    assert expected == before
    saved["dt"] = .01
    with pytest.raises(ValueError, match=r"dt \(checkpoint=0.01, requested=0.02, tolerance=1e-9 s\)"):
        module.validate_m1_contract(payload, spec, expected, resume=resume)


def test_real_preflight_reads_float32_checkpoint_without_changing_config_or_file(tmp_path, monkeypatch):
    payload, _, _, checkpoint = checkpoint_fixture(tmp_path)
    cfg, ppo = configs()
    teacher, reference = tmp_path / "teacher.pt", tmp_path / "posture.json"
    teacher.write_bytes(b"teacher_fixture")
    reference.write_bytes(b"reference_fixture")
    cfg.handoff.teacher_checkpoint, cfg.handoff.posture_reference = str(teacher), str(reference)
    saved = payload["handoff_contract"]
    saved["dt"] = CONTROL_DT_FLOAT32
    saved["environment"]["domain_rand"]["push_interval"] = math.ceil(cfg.domain_rand.push_interval_s / saved["dt"])
    for key, path in (("teacher_sha256", teacher), ("reference_sha256", reference)):
        saved[key] = saved["environment"]["b2_initialization"][key] = file_sha256(path)
    torch.save(payload, checkpoint)
    before_hash = file_sha256(checkpoint)
    monkeypatch.setitem(sys.modules, "legged_gym.utils.helpers", NS(class_to_dict=as_dict))
    module = load_script("mixed_initialization")
    result = module.validate_m1_initialization(cfg, ppo, NS(student_checkpoint=str(checkpoint)),
                                                NS(resume=False, resume_path=None))
    assert result["source_control_dt"] == CONTROL_DT_FLOAT32
    assert result["requested_control_dt"] == .02
    assert cfg.sim.dt == .005 and not hasattr(cfg.domain_rand, "push_interval")
    assert file_sha256(checkpoint) == before_hash
    module.validate_m1_runtime(NS(cfg=cfg, num_envs=4096, dt=CONTROL_DT_FLOAT32, num_obs=738,
                                 num_actions=29, num_privileged_obs=160), ppo)


def test_actual_m1_runner_resume_accepts_dt_roundoff_but_b2_remains_exact(tmp_path):
    _, _, _, source = checkpoint_fixture(tmp_path)
    _, ppo = configs()
    train = as_dict(ppo)
    train["runner"].update(num_steps_per_env=4)
    train["algorithm"].update(min_valid_samples=1, num_learning_epochs=1, num_mini_batches=1)
    train.update(task="carrybox_teacher_warmup_mixed", handoff_contract={})
    run = TeacherWarmupOnPolicyRunner(SyntheticVecEnv(source), train, str(source), str(tmp_path / "run"), "cpu")
    run.load_init_policy(str(source))
    run.learn(1)
    run.writer.close()
    checkpoint = tmp_path / "run/model_1.pt"
    data = torch.load(checkpoint, weights_only=False)
    data["handoff_contract"]["dt"] = CONTROL_DT_FLOAT32
    portable = tmp_path / "portable.pt"
    torch.save(data, portable)
    resumed = TeacherWarmupOnPolicyRunner(SyntheticVecEnv(source), train, str(portable), str(tmp_path / "resume"), "cpu")
    resumed.load(str(portable))
    assert resumed.current_learning_iteration == 1
    assert resumed.contract["dt"] == .02
    data["handoff_contract"]["dt"] = .01
    incompatible = tmp_path / "different_physics.pt"
    torch.save(data, incompatible)
    with pytest.raises(ValueError, match="Resume contract"):
        resumed.load(str(incompatible))
    # Legacy tasks retain the existing exact resume comparison.
    data["handoff_contract"].pop("task")
    data["handoff_contract"]["dt"] = CONTROL_DT_FLOAT32
    legacy = tmp_path / "legacy_b2.pt"
    torch.save(data, legacy)
    resumed.contract.pop("task")
    with pytest.raises(ValueError, match="Resume contract"):
        resumed.load(str(legacy))
    resumed.writer.close()
