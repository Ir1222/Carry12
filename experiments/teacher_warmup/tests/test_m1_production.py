"""M1 production settings and smoke plumbing; no synthetic fixture is a GPU measurement."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest
import torch

from test_mixed_loco import configs, as_dict, checkpoint_fixture
from test_teacher_warmup import ROOT, load_configs, load_script, SyntheticVecEnv
from rsl_rl.runners.teacher_warmup_on_policy_runner import TeacherWarmupOnPolicyRunner
from rsl_rl.m1_performance import summarize


def smoke_script():
    spec = importlib.util.spec_from_file_location("m1_smoke", ROOT / "experiments/teacher_warmup/m1_smoke.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("count", [1024, 2048, 4096, 8192])
def test_cli_env_count_and_independent_m1_validation(count):
    cfg, train = configs()
    assert cfg.env.num_envs == 4096 and train.runner.max_iterations == 10000
    assert train.runner.num_steps_per_env == 100 and train.runner.save_interval == 500
    assert train.algorithm.learning_rate == 5e-5 and train.algorithm.min_valid_samples == 2048
    tree = ast.parse((ROOT / "legged_gym/legged_gym/utils/helpers.py").read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "update_cfg_from_args"]
    namespace = {}
    exec(compile(tree, "helpers.py", "exec"), namespace)
    args = NS(num_envs=count, seed=3, play_dataset=False, max_iterations=40, resume=False,
              resume_path=None, finetune_path=None, experiment_name=None, run_name="smoke",
              load_run=None, checkpoint=None)
    namespace["update_cfg_from_args"](cfg, train, args)
    validate = load_script("mixed_initialization")
    validate.validate_m1_training_settings(cfg, train)
    validate.validate_m1_runtime(NS(cfg=cfg, num_envs=count, dt=.02, num_obs=738,
                                    num_actions=29, num_privileged_obs=160), train)
    assert cfg.env.num_envs == count and train.runner.max_iterations == 40
    base, b2_train, _ = load_configs()
    own = NS(teacher_checkpoint="teacher.pt", posture_reference="pose.json", ablation=None,
             command_suite=None, fixed_command=None,
             handoff_config=str(ROOT / "experiments/teacher_warmup/config_B2_gait_vx08.json"))
    load_script("train_teacher_warmup_loco").apply_configuration(base, b2_train, own)
    assert base.env.num_envs == 512
    assert b2_train.runner.save_interval == b2_train.runner.num_steps_per_env == 100


@pytest.mark.parametrize("group,key,value", [("env", "num_envs", 0), ("runner", "max_iterations", 0),
    ("runner", "save_interval", 100), ("runner", "num_steps_per_env", 99),
    ("algorithm", "learning_rate", 1e-4), ("algorithm", "min_valid_samples", 1)])
def test_m1_rejects_wrong_production_settings(group, key, value):
    cfg, train = configs()
    target = cfg.env if group == "env" else getattr(train, group)
    setattr(target, key, value)
    with pytest.raises(ValueError, match="M1"):
        load_script("mixed_initialization").validate_m1_training_settings(cfg, train)


def test_m1_resume_changes_execution_metadata_but_rejects_rollout_or_semantic_change():
    base = dict(task="carrybox_teacher_warmup_mixed", handoff={"adapter": "rate_limited"},
                environment=dict(sim={"dt": .005}, b2_execution=dict(num_envs=512,
                    save_interval=100, num_steps_per_env=100, seed=1, rl_device="cuda:0")))
    current = copy.deepcopy(base)
    current["environment"].pop("b2_execution")
    current["environment"]["m1_execution"] = dict(num_envs=4096, save_interval=500,
                                                num_steps_per_env=100, seed=3, rl_device="cuda:1")
    normalize = TeacherWarmupOnPolicyRunner._resume_contract
    assert normalize(base) == normalize(current)
    assert "b2_execution" in base["environment"]  # No mutation of saved contract.
    current["environment"]["m1_execution"]["num_steps_per_env"] = 99
    assert normalize(base) != normalize(current)
    current["environment"]["m1_execution"]["num_steps_per_env"] = 100
    current["handoff"]["adapter"] = "hard"
    assert normalize(base) != normalize(current)
    legacy = copy.deepcopy(base)
    legacy.pop("task")
    assert normalize(legacy) == legacy  # B2 still compares all execution metadata.


def test_tensorboard_each_iteration_and_save_every_500_are_independent(tmp_path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    _, _, _, source = checkpoint_fixture(tmp_path)
    _, ppo = configs()
    train = as_dict(ppo)
    train["runner"].update(num_steps_per_env=4)  # Small actual masked-PPO fixture.
    train["algorithm"].update(min_valid_samples=1, num_learning_epochs=1, num_mini_batches=1)
    train.update(task="carrybox_teacher_warmup_mixed", handoff_contract={})
    run = TeacherWarmupOnPolicyRunner(SyntheticVecEnv(source), train, str(source), str(tmp_path / "run"), "cpu")
    run.load_init_policy(str(source))
    run.current_learning_iteration = 498
    run.learn(3)
    run.writer.close()
    assert sorted(p.name for p in (tmp_path / "run").glob("model_*.pt")) == ["model_500.pt", "model_501.pt"]
    events = EventAccumulator(str(tmp_path / "run")).Reload()
    assert [row.step for row in events.Scalars("Perf/fps")] == [498, 499, 500]
    assert len(events.Scalars("Train/valid_student_samples")) == 3


def test_smoke_commands_use_actual_training_and_user_selected_checkpoint(tmp_path):
    module = smoke_script()
    assert module.DEFAULT_ENV_COUNTS == (1024, 2048, 4096)
    args = NS(teacher_checkpoint="teacher.pt", posture_reference="pose.json", init_policy_path="chosen_best_B2.pt",
              iterations=40, seed=1, rl_device="cuda:0")
    for count in module.DEFAULT_ENV_COUNTS:
        command = module.build_command(args, count, tmp_path / (str(count) + ".json"))
        assert command[command.index("--num_envs") + 1] == str(count)
        assert command[command.index("--init_policy_path") + 1] == "chosen_best_B2.pt"
        assert command[command.index("--max_iterations") + 1] == "40"
        assert "--smoke_report" in command and "--handoff_config" not in command


def test_smoke_summary_requires_ppo_and_mixed_coverage_and_absent_values_stay_absent():
    assert summarize([])["peak_torch_allocated_bytes"] is None
    row = dict(skipped_update=True, mixed_steps=0, ramp_steps=0, iteration_seconds=2.,
               env_steps=1000, rollout_seconds=1.8, update_seconds=.1, valid_student_samples=0,
               peak_torch_allocated_bytes=10, peak_torch_reserved_bytes=20,
               device_used_bytes=30, device_free_bytes=70)
    assert summarize([row])["status"] == "incomplete_student_coverage"
    assert summarize([row])["student_update_env_steps_per_second"] is None
    row.update(skipped_update=False, mixed_steps=10, valid_student_samples=20)
    result = summarize([row])
    assert result["status"] == "measured_student_ppo" and result["ppo_updates"] == 1
    assert result["overall_env_steps_per_second"] == 500


def test_smoke_probe_reports_three_unmeasured_cases(tmp_path, monkeypatch):
    module = smoke_script()
    out = tmp_path / "probe"
    monkeypatch.setattr(module, "runtime_probe", lambda: ({"cuda_available": False}, ["CUDA is unavailable"]))
    monkeypatch.setattr(sys, "argv", ["m1_smoke.py", "--probe_only", "--output", str(out)])
    assert module.main() == 2
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "blocked" and len(report["cases"]) == 3
    assert all(case["env_steps_per_second"] is None for case in report["cases"])
    assert report["blockers"] == ["CUDA is unavailable"]



def test_cuda_recorder_timing_memory_and_json_with_synthetic_backend(tmp_path, monkeypatch):
    import rsl_rl.m1_performance as module
    backend = NS(is_available=lambda: True, synchronize=lambda device: None,
                 get_device_properties=lambda device: NS(name="synthetic_test_backend", total_memory=1024),
                 mem_get_info=lambda device: (512, 1024), memory_allocated=lambda device: 64,
                 memory_reserved=lambda device: 128, max_memory_allocated=lambda device: 128,
                 max_memory_reserved=lambda device: 256)
    monkeypatch.setattr(torch, "cuda", backend)
    ticks = iter([0., 2., 3., 4.])
    monkeypatch.setattr(module, "time", NS(perf_counter=lambda: next(ticks)))
    report = tmp_path / "synthetic_recorder.json"
    recorder = module.PerformanceSmokeRecorder(report, "cuda:0", 1024, 100, 500)
    recorder.begin_iteration()
    recorder.rollout_done()
    recorder.update_done()
    recorder.record(1, dict(valid_samples=4096, skipped_update=False), {"vx/all/steps": 50}, 0., [1., 2.])
    result = recorder.finish("synthetic_run", .5)
    assert result["configuration"]["save_interval"] == 500
    assert result["summary"]["mean_rollout_seconds"] == 2.
    assert result["summary"]["mean_ppo_update_seconds"] == 1.
    assert result["summary"]["overall_env_steps_per_second"] == 102400 / 4
    assert result["summary"]["peak_torch_allocated_bytes"] == 128
    assert result["summary"]["max_sampled_device_used_bytes"] == 512
    assert json.loads(report.read_text(encoding="utf-8"))["final_checkpoint_seconds"] == .5
    assert len(report.with_suffix(".jsonl").read_text(encoding="utf-8").splitlines()) == 1
    recorder.rows[0]["losses_finite"] = False
    assert module.summarize(recorder.rows)["status"] == "nonfinite_training"


def test_smoke_worker_failure_records_oom_without_launching_next_size(tmp_path, monkeypatch):
    module = smoke_script()
    monkeypatch.setattr(module, "gpu_sample", lambda: None)
    monkeypatch.setattr(module, "build_command", lambda *args: [sys.executable, "-c",
                        "raise RuntimeError('CUDA out of memory')"])
    result = module.run_case(NS(timeout_s=10), 4096, tmp_path)
    assert result["status"] == "out_of_memory" and result["returncode"] != 0
    assert result["nvidia_smi_peak_device_usage"] is None
    assert "measurement" not in result
