"""Deterministic CPU command/contract tests. These do not validate physical locomotion."""
import copy
import hashlib
import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from test_teacher_warmup import (ROOT, NAMES, definitions, load_configs, load_env_class,
                                 load_script, supervisor_fixture, SyntheticVecEnv)
import sys
sys.path.insert(0, str(ROOT / "legged_gym/legged_gym/scripts"))

from rsl_rl.teacher_handoff import checkpoint_actor
from rsl_rl.runners.teacher_warmup_on_policy_runner import TeacherWarmupOnPolicyRunner
from rsl_rl.mixed_command_metrics import MixedCommandMetrics


def configs():
    cfg, ppo, _ = load_configs()
    ns = dict(WarmupCfg=type(cfg), WarmupPPO=type(ppo))
    definitions("legged_gym/legged_gym/envs/g1/carrybox_teacher_warmup_mixed_config.py", ns)
    return ns["G1Cfg"](), ns["G1CfgPPO"]()


def as_dict(obj):
    if isinstance(obj, (list, tuple)):
        return [as_dict(v) for v in obj]
    if not hasattr(obj, "__dict__"):
        return obj
    return {k: as_dict(getattr(obj, k)) for k in dir(obj) if not k.startswith("_") and not callable(getattr(obj, k))}


@pytest.fixture
def env():
    warmup = load_env_class()
    # Reuse actual inherited sampler with deterministic CPU Torch RNG.
    warmup._sample_carry_commands.__globals__["torch_rand_float"] = lambda lo, hi, shape, device: lo + (hi-lo) * torch.rand(shape, device=device)
    ns = dict(Warmup=warmup, FullCarryBox=warmup._reset_task.__globals__["FullCarryBox"],
              torch=torch, math=math, TEACHER_WARMUP=0, STUDENT_CARRY=2)
    definitions("legged_gym/legged_gym/envs/g1/carrybox_teacher_warmup_mixed.py", ns)
    result = ns["LeggedRobot"].__new__(ns["LeggedRobot"])
    result.__dict__.update(vars(supervisor_fixture()))
    result.cfg, _ = configs()
    result.carry_command_mode = torch.zeros(3, dtype=torch.long)
    result.control_mask = torch.zeros(3, dtype=torch.bool)
    # Remove the supervisor fixture's no-op timer sampler.
    del result._sample_carry_command_resample_time
    result._init_mixed_commands()
    result.gait_enabled = False
    result.commands[:, 0] = .8
    result.carry_policy_commands.copy_(result.commands)
    return result


def tick(env):
    env.control_mask = env.phase != 0
    env._update_carry_heading_commands()
    env._advance_mixed_commands()


def test_legacy_configs_unchanged_and_m1_exact():
    for file, sha in (("config_4090.json", "a0421cbf1bbc79a6a5cacf084794750622d9ea9ce00dc407a170c9f43bda4d54"),
                      ("config_B2_gait_vx08.json", "bb999d220f92e82bdf810e6791e207d24b00fe35b7e1ce74816714bc451e328f")):
        data = (ROOT / "experiments/teacher_warmup" / file).read_bytes()
        # Git checkout can normalize CRLF independently of the original JSON.
        assert hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest() == sha
        assert json.loads(data)
    cfg, ppo = configs()
    b2 = json.loads((ROOT / "experiments/teacher_warmup/config_B2_gait_vx08.json").read_text())
    for section, target in (("handoff", cfg.handoff), ("algorithm", ppo.algorithm),
                            ("rewards", cfg.rewards.scales), ("gait", cfg.gait)):
        for key, value in b2[section].items():
            if key != "command_suite":
                assert getattr(target, key) == value
    assert cfg.handoff.command_suite == "mixed"
    assert cfg.handoff.mixed_command_hold_s == 2.
    assert cfg.handoff.mixed_command_slew_rates == [.6, .5, .6]
    assert cfg.env.num_envs == 4096
    assert ppo.runner.max_iterations == 10000
    assert ppo.runner.num_steps_per_env == 100
    assert ppo.runner.save_interval == 500
    legacy, _, _ = load_configs()
    assert legacy.handoff.ablation == "A3" and legacy.rewards.scales.gait_air_time == 0


def test_first_student_hold_transition_and_single_release(env):
    torch.manual_seed(4)
    for _ in range(130):
        tick(env)
    assert env.mixed_target_samples.eq(0).all()  # No sampling during teacher.
    env.phase[0] = 1
    env._set_student_commands(torch.tensor([0]))
    assert env.applied_command[0].tolist() == pytest.approx([.8, 0., 0.])
    for _ in range(150):
        tick(env)
        assert env.applied_command[0].tolist() == pytest.approx([.8, 0., 0.])
    assert env.student_control_elapsed[0] == 3.
    assert not env.mixed_released.any()  # Even a transition longer than hold cannot release.
    env.phase[0] = 2
    env._advance_mixed_commands()
    assert env.mixed_target_samples.tolist() == [1, 0, 0]
    for _ in range(40):
        tick(env)
    assert env.mixed_target_samples.tolist() == [1, 0, 0]


def test_hold_exact_ticks_and_axis_slew_independent(env):
    env.phase[0] = 2
    env._set_student_commands(torch.tensor([0]))
    for _ in range(99):
        tick(env)
        assert env.applied_command[0].tolist() == pytest.approx([.8, 0., 0.])
        assert not env.mixed_released.any()
    tick(env)
    assert env.mixed_released.tolist() == [True, False, False]
    env.target_command[0] = torch.tensor([-.6, .5, -.7])
    limit = torch.tensor([.6, .5, .6]) * env.dt
    for _ in range(120):
        before = env.applied_command.clone()
        tick(env)
        assert ((env.applied_command - before).abs() <= limit + 1e-7).all()
        assert env.applied_command[1:].eq(torch.tensor([.8, 0., 0.])).all()
    assert torch.allclose(env.applied_command[0], env.target_command[0])


def test_target_timer_starts_at_sampling_not_settling_and_async(env):
    torch.manual_seed(13)
    env.phase[:] = torch.tensor([2, 1, 0])
    env.student_control_steps[0] = 100
    env.student_control_elapsed[0] = 2.
    env._advance_mixed_commands()
    interval = float(env.carry_command_resample_time[0])
    assert 4 <= interval <= 6
    assert env.mixed_target_samples.tolist() == [1, 0, 0]
    # Deliberately make ramp last far longer than the resample interval.
    env.cfg.handoff.mixed_command_slew_rates = [1e-5] * 3
    env.target_command[0] = torch.tensor([-.6, .5, -.7])
    for _ in range(math.ceil(interval / env.dt) - 1):
        tick(env)
    assert env.mixed_target_samples[0] == 1
    assert env.mixed_ramp_active[0]
    tick(env)
    assert env.mixed_target_samples.tolist() == [2, 0, 0]
    env.phase[1] = 2
    tick(env)
    assert env.mixed_target_samples.tolist() == [2, 1, 0]


def test_reset_clears_every_manager_field_for_only_requested_environment(env):
    env.phase[:] = 2
    env.student_control_elapsed[:] = 2
    env._advance_mixed_commands()
    before = env.target_command[1:].clone()
    env._reset_mixed_commands(torch.tensor([0]))
    assert torch.equal(env.target_command[1:], before)
    assert env.applied_command[0].tolist() == pytest.approx([.8, 0., 0.])
    for name in ("student_control_elapsed", "student_control_steps", "mixed_released", "mixed_ramp_active",
                 "mixed_target_samples", "mixed_target_changed", "carry_command_resample_time"):
        assert getattr(env, name)[0] == 0
    assert env.carry_command_mode[0] == -1


def test_original_sampler_distribution_dead_zones_and_ranges(env):
    cfg = env.cfg.commands
    assert cfg.carry_command_mode_probabilities == [.1, .1, .1, .1, .6]
    assert sum(cfg.carry_command_mode_probabilities) == 1
    assert cfg.carry_mixed_ranges == [[-.6, 1.2], [-.5, .5], [-.7, .7]]
    # No sampler override exists in the new task.
    assert "_sample_carry_commands" not in type(env).__dict__
    torch.manual_seed(37)
    samples, modes = [], []
    for _ in range(300):
        env._sample_mixed_targets(torch.arange(3))
        samples.append(env.target_command.clone())
        modes.append(env.carry_command_mode.clone())
    commands, modes = torch.cat(samples), torch.cat(modes)
    assert (torch.bincount(modes, minlength=5) > 50).all()
    for axis, limit in enumerate(cfg.carry_mixed_ranges):
        assert commands[:, axis].min() >= limit[0]
        assert commands[:, axis].max() <= limit[1]
    assert commands[modes == 0].eq(0).all()
    for mode in (1, 2, 3):
        axis = mode - 1
        assert commands[modes == mode, axis].abs().min() >= .1
        assert commands[modes == mode][:, [i for i in range(3) if i != axis]].eq(0).all()


def test_reward_observation_snapshot_temporal_consistency(env, monkeypatch):
    warmup = type(env).__bases__[0]
    seen = {}
    def reward(self):
        seen["reward"] = self.carry_policy_commands[:, :3].clone()
        self.last_step_snapshot = {"command": self.control_command.clone()}
    def observation(self):
        seen["actor_next"] = self.carry_policy_commands[:, :3].clone()
    monkeypatch.setattr(warmup, "compute_reward", reward)
    monkeypatch.setattr(warmup, "compute_observations", observation)
    env.phase[:] = 2
    env.student_control_elapsed[:] = 2
    env._advance_mixed_commands()
    env.control_mask[:] = True
    env.carry_history_valid = torch.ones(3, dtype=torch.bool)
    env.box_relative_pos = torch.zeros(3, 3)
    env.box_relative_velocity = torch.zeros(3, 3)
    env.hand_slip = torch.zeros(3, 2)
    env.control_command = env.applied_command.clone()
    physical = env.control_command.clone()
    target = env.target_command.clone()
    env._control_target = target.clone()
    env._control_mode = env.carry_command_mode.clone()
    env._control_ramp = env.mixed_ramp_active.clone()
    env._control_hold = ~env.mixed_released.clone()
    env._control_target_changed = env.mixed_target_changed.clone()
    # Heading callback must preserve raw vy/yaw and command provided to physical action.
    env._update_carry_heading_commands()
    assert torch.equal(env.carry_policy_commands[:, :3], physical)
    env.compute_reward()
    assert torch.equal(seen["reward"], physical)
    env.carry_command_resample_time[:] = 0
    env.compute_observations()
    assert torch.equal(seen["actor_next"], env.applied_command)
    assert torch.equal(env.last_step_snapshot["command"], physical)
    assert torch.equal(env.last_step_snapshot["target_command"], target)
    assert not torch.equal(env.target_command, target)


def test_configuration_precedence_and_initialization_arguments():
    script = load_script("train_teacher_warmup_loco")
    cfg, ppo = configs()
    own = NS(teacher_checkpoint="teacher.pt", student_checkpoint=None, init_policy_path="b2.pt",
             ablation=None, command_suite=None, fixed_command=None, posture_reference="pose.json", handoff_config=None)
    script.apply_configuration(cfg, ppo, own)
    assert cfg.handoff.ablation == "A5" and cfg.handoff.command_suite == "mixed"
    assert ppo.runner.run_name == "M1_mixed"
    own.command_suite, own.fixed_command = "fixed", [0., 0., 0.]
    script.apply_configuration(cfg, ppo, own)
    assert cfg.handoff.command_suite == "fixed" and cfg.handoff.fixed_command == [0., 0., 0.]
    args = NS(init_policy_path=None, finetune_path=None, resume=False, resume_path=None)
    script.validate_initialization_arguments(args, own)
    assert own.student_checkpoint == "b2.pt"
    with pytest.raises(ValueError, match="Conflicting"):
        script.validate_initialization_arguments(args, own)
    own.init_policy_path = own.student_checkpoint = None
    args.resume = True
    with pytest.raises(ValueError, match="Resume requires"):
        script.validate_initialization_arguments(args, own)
    args.resume_path = "m1.pt"
    script.validate_initialization_arguments(args, own)
    args.resume = False
    with pytest.raises(ValueError, match="New runs require"):
        script.validate_initialization_arguments(args, own)


def checkpoint_fixture(tmp_path):
    from rsl_rl.modules import ActorCritic
    model = ActorCritic(738, 160, 6, 29, actor_hidden_dims=[512, 256, 256], critic_hidden_dims=[512, 256, 128])
    cfg, ppo = configs()
    h = as_dict(cfg.handoff)
    h.pop("teacher_checkpoint")
    h.pop("posture_reference")
    expected = dict(teacher_sha256="teacher", reference_sha256="pose", dof_names=NAMES,
                    actor_obs=738, critic_obs=160, dt=.02, action_scale=.25, handoff=h,
                    rewards=as_dict(cfg.rewards), algorithm=as_dict(ppo.algorithm), policy=as_dict(ppo.policy),
                    environment={k: as_dict(getattr(cfg, k)) for k in load_script("mixed_initialization").ENV_KEYS + ("gait",)})
    source = copy.deepcopy(expected)
    source["schema"] = 1
    for key in load_script("mixed_initialization").MANAGER_KEYS:
        source["handoff"].pop(key)
    source["handoff"]["command_suite"] = "fixed"
    source["environment"].update(b2_initialization=dict(teacher_sha256="teacher", reference_sha256="pose",
                               actor_dims=[738, 512, 256, 256, 29], original_iteration=101), gait_geometry={"mesh_sha256": "mesh"})
    optimizer = torch.optim.Adam(model.parameters())
    model.actor(torch.zeros(1, 738)).sum().backward()
    optimizer.step()
    payload = dict(iter=17, model_state_dict=model.state_dict(), handoff_contract=source,
                   optimizer_state_dict=optimizer.state_dict(), timesteps=17 * 100 * 512)
    path = tmp_path / "arbitrary_best_B2.pt"
    torch.save(payload, path)
    _, spec = checkpoint_actor(path)
    return payload, spec, expected, path


def test_m1_accepts_trained_b2_and_rejects_incompatible_contracts(tmp_path):
    payload, spec, expected, _ = checkpoint_fixture(tmp_path)
    validate = load_script("mixed_initialization").validate_m1_contract
    validate(payload, spec, expected)
    mutations = [("teacher_sha256", "different"), ("reference_sha256", None), ("dt", .01),
                 ("action_scale", .5), ("critic_obs", 126), ("dof_names", NAMES[::-1]),
                 ("rewards", {}), ("algorithm", {}), ("policy", {})]
    for key, value in mutations:
        bad = copy.deepcopy(payload)
        bad["handoff_contract"][key] = value
        with pytest.raises(ValueError, match="mismatch"):
            validate(bad, spec, expected)
    for key, value in (("adapter", "hard"), ("warmup_vx", .4)):
        bad = copy.deepcopy(payload)
        bad["handoff_contract"]["handoff"][key] = value
        with pytest.raises(ValueError, match="handoff"):
            validate(bad, spec, expected)
    for key in ("b2_initialization", "gait_geometry"):
        bad = copy.deepcopy(payload)
        bad["handoff_contract"]["environment"].pop(key)
        with pytest.raises(ValueError):
            validate(bad, spec, expected)
    for bad in (dict(payload, iter=0), dict(payload, handoff_contract=None)):
        with pytest.raises(ValueError, match="trained"):
            validate(bad, spec, expected)
    with pytest.raises(ValueError, match="738D"):
        validate(payload, dict(spec, critic_input=126), expected)


def test_m1_resume_contract_rejects_b2_and_manager_changes(tmp_path):
    payload, spec, expected, _ = checkpoint_fixture(tmp_path)
    module = load_script("mixed_initialization")
    with pytest.raises(ValueError, match="M1 checkpoint"):
        module.validate_m1_contract(payload, spec, expected, resume=True)
    payload["handoff_contract"]["task"] = module.MIXED_TASK
    payload["handoff_contract"]["handoff"] = copy.deepcopy(expected["handoff"])
    module.validate_m1_contract(payload, spec, expected, resume=True)
    # A valid early M1 save may have skipped all PPO updates while teacher warms up.
    early = dict(payload, iter=1, optimizer_state_dict={"state": {}}, timesteps=100)
    module.validate_m1_contract(early, spec, expected, resume=True)
    payload["handoff_contract"]["handoff"]["mixed_command_hold_s"] = 1.
    with pytest.raises(ValueError, match="handoff"):
        module.validate_m1_contract(payload, spec, expected, resume=True)


def test_actual_actor_std_only_initialization_and_fresh_adam(tmp_path):
    payload, _, _, path = checkpoint_fixture(tmp_path)
    cfg, ppo = configs()
    train = as_dict(ppo)
    train.update(handoff_contract={}, task="carrybox_teacher_warmup_mixed")
    synthetic = SyntheticVecEnv(path)
    run = TeacherWarmupOnPolicyRunner(synthetic, train, str(path), str(tmp_path / "run"), "cpu")
    critic_before = {k: v.clone() for k, v in run.alg.actor_critic.state_dict().items() if k.startswith("critic.")}
    run.load_init_policy(str(path))
    for key, value in run.alg.actor_critic.state_dict().items():
        if key.startswith("actor.") or key == "std":
            assert torch.equal(value, payload["model_state_dict"][key])
        elif key.startswith("critic."):
            assert torch.equal(value, critic_before[key])
    assert not run.alg.optimizer.state
    assert run.current_learning_iteration == 0
    run.writer.close()


def test_absent_modes_missing_gait_and_ramp_metrics():
    metrics = MixedCommandMetrics()
    assert metrics.scalars() == {}
    snap = dict(target_mode=torch.tensor([3, -1]), command_hold=torch.tensor([False, True]),
                command_ramp=torch.tensor([True, False]), target_sampled=torch.tensor([True, False]),
                command=torch.tensor([[.7, 0., .1], [.8, 0., 0.]]),
                target_command=torch.tensor([[0., 0., .4], [.8, 0., 0.]]), actual=torch.zeros(2, 3),
                gait_clearance=torch.full((2, 2), float("nan")))
    metrics.update(snap, torch.ones(2, dtype=torch.bool))
    values = metrics.scalars()
    assert values["hold_ratio"] == values["ramp_ratio"] == .5
    assert values["yaw/target_samples"] == 1
    assert values["yaw/ramping/applied_rmse_vx"] == pytest.approx(.7)
    assert values["yaw/ramping/target_rmse_vx"] == 0
    assert "yaw/settled/applied_rmse_vx" not in values
    assert not any("gait_clearance" in key or key.startswith("stand/") for key in values)


@pytest.mark.parametrize("suite,command,mode", [
    ("fixed08", [.8, 0., 0.], 1), ("stand", [0., 0., 0.], 0),
    ("vx_forward", [.8, 0., 0.], 1), ("vx_backward", [-.4, 0., 0.], 1),
    ("vy_left", [0., .3, 0.], 2), ("vy_right", [0., -.3, 0.], 2),
    ("yaw_left", [0., 0., .4], 3), ("yaw_right", [0., 0., -.4], 3)])
def test_fixed_evaluation_suites_preserve_handoff_and_mode(env, suite, command, mode):
    cfg, ppo = configs()
    saved = dict(handoff=as_dict(cfg.handoff), task="carrybox_teacher_warmup_mixed")
    saved["handoff"].pop("teacher_checkpoint")
    saved["handoff"].pop("posture_reference")
    own = NS(teacher_checkpoint="teacher.pt", posture_reference="pose.json", ablation=None,
             command_suite=None, fixed_command=None, handoff_config=None, eval_suite=suite)
    load_script("eval_teacher_warmup_loco").configure_evaluation(cfg, ppo, own, saved, True)
    assert cfg.handoff.fixed_command == command
    assert cfg.handoff.adapter == "rate_limited"
    env.cfg = cfg
    env.phase[0] = 2
    for _ in range(99):
        tick(env)
    assert env.applied_command[0].tolist() == pytest.approx([.8, 0., 0.])
    tick(env)
    assert env.target_command[0].tolist() == pytest.approx(command)
    assert env.carry_command_mode[0] == mode
    assert env.mixed_target_samples[0] == 1
    for _ in range(250):
        tick(env)
    assert env.mixed_target_samples[0] == 1
    assert torch.allclose(env.applied_command[0], torch.tensor(command))


def test_m1_preflight_no_json_and_actor_architecture_contract(tmp_path, monkeypatch):
    import sys
    from rsl_rl.teacher_handoff import file_sha256
    payload, _, _, path = checkpoint_fixture(tmp_path)
    cfg, ppo = configs()
    teacher, pose = tmp_path / "teacher.pt", tmp_path / "pose.json"
    teacher.write_bytes(b"teacher")
    pose.write_bytes(b"reference")
    cfg.handoff.teacher_checkpoint, cfg.handoff.posture_reference = str(teacher), str(pose)
    source = payload["handoff_contract"]
    source["teacher_sha256"] = source["environment"]["b2_initialization"]["teacher_sha256"] = file_sha256(teacher)
    source["reference_sha256"] = source["environment"]["b2_initialization"]["reference_sha256"] = file_sha256(pose)
    torch.save(payload, path)
    monkeypatch.setitem(sys.modules, "legged_gym.utils.helpers", NS(class_to_dict=as_dict))
    args = NS(resume=False, resume_path=None)
    own = NS(student_checkpoint=str(path))
    module = load_script("mixed_initialization")
    provenance = module.validate_m1_initialization(cfg, ppo, own, args)
    assert provenance["source_kind"] == "trained_B2"
    assert provenance["source_sha256"] == file_sha256(path)
    assert provenance["source_iteration"] == 17
    assert provenance["critic_initialization"] == "fresh"
    cfg.handoff.mixed_command_hold_s = 3.5  # Parameters remain configurable.
    cfg.handoff.mixed_command_slew_rates = [.3, .4, .5]
    module.validate_m1_initialization(cfg, ppo, own, args)
    cfg.handoff.warmup_vx = .4
    with pytest.raises(ValueError, match="warmup_vx"):
        module.validate_m1_initialization(cfg, ppo, own, args)


def test_real_runner_m1_snapshot_metrics_and_resume(tmp_path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    payload, _, _, path = checkpoint_fixture(tmp_path)
    cfg, ppo = configs()
    train = as_dict(ppo)
    train["runner"].update(num_steps_per_env=8, save_interval=1)
    train["algorithm"].update(min_valid_samples=2, num_learning_epochs=1, num_mini_batches=2)
    train.update(handoff_contract={"mixed_command_hold_s": 2.}, task="carrybox_teacher_warmup_mixed",
                 initialization_provenance={"source_kind": "synthetic_test_B2"})
    synthetic = SyntheticVecEnv(path)
    step = synthetic.step
    def measured_step(action):
        result = step(action)
        synthetic.last_step_snapshot = dict(command=torch.zeros(2, 3), actual=torch.zeros(2, 3),
            target_command=torch.zeros(2, 3), target_mode=torch.full((2,), 3),
            target_sampled=torch.ones(2, dtype=torch.bool), command_hold=torch.zeros(2, dtype=torch.bool),
            command_ramp=torch.ones(2, dtype=torch.bool))
        return result
    synthetic.step = measured_step
    run = TeacherWarmupOnPolicyRunner(synthetic, train, str(path), str(tmp_path / "run"), "cpu")
    run.load_init_policy(str(path))
    run.learn(1)
    run.writer.close()
    events = EventAccumulator(str(tmp_path / "run")).Reload()
    tags = events.Tags()["scalars"]
    for tag in ("Train/mean_reward", "Train/mean_episode_length", "Train/valid_student_samples",
                "Mixed/hold_ratio", "Mixed/ramp_ratio", "Mixed/yaw/ramping/applied_rmse_vx"):
        assert tag in tags
    assert "Mixed/yaw/settled/applied_rmse_vx" not in tags
    saved = tmp_path / "run/model_1.pt"
    data = torch.load(saved, weights_only=False)
    assert data["initialization_provenance"] == train["initialization_provenance"]
    resumed = TeacherWarmupOnPolicyRunner(SyntheticVecEnv(path), train, str(saved), str(tmp_path / "resume"), "cpu")
    resumed.load(str(saved))
    assert resumed.current_learning_iteration == 1 and resumed.alg.optimizer.state
    resumed.contract["handoff"]["mixed_command_hold_s"] = 1.
    with pytest.raises(ValueError, match="Resume contract"):
        resumed.load(str(saved))
    resumed.writer.close()


def test_evaluation_comparison_rejects_different_conditions(tmp_path):
    module_path = ROOT / "experiments/teacher_warmup/compare_mixed.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("compare_mixed", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dirs = [tmp_path / "b2", tmp_path / "m1"]
    manifest = dict(task="carrybox_teacher_warmup_mixed", status="complete", seed=1, episodes=2,
                    num_envs=1, dt=.02, carry_seconds=45., teacher_sha256="teacher", dof_names=NAMES,
                    action_scale=.25, reference_sha256="pose", rewards={}, gait_configuration={},
                    gait_geometry={}, physical_reset_configuration={}, evaluation_command_override={},
                    handoff={"adapter": "rate_limited"})
    for directory in dirs:
        directory.mkdir()
        (directory / "manifest.json").write_text(json.dumps(manifest))
        (directory / "summary.json").write_text(json.dumps(dict(pickup_success=.8)))
        (directory / "mixed_summary.json").write_text(json.dumps({"yaw/settled/applied_rmse_vx": .2}))
    report = module.compare(*dirs)
    assert report["reports"]["summary.json"]["pickup_success"]["delta"] == 0
    (dirs[1] / "manifest.json").write_text(json.dumps(dict(manifest, seed=2)))
    with pytest.raises(ValueError, match="seed"):
        module.compare(*dirs)



def test_inherited_supervisor_starts_first_student_at_08_and_holds(env):
    # Executes the real supervisor with prescribed state tensors; no physics claim.
    env.cfg.handoff.stable_s = .06
    env.cfg.handoff.transition_duration_s = .04
    first_student_step = None
    commands = []
    for physical_step in range(110):
        controlled = env.phase != 0
        env.control_mask = controlled.clone()
        if controlled[0] and first_student_step is None:
            first_student_step = physical_step
        if controlled[0]:
            commands.append(env.carry_policy_commands[0, :3].clone())
        env._update_carry_heading_commands()
        env.check_termination()
        env._advance_mixed_commands()
    assert first_student_step is not None
    assert len(commands) >= 101
    assert torch.stack(commands[:100]).eq(torch.tensor([.8, 0., 0.])).all()
    assert env.mixed_target_samples.eq(1).all()
