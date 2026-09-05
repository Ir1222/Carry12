import math
from unittest import TestCase

import torch

from nforce_test_support import RolloutEnv, command_env, trial
from evaluation.nforce_trial import _assert_no_force_state, _reset_for_trial


def test_complete_trial_has_ten_warmup_steps_and_250_samples():
    samples, summary = trial()
    assert samples[0]["policy_step"] == 20  # 10 confirmation + 10 extra warmup.
    assert samples[-1]["policy_step"] == 269
    assert len(samples) == summary["steady_carry_steps"] == 250
    assert summary["steady_carry_duration_s"] == 5.0
    assert summary["termination_reason"] == "steady_carry_complete"
    assert summary["confirmed_carry_fraction_steady"] == 1.0
    assert summary["force_scheduled"] == 0


def test_zero_warmup_samples_on_confirmation_step():
    samples, _ = trial(warmup=0, duration=0.04)
    assert [row["policy_step"] for row in samples] == [10, 11]


def test_interrupted_warmup_waits_for_new_confirmation():
    samples, _ = trial(RolloutEnv(contact=lambda step: step != 15), duration=0.04)
    # Steps 16..25 re-confirm contact; steps 26..35 supply the extra warmup.
    assert [row["policy_step"] for row in samples] == [35, 36]


def test_confirmation_loss_during_steady_keeps_samples():
    samples, summary = trial(RolloutEnv(contact=lambda step: step != 21), duration=0.06)
    assert [row["policy_step"] for row in samples] == [20, 21, 22]
    assert [row["confirmed_carry"] for row in samples] == [1, 0, 0]
    assert math.isclose(summary["confirmed_carry_fraction_steady"], 1 / 3)
    assert summary["termination_reason"] == "steady_carry_complete"


def test_no_carry_timeout_has_empty_trace_and_nan_metrics():
    samples, summary = trial(RolloutEnv(contact=lambda step: False, terminal_step=40, timeout=True))
    assert not samples
    assert summary["steady_carry_achieved"] == summary["carry_achieved"] == 0
    assert math.isnan(summary["vx_mae"])
    assert summary["timeout"] == 1
    assert summary["termination_reason"] == "timeout"


def test_partial_trial_stops_on_latched_box_failure():
    samples, summary = trial(RolloutEnv(failure_step=23))
    assert len(samples) == 4  # Include the post-step failure sample, as on main.
    assert summary["box_failure"] == 1
    assert summary["termination_reason"] == "dropped_to_ground"
    assert summary["steady_carry_duration_s"] == 0.08


def test_terminal_snapshot_survives_automatic_reset():
    samples, summary = trial(RolloutEnv(terminal_step=23))
    assert len(samples) == 3  # Never sample the auto-reset observation on a done step.
    assert summary["humanoid_failure"] == 1
    assert summary["humanoid_failure_reason"] == "head_low"
    assert summary["carry_achieved"] == summary["final_confirmed_carry"] == 1
    assert math.isclose(summary["final_base_yaw_rad"], 0.7, abs_tol=1e-6)
    assert summary["termination_reason"] == "head_low"


def test_partial_timeout_retains_samples_and_terminal_carry():
    samples, summary = trial(RolloutEnv(terminal_step=23, timeout=True))
    assert len(samples) == 3
    assert summary["timeout"] == 1
    assert summary["humanoid_failure"] == 0
    assert summary["final_confirmed_carry"] == 1


def test_rollout_guard_ends_when_environment_never_reports_done():
    env = RolloutEnv(contact=lambda step: False)
    env.max_episode_length = 30
    samples, summary = trial(env)
    assert not samples
    assert env.step_id == 32
    assert summary["termination_reason"] == "rollout_limit"


def test_reset_retains_policy_command_and_does_not_rebuild_history():
    env = RolloutEnv(command=(0.4, 0.2, 0.1))
    base_reset = env.reset

    def reset_with_feedback():
        obs, privileged = base_reset()
        env.carry_policy_commands[:, 2] = 0.15
        obs[:, -3:] = env.carry_policy_commands
        obs[:, :3] = torch.tensor([1.0, 2.0, 3.0])
        return obs, privileged

    env.reset = reset_with_feedback
    obs = _reset_for_trial(env, 1, env.command, lambda seed: None)
    torch.testing.assert_close(obs[:, -3:], torch.tensor([[0.4, 0.0, 0.15]]))
    torch.testing.assert_close(obs[:, :3], torch.tensor([[1.0, 2.0, 3.0]]))
    assert obs is env.obs


def test_target_reset_and_heading_keep_raw_commands_without_resampling():
    env = command_env((0.4, 0.2, 0.1))

    def unexpected(*args):
        raise AssertionError("Evaluation called a training command sampler")

    env._sample_episode_vx_commands = unexpected
    env._resample_carry_yaw_commands = unexpected
    env._sample_carry_yaw_resample_time = unexpected
    for _ in range(2):
        env._reset_task(torch.tensor([0]))
        env.is_stage_carry[:] = False
        env.carry_heading_initialized[:] = False
        env._update_carry_heading_commands()
        torch.testing.assert_close(env.carry_policy_commands, torch.tensor([[0.4, 0.0, 0.1]]))
        env.is_stage_carry[:] = True
        for _ in range(350):  # Beyond the training yaw-resampling interval.
            env._update_carry_heading_commands()
            torch.testing.assert_close(env.commands[:, :3], torch.tensor([[0.4, 0.2, 0.1]]))
        assert env.carry_heading_error.item() > 0
        assert math.isclose(env.carry_policy_commands[0, 2].item(), 0.4, abs_tol=1e-6)
        assert env.carry_policy_commands[0, 1].item() == 0.0


def test_target_carry_gate_and_evaluator_contact_confirmation():
    env = command_env()
    env.box_states = torch.zeros(1, 13)
    env._box_size = torch.tensor([[0.35, 0.35, 0.30]])
    env.platform_pos = torch.zeros(1, 3)
    env.object2start_dist_xy = torch.zeros(1)
    assert not env._compute_is_stage_carry().item()
    env.box_states[:, 2] = 0.21
    assert env._compute_is_stage_carry().item()  # Lift alone suffices in this branch.
    env.box_states[:, 2] = 0
    env.object2start_dist_xy[:] = 0.6
    assert env._compute_is_stage_carry().item()  # Or displacement alone.
    detector = RolloutEnv()
    detector.contact_forces[:] = 2.0
    for _ in range(9):
        detector._update_confirmed_carry_detector()
        assert not detector.confirmed_carry_buf.item()
    detector._update_confirmed_carry_detector()
    assert detector.confirmed_carry_buf.item()
    detector.box_states[:, 7] = 1.0
    detector._update_confirmed_carry_detector()
    assert not detector.confirmed_carry_buf.item()  # Strict relative-velocity threshold.
    detector.box_states[:, 7] = 0
    detector.box_states[:, 10] = 3.0
    detector._update_confirmed_carry_detector()
    assert not detector.clean_carry_condition_buf.item()
    detector.box_states[:, 10] = 0
    detector.contact_forces[:, 1] = 0
    detector._update_confirmed_carry_detector()
    assert not detector.clean_carry_condition_buf.item()


def test_nominal_config_disables_randomness_and_preserves_training_config():
    from nforce_test_support import carrybox_configs

    cfg = command_env().cfg
    assert cfg.env.num_envs == 1 and cfg.env.test
    assert not cfg.noise.add_noise
    assert not cfg.asset.box.random_size and not cfg.asset.box.random_density
    assert cfg.asset.box.reset_mode == "default"
    assert cfg.commands.resampling_time == 0 and not cfg.commands.resample_carry_commands
    for key in ("randomize_actuation_offset", "randomize_motor_strength", "randomize_payload_mass",
                "randomize_com_displacement", "randomize_link_mass", "randomize_friction",
                "randomize_restitution", "randomize_kp", "randomize_kd",
                "randomize_initial_joint_pos", "disturbance", "delay", "push_robots"):
        assert not getattr(cfg.domain_rand, key, False), key
    training_cfg, _ = carrybox_configs()
    assert training_cfg.commands.resample_carry_commands
    assert training_cfg.env.num_envs == 4096


def test_nforce_rejects_force_state_and_enabled_resampling():
    for field in ("clean_eval_force_tensor", "clean_eval_remaining_physics_steps",
                  "clean_eval_event_count_buf", "disturbance"):
        env = RolloutEnv()
        getattr(env, field).fill_(1)
        with TestCase().assertRaises(AssertionError):
            _assert_no_force_state(env)
    env = RolloutEnv()
    env.cfg.commands.resample_carry_commands = True
    with TestCase().assertRaisesRegex(AssertionError, "resampling"):
        _assert_no_force_state(env)
