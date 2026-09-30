import importlib.util
import math
from pathlib import Path
import sys
import types
from unittest.mock import patch

import pytest
import torch

from experiments.carrybox_locomotion_eval.evaluation.command_suite import (
    STEP_COMMANDS, CommandCondition, build_evaluation_suite,
)
from experiments.carrybox_locomotion_eval.evaluation.metrics import (
    aggregate_protocols, step_response_metrics, write_csv,
)
from experiments.carrybox_locomotion_eval.evaluation.readiness import (
    ReadinessConfig, ReadinessGate,
)


def config():
    return ReadinessConfig(.04, .10, 1., .03, .01, .35,
                           (.35, .45, .45), .15, 1., 70.)


def evidence(**changes):
    value = {"physical_state_finite": 1, "left_hand_force_n": 2.,
             "right_hand_force_n": 2., "left_hand_side_error_m": .01,
             "right_hand_side_error_m": .01, "left_hand_face_excess_m": 0.,
             "right_hand_face_excess_m": 0.,
             "left_hand_tangential_slip_mps": .1,
             "right_hand_tangential_slip_mps": .1,
             "box_relative_vx_mps": .1, "box_relative_vy_mps": .1,
             "box_relative_vz_mps": .1, "box_bottom_m": .4,
             "robot_box_distance": .3, "box_tilt_deg": 5.}
    value.update(changes)
    return value


def test_raw_bilateral_continuity_and_geometry_are_required():
    gate = ReadinessGate(config(), .02)
    assert not gate.update(evidence(left_hand_force_n=0.))
    assert not gate.update(evidence(right_hand_force_n=0.))
    assert gate.streak == 0
    assert not gate.update(evidence(left_hand_face_excess_m=.001))
    assert not gate.update(evidence(left_hand_tangential_slip_mps=.36))
    assert not gate.update(evidence(box_relative_vz_mps=.46))
    assert not gate.update(evidence())
    assert gate.update(evidence())
    assert gate.longest_stable_s == pytest.approx(.04)


def test_nonfinite_and_timeout_never_release_gate():
    gate = ReadinessGate(config(), .02)
    for _ in range(5):
        assert not gate.update(evidence(box_relative_vx_mps=float("nan")))
    assert gate.timed_out
    assert gate.last_checks["finite"] is False


def test_context_grid_and_sequences_are_unique_and_bounded():
    conditions = build_evaluation_suite(
        seeds=(1, 2, 3), carry_motion_ids=(0, 1, 2),
        carry_phases=(.25, .5, .75), protocol="both")
    constant = [c for c in conditions if c.protocol == "constant"]
    steps = [c for c in conditions if c.protocol == "step"]
    assert len(constant) == 27 * 52
    assert len(steps) == 27 * 4
    assert len({c.trial_id for c in conditions}) == len(conditions)
    assert all(len(c.commands) == 5 for c in steps)
    assert all(c.commands[-1] == (0., 0., 0.) for c in steps)
    assert steps[0].commands == STEP_COMMANDS["vx"]
    with pytest.raises(ValueError):
        build_evaluation_suite(seeds=(1,), carry_motion_ids=(0,),
                               carry_phases=(.5,), modes=("stand",), protocol="step")


def test_step_transient_reversal_and_censored_settling():
    def series(values):
        return [{"command_time_s": (i+1)*.02,
                 "actual_vx_training_frame": value,
                 "actual_vy_training_frame": 0.,
                 "actual_yaw_rate_training_frame": 0.}
                for i, value in enumerate(values)]
    forward = step_response_metrics(series([.7, 1.1, 1., 1., 1.]),
                                    (0., 0., 0.), (1., 0., 0.),
                                    policy_dt=.02, completed=True,
                                    onset_actual=(0., 0., 0.), hold_s=.04)
    assert forward["vx_overshoot"] == pytest.approx(.1)
    assert forward["vx_settling_time_s"] == pytest.approx(.06)
    reverse = step_response_metrics(series([-.2, -.6, -.5, -.5]),
                                    (.5, 0., 0.), (-.5, 0., 0.),
                                    policy_dt=.02, completed=True,
                                    onset_actual=(.5, 0., 0.), hold_s=.04)
    assert reverse["vx_overshoot"] == pytest.approx(.1)
    assert reverse["vx_overshoot_signed"] == pytest.approx(-.1)
    censored = step_response_metrics(series([.7, 1.]),
                                     (0., 0., 0.), (1., 0., 0.),
                                     policy_dt=.02, completed=False,
                                     onset_actual=(0., 0., 0.), hold_s=.04)
    assert censored["overshoot_scope"] == "observed_prefix"
    assert math.isnan(censored["vx_settling_time_s"])


def test_attempt_denominator_not_multiplied_by_step_segments():
    attempts = [{"trial_id": "A", "protocol": "step", "mode": "vx", "initialization_success": 1,
                 "trial_completed": 1, "measurement_steps": 10, "settle_duration_s": .2},
                {"trial_id": "B", "protocol": "step", "mode": "vx", "initialization_success": 0,
                 "trial_completed": 0, "measurement_steps": 0, "settle_duration_s": 2.}]
    segments = [{"trial_id": "A", "protocol": "step", "sequence_family": "vx",
                 "segment_status": "completed", "measurement_steps": 2,
                 "measurement_duration_s": .04} for _ in range(5)]
    segments.extend({"trial_id": "B", "protocol": "step", "sequence_family": "vx",
                     "segment_status": "not_reached", "measurement_steps": 0,
                     "measurement_duration_s": 0.} for _ in range(5))
    row = aggregate_protocols(attempts, segments)[0]
    assert row["number_of_trials"] == 2
    assert row["number_of_segments"] == 10
    assert row["initialization_success_rate"] == .5
    assert row["conditional_completion_rate"] == 1.
    assert row["end_to_end_completion_rate"] == .5


def test_mixed_corner_breakdown_and_variable_failure_schema(tmp_path):
    attempts = [
        {"trial_id": "C", "protocol": "constant", "mode": "mixed",
         "case_type": "corner", "initialization_success": 1,
         "trial_completed": 1, "measurement_steps": 2,
         "settle_duration_s": .2, "survival_duration_s": 3.2,
         "termination_reason": "completed"},
        {"trial_id": "I", "protocol": "constant", "mode": "mixed",
         "case_type": "interior", "initialization_success": 0,
         "trial_completed": 0, "measurement_steps": 0,
         "settle_duration_s": 2., "survival_duration_s": 2.,
         "termination_reason": "initialization_timeout"},
    ]
    segments = [
        {"trial_id": "C", "protocol": "constant", "sequence_family": "mixed",
         "segment_status": "completed", "measurement_steps": 2,
         "measurement_duration_s": .04, "final_confirmed_carry": 1},
        {"trial_id": "I", "protocol": "constant", "sequence_family": "mixed",
         "segment_status": "not_reached", "measurement_steps": 0,
         "measurement_duration_s": 0., "final_confirmed_carry": 0},
    ]
    grouped = aggregate_protocols(attempts, segments)
    by_case = {r["case_type"]: r for r in grouped if r["mode"] == "mixed"}
    assert by_case["all"]["number_of_trials"] == 2
    assert by_case["corner"]["end_to_end_completion_rate"] == 1.
    assert by_case["interior"]["end_to_end_completion_rate"] == 0.
    assert by_case["all"]["survival_duration_s_mean"] == pytest.approx(2.6)
    output = tmp_path / "summary.csv"
    write_csv(str(output), [{"trial_id": "I", "init": 0},
                            {"trial_id": "C", "init": 1, "vx_mae": .1}])
    assert "vx_mae" in output.read_text()


def test_command_switch_updates_only_latest_frame_without_gym_import():
    root = Path(__file__).resolve().parents[1]
    base = type("FakeBase", (), {})
    modules = {"legged_gym": types.ModuleType("legged_gym"),
               "legged_gym.envs": types.ModuleType("legged_gym.envs"),
               "legged_gym.envs.g1": types.ModuleType("legged_gym.envs.g1")}
    modules["legged_gym.envs.g1"].carrybox_locomotion = types.SimpleNamespace(LeggedRobot=base)
    spec = importlib.util.spec_from_file_location(
        "_carry_eval_test", root / "envs" / "carrybox_locomotion_eval_env.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    env = object.__new__(module.CarryBoxLocomotionEvalEnv)
    env.num_envs, env.num_task_obs, env.device = 1, 3, "cpu"
    env.commands = torch.zeros(1, 3)
    env.carry_policy_commands = torch.zeros(1, 3)
    env._eval_requested_command = torch.zeros(1, 3)
    env._eval_command_active = False
    env.obs_buf = torch.arange(18., dtype=torch.float).reshape(1, 18)
    env.privileged_obs_buf = torch.arange(9., dtype=torch.float).reshape(1, 9)
    history = env.obs_buf[:, :-3].clone()
    env.set_evaluation_command((.4, -.2, .3))
    assert torch.equal(env.obs_buf[:, :-3], history)
    assert torch.allclose(env.obs_buf[:, -3:], torch.tensor([[.4, -.2, .3]]))
    assert torch.allclose(env.privileged_obs_buf[:, -3:], env.obs_buf[:, -3:])
    env.assert_evaluation_command(env.obs_buf)


def test_rollout_waits_for_contact_and_counts_reset_failure_once():
    from experiments.carrybox_locomotion_eval.evaluation import trial
    from .test_metrics import sample as metric_sample

    rewards = types.SimpleNamespace(
        hand_contact_threshold=1., carry_hand_side_tolerance=.03,
        carry_hand_face_margin=.01, carry_hand_slip_tolerance=.35,
        carry_box_relative_velocity_tolerance=(.35, .45, .45),
        box_drop_height=.15, robot_box_max_distance=1.,
        box_tilt_termination_deg=70.)

    class FakeEnv:
        dt, num_envs, num_actions, device = .02, 1, 2, "cpu"

        def __init__(self):
            self.cfg = types.SimpleNamespace(rewards=rewards)
            self.obs = torch.zeros(1, 3)
            self.disturbance = torch.zeros(1, 3)
            self.base_lin_vel_yaw = torch.zeros(1, 2)
            self.base_yaw_rate_world = torch.zeros(1)
            self.eval_last_termination_reason = [""]
            self.commands_seen = []
            self.reset_count = 0
            self.fail_step = None

        def set_evaluation_reset(self, motion, phase):
            self.motion, self.phase = motion, phase

        def set_evaluation_command(self, command):
            self.current = tuple(command)
            self.obs[0] = torch.tensor(command)

        def get_observations(self):
            return self.obs

        def clear_evaluation_outcome(self):
            self.eval_last_termination_reason = [""]

        def assert_evaluation_command(self, obs):
            assert torch.allclose(obs[0], torch.tensor(self.current))

        def reset(self):
            self.reset_count += 1
            self.step_count = 0
            self.obs.zero_()
            return self.obs, None

        def step(self, actions):
            self.commands_seen.append(self.current)
            self.step_count += 1
            self.base_lin_vel_yaw[0, 0] = self.current[0]
            failed = ((self.reset_count == 1 and self.step_count == 1) or
                      self.step_count == self.fail_step)
            if failed:
                self.eval_last_termination_reason = ["box_drop"]
            return (self.obs, None, None, torch.tensor([failed]),
                    None, None, None, None)

    def fake_sample(env, condition, **kwargs):
        row = metric_sample(float(env.base_lin_vel_yaw[0, 0]))
        row.update(evidence())
        row.update({"command_time_s": kwargs["command_time_s"],
                    "phase": kwargs["phase"],
                    "actual_vx_training_frame": float(env.base_lin_vel_yaw[0, 0])})
        return row

    env = FakeEnv()
    observed_inputs = []
    def policy(obs):
        observed_inputs.append(tuple(float(v) for v in obs[0]))
        return torch.zeros(1, 2)

    conditions = [CommandCondition("A", "vx", .4, 0., 0., 1, 0, .5),
                  CommandCondition("B", "vx", .4, 0., 0., 2, 1, .75)]
    with patch.object(trial, "_carry_sample", return_value=({}, None, None)), \
         patch.object(trial, "_sample", side_effect=fake_sample), \
         patch.object(trial, "assert_observation_compatibility"):
        attempts, segments = trial.run_suite(
            env, policy, conditions, warmup_s=.02, duration_s=.04,
            stable_hold_s=.04, settle_timeout_s=.10, seed_fn=lambda seed: None)
    assert attempts[0]["initialization_success"] == 0
    assert attempts[0]["termination_phase"] == "SETTLE"
    assert attempts[0]["termination_reason"] == "box_drop"
    assert math.isnan(segments[0]["vx_mae"])
    assert attempts[1]["initialization_success"] == 1
    assert attempts[1]["trial_completed"] == 1
    assert (env.motion, env.phase) == (1, .75)
    assert env.commands_seen == [(0., 0., 0.)] * 3 + [(.4, 0., 0.)] * 3
    assert observed_inputs[:3] == [(0., 0., 0.)] * 3
    assert len(observed_inputs[3:]) == 3
    assert all(values == pytest.approx((.4, 0., 0.)) for values in observed_inputs[3:])

    env.fail_step = 6  # Second segment's first physics step.
    sequence = CommandCondition("C", "vx", .4, 0., 0., 3, 2, .25,
                                case_type="sequence", protocol="step",
                                segment_commands=STEP_COMMANDS["vx"])
    with patch.object(trial, "_carry_sample", return_value=({}, None, None)), \
         patch.object(trial, "_sample", side_effect=fake_sample), \
         patch.object(trial, "assert_observation_compatibility"):
        trace, result, rows = trial.run_trial(
            env, policy, sequence, warmup_s=.02, duration_s=.04,
            stable_hold_s=.04, settle_timeout_s=.10, seed_fn=lambda seed: None)
    assert result["initialization_success"] == 1
    assert result["trial_completed"] == 0
    assert result["termination_phase"] == "WARMUP"
    assert [r["segment_status"] for r in rows] == [
        "completed", "terminated", "not_reached", "not_reached", "not_reached"]
    assert len([row for row in trace if row["phase"] == "SETTLE"]) == 2
    assert any(row["phase"] == "MEASURE" for row in trace)


def test_real_sampling_shape_and_readiness_schema_with_identity_quaternions():
    from experiments.carrybox_locomotion_eval.evaluation import trial
    from experiments.carrybox_locomotion_eval.evaluation.metrics import (
        LOWER_BODY_TRACE_METRICS, PRESERVATION_METRICS,
    )
    gym_utils = types.ModuleType("isaacgym.torch_utils")
    gym_utils.quat_rotate_inverse = lambda quat, vector: vector
    heading_utils = types.ModuleType("legged_gym.utils.torch_utils")
    heading_utils.calc_heading_quat = lambda quat: quat
    fake_modules = {"isaacgym": types.ModuleType("isaacgym"),
                    "isaacgym.torch_utils": gym_utils,
                    "legged_gym": types.ModuleType("legged_gym"),
                    "legged_gym.utils": types.ModuleType("legged_gym.utils"),
                    "legged_gym.utils.torch_utils": heading_utils}
    c = types.SimpleNamespace(
        carry_arm_range_lower=[-1.] * 3, carry_arm_range_upper=[1.] * 3,
        carry_box_relative_position_lower=[0., -1., -1.],
        carry_box_relative_position_upper=[1., 1., 1.],
        carry_hand_face_margin=.01, hand_contact_threshold=1.,
        carry_hand_side_tolerance=.03, carry_hand_slip_tolerance=.35,
        carry_box_relative_velocity_tolerance=[.35, .45, .45],
        box_drop_height=.15, robot_box_max_distance=1.,
        box_tilt_termination_deg=70.)
    env = types.SimpleNamespace(cfg=types.SimpleNamespace(rewards=c), dt=.02)
    env.rigid_body_states = torch.zeros(1, 6, 13)
    env.rigid_body_states[..., 6] = 1.
    env.box_states = torch.zeros(1, 13)
    env.box_states[0, :3] = torch.tensor([.3, 0., .5])
    env.box_states[0, 6] = 1.
    env.rigid_body_states[0, 1, :3] = torch.tensor([.3, .15, .5])
    env.rigid_body_states[0, 2, :3] = torch.tensor([.3, -.15, .5])
    env._box_size = torch.tensor([[.3, .3, .3]])
    env.carry_torso_index = env.upper_body_index = 0
    env.carry_palm_indices = torch.tensor([1, 2])
    env.hand_colli_indices = torch.tensor([3, 4])
    env.contact_forces = torch.zeros(1, 6, 3)
    env.contact_forces[0, 3:5, 2] = 2.
    env.dof_pos = torch.zeros(1, 3)
    env.dof_vel = torch.zeros(1, 3)
    env.carry_arm_indices = env.carry_waist_indices = torch.tensor([0, 1, 2])
    env.carry_hip_error = torch.zeros(1, 4)
    env.carry_foot_heading_error = torch.zeros(1, 2)
    env.carry_feet_width = env.carry_feet_width_violation = torch.zeros(1)
    env.carry_knee_width = env.carry_knee_width_violation = torch.zeros(1)
    env.carry_foot_heading_excess = torch.zeros(1, 2)
    env.carry_torso_pelvis_rpy = env.carry_waist_error = torch.zeros(1, 3)
    env.carry_torso_pelvis_rotvec_error = torch.zeros(1, 3)
    env.base_lin_vel_yaw = torch.zeros(1, 3)
    env.base_yaw_rate_world = torch.zeros(1)
    env.carry_policy_commands = torch.tensor([[.4, .2, .3]])
    env.hand_contact_filt = torch.ones(1, 2, dtype=torch.bool)
    env.projected_gravity_box = torch.tensor([[0., 0., -1.]])
    env.robot2object_dist = torch.tensor([.3])
    env.root_states = torch.zeros(1, 13)
    env.root_states[0, 6] = 1.
    env.yaw = env.carry_heading_ref = env.carry_heading_error = torch.zeros(1)
    env.base_lin_vel = env.base_ang_vel = torch.zeros(1, 3)
    env.feet_indices = torch.tensor([5])
    env.feet_vel = torch.zeros(1, 1, 3)
    env.torques = torch.zeros(1, 3)
    condition = CommandCondition("S", "mixed", .4, .2, .3, 1, 0, .5)
    with patch.dict(sys.modules, fake_modules):
        _, hand, box = trial._carry_sample(env)
        carry, _, _ = trial._carry_sample(env, hand, box)
        assert set(PRESERVATION_METRICS).issubset(carry)
        assert set(LOWER_BODY_TRACE_METRICS).issubset(carry)
        value = trial._sample(
            env, condition, policy_step=1, time_s=.02,
            actions=torch.zeros(1, 3), previous_actions=torch.zeros(1, 3),
            carry_sample=carry, phase="SETTLE", segment_index=-1,
            command_time_s=float("nan"), command_onset_s=float("nan"),
            readiness_config=config())
    assert tuple(value) != ()
    assert set(value) == set(trial.TRACE_FIELDS)
    assert value["ready_all"] == 1
    assert value["normalized_vector_error"] == pytest.approx(
        math.sqrt((.4 / 1.2)**2 + (.2 / .5)**2 + (.3 / .7)**2))
