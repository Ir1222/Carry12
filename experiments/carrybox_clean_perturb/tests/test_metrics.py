import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))

from evaluation.metrics import sample_policy_metrics, summarize_trial  # noqa: E402


def _condition():
    return SimpleNamespace(
        trial_id="T0001",
        seed=1,
        profile="half_sine",
        direction="+box_x",
        beta=0.1,
        hold_duration_s=1.0,
        pulse_duration_s=0.1,
        ramp_up_s=0.15,
        ramp_down_s=0.15,
    )


class SummaryEnv:
    def __init__(self, force_scheduled=False):
        self.dt = 0.02
        self.box_masses = torch.tensor([10.0])
        self.clean_eval_has_terminal_snapshot = torch.tensor([False])
        self.clean_eval_terminal_confirmed_carry_buf = torch.tensor([False])
        values = {
            "clean_eval_peak_force_N": 9.81 if force_scheduled else 0.0,
            "clean_eval_impulse_Ns": 1.0 if force_scheduled else 0.0,
            "clean_eval_force_duration_s": 0.1 if force_scheduled else 0.0,
            "clean_eval_event_count_buf": 1 if force_scheduled else 0,
            "clean_eval_humanoid_failure_buf": 0,
            "clean_eval_box_failure_buf": 0,
            "clean_eval_timeout_buf": 1,
            "clean_eval_carry_achieved_buf": 1,
            "clean_eval_recovery_success_buf": 0,
            "clean_eval_recovery_time_s": float("nan"),
        }
        for name, value in values.items():
            setattr(self, name, torch.tensor([value]))
        self.confirmed_carry_buf = torch.tensor([True])
        self.clean_eval_humanoid_failure_reason = [""]
        self.clean_eval_box_failure_reason = [""]

    def summary_scalar(self, name, env_id=0):
        return getattr(self, name)[env_id]

    def summary_reason(self, name, env_id=0):
        return getattr(self, name)[env_id]


def _sample(phase, left=1, right=1):
    return {
        "phase": phase,
        "box_displacement_along_force": 0.1,
        "robot_displacement_along_force": 0.2,
        "robot_forward_displacement_from_start": 0.3,
        "robot_lateral_displacement_from_start": 0.4,
        "box_velocity_along_force": 0.5,
        "robot_velocity_along_force": 0.6,
        "left_hand_contact_proxy": left,
        "right_hand_contact_proxy": right,
        "max_hand_box_relative_speed": 0.7,
        "confirmed_carry": 1,
        "vx_tracking_error": 0.1,
        "vy_tracking_error": 0.2,
        "yaw_rate_tracking_error": 0.3,
        "command_vx": 0.6,
        "command_vy": 0.0,
        "command_yaw_rate": 0.4,
        "base_yaw_rad": 0.5,
        "base_yaw_delta_from_start_rad": 0.25,
        "base_yaw_rate_body_rad_s": 0.35,
    }


def test_no_force_response_metrics_are_nan_and_nominal_phase_is_used():
    summary = summarize_trial(
        _condition(),
        "model.pt",
        {"sha1": "abc", "json": "{}"},
        [_sample("wait_carry"), _sample("confirmed_carry")],
        SummaryEnv(force_scheduled=False),
        {"termination_reason": "timeout"},
    )
    assert "physical_failure" not in summary
    assert summary["timeout"] == 1
    assert summary["force_scheduled"] == 0
    assert math.isnan(summary["contact_loss"])
    assert math.isnan(summary["recovery_success"])
    assert summary["vx_tracking_error_nominal_mean"] == 0.1
    assert summary["base_abs_yaw_delta_from_start_nominal_max"] == 0.25


def test_contact_loss_only_uses_force_response_window():
    summary = summarize_trial(
        _condition(),
        "model.pt",
        {"sha1": "abc", "json": "{}"},
        [_sample("wait_carry", left=0), _sample("force"), _sample("post_force", right=0)],
        SummaryEnv(force_scheduled=True),
        {"termination_reason": ""},
    )
    assert summary["force_scheduled"] == 1
    assert summary["contact_loss"] == 1


def test_sample_policy_metrics_uses_fixed_start_yaw():
    env = SimpleNamespace(
        box_states=torch.zeros((1, 13)),
        root_states=torch.zeros((1, 13)),
        rigid_body_states=torch.zeros((1, 3, 13)),
        contact_forces=torch.zeros((1, 3, 3)),
        left_hand_contact_proxy_index=1,
        right_hand_contact_proxy_index=2,
        left_hand_contact_proxy=torch.tensor([True]),
        right_hand_contact_proxy=torch.tensor([True]),
        confirmed_carry_buf=torch.tensor([True]),
        commands=torch.tensor([[0.6, 0.0, 0.4]]),
        base_lin_vel=torch.tensor([[0.5, 0.0, 0.0]]),
        base_ang_vel=torch.tensor([[0.0, 0.0, 0.3]]),
        yaw=torch.tensor([[0.5]]),
    )
    env.root_states[0, 0] = 1.0
    row = sample_policy_metrics(
        env,
        torch.tensor([1.0, 0.0, 0.0]),
        {
            "box_pos": torch.zeros(3),
            "robot_pos": torch.zeros(3),
            "robot_yaw": 0.2,
        },
        "confirmed_carry",
    )
    assert math.isclose(row["base_yaw_delta_from_start_rad"], 0.3, abs_tol=1e-6)
    assert math.isclose(
        row["robot_forward_displacement_from_start"], math.cos(0.2), abs_tol=1e-6
    )
