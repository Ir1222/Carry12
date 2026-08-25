"""Metric helpers for clean CarryBox perturbation evaluation."""

import math

import torch


def _yaw_scalar(env, env_id):
    if not hasattr(env, "yaw"):
        return float("nan")
    return float(env.yaw[env_id].reshape(-1)[0].item())


def _wrap_angle(value):
    return math.atan2(math.sin(float(value)), math.cos(float(value)))


def sample_policy_metrics(env, direction_world, force_start, phase, env_id=0):
    direction = direction_world
    box_delta = env.box_states[env_id, 0:3] - force_start["box_pos"]
    robot_delta = env.root_states[env_id, 0:3] - force_start["robot_pos"]
    box_vel = env.box_states[env_id, 7:10]
    robot_vel = env.root_states[env_id, 7:10]
    left_index = int(env.left_hand_contact_proxy_index)
    right_index = int(env.right_hand_contact_proxy_index)
    left_norm = torch.linalg.vector_norm(env.contact_forces[env_id, left_index, :])
    right_norm = torch.linalg.vector_norm(env.contact_forces[env_id, right_index, :])
    left_rel = torch.linalg.vector_norm(
        env.rigid_body_states[env_id, left_index, 7:10] - box_vel
    )
    right_rel = torch.linalg.vector_norm(
        env.rigid_body_states[env_id, right_index, 7:10] - box_vel
    )
    yaw = _yaw_scalar(env, env_id)
    start_yaw = force_start.get("robot_yaw", yaw)
    forward_xy = torch.tensor(
        [math.cos(float(start_yaw)), math.sin(float(start_yaw))],
        dtype=robot_delta.dtype,
        device=robot_delta.device,
    )
    left_xy = torch.tensor(
        [-math.sin(float(start_yaw)), math.cos(float(start_yaw))],
        dtype=robot_delta.dtype,
        device=robot_delta.device,
    )

    return {
        "phase": phase,
        "box_displacement_along_force": float(torch.dot(box_delta, direction).item()),
        "robot_displacement_along_force": float(torch.dot(robot_delta, direction).item()),
        "robot_forward_displacement_from_start": float(
            torch.dot(robot_delta[0:2], forward_xy).item()
        ),
        "robot_lateral_displacement_from_start": float(
            torch.dot(robot_delta[0:2], left_xy).item()
        ),
        "box_velocity_along_force": float(torch.dot(box_vel, direction).item()),
        "robot_velocity_along_force": float(torch.dot(robot_vel, direction).item()),
        "left_hand_contact_proxy": int(env.left_hand_contact_proxy[env_id].item()),
        "right_hand_contact_proxy": int(env.right_hand_contact_proxy[env_id].item()),
        "left_hand_contact_proxy_norm_N": float(left_norm.item()),
        "right_hand_contact_proxy_norm_N": float(right_norm.item()),
        "max_hand_box_relative_speed": float(max(left_rel.item(), right_rel.item())),
        "confirmed_carry": int(env.confirmed_carry_buf[env_id].item()),
        "vx_tracking_error": float(
            abs(env.commands[env_id, 0] - env.base_lin_vel[env_id, 0]).item()
        ),
        "vy_tracking_error": float(
            abs(env.commands[env_id, 1] - env.base_lin_vel[env_id, 1]).item()
        ),
        "yaw_rate_tracking_error": float(
            abs(env.commands[env_id, 2] - env.base_ang_vel[env_id, 2]).item()
        ),
        "command_vx": float(env.commands[env_id, 0].item()),
        "command_vy": float(env.commands[env_id, 1].item()),
        "command_yaw_rate": float(env.commands[env_id, 2].item()),
        "base_yaw_rad": yaw,
        "base_yaw_delta_from_start_rad": _wrap_angle(yaw - float(start_yaw)),
        "base_yaw_rate_body_rad_s": float(env.base_ang_vel[env_id, 2].item()),
    }


def _mean(values):
    values = [float(value) for value in values if not math.isnan(float(value))]
    if not values:
        return float("nan")
    return sum(values) / len(values)


def _phase_mean(samples, phase, key):
    return _mean(row[key] for row in samples if row.get("phase") == phase)


def summarize_trial(condition, checkpoint, signature, samples, env, termination):
    final_sample = samples[-1] if samples else {}
    force_samples = [row for row in samples if row.get("phase") == "force"]
    post_samples = [row for row in samples if row.get("phase") == "post_force"]
    nominal_samples = [
        row
        for row in samples
        if row.get("phase")
        in ("wait_carry", "pre_force", "confirmed_carry", "nominal_locomotion")
    ]
    yaw_drift_samples = [
        row
        for row in samples
        if row.get("phase") in ("nominal_locomotion", "confirmed_carry", "pre_force")
    ]
    if not yaw_drift_samples:
        yaw_drift_samples = nominal_samples
    response_samples = force_samples + post_samples
    max_rel_speed = max(
        (row.get("max_hand_box_relative_speed", 0.0) for row in samples),
        default=float("nan"),
    )

    def scalar(name):
        value = env.summary_scalar(name, env_id=0) if hasattr(env, "summary_scalar") else getattr(env, name)[0]
        return float(value.item())

    def bool_scalar(name):
        value = env.summary_scalar(name, env_id=0) if hasattr(env, "summary_scalar") else getattr(env, name)[0]
        return int(bool(value.item()))

    force_scheduled = int(scalar("clean_eval_event_count_buf") > 0)
    contact_loss = (
        int(
            any(
                row.get("left_hand_contact_proxy", 0) == 0
                or row.get("right_hand_contact_proxy", 0) == 0
                for row in response_samples
            )
        )
        if force_scheduled and response_samples
        else float("nan")
    )
    humanoid_failure_reason = env.summary_reason(
        "clean_eval_humanoid_failure_reason", env_id=0
    )
    box_failure_reason = env.summary_reason(
        "clean_eval_box_failure_reason", env_id=0
    )

    if bool(env.clean_eval_has_terminal_snapshot[0].item()):
        final_confirmed_carry = int(env.clean_eval_terminal_confirmed_carry_buf[0].item())
    else:
        final_confirmed_carry = int(env.confirmed_carry_buf[0].item())

    return {
        "trial_id": condition.trial_id,
        "checkpoint": checkpoint,
        "seed": condition.seed,
        "profile": condition.profile,
        "direction": condition.direction,
        "beta": condition.beta,
        "hold_duration": condition.hold_duration_s,
        "pulse_duration": condition.pulse_duration_s,
        "ramp_up_s": condition.ramp_up_s,
        "ramp_down_s": condition.ramp_down_s,
        "box_mass": float(env.box_masses[0].item()),
        "peak_force_N": scalar("clean_eval_peak_force_N"),
        "impulse_Ns": scalar("clean_eval_impulse_Ns"),
        "force_duration": scalar("clean_eval_force_duration_s"),
        "humanoid_failure": bool_scalar("clean_eval_humanoid_failure_buf"),
        "humanoid_failure_reason": humanoid_failure_reason,
        "box_failure": bool_scalar("clean_eval_box_failure_buf"),
        "box_failure_reason": box_failure_reason,
        "timeout": bool_scalar("clean_eval_timeout_buf"),
        "carry_achieved": bool_scalar("clean_eval_carry_achieved_buf"),
        "force_scheduled": force_scheduled,
        "termination_reason": termination["termination_reason"],
        "left_hand_contact_proxy": final_sample.get("left_hand_contact_proxy", 0),
        "right_hand_contact_proxy": final_sample.get("right_hand_contact_proxy", 0),
        "contact_loss": contact_loss,
        "max_hand_box_relative_speed": max_rel_speed,
        "box_displacement_along_force": final_sample.get(
            "box_displacement_along_force", float("nan")
        ),
        "robot_displacement_along_force": final_sample.get(
            "robot_displacement_along_force", float("nan")
        ),
        "robot_forward_displacement_from_start": final_sample.get(
            "robot_forward_displacement_from_start", float("nan")
        ),
        "robot_lateral_displacement_from_start": final_sample.get(
            "robot_lateral_displacement_from_start", float("nan")
        ),
        "box_velocity_along_force": final_sample.get(
            "box_velocity_along_force", float("nan")
        ),
        "robot_velocity_along_force": final_sample.get(
            "robot_velocity_along_force", float("nan")
        ),
        "final_confirmed_carry": final_confirmed_carry,
        "recovery_success": (
            bool_scalar("clean_eval_recovery_success_buf")
            if force_scheduled
            else float("nan")
        ),
        "recovery_time": (
            scalar("clean_eval_recovery_time_s")
            if force_scheduled
            else float("nan")
        ),
        "vx_tracking_error_nominal_mean": _mean(
            row["vx_tracking_error"] for row in nominal_samples
        ),
        "vy_tracking_error_nominal_mean": _mean(
            row["vy_tracking_error"] for row in nominal_samples
        ),
        "yaw_rate_tracking_error_nominal_mean": _mean(
            row["yaw_rate_tracking_error"] for row in nominal_samples
        ),
        "command_vx": final_sample.get("command_vx", float("nan")),
        "command_vy": final_sample.get("command_vy", float("nan")),
        "command_yaw_rate": final_sample.get("command_yaw_rate", float("nan")),
        "final_base_yaw_rad": final_sample.get("base_yaw_rad", float("nan")),
        "final_base_yaw_delta_from_start_rad": final_sample.get(
            "base_yaw_delta_from_start_rad", float("nan")
        ),
        "base_yaw_rate_body_nominal_mean": _mean(
            row["base_yaw_rate_body_rad_s"] for row in yaw_drift_samples
        ),
        "base_abs_yaw_rate_body_nominal_mean": _mean(
            abs(row["base_yaw_rate_body_rad_s"]) for row in yaw_drift_samples
        ),
        "base_yaw_rate_integral_nominal_rad": sum(
            row["base_yaw_rate_body_rad_s"] * float(env.dt)
            for row in yaw_drift_samples
        ),
        "base_abs_yaw_delta_from_start_nominal_max": max(
            (abs(row["base_yaw_delta_from_start_rad"]) for row in yaw_drift_samples),
            default=float("nan"),
        ),
        "vx_tracking_error_post_force_mean": _phase_mean(
            post_samples, "post_force", "vx_tracking_error"
        ),
        "vy_tracking_error_post_force_mean": _phase_mean(
            post_samples, "post_force", "vy_tracking_error"
        ),
        "yaw_rate_tracking_error_post_force_mean": _phase_mean(
            post_samples, "post_force", "yaw_rate_tracking_error"
        ),
        "initial_state_signature_sha1": signature["sha1"],
        "initial_state_signature_json": signature["json"],
    }
