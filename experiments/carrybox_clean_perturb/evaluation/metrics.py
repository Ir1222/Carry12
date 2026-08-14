"""Metric helpers for clean CarryBox perturbation evaluation."""

import math

import torch


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

    return {
        "phase": phase,
        "box_displacement_along_force": float(torch.dot(box_delta, direction).item()),
        "robot_displacement_along_force": float(torch.dot(robot_delta, direction).item()),
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
    }


def _mean(values):
    values = [float(value) for value in values if not math.isnan(float(value))]
    if not values:
        return float("nan")
    return sum(values) / len(values)


def _phase_mean(samples, phase, key):
    return _mean(row[key] for row in samples if row.get("phase") == phase)


def summarize_trial(condition, checkpoint, signature, samples, env, failure):
    final_sample = samples[-1] if samples else {}
    force_samples = [row for row in samples if row.get("phase") == "force"]
    post_samples = [row for row in samples if row.get("phase") == "post_force"]
    nominal_samples = [
        row for row in samples if row.get("phase") in ("wait_carry", "pre_force")
    ]
    contact_loss = any(
        row.get("left_hand_contact_proxy", 0) == 0
        or row.get("right_hand_contact_proxy", 0) == 0
        for row in force_samples + post_samples
    )
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
        "physical_failure": int(bool(failure["physical_failure"])),
        "termination_reason": failure["termination_reason"],
        "left_hand_contact_proxy": final_sample.get("left_hand_contact_proxy", 0),
        "right_hand_contact_proxy": final_sample.get("right_hand_contact_proxy", 0),
        "contact_loss": int(contact_loss),
        "max_hand_box_relative_speed": max_rel_speed,
        "box_displacement_along_force": final_sample.get(
            "box_displacement_along_force", float("nan")
        ),
        "robot_displacement_along_force": final_sample.get(
            "robot_displacement_along_force", float("nan")
        ),
        "box_velocity_along_force": final_sample.get(
            "box_velocity_along_force", float("nan")
        ),
        "robot_velocity_along_force": final_sample.get(
            "robot_velocity_along_force", float("nan")
        ),
        "final_confirmed_carry": final_confirmed_carry,
        "recovery_success": bool_scalar("clean_eval_recovery_success_buf"),
        "recovery_time": scalar("clean_eval_recovery_time_s"),
        "vx_tracking_error_nominal_mean": _mean(
            row["vx_tracking_error"] for row in nominal_samples
        ),
        "vy_tracking_error_nominal_mean": _mean(
            row["vy_tracking_error"] for row in nominal_samples
        ),
        "yaw_rate_tracking_error_nominal_mean": _mean(
            row["yaw_rate_tracking_error"] for row in nominal_samples
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
