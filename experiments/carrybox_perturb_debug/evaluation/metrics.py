"""Metric helpers for carrybox perturbation evaluation."""

import math

import torch


def vector_to_tuple(tensor):
    return tuple(float(value) for value in tensor.detach().cpu().tolist())


def sample_policy_metrics(env, direction_world, force_on_box_start, env_id=0):
    direction = direction_world
    box_delta = env.box_states[env_id, 0:3] - force_on_box_start["box_pos"]
    robot_delta = env.root_states[env_id, 0:3] - force_on_box_start["robot_pos"]
    box_vel = env.box_states[env_id, 7:10]
    robot_vel = env.root_states[env_id, 7:10]
    left_index = int(env.left_hand_net_contact_force_index)
    right_index = int(env.right_hand_net_contact_force_index)
    box_vel_world = env.box_states[env_id, 7:10]
    left_rel = torch.linalg.vector_norm(
        env.rigid_body_states[env_id, left_index, 7:10] - box_vel_world
    )
    right_rel = torch.linalg.vector_norm(
        env.rigid_body_states[env_id, right_index, 7:10] - box_vel_world
    )
    threshold = float(env.cfg.carry_phase.contact_force_threshold)
    left_norm = torch.linalg.vector_norm(env.contact_forces[env_id, left_index, :])
    right_norm = torch.linalg.vector_norm(env.contact_forces[env_id, right_index, :])
    return {
        "box_displacement_along_force": float(torch.dot(box_delta, direction).item()),
        "robot_displacement_along_force": float(torch.dot(robot_delta, direction).item()),
        "box_velocity_along_force": float(torch.dot(box_vel, direction).item()),
        "robot_velocity_along_force": float(torch.dot(robot_vel, direction).item()),
        "left_hand_box_relative_speed": float(left_rel.item()),
        "right_hand_box_relative_speed": float(right_rel.item()),
        "max_hand_box_relative_speed": float(max(left_rel.item(), right_rel.item())),
        "left_contact": int(left_norm.item() > threshold),
        "right_contact": int(right_norm.item() > threshold),
        "confirmed_carry": int(env.confirmed_carry_buf[env_id].item()),
    }


def summarize_trial(condition, checkpoint, signature, trace_rows, samples, env, failure):
    final_sample = samples[-1] if samples else {}
    max_rel_speed = max(
        (row.get("max_hand_box_relative_speed", 0.0) for row in samples),
        default=float("nan"),
    )
    contact_loss = any(
        row.get("left_contact", 0) == 0 or row.get("right_contact", 0) == 0
        for row in samples
    )
    impulse = float("nan")
    if trace_rows:
        impulse = float(trace_rows[-1].get("force_impulse_Ns", float("nan")))
    elif hasattr(env, "box_perturb_eval_impulse_Ns"):
        impulse = float(env.box_perturb_eval_impulse_Ns[0].item())
    peak_force = float(env.box_perturb_peak_force_N[0].item())
    if math.isnan(peak_force) or peak_force == 0.0:
        peak_force = float(condition.beta) * float(env.box_masses[0].item()) * 9.81

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
        "peak_force_N": peak_force,
        "impulse_Ns": impulse,
        "physical_failure": int(bool(failure["physical_failure"])),
        "termination_reason": failure["termination_reason"],
        "contact_loss": int(contact_loss),
        "max_hand_box_relative_speed": max_rel_speed,
        "box_displacement_along_force": final_sample.get(
            "box_displacement_along_force", float("nan")
        ),
        "robot_displacement_along_force": final_sample.get(
            "robot_displacement_along_force", float("nan")
        ),
        "final_confirmed_carry": final_sample.get("confirmed_carry", 0),
        "task_success": int(env.success_buf[0].item()) if hasattr(env, "success_buf") else 0,
        "object2goal_distance_final": float(env.object2goal_dist_xyz[0].item())
        if hasattr(env, "object2goal_dist_xyz")
        else float("nan"),
        "initial_state_signature_sha1": signature["sha1"],
        "initial_state_signature_json": signature["json"],
    }
