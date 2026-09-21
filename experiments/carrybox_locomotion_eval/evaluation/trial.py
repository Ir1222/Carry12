"""RESET_CARRY_REFERENCE -> WARMUP -> MEASURE -> END rollout."""

import math

import torch
from isaacgym.torch_utils import quat_rotate_inverse

from legged_gym.utils.torch_utils import calc_heading_quat

from .inference import assert_observation_compatibility
from .metrics import (
    LOWER_BODY_TRACE_METRICS, PRESERVATION_METRICS, TRACKING_SCALES,
    summarize_trial,
)


TRACE_FIELDS = (
    "time_s", "policy_step", "mode",
    "command_vx", "command_vy", "command_yaw_rate",
    "actual_vx_training_frame", "actual_vy_training_frame",
    "actual_yaw_rate_training_frame",
    "vx_error", "vy_error", "yaw_rate_error", "normalized_vector_error",
    "xy_speed", "box_vx", "box_vy", "box_yaw_rate",
    "box_vx_heading", "box_vy_heading", "box_yaw_rate_world",
    "box_vx_error", "box_vy_error", "box_yaw_rate_error",
    "robot_box_relative_linear_velocity_norm",
    "bilateral_contact", "grasp_loss", "robot_box_distance",
    "box_tilt_deg", "confirmed_carry", "base_pos_x", "base_pos_y",
    "base_yaw", "carry_heading_ref", "carry_heading_error",
    "legacy_body_vx", "legacy_body_vy", "legacy_body_yaw_rate",
    "action_delta_rms", "action_rate_rms", "torque_rms", "feet_slip",
) + PRESERVATION_METRICS + LOWER_BODY_TRACE_METRICS


def duration_steps(seconds, policy_dt, *, allow_zero=False):
    steps = int(round(float(seconds) / float(policy_dt)))
    return max(0 if allow_zero else 1, steps)


def _scalar(tensor, env_id=0):
    return float(tensor[env_id].reshape(-1)[0].item())


def _carry_sample(env, previous_hand_box=None, previous_box_pos=None):
    """Evaluation-only measurements, independent of training logging buffers."""
    c = env.cfg.rewards
    torso = env.rigid_body_states[:1, env.carry_torso_index]
    box = env.box_states[:1]
    box_pos = quat_rotate_inverse(torso[:, 3:7], box[:, :3] - torso[:, :3])
    hands = env.rigid_body_states[:1, env.carry_palm_indices, :3] - box[:, None, :3]
    hand_box = quat_rotate_inverse(box[:, 3:7].expand(2, -1), hands.reshape(2, 3))
    side_error = (hand_box[:, 1] * hand_box.new_tensor([1., -1.]) - env._box_size[0, 1] / 2).abs()
    arms = env.dof_pos[0, env.carry_arm_indices]
    arm_error = ((arms.new_tensor(c.carry_arm_range_lower) - arms).clamp_min(0)
                 + (arms - arms.new_tensor(c.carry_arm_range_upper)).clamp_min(0))
    position_error = ((box_pos.new_tensor(c.carry_box_relative_position_lower) - box_pos).clamp_min(0)
                      + (box_pos - box_pos.new_tensor(c.carry_box_relative_position_upper)).clamp_min(0))
    if previous_hand_box is None:
        slip = [float("nan"), float("nan")]
        speed = float("nan")
    else:
        slip = ((hand_box - previous_hand_box)[:, [0, 2]] / env.dt).norm(dim=-1).tolist()
        speed = ((box_pos - previous_box_pos) / env.dt).norm().item()
    metrics = {
        "left_hand_side_error_m": side_error[0].item(),
        "right_hand_side_error_m": side_error[1].item(),
        "left_hand_tangential_slip_mps": slip[0],
        "right_hand_tangential_slip_mps": slip[1],
        "arm_range_violation_rad": arm_error.max().item(),
        "box_relative_region_violation_m": position_error.norm().item(),
        "box_relative_motion_error_mps": speed,
    }
    hip = env.carry_hip_error[0]
    foot_heading = env.carry_foot_heading_error[0]
    waist = env.dof_pos[0, env.carry_waist_indices]
    torso_rpy = env.carry_torso_pelvis_rpy[0]
    metrics.update({
        "left_hip_roll_error_rad": hip[0].item(),
        "right_hip_roll_error_rad": hip[2].item(),
        "left_hip_yaw_error_rad": hip[1].item(),
        "right_hip_yaw_error_rad": hip[3].item(),
        "feet_width_m": env.carry_feet_width[0].item(),
        "feet_width_violation_m": env.carry_feet_width_violation[0].item(),
        "knee_width_m": env.carry_knee_width[0].item(),
        "knee_width_violation_m": env.carry_knee_width_violation[0].item(),
        "left_foot_yaw_error_rad": foot_heading[0].item(),
        "right_foot_yaw_error_rad": foot_heading[1].item(),
        "foot_heading_violation_rad": env.carry_foot_heading_excess[0].mean().item(),
        "waist_yaw_rad": waist[0].item(),
        "waist_roll_rad": waist[1].item(),
        "waist_pitch_rad": waist[2].item(),
        "torso_pelvis_relative_yaw_rad": torso_rpy[2].item(),
        "torso_pelvis_relative_roll_rad": torso_rpy[0].item(),
        "torso_pelvis_relative_pitch_rad": torso_rpy[1].item(),
    })
    return metrics, hand_box, box_pos


def _sample(env, condition, *, policy_step, time_s, actions, previous_actions, carry_sample):
    env_id = 0
    command = env.carry_policy_commands[env_id, :3]
    actual = torch.stack(
        (
            env.base_lin_vel_yaw[env_id, 0],
            env.base_lin_vel_yaw[env_id, 1],
            env.base_yaw_rate_world[env_id],
        )
    )
    error = actual - command

    heading_quat = calc_heading_quat(
        env.rigid_body_states[env_id : env_id + 1, env.upper_body_index, 3:7]
    )
    box_velocity_heading = quat_rotate_inverse(
        heading_quat, env.box_states[env_id : env_id + 1, 7:10]
    )[0]
    box_yaw_rate = env.box_states[env_id, 12]
    box_error = torch.stack(
        (
            box_velocity_heading[0] - command[0],
            box_velocity_heading[1] - command[1],
            box_yaw_rate - command[2],
        )
    )

    robot_world_velocity = env.rigid_body_states[
        env_id, env.upper_body_index, 7:10
    ]
    relative_velocity = torch.linalg.vector_norm(
        env.box_states[env_id, 7:9] - robot_world_velocity[:2]
    )
    bilateral = torch.all(env.hand_contact_filt[env_id])
    grasp_loss = ~torch.any(env.hand_contact_filt[env_id])
    tilt_deg = torch.rad2deg(
        torch.acos(torch.clamp(-env.projected_gravity_box[env_id, 2], -1.0, 1.0))
    )
    box_bottom = env.box_states[env_id, 2] - 0.5 * env._box_size[env_id, 2]
    confirmed_carry = (
        bilateral
        & (box_bottom >= env.cfg.rewards.box_drop_height)
        & (env.robot2object_dist[env_id] <= env.cfg.rewards.robot_box_max_distance)
        & (tilt_deg <= env.cfg.rewards.box_tilt_termination_deg)
    )

    normalized = torch.stack(
        (
            error[0] / TRACKING_SCALES["vx"],
            error[1] / TRACKING_SCALES["vy"],
            error[2] / TRACKING_SCALES["yaw_rate"],
        )
    )
    delta = actions[env_id] - previous_actions[env_id]
    feet_contact = env.contact_forces[env_id, env.feet_indices, 2] > 1.0
    feet_slip = torch.sum(
        torch.linalg.vector_norm(env.feet_vel[env_id, :, :2], dim=-1)
        * feet_contact
    )
    sample = {
        "time_s": float(time_s),
        "policy_step": int(policy_step),
        "mode": condition.mode,
        "command_vx": float(command[0].item()),
        "command_vy": float(command[1].item()),
        "command_yaw_rate": float(command[2].item()),
        "actual_vx_training_frame": float(actual[0].item()),
        "actual_vy_training_frame": float(actual[1].item()),
        "actual_yaw_rate_training_frame": float(actual[2].item()),
        "vx_error": float(error[0].item()),
        "vy_error": float(error[1].item()),
        "yaw_rate_error": float(error[2].item()),
        "normalized_vector_error": float(torch.linalg.vector_norm(normalized).item()),
        "xy_speed": float(torch.linalg.vector_norm(actual[:2]).item()),
        "box_vx": float(box_velocity_heading[0].item()),
        "box_vy": float(box_velocity_heading[1].item()),
        "box_yaw_rate": float(box_yaw_rate.item()),
        "box_vx_heading": float(box_velocity_heading[0].item()),
        "box_vy_heading": float(box_velocity_heading[1].item()),
        "box_yaw_rate_world": float(box_yaw_rate.item()),
        "box_vx_error": float(box_error[0].item()),
        "box_vy_error": float(box_error[1].item()),
        "box_yaw_rate_error": float(box_error[2].item()),
        "robot_box_relative_linear_velocity_norm": float(relative_velocity.item()),
        "bilateral_contact": int(bilateral.item()),
        "grasp_loss": int(grasp_loss.item()),
        "robot_box_distance": _scalar(env.robot2object_dist, env_id),
        "box_tilt_deg": float(tilt_deg.item()),
        "confirmed_carry": int(confirmed_carry.item()),
        "base_pos_x": float(env.root_states[env_id, 0].item()),
        "base_pos_y": float(env.root_states[env_id, 1].item()),
        "base_yaw": _scalar(env.yaw, env_id),
        "carry_heading_ref": _scalar(env.carry_heading_ref, env_id),
        "carry_heading_error": _scalar(env.carry_heading_error, env_id),
        "legacy_body_vx": float(env.base_lin_vel[env_id, 0].item()),
        "legacy_body_vy": float(env.base_lin_vel[env_id, 1].item()),
        "legacy_body_yaw_rate": float(env.base_ang_vel[env_id, 2].item()),
        "action_delta_rms": float(torch.sqrt(torch.mean(delta.square())).item()),
        "action_rate_rms": float(
            (torch.sqrt(torch.mean(delta.square())) / env.dt).item()
        ),
        "torque_rms": float(
            torch.sqrt(torch.mean(env.torques[env_id].square())).item()
        ),
        "feet_slip": float(feet_slip.item()),
    }
    sample.update(carry_sample)
    if tuple(sample) != TRACE_FIELDS:
        raise AssertionError("Trace schema changed unexpectedly")
    return sample


def run_trial(env, policy, condition, *, warmup_s, duration_s, seed_fn):
    command = (condition.vx, condition.vy, condition.yaw_rate)
    seed_fn(condition.seed)
    env.set_evaluation_command(command)
    env.clear_evaluation_outcome()
    obs, _ = env.reset()
    assert_observation_compatibility(env, obs, command)
    if env.eval_last_termination_reason[0]:
        raise RuntimeError(
            "The mandatory zero-action reset step terminated: "
            f"{env.eval_last_termination_reason[0]}"
        )

    policy_dt = float(env.dt)
    warmup_steps = duration_steps(warmup_s, policy_dt, allow_zero=True)
    measure_steps = duration_steps(duration_s, policy_dt)
    total_steps = warmup_steps + measure_steps
    previous_actions = torch.zeros(
        env.num_envs, env.num_actions, device=env.device
    )
    samples = []
    executed_steps = 0
    termination_reason = "completed"

    print(
        f"[{condition.trial_id}] mode={condition.mode} command={command} "
        f"seed={condition.seed}"
    )
    _, previous_hand_box, previous_box_pos = _carry_sample(env)
    for step_index in range(total_steps):
        with torch.inference_mode():
            actions = policy(obs.detach())
        obs, _, _, dones, _, _, _, _ = env.step(actions.detach())
        executed_steps += 1
        env.assert_evaluation_command(obs)
        if torch.count_nonzero(env.disturbance).item() != 0:
            raise AssertionError("No-force evaluation produced a robot disturbance")
        if bool(dones[0].item()):
            termination_reason = env.eval_last_termination_reason[0] or "termination"
            break
        carry_sample, previous_hand_box, previous_box_pos = _carry_sample(
            env, previous_hand_box, previous_box_pos)
        if step_index >= warmup_steps:
            samples.append(
                _sample(
                    env,
                    condition,
                    policy_step=executed_steps,
                    time_s=executed_steps * policy_dt,
                    actions=actions,
                    previous_actions=previous_actions,
                    carry_sample=carry_sample,
                )
            )
        previous_actions.copy_(actions)

    summary = summarize_trial(
        condition,
        samples,
        policy_dt=policy_dt,
        requested_steps=measure_steps,
        executed_steps=executed_steps,
        termination_reason=termination_reason,
    )
    print(
        f"[{condition.trial_id}] completed={summary['trial_completed']} "
        f"reason={termination_reason} samples={len(samples)}/{measure_steps}"
    )
    return samples, summary


def run_suite(env, policy, conditions, *, warmup_s, duration_s, seed_fn,
              trace_writer=None):
    summaries = []
    for condition in conditions:
        samples, summary = run_trial(
            env,
            policy,
            condition,
            warmup_s=warmup_s,
            duration_s=duration_s,
            seed_fn=seed_fn,
        )
        if trace_writer is not None:
            trace_writer(condition.trial_id, samples)
        summaries.append(summary)
    return summaries
