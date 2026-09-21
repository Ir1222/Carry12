"""Carry-only loaded-locomotion specialist for the Unitree G1."""

import math

import numpy as np
import torch

from isaacgym.torch_utils import (
    quat_conjugate, quat_mul, quat_rotate_inverse, torch_rand_float,
)
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.math import wrap_to_pi
from legged_gym.utils.torch_utils import calc_heading_quat

from .carrybox import LeggedRobot as CarryBoxBase


def range_violation(value, lower, upper):
    """No center preference anywhere inside the closed feasible interval."""
    return (lower - value).clamp_min(0.0) + (value - upper).clamp_min(0.0)


def constraint_reward(normalized_violation):
    """C1 dead-zone reward: 1/(1+sum(excess/softness)^2), no Gaussian."""
    return 1.0 / (1.0 + normalized_violation.flatten(1).square().sum(-1))


def deadzone_gaussian_reward(error, deadzone, softness):
    """Bounded reward with a flat feasible dead zone and Gaussian falloff."""
    excess = (error.abs() - deadzone).clamp_min(0.0)
    return torch.exp(-torch.mean((excess / softness).square(), dim=-1))


def interval_gaussian_reward(value, lower, upper, softness):
    """Bounded reward equal to one throughout a closed feasible interval."""
    excess = range_violation(value, lower, upper)
    return torch.exp(-(excess / softness).square())


def quaternion_projected_heading(quaternion):
    """Heading of the quaternion's forward axis projected onto world XY."""
    quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    x, y, z, w = quaternion.unbind(-1)
    forward_x = 1.0 - 2.0 * (y.square() + z.square())
    forward_y = 2.0 * (x * y + w * z)
    return torch.atan2(forward_y, forward_x)


def relative_projected_heading(orientations, reference_orientation):
    """Wrapped heading difference for one or more bodies per environment."""
    reference_heading = quaternion_projected_heading(reference_orientation)
    if orientations.ndim == reference_orientation.ndim + 1:
        reference_heading = reference_heading.unsqueeze(-1)
    return wrap_to_pi(
        quaternion_projected_heading(orientations) - reference_heading
    )


def quaternion_to_rpy(quaternion):
    """Convert normalized XYZW quaternions to wrapped XYZ Euler diagnostics."""
    quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    x, y, z, w = quaternion.unbind(-1)
    roll = torch.atan2(2.0 * (w * x + y * z),
                       1.0 - 2.0 * (x.square() + y.square()))
    pitch = torch.asin((2.0 * (w * y - z * x)).clamp(-1.0, 1.0))
    yaw = torch.atan2(2.0 * (w * z + x * y),
                      1.0 - 2.0 * (y.square() + z.square()))
    return torch.stack((roll, pitch, yaw), dim=-1)


def quaternion_rotation_vector(quaternion):
    """Principal SO(3) log for XYZW quaternions, invariant to q versus -q."""
    quaternion = quaternion / quaternion.norm(
        dim=-1, keepdim=True).clamp_min(1.0e-12)
    quaternion = torch.where(
        quaternion[..., 3:4] < 0.0, -quaternion, quaternion)
    vector = quaternion[..., :3]
    sin_half_angle = vector.norm(dim=-1, keepdim=True)
    half_angle = torch.atan2(
        sin_half_angle, quaternion[..., 3:4].clamp(0.0, 1.0))
    scale = 2.0 * half_angle / sin_half_angle.clamp_min(1.0e-12)
    scale = torch.where(sin_half_angle > 1.0e-7, scale, 2.0)
    return vector * scale


class LeggedRobot(CarryBoxBase):
    """CarryBox variant that starts in, and remains in, the carry phase."""

    _CARRY_SKILL = "carryWith"

    def _parse_cfg(self, cfg):
        """Parse the parent configuration without requiring positive-only vx."""
        self.dt = cfg.control.decimation * self.sim_params.dt
        self.obs_scales = cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(cfg.rewards.scales)
        self.command_ranges = class_to_dict(cfg.commands.ranges)
        if cfg.terrain.mesh_type not in ["heightfield", "trimesh"]:
            cfg.terrain.curriculum = False
        self.max_episode_length_s = cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
        cfg.domain_rand.push_interval = np.ceil(
            cfg.domain_rand.push_interval_s / self.dt
        )

    def _init_buffers(self):
        super()._init_buffers()
        self.carry_command_resample_time = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device,
            requires_grad=False
        )
        self.grasp_loss_time = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device,
            requires_grad=False
        )
        self.carry_command_mode = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device,
            requires_grad=False
        )

        rewards = self.cfg.rewards
        self.carry_hand_side_sign = self.dof_pos.new_tensor([1.0, -1.0])
        self.carry_arm_range_lower = self.dof_pos.new_tensor(rewards.carry_arm_range_lower)
        self.carry_arm_range_upper = self.dof_pos.new_tensor(rewards.carry_arm_range_upper)
        self.carry_box_relative_position_lower = self.dof_pos.new_tensor(rewards.carry_box_relative_position_lower)
        self.carry_box_relative_position_upper = self.dof_pos.new_tensor(rewards.carry_box_relative_position_upper)
        self.carry_box_position_violation_scale = self.dof_pos.new_tensor(rewards.carry_box_position_violation_scale)
        self.carry_box_relative_velocity_tolerance = self.dof_pos.new_tensor(rewards.carry_box_relative_velocity_tolerance)
        self.carry_box_velocity_violation_scale = self.dof_pos.new_tensor(rewards.carry_box_velocity_violation_scale)

        def body_index(name):
            index = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], name
            )
            if index < 0:
                raise ValueError("Carry preservation body is missing: " + name)
            return index

        # This index deliberately does not replace upper_body_index (pelvis),
        # which defines the pretrained actor's observation/command semantics.
        self.carry_torso_index = body_index(rewards.carry_torso_link)
        self.carry_palm_indices = torch.tensor(
            [body_index(name) for name in rewards.carry_hand_links],
            device=self.device, dtype=torch.long,
        )
        self.carry_arm_indices = torch.tensor(
            [self.dof_names.index(name) for name in rewards.carry_arm_joint_names],
            device=self.device, dtype=torch.long,
        )
        self.carry_leg_indices = torch.tensor(
            [self.dof_names.index(name) for name in rewards.carry_leg_joint_names],
            device=self.device, dtype=torch.long,
        )
        self.carry_leg_range_lower = self.dof_pos.new_tensor(rewards.carry_leg_range_lower)
        self.carry_leg_range_upper = self.dof_pos.new_tensor(rewards.carry_leg_range_upper)
        self.carry_leg_violation_scale = self.dof_pos.new_tensor(rewards.carry_leg_violation_scale)
        if any(t.shape != (12,) for t in (
            self.carry_leg_indices, self.carry_leg_range_lower,
            self.carry_leg_range_upper, self.carry_leg_violation_scale,
        )):
            raise ValueError("Carry leg names, bounds and softness must each have 12 entries")
        if not torch.all(self.carry_leg_range_lower < self.carry_leg_range_upper) or not torch.all(self.carry_leg_violation_scale > 0):
            raise ValueError("Carry leg bounds must be ordered and softness positive")

        self.carry_hip_indices = torch.tensor(
            [self.dof_names.index(name) for name in rewards.carry_hip_joint_names],
            device=self.device, dtype=torch.long,
        )
        self.carry_hip_target = self.default_dof_pos[self.carry_hip_indices].clone()
        self.carry_hip_deadzone = self.dof_pos.new_tensor(
            rewards.carry_hip_posture_deadzone)
        self.carry_hip_softness = self.dof_pos.new_tensor(
            rewards.carry_hip_posture_softness)
        self.carry_foot_indices = torch.tensor(
            [body_index(name) for name in rewards.carry_foot_links],
            device=self.device, dtype=torch.long,
        )
        self.carry_knee_indices = torch.tensor(
            [body_index(name) for name in rewards.carry_knee_links],
            device=self.device, dtype=torch.long,
        )
        self.carry_waist_indices = torch.tensor(
            [self.dof_names.index(name) for name in rewards.carry_waist_joint_names],
            device=self.device, dtype=torch.long,
        )
        self.carry_waist_reference_target = self.dof_pos.new_tensor(
            rewards.carry_waist_reference_target)
        self.carry_waist_reference_deadzone = self.dof_pos.new_tensor(
            rewards.carry_waist_reference_deadzone)
        self.carry_waist_reference_softness = self.dof_pos.new_tensor(
            rewards.carry_waist_reference_softness)
        self.carry_torso_pelvis_reference_quat = self.dof_pos.new_tensor(
            rewards.carry_torso_pelvis_reference_quat)
        self.carry_torso_pelvis_reference_quat /= (
            self.carry_torso_pelvis_reference_quat.norm().clamp_min(1.0e-12)
        )
        self.carry_torso_pelvis_alignment_deadzone = self.dof_pos.new_tensor(
            rewards.carry_torso_pelvis_alignment_deadzone)
        self.carry_torso_pelvis_alignment_softness = self.dof_pos.new_tensor(
            rewards.carry_torso_pelvis_alignment_softness)
        self.carry_feet_width_range = self.dof_pos.new_tensor(
            rewards.carry_feet_width_range)
        self.carry_knee_width_range = self.dof_pos.new_tensor(
            rewards.carry_knee_width_range)
        self.carry_foot_heading_deadzone = self.dof_pos.new_full(
            (2,), rewards.carry_foot_heading_deadzone)
        self.carry_foot_heading_softness = self.dof_pos.new_full(
            (2,), rewards.carry_foot_heading_softness)
        if any(t.shape != (4,) for t in (
            self.carry_hip_indices, self.carry_hip_target,
            self.carry_hip_deadzone, self.carry_hip_softness,
        )):
            raise ValueError("Carry hip names, targets, dead zones and softness must have four entries")
        if self.carry_foot_indices.shape != (2,) or self.carry_knee_indices.shape != (2,):
            raise ValueError("Carry foot and knee link lists must contain left and right entries")
        expected_waist_names = (
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        )
        if tuple(rewards.carry_waist_joint_names) != expected_waist_names:
            raise ValueError(
                "Carry waist reference requires exactly yaw, roll and pitch, in that order"
            )
        if any(t.shape != (3,) for t in (
            self.carry_waist_indices,
            self.carry_waist_reference_target,
            self.carry_waist_reference_deadzone,
            self.carry_waist_reference_softness,
            self.carry_torso_pelvis_alignment_deadzone,
            self.carry_torso_pelvis_alignment_softness,
        )) or self.carry_torso_pelvis_reference_quat.shape != (4,):
            raise ValueError(
                "Carry waist/alignment targets, dead zones and softness have invalid shapes"
            )
        if (not torch.all(self.carry_hip_deadzone >= 0)
                or not torch.all(self.carry_hip_softness > 0)
                or not torch.all(self.carry_foot_heading_softness > 0)
                or not torch.all(self.carry_waist_reference_deadzone >= 0)
                or not torch.all(self.carry_waist_reference_softness > 0)
                or not torch.all(self.carry_torso_pelvis_alignment_deadzone >= 0)
                or not torch.all(self.carry_torso_pelvis_alignment_softness > 0)
                or rewards.carry_foot_heading_deadzone < 0
                or rewards.carry_feet_width_softness <= 0
                or rewards.carry_knee_width_softness <= 0
                or not torch.all(self.carry_feet_width_range[0] < self.carry_feet_width_range[1])
                or not torch.all(self.carry_knee_width_range[0] < self.carry_knee_width_range[1])):
            raise ValueError("Carry lower-body dead zones, softness and intervals are invalid")

        self.previous_hand_box = self.dof_pos.new_zeros(self.num_envs, 2, 3)
        self.previous_box_relative_pos = self.dof_pos.new_zeros(self.num_envs, 3)
        self.carry_history_valid = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        # Episode aggregates; count actual steps, not randomized episode ages.
        self.carry_log_sums = self.dof_pos.new_zeros(self.num_envs, 9)
        self.carry_log_steps = self.dof_pos.new_zeros(self.num_envs, 2)  # all / valid temporal
        self.carry_lower_log_sums = self.dof_pos.new_zeros(self.num_envs, 38)
        self.carry_lower_log_steps = self.dof_pos.new_zeros(self.num_envs)

    def _reset_actors(self, env_ids):
        """Reset robot state from one coherent CarryWith reference frame."""
        num_envs = len(env_ids)
        if num_envs == 0:
            return

        if self.box_cfg.fixed_carry_reset:
            motion_id = int(self.box_cfg.fixed_carry_motion_id)
            if not 0 <= motion_id < self.motionlib.num_motion[self._CARRY_SKILL]:
                raise ValueError(
                    f"fixed_carry_motion_id={motion_id} is outside the "
                    f"available range [0, "
                    f"{self.motionlib.num_motion[self._CARRY_SKILL] - 1}]"
                )
            phase_value = float(self.box_cfg.fixed_carry_phase)
            if not 0.0 <= phase_value <= 1.0:
                raise ValueError("fixed_carry_phase must be in [0, 1]")
            motion_ids = torch.full(
                (num_envs,), motion_id, dtype=torch.long, device=self.device
            )
            phases = torch.full(
                (num_envs,), phase_value, dtype=torch.float, device=self.device
            )
        else:
            phase_range = self.box_cfg.carry_reset_phase_range
            if not (
                len(phase_range) == 2
                and 0.0 <= phase_range[0] <= phase_range[1] <= 1.0
            ):
                raise ValueError(
                    "carry_reset_phase_range must be ordered within [0, 1]"
                )
            motion_ids = self.motionlib.sample_motions(
                self._CARRY_SKILL, num_envs
            )
            phases = torch_rand_float(
                phase_range[0], phase_range[1], (num_envs, 1),
                device=self.device
            ).squeeze(1)

        motion_lengths = self.motionlib.motion_len[
            self._CARRY_SKILL
        ][motion_ids].to(dtype=torch.float)
        motion_times = phases * (motion_lengths - 1.0)
        root_pos, root_rot, _, _, dof_pos, _, _ = (
            self.motionlib.get_motion_state(
                self._CARRY_SKILL, motion_ids, motion_times
            )
        )

        self.root_states[env_ids, :3] = root_pos + self.env_origins[env_ids]
        self.root_states[env_ids, 3:7] = root_rot
        self.root_states[env_ids, 7:13] = 0.0
        self.dof_pos[env_ids] = dof_pos
        self.dof_vel[env_ids] = 0.0

        self._reset_ref_env_ids[self._CARRY_SKILL] = env_ids
        self._reset_ref_motion_ids[self._CARRY_SKILL] = motion_ids
        self._reset_ref_motion_times[self._CARRY_SKILL] = motion_times

    def _reset_boxes(self, env_ids):
        """Reset the box from the exact CarryWith frame used by the robot."""
        curr_env_ids = self._reset_ref_env_ids[self._CARRY_SKILL]
        motion_ids = self._reset_ref_motion_ids[self._CARRY_SKILL]
        motion_times = self._reset_ref_motion_times[self._CARRY_SKILL]
        box_pos, box_rot, _, _ = self.motionlib.get_obj_motion_state(
            skill=self._CARRY_SKILL,
            motion_ids=motion_ids,
            motion_times=motion_times,
        )

        self.box_states[curr_env_ids, :3] = (
            box_pos + self.env_origins[curr_env_ids]
        )
        self.box_states[curr_env_ids, 3:7] = box_rot
        self.box_states[curr_env_ids, 7:13] = 0.0
        self.platform_pos[curr_env_ids] = self.platform_default_pos[curr_env_ids]

        self.thresh_tag[env_ids] = torch_rand_float(
            self.box_cfg.thresh_tag[0], self.box_cfg.thresh_tag[1],
            (len(env_ids), 1), device=self.device
        ).squeeze(1)
        self.far_pos_offset[env_ids] = torch_rand_float(
            -self.box_cfg.far_pos_offset, self.box_cfg.far_pos_offset,
            (len(env_ids), 3), device=self.device
        )
        self.far_pos_offset[env_ids, 2] *= 2.0
        self.has_seen_tag[env_ids] = True
        self.can_see_tag[env_ids] = False

    def _reset_task(self, env_ids):
        self.tar_platform_states[env_ids] = (
            self.tar_platform_default_states[env_ids]
        )
        self._sample_carry_commands(env_ids)
        self._sample_carry_command_resample_time(env_ids)

    def reset_idx(self, env_ids):
        """Use the parent bookkeeping, then restore carry-only state."""
        if len(env_ids) == 0:
            return
        super().reset_idx(env_ids)
        sums = self.carry_log_sums[env_ids].sum(0)
        steps = self.carry_log_steps[env_ids].sum(0)[[0, 0, 1, 0, 0, 1, 0, 0, 0]]
        means = torch.where(steps > 0, sums / steps.clamp_min(1), float("nan"))
        self.extras["episode"].update({
            "carry/bilateral_contact": means[0],
            "carry/hand_surface_violation": means[1],
            "carry/hand_slip": means[2],
            "carry/arm_violation": means[3],
            "carry/box_position_violation": means[4],
            "carry/box_relative_velocity": means[5],
            "carry/leg_range_violation": means[6],
            "carry/stance_width": means[7],
            "carry/stance_width_violation": means[8],
        })
        lower_count = self.carry_lower_log_steps[env_ids].sum()
        lower_sums = self.carry_lower_log_sums[env_ids].sum(0)
        lower_means = torch.where(
            lower_count > 0,
            lower_sums / lower_count.clamp_min(1),
            torch.full_like(lower_sums, float("nan")),
        )
        self.extras["episode"].update({
            "carry/hip_roll_abs_mean": lower_means[0],
            "carry/hip_yaw_abs_mean": lower_means[1],
            "carry/left_hip_roll_abs": lower_means[2],
            "carry/right_hip_roll_abs": lower_means[3],
            "carry/left_hip_yaw_abs": lower_means[4],
            "carry/right_hip_yaw_abs": lower_means[5],
            "carry/feet_width_mean": lower_means[6],
            "carry/feet_width_violation": lower_means[7],
            "carry/knee_width_mean": lower_means[8],
            "carry/knee_width_violation": lower_means[9],
            "carry/left_foot_yaw_error": lower_means[10],
            "carry/right_foot_yaw_error": lower_means[11],
            "carry/foot_heading_violation": lower_means[12],
            "carry/waist_yaw_abs": lower_means[13],
            "carry/waist_roll_abs": lower_means[14],
            "carry/waist_pitch_abs": lower_means[15],
            "carry/waist_yaw_rms": lower_means[16].clamp_min(0.0).sqrt(),
            "carry/waist_roll_rms": lower_means[17].clamp_min(0.0).sqrt(),
            "carry/waist_pitch_rms": lower_means[18].clamp_min(0.0).sqrt(),
            "carry/torso_pelvis_relative_yaw_abs": lower_means[19],
            "carry/torso_pelvis_relative_roll_abs": lower_means[20],
            "carry/torso_pelvis_relative_pitch_abs": lower_means[21],
            "carry/torso_pelvis_relative_yaw_rms": lower_means[22].clamp_min(0.0).sqrt(),
            "carry/torso_pelvis_relative_roll_rms": lower_means[23].clamp_min(0.0).sqrt(),
            "carry/torso_pelvis_relative_pitch_rms": lower_means[24].clamp_min(0.0).sqrt(),
            "carry/waist_reference_reward": lower_means[25],
            "carry/waist_yaw_error_abs": lower_means[26],
            "carry/waist_roll_error_abs": lower_means[27],
            "carry/waist_pitch_error_abs": lower_means[28],
            "carry/waist_yaw_excess": lower_means[29],
            "carry/waist_roll_excess": lower_means[30],
            "carry/waist_pitch_excess": lower_means[31],
            "carry/torso_pelvis_alignment_reward": lower_means[32],
            "carry/torso_pelvis_rotvec_x_abs": lower_means[33],
            "carry/torso_pelvis_rotvec_y_abs": lower_means[34],
            "carry/torso_pelvis_rotvec_z_abs": lower_means[35],
            "carry/torso_pelvis_alignment_excess": lower_means[36],
            "carry/torso_pelvis_alignment_error": lower_means[37],
        })
        self.carry_log_sums[env_ids] = 0.0
        self.carry_log_steps[env_ids] = 0.0
        self.carry_lower_log_sums[env_ids] = 0.0
        self.carry_lower_log_steps[env_ids] = 0.0
        self.carry_history_valid[env_ids] = False
        self.carry_policy_commands[env_ids, :3] = self.commands[env_ids, :3]
        self.is_stage_carry[env_ids] = True
        self.carry_velocity_active[env_ids] = True
        self.carry_tracking_started[env_ids] = True
        self.grasp_loss_time[env_ids] = 0.0
        self.has_seen_tag[env_ids] = True
        self.can_see_tag[env_ids] = False

        invalid_timer = self.carry_command_resample_time[env_ids] <= 0.0
        if invalid_timer.any():
            self._sample_carry_command_resample_time(env_ids[invalid_timer])

    def _compute_is_stage_carry(self):
        return torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def _compute_carry_velocity_active(self):
        return torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def _sample_carry_commands(self, env_ids):
        """Sample stand/pure-axis/mixed commands from the fixed V1 mixture."""
        num_envs = len(env_ids)
        if num_envs == 0:
            return

        probabilities = torch.tensor(
            self.cfg.commands.carry_command_mode_probabilities,
            dtype=torch.float, device=self.device
        )
        if probabilities.shape != (5,) or not torch.isclose(
            probabilities.sum(),
            torch.tensor(1.0, device=self.device),
        ):
            raise ValueError(
                "carry_command_mode_probabilities must contain five values "
                "that sum to one"
            )
        modes = torch.multinomial(
            probabilities, num_samples=num_envs, replacement=True
        )
        self.carry_command_mode[env_ids] = modes
        self.commands[env_ids] = 0.0

        def sample_axis(mode, axis, value_range):
            selected = env_ids[modes == mode]
            if len(selected) > 0:
                self.commands[selected, axis] = torch_rand_float(
                    value_range[0], value_range[1], (len(selected), 1),
                    device=self.device
                ).squeeze(1)

        sample_axis(1, 0, self.cfg.commands.carry_vx_range)
        sample_axis(2, 1, self.cfg.commands.carry_vy_range)
        sample_axis(3, 2, self.cfg.commands.carry_yaw_rate_range)

        mixed_env_ids = env_ids[modes == 4]
        if len(mixed_env_ids) > 0:
            mixed_ranges = self.cfg.commands.carry_mixed_ranges
            for axis in range(3):
                self.commands[mixed_env_ids, axis] = torch_rand_float(
                    mixed_ranges[axis][0], mixed_ranges[axis][1],
                    (len(mixed_env_ids), 1), device=self.device
                ).squeeze(1)

    def _sample_carry_command_resample_time(self, env_ids):
        if len(env_ids) == 0:
            return
        interval = self.cfg.commands.carry_command_resample_interval_s
        self.carry_command_resample_time[env_ids] = torch_rand_float(
            interval[0], interval[1], (len(env_ids), 1),
            device=self.device
        ).squeeze(1)

    def _update_carry_heading_commands(self):
        """Resample commands when due and expose raw [vx, vy, yaw_rate]."""
        self.carry_command_resample_time -= self.dt
        due_env_ids = (
            self.carry_command_resample_time <= 0.0
        ).nonzero(as_tuple=False).flatten()
        if len(due_env_ids) > 0:
            self._sample_carry_commands(due_env_ids)
            self._sample_carry_command_resample_time(due_env_ids)

        self.carry_policy_commands[:, :3] = self.commands[:, :3]

        entering_carry = ~self.carry_heading_initialized
        self.carry_heading_ref[entering_carry] = self.yaw[entering_carry]
        self.carry_heading_initialized[entering_carry] = True
        self.carry_heading_ref[:] = wrap_to_pi(
            self.carry_heading_ref + self.commands[:, 2] * self.dt
        )
        self.carry_heading_error[:] = wrap_to_pi(
            self.carry_heading_ref - self.yaw
        )

    def _command_is_moving(self):
        command = self.carry_policy_commands[:, :3]
        return (
            torch.norm(command[:, :2], dim=-1) > 0.05
        ) | (torch.abs(command[:, 2]) > 0.05)

    def _update_hand_contact_state(self):
        current_contact = torch.norm(
            self.contact_forces[:, self.hand_colli_indices], dim=-1
        ) > self.cfg.rewards.hand_contact_threshold
        self.hand_contact_filt = torch.logical_or(
            current_contact, self.last_hand_contacts
        )
        self.last_hand_contacts = current_contact

    def check_termination(self):
        super().check_termination()
        self._update_hand_contact_state()

        box_bottom = self.box_states[:, 2] - 0.5 * self._box_size[:, 2]
        self.reset_buf |= box_bottom < self.cfg.rewards.box_drop_height
        self.reset_buf |= (
            self.robot2object_dist
            > self.cfg.rewards.robot_box_max_distance
        )

        tilt_cos = math.cos(
            math.radians(self.cfg.rewards.box_tilt_termination_deg)
        )
        self.reset_buf |= self.projected_gravity_box[:, 2] > -tilt_cos

        complete_grasp_loss = ~torch.any(self.hand_contact_filt, dim=-1)
        self.grasp_loss_time = torch.where(
            complete_grasp_loss,
            self.grasp_loss_time + self.dt,
            torch.zeros_like(self.grasp_loss_time),
        )
        self.reset_buf |= (
            self.grasp_loss_time > self.cfg.rewards.grasp_loss_grace_s
        )

    def _reward_feet_air_time(self):
        reward = torch.sum(
            (self.feet_air_time - 0.5) * self.first_contacts, dim=1
        )
        return reward * self._command_is_moving()

    def _reward_carry_bilateral_contact(self):
        contact = torch.all(self.hand_contact_filt, dim=-1).to(torch.float)
        self.carry_log_sums[:, 0] += contact
        return contact

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        # CarryBoxBase retains extras on non-reset steps, and the locomotion
        # runner appends every present "episode" entry. Clear it once per step
        # so a completed episode is logged only once (as before this refactor).
        self.extras.pop("episode", None)

    def compute_reward(self):
        self._update_carry_state()
        super().compute_reward()
        self.carry_log_steps[:, 0] += 1
        self.carry_log_steps[:, 1] += self.carry_history_valid
        # Rewards above use the pre-step validity; reset invalidates only reset envs.
        self.carry_history_valid[:] = True

    def _update_carry_state(self):
        """Sample local positions once per policy step, before rewards and resets."""
        torso = self.rigid_body_states[:, self.carry_torso_index]
        self.box_relative_pos = quat_rotate_inverse(
            torso[:, 3:7], self.box_states[:, :3] - torso[:, :3])
        offset = self.rigid_body_states[:, self.carry_palm_indices, :3] - self.box_states[:, None, :3]
        box_quat = self.box_states[:, None, 3:7].expand(-1, 2, -1).reshape(-1, 4)
        self.hand_box = quat_rotate_inverse(box_quat, offset.reshape(-1, 3)).reshape(-1, 2, 3)

        hand_velocity = (self.hand_box - self.previous_hand_box) / self.dt
        self.hand_slip = hand_velocity[..., [0, 2]].norm(dim=-1)
        self.hand_slip[~self.carry_history_valid] = 0.0
        self.box_relative_velocity = (self.box_relative_pos - self.previous_box_relative_pos) / self.dt
        self.box_relative_velocity[~self.carry_history_valid] = 0.0
        self.previous_hand_box.copy_(self.hand_box)
        self.previous_box_relative_pos.copy_(self.box_relative_pos)
        self._update_lower_body_state(torso)

    def _update_lower_body_state(self, torso):
        """Cache lower-body geometry and diagnostics once per policy step."""
        pelvis_quat = self.root_states[:, 3:7]
        heading_quat = calc_heading_quat(pelvis_quat)

        self.carry_hip_error = (
            self.dof_pos[:, self.carry_hip_indices] - self.carry_hip_target
        )

        feet = self.rigid_body_states[:, self.carry_foot_indices]
        knees = self.rigid_body_states[:, self.carry_knee_indices]
        feet_separation = quat_rotate_inverse(
            heading_quat, feet[:, 0, :3] - feet[:, 1, :3])
        knee_separation = quat_rotate_inverse(
            heading_quat, knees[:, 0, :3] - knees[:, 1, :3])
        self.carry_feet_width = feet_separation[:, 1].abs()
        self.carry_knee_width = knee_separation[:, 1].abs()
        self.carry_feet_width_violation = range_violation(
            self.carry_feet_width,
            self.carry_feet_width_range[0], self.carry_feet_width_range[1],
        )
        self.carry_knee_width_violation = range_violation(
            self.carry_knee_width,
            self.carry_knee_width_range[0], self.carry_knee_width_range[1],
        )
        leg_range_excess = range_violation(
            self.dof_pos[:, self.carry_leg_indices],
            self.carry_leg_range_lower, self.carry_leg_range_upper,
        )
        self.carry_log_sums[:, 6] += leg_range_excess.mean(-1)
        self.carry_log_sums[:, 7] += self.carry_feet_width
        self.carry_log_sums[:, 8] += self.carry_feet_width_violation

        self.carry_foot_heading_error = relative_projected_heading(
            feet[:, :, 3:7], pelvis_quat)
        self.carry_foot_heading_excess = (
            self.carry_foot_heading_error.abs() - self.carry_foot_heading_deadzone
        ).clamp_min(0.0)

        waist = self.dof_pos[:, self.carry_waist_indices]
        relative_torso_quat = quat_mul(
            quat_conjugate(pelvis_quat), torso[:, 3:7])
        self.carry_torso_pelvis_rpy = quaternion_to_rpy(relative_torso_quat)
        self.carry_waist_error = waist - self.carry_waist_reference_target
        self.carry_waist_excess = (
            self.carry_waist_error.abs() - self.carry_waist_reference_deadzone
        ).clamp_min(0.0)
        self.carry_waist_reference_reward = deadzone_gaussian_reward(
            self.carry_waist_error,
            self.carry_waist_reference_deadzone,
            self.carry_waist_reference_softness,
        )
        reference_quat = self.carry_torso_pelvis_reference_quat.unsqueeze(
            0).expand_as(relative_torso_quat)
        relative_error_quat = quat_mul(
            quat_conjugate(reference_quat), relative_torso_quat)
        self.carry_torso_pelvis_rotvec_error = quaternion_rotation_vector(
            relative_error_quat)
        self.carry_torso_pelvis_alignment_excess = (
            self.carry_torso_pelvis_rotvec_error.abs()
            - self.carry_torso_pelvis_alignment_deadzone
        ).clamp_min(0.0)
        self.carry_torso_pelvis_alignment_reward = deadzone_gaussian_reward(
            self.carry_torso_pelvis_rotvec_error,
            self.carry_torso_pelvis_alignment_deadzone,
            self.carry_torso_pelvis_alignment_softness,
        )

        hip_abs = self.carry_hip_error.abs()
        foot_heading_abs = self.carry_foot_heading_error.abs()
        waist_abs = waist.abs()
        torso_abs = self.carry_torso_pelvis_rpy.abs()
        waist_error_abs = self.carry_waist_error.abs()
        torso_rotvec_abs = self.carry_torso_pelvis_rotvec_error.abs()
        self.carry_lower_log_sums += torch.stack((
            hip_abs[:, [0, 2]].mean(-1),
            hip_abs[:, [1, 3]].mean(-1),
            hip_abs[:, 0], hip_abs[:, 2],
            hip_abs[:, 1], hip_abs[:, 3],
            self.carry_feet_width, self.carry_feet_width_violation,
            self.carry_knee_width, self.carry_knee_width_violation,
            foot_heading_abs[:, 0], foot_heading_abs[:, 1],
            self.carry_foot_heading_excess.mean(-1),
            waist_abs[:, 0], waist_abs[:, 1], waist_abs[:, 2],
            waist[:, 0].square(), waist[:, 1].square(), waist[:, 2].square(),
            torso_abs[:, 2], torso_abs[:, 0], torso_abs[:, 1],
            self.carry_torso_pelvis_rpy[:, 2].square(),
            self.carry_torso_pelvis_rpy[:, 0].square(),
            self.carry_torso_pelvis_rpy[:, 1].square(),
            self.carry_waist_reference_reward,
            waist_error_abs[:, 0], waist_error_abs[:, 1], waist_error_abs[:, 2],
            self.carry_waist_excess[:, 0], self.carry_waist_excess[:, 1],
            self.carry_waist_excess[:, 2],
            self.carry_torso_pelvis_alignment_reward,
            torso_rotvec_abs[:, 0], torso_rotvec_abs[:, 1],
            torso_rotvec_abs[:, 2],
            self.carry_torso_pelvis_alignment_excess.norm(dim=-1),
            self.carry_torso_pelvis_rotvec_error.norm(dim=-1),
        ), dim=-1)
        self.carry_lower_log_steps += 1

    def _reward_carry_hand_box_surface(self):
        c = self.cfg.rewards
        half = 0.5 * self._box_size[:, None, :]
        # Left +Y / right -Y, with freedom along each actual box face.
        side_error = (self.carry_hand_side_sign * self.hand_box[..., 1] - half[..., 1]).abs()
        side = (side_error - c.carry_hand_side_tolerance).clamp_min(0.0)
        face = (self.hand_box[..., [0, 2]].abs() - half[..., [0, 2]]
                - c.carry_hand_face_margin).clamp_min(0.0)
        excess = torch.cat((side.unsqueeze(-1), face), dim=-1)
        self.carry_log_sums[:, 1] += excess.norm(dim=-1).mean(-1)
        return constraint_reward(excess / c.carry_hand_surface_violation_scale)

    def _reward_carry_hand_slip(self):
        c = self.cfg.rewards
        excess = (self.hand_slip - c.carry_hand_slip_tolerance).clamp_min(0.0)
        self.carry_log_sums[:, 2] += self.hand_slip.mean(-1)
        return constraint_reward(excess / c.carry_hand_slip_violation_scale) * self.carry_history_valid

    def _reward_carry_relative_velocity(self):
        excess = (self.box_relative_velocity.abs() - self.carry_box_relative_velocity_tolerance).clamp_min(0.0)
        self.carry_log_sums[:, 5] += self.box_relative_velocity.norm(dim=-1)
        return constraint_reward(excess / self.carry_box_velocity_violation_scale) * self.carry_history_valid

    def _reward_carry_relative_position(self):
        excess = range_violation(self.box_relative_pos,
                                 self.carry_box_relative_position_lower, self.carry_box_relative_position_upper)
        self.carry_log_sums[:, 4] += excess.norm(dim=-1)
        return constraint_reward(excess / self.carry_box_position_violation_scale)

    def _reward_carry_box_tilt(self):
        return torch.sum(torch.square(self.projected_gravity_box[:, :2]), dim=-1)

    def _reward_carry_arm_range(self):
        excess = range_violation(self.dof_pos[:, self.carry_arm_indices],
                                 self.carry_arm_range_lower, self.carry_arm_range_upper)
        self.carry_log_sums[:, 3] += excess.amax(-1)
        return constraint_reward(excess / self.cfg.rewards.carry_arm_violation_scale)

    def _reward_carry_hip_posture(self):
        return deadzone_gaussian_reward(
            self.carry_hip_error, self.carry_hip_deadzone,
            self.carry_hip_softness,
        )

    def _reward_carry_waist_reference(self):
        return self.carry_waist_reference_reward

    def _reward_carry_torso_pelvis_alignment(self):
        return self.carry_torso_pelvis_alignment_reward

    def _reward_carry_foot_heading(self):
        return deadzone_gaussian_reward(
            self.carry_foot_heading_error,
            self.carry_foot_heading_deadzone,
            self.carry_foot_heading_softness,
        )

    def _reward_carry_feet_width(self):
        c = self.cfg.rewards
        return interval_gaussian_reward(
            self.carry_feet_width,
            self.carry_feet_width_range[0], self.carry_feet_width_range[1],
            c.carry_feet_width_softness,
        )

    def _reward_carry_knee_width(self):
        c = self.cfg.rewards
        return interval_gaussian_reward(
            self.carry_knee_width,
            self.carry_knee_width_range[0], self.carry_knee_width_range[1],
            c.carry_knee_width_softness,
        )

    def _reward_carry_leg_range(self):
        excess = range_violation(self.dof_pos[:, self.carry_leg_indices],
                                 self.carry_leg_range_lower, self.carry_leg_range_upper)
        return constraint_reward(excess / self.carry_leg_violation_scale)

    def _reward_carry_stance_width(self):
        # Compatibility alias for old configs. V1 sets this scale to zero so it
        # cannot double-count the new two-sided feasible interval.
        return self._reward_carry_feet_width()

    def _reward_zero_command_stillness(self):
        stillness = torch.exp(
            -4.0 * torch.sum(torch.square(self.base_lin_vel_yaw[:, :2]), dim=-1)
            -2.0 * torch.square(self.base_yaw_rate_world)
        )
        return stillness * ~self._command_is_moving()
