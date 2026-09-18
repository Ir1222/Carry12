"""Carry-only loaded-locomotion specialist for the Unitree G1."""

import math

import numpy as np
import torch

from isaacgym.torch_utils import (
    quat_conjugate, quat_mul, quat_rotate_inverse, torch_rand_float,
)
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.math import wrap_to_pi

from .carrybox import LeggedRobot as CarryBoxBase


def quaternion_log(q):
    """Shortest SO(3) rotation vector for an XYZW quaternion (q and -q agree)."""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    q = torch.where(q[..., 3:4] < 0.0, -q, q)
    length = q[..., :3].norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(length, q[..., 3:4].clamp_min(0.0))
    factor = torch.where(length > 1.0e-7, angle / length.clamp_min(1.0e-12),
                         torch.full_like(length, 2.0))
    return q[..., :3] * factor


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
        if not math.isclose(self.dt, rewards.carry_reference_policy_dt, abs_tol=1e-8):
            raise ValueError("Recalibrate carry motion widths for the changed policy dt")

        # Small calibrated constants live in the task config; no analysis files
        # or calibration objects are needed by the training environment.
        self.carry_hand_directions = self.dof_pos.new_tensor([
            rewards.carry_hand_direction_left, rewards.carry_hand_direction_right,
        ])
        self.carry_hand_side_sign = self.dof_pos.new_tensor([1.0, -1.0])
        self.carry_hand_normal_sigma = self.dof_pos.new_tensor(rewards.carry_hand_normal_sigma)
        self.carry_hand_direction_sigma = self.dof_pos.new_tensor(rewards.carry_hand_direction_sigma)
        self.carry_arm_target = self.dof_pos.new_tensor(rewards.carry_arm_target)
        self.carry_arm_sigma = self.dof_pos.new_tensor(rewards.carry_arm_sigma)
        self.carry_box_relative_position_target = self.dof_pos.new_tensor(
            rewards.carry_box_relative_position_target)
        self.carry_box_relative_position_sigma = self.dof_pos.new_tensor(
            rewards.carry_box_relative_position_sigma)
        self.carry_box_relative_orientation_target = self.dof_pos.new_tensor(
            rewards.carry_box_relative_orientation_target)
        self.carry_box_relative_orientation_sigma = self.dof_pos.new_tensor(
            rewards.carry_box_relative_orientation_sigma)
        self.carry_box_relative_velocity_sigma = self.dof_pos.new_tensor(
            rewards.carry_box_relative_velocity_sigma)

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
        self.carry_previous_relative_pos = torch.zeros(
            self.num_envs, 3, device=self.device, dtype=self.dof_pos.dtype
        )
        self.carry_history_valid = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.carry_error_sums = {
            name: self.dof_pos.new_zeros(self.num_envs) for name in (
                "left_hand_side_error_m", "right_hand_side_error_m",
                "box_relative_position_error_m", "box_relative_orientation_error_rad",
                "box_relative_motion_error_mps", "arm_reference_error_rad",
            )
        }
        # Count actual samples: the runner randomizes episode_length_buf at startup.
        self.carry_error_steps = self.dof_pos.new_zeros(self.num_envs)
        self.carry_motion_samples = self.dof_pos.new_zeros(self.num_envs)

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
        for name, sums in self.carry_error_sums.items():
            counts = (self.carry_motion_samples if name == "box_relative_motion_error_mps"
                      else self.carry_error_steps)
            count = counts[env_ids].sum()
            self.extras["episode"]["carry/" + name] = torch.where(
                count > 0, sums[env_ids].sum() / count.clamp_min(1),
                torch.full_like(count, float("nan")),
            )
            sums[env_ids] = 0.0
        self.carry_error_steps[env_ids] = 0.0
        self.carry_motion_samples[env_ids] = 0.0
        self.carry_previous_relative_pos[env_ids] = 0.0
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
        return torch.all(self.hand_contact_filt, dim=-1).to(torch.float)

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        # CarryBoxBase retains extras on non-reset steps, and the locomotion
        # runner appends every present "episode" entry. Clear it once per step
        # so a completed episode is logged only once (as before this refactor).
        self.extras.pop("episode", None)
        self.carry_error_steps += 1

    def _reward_carry_hand_box_surface(self):
        hand_from_box = (self.rigid_body_states[:, self.carry_palm_indices, :3]
                         - self.box_states[:, None, :3])
        # Isaac Gym's inverse rotation takes flat batches of vectors/quaternions.
        box_quat = self.box_states[:, None, 3:7].expand(-1, 2, -1).reshape(-1, 4)
        torso_quat = self.rigid_body_states[:, self.carry_torso_index, 3:7]
        torso_quat = torso_quat[:, None, :].expand(-1, 2, -1).reshape(-1, 4)
        hand_box = quat_rotate_inverse(box_quat, hand_from_box.reshape(-1, 3)).reshape(-1, 2, 3)
        hand_torso = quat_rotate_inverse(torso_quat, hand_from_box.reshape(-1, 3)).reshape(-1, 2, 3)

        # Left = +Y face, right = -Y face, using each randomized box's dimensions.
        half = 0.5 * self._box_size[:, None, :]
        normal_error = self.carry_hand_side_sign * hand_box[..., 1] - half[..., 1]
        overflow = (hand_box[..., [0, 2]].abs() - half[..., [0, 2]]).clamp_min(0.0)

        # CarryWith box-to-palm rays are expressed in the torso frame.
        length = hand_torso.norm(dim=-1)
        direction = hand_torso / length.unsqueeze(-1).clamp_min(1.0e-9)
        desired = self.carry_hand_directions.unsqueeze(0).expand_as(direction)
        cross = torch.cross(direction, desired, dim=-1).norm(dim=-1)
        dot = (direction * desired).sum(dim=-1)
        direction_error = torch.atan2(cross, dot)
        direction_error = torch.where(length > 1.0e-9, direction_error,
                                      torch.full_like(direction_error, math.pi))
        energy = (normal_error / self.carry_hand_normal_sigma).square()
        energy += (overflow / self.carry_hand_normal_sigma[None, :, None]).square().sum(-1)
        energy += (direction_error / self.carry_hand_direction_sigma).square()
        self.carry_error_sums["left_hand_side_error_m"] += normal_error[:, 0].abs()
        self.carry_error_sums["right_hand_side_error_m"] += normal_error[:, 1].abs()
        return torch.exp(-0.25 * energy.sum(-1))

    def _reward_carry_relative_velocity(self):
        torso = self.rigid_body_states[:, self.carry_torso_index]
        relative_pos = quat_rotate_inverse(
            torso[:, 3:7], self.box_states[:, :3] - torso[:, :3])
        # Differentiate in the rotating torso frame, including during yaw motion.
        velocity = (relative_pos - self.carry_previous_relative_pos) / self.dt
        velocity = torch.where(
            self.carry_history_valid[:, None], velocity, torch.zeros_like(velocity))
        reward = torch.exp(
            -0.5 * (velocity / self.carry_box_relative_velocity_sigma).square().sum(-1))
        reward *= self.carry_history_valid  # No fabricated derivative after reset.
        self.carry_error_sums["box_relative_motion_error_mps"] += velocity.norm(dim=-1)
        self.carry_motion_samples += self.carry_history_valid
        # The parent's reward dispatch calls this stateful term once per step.
        self.carry_previous_relative_pos.copy_(relative_pos)
        self.carry_history_valid[:] = True
        return reward

    def _reward_carry_relative_position(self):
        torso = self.rigid_body_states[:, self.carry_torso_index]
        relative_pos = quat_rotate_inverse(
            torso[:, 3:7], self.box_states[:, :3] - torso[:, :3])
        error = relative_pos - self.carry_box_relative_position_target
        self.carry_error_sums["box_relative_position_error_m"] += error.norm(dim=-1)
        return torch.exp(
            -0.5 * (error / self.carry_box_relative_position_sigma).square().sum(-1))

    def _reward_carry_relative_orientation(self):
        torso_quat = self.rigid_body_states[:, self.carry_torso_index, 3:7]
        relative_quat = quat_mul(quat_conjugate(torso_quat), self.box_states[:, 3:7])
        target = self.carry_box_relative_orientation_target.expand_as(relative_quat)
        error = quaternion_log(quat_mul(quat_conjugate(target), relative_quat))
        self.carry_error_sums["box_relative_orientation_error_rad"] += error.norm(dim=-1)
        # Axis-wise SO(3) widths preserve the deliberately larger yaw tolerance.
        return torch.exp(
            -0.5 * (error / self.carry_box_relative_orientation_sigma).square().sum(-1))

    def _reward_carry_box_tilt(self):
        return torch.sum(torch.square(self.projected_gravity_box[:, :2]), dim=-1)

    def _reward_carry_arm_pose(self):
        # Exactly the 14 named arm joints; waist and legs remain unsupervised.
        error = self.dof_pos[:, self.carry_arm_indices] - self.carry_arm_target
        self.carry_error_sums["arm_reference_error_rad"] += error.square().mean(-1).sqrt()
        return torch.exp(-0.5 * (error / self.carry_arm_sigma).square().mean(-1))

    def _reward_zero_command_stillness(self):
        stillness = torch.exp(
            -4.0 * torch.sum(torch.square(self.base_lin_vel_yaw[:, :2]), dim=-1)
            -2.0 * torch.square(self.base_yaw_rate_world)
        )
        return stillness * ~self._command_is_moving()
