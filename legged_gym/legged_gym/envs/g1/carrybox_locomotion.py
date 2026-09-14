"""Carry-only loaded-locomotion specialist for the Unitree G1."""

import math

import numpy as np
import torch

from isaacgym.torch_utils import quat_rotate_inverse, torch_rand_float

from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.math import wrap_to_pi

from .carrybox import LeggedRobot as CarryBoxBase


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

        carry_start = int(
            self.motionlib.motion_start_ids[self._CARRY_SKILL][0].item()
        )
        carry_end = int(
            self.motionlib.motion_end_ids[self._CARRY_SKILL][-1].item()
        )
        self.carry_ref_dof_pos = torch.median(
            self.motionlib.motion_dof_pos[carry_start:carry_end], dim=0
        ).values

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

    def _reward_carry_hand_box_surface(self):
        hand_pos_world = self.rigid_body_states[
            :, self.hand_colli_indices, :3
        ]
        hand_from_box = hand_pos_world - self.box_states[:, None, :3]
        box_quat = self.box_states[:, None, 3:7].expand(
            -1, hand_from_box.shape[1], -1
        )
        hand_pos_box = quat_rotate_inverse(
            box_quat.reshape(-1, 4), hand_from_box.reshape(-1, 3)
        ).reshape_as(hand_from_box)
        half_size = 0.5 * self._box_size[:, None, :]

        signed_face_distance = torch.abs(hand_pos_box) - half_size
        outside_distance = torch.norm(
            torch.clamp(signed_face_distance, min=0.0), dim=-1
        )
        inside_distance = torch.min(
            half_size - torch.abs(hand_pos_box), dim=-1
        ).values
        is_outside = torch.any(signed_face_distance > 0.0, dim=-1)
        surface_distance = torch.where(
            is_outside, outside_distance, inside_distance
        )
        sigma = self.cfg.rewards.carry_hand_surface_sigma
        return torch.exp(-torch.square(surface_distance / sigma)).mean(dim=-1)

    def _reward_carry_relative_velocity(self):
        robot_vel_xy = self.rigid_body_states[
            :, self.upper_body_index, 7:9
        ]
        error = torch.sum(
            torch.square(self.box_states[:, 7:9] - robot_vel_xy), dim=-1
        )
        return torch.exp(-5.0 * error)

    def _reward_carry_relative_position(self):
        reward = torch.exp(-0.5 * self.robot2object_dist)
        reward[
            self.robot2object_dist < self.cfg.rewards.thresh_robot2object
        ] = 1.0
        return reward

    def _reward_carry_box_tilt(self):
        return torch.sum(torch.square(self.projected_gravity_box[:, :2]), dim=-1)

    def _reward_carry_upper_body_pose(self):
        indices = torch.cat((self.arm_joint_indices, self.waist_joint_indices))
        error = self.dof_pos[:, indices] - self.carry_ref_dof_pos[indices]
        return torch.sum(torch.square(error), dim=-1)

    def _reward_zero_command_stillness(self):
        stillness = torch.exp(
            -4.0 * torch.sum(torch.square(self.base_lin_vel_yaw[:, :2]), dim=-1)
            -2.0 * torch.square(self.base_yaw_rate_world)
        )
        return stillness * ~self._command_is_moving()
