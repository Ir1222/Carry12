"""Carry-only loaded-locomotion specialist for the Unitree G1."""

import math

import numpy as np
import torch

from isaacgym.torch_utils import (
    quat_rotate_inverse, torch_rand_float,
)
from legged_gym.carry_constraint_metrics import METRIC_NAMES, TEMPORAL_METRICS
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.math import wrap_to_pi

from .carrybox import LeggedRobot as CarryBoxBase


def range_violation(value, lower, upper):
    """No center preference anywhere inside the closed feasible interval."""
    return (lower - value).clamp_min(0.0) + (value - upper).clamp_min(0.0)


def constraint_reward(normalized_violation):
    """C1 dead-zone reward: 1/(1+sum(excess/softness)^2), no Gaussian."""
    return 1.0 / (1.0 + normalized_violation.flatten(1).square().sum(-1))


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
        # Static engineering regions only. No motion/calibration lookup at runtime.
        self.carry_hand_side_sign = self.dof_pos.new_tensor([1.0, -1.0])
        for name, shape in (
            ("carry_arm_range_lower", (14,)), ("carry_arm_range_upper", (14,)),
            ("carry_box_relative_position_lower", (3,)),
            ("carry_box_relative_position_upper", (3,)),
            ("carry_box_position_violation_scale", (3,)),
            ("carry_box_relative_velocity_tolerance", (3,)),
            ("carry_box_velocity_violation_scale", (3,)),
        ):
            value = self.dof_pos.new_tensor(getattr(rewards, name))
            if value.shape != shape or not bool(torch.isfinite(value).all()):
                raise ValueError("Invalid carry constraint: " + name)
            if name.endswith(("_scale", "_tolerance")) and not bool((value > 0).all()):
                raise ValueError("Carry scales/tolerances must be positive: " + name)
            setattr(self, name, value)
        for prefix in ("carry_arm_range", "carry_box_relative_position"):
            if not bool((getattr(self, prefix + "_lower") < getattr(self, prefix + "_upper")).all()):
                raise ValueError("Carry range must be ordered: " + prefix)
        for name in ("carry_hand_side_tolerance", "carry_hand_face_margin",
                     "carry_hand_surface_violation_scale", "carry_hand_slip_tolerance",
                     "carry_hand_slip_violation_scale", "carry_arm_violation_scale"):
            value = getattr(rewards, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Carry tolerance/scale must be positive: " + name)
        expected_arms = [side + "_" + joint + "_joint"
                         for side in ("left", "right")
                         for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw",
                                       "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
        if list(rewards.carry_arm_joint_names) != expected_arms:
            raise ValueError("Carry guardrail requires exactly the 14 arm joints")
        if list(rewards.carry_hand_links) != ["left_palm_link", "right_palm_link"]:
            raise ValueError("Carry hands must be ordered left, right")

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
        self.carry_previous_hand_box = self.dof_pos.new_zeros(self.num_envs, 2, 3)
        self.carry_motion_metric_valid = self.carry_history_valid.clone()
        self.carry_metrics = {name: self.dof_pos.new_zeros(self.num_envs) for name in METRIC_NAMES}
        self.carry_error_sums = {name: torch.zeros_like(value) for name, value in self.carry_metrics.items()}
        # Count actual samples, independent of randomized episode_length_buf.
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
            counts = (self.carry_motion_samples if name in TEMPORAL_METRICS
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
        self.carry_previous_hand_box[env_ids] = 0.0
        self.carry_motion_metric_valid[env_ids] = False
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

    def compute_reward(self):
        # Parent calls once per policy step, after contact filtering/termination and
        # before reset. Sample history once even if individual reward scales are 0.
        self._update_carry_constraints()
        super().compute_reward()

    def _update_carry_constraints(self):
        c = self.cfg.rewards
        torso = self.rigid_body_states[:, self.carry_torso_index]
        relative_pos = quat_rotate_inverse(torso[:, 3:7], self.box_states[:, :3] - torso[:, :3])
        offset = self.rigid_body_states[:, self.carry_palm_indices, :3] - self.box_states[:, None, :3]
        box_quat = self.box_states[:, None, 3:7].expand(-1, 2, -1).reshape(-1, 4)
        hand_box = quat_rotate_inverse(box_quat, offset.reshape(-1, 3)).reshape(-1, 2, 3)
        half = 0.5 * self._box_size[:, None, :]
        side_error = (self.carry_hand_side_sign * hand_box[..., 1] - half[..., 1]).abs()
        self.carry_side_violation = (side_error - c.carry_hand_side_tolerance).clamp_min(0.0)
        face_overflow = (hand_box[..., [0, 2]].abs() - half[..., [0, 2]]).clamp_min(0.0)
        self.carry_face_violation = (face_overflow - c.carry_hand_face_margin).clamp_min(0.0)

        valid = self.carry_history_valid
        velocity = (relative_pos - self.carry_previous_relative_pos) / self.dt
        velocity = torch.where(valid[:, None], velocity, torch.zeros_like(velocity))
        hand_velocity = (hand_box - self.carry_previous_hand_box) / self.dt
        hand_velocity = torch.where(valid[:, None, None], hand_velocity, torch.zeros_like(hand_velocity))
        slip = hand_velocity[..., [0, 2]].norm(dim=-1)
        self.carry_slip_violation = (slip - c.carry_hand_slip_tolerance).clamp_min(0.0)
        self.carry_velocity_violation = (velocity.abs() - self.carry_box_relative_velocity_tolerance).clamp_min(0.0)
        self.carry_position_violation = range_violation(
            relative_pos, self.carry_box_relative_position_lower, self.carry_box_relative_position_upper)
        arms = self.dof_pos[:, self.carry_arm_indices]
        self.carry_arm_violation = range_violation(arms, self.carry_arm_range_lower, self.carry_arm_range_upper)
        arm_clearance = torch.minimum(arms - self.carry_arm_range_lower, self.carry_arm_range_upper - arms).amin(-1)
        position_clearance = torch.minimum(relative_pos - self.carry_box_relative_position_lower,
                                           self.carry_box_relative_position_upper - relative_pos).amin(-1)

        m = self.carry_metrics
        for i, side in enumerate(("left", "right")):
            m[side + "_hand_side_error_m"] = side_error[:, i]
            m[side + "_hand_side_violation_m"] = self.carry_side_violation[:, i]
            m[side + "_hand_face_overflow_m"] = face_overflow[:, i].norm(dim=-1)
            m[side + "_hand_face_violation_m"] = self.carry_face_violation[:, i].norm(dim=-1)
            m[side + "_hand_tangential_slip_mps"] = slip[:, i]
            m[side + "_hand_slip_violation_mps"] = self.carry_slip_violation[:, i]
        m["arm_range_violation_rad"] = self.carry_arm_violation.amax(-1)
        m["arm_range_clearance_rad"] = arm_clearance
        m["box_relative_region_violation_m"] = self.carry_position_violation.norm(dim=-1)
        m["box_relative_region_clearance_m"] = position_clearance
        m["box_relative_motion_error_mps"] = velocity.norm(dim=-1)
        m["box_relative_velocity_violation_mps"] = self.carry_velocity_violation.norm(dim=-1)
        m["bilateral_contact_rate"] = self._reward_carry_bilateral_contact()
        for i, axis in enumerate(("x", "y", "z")):
            m["box_relative_position_" + axis + "_m"] = relative_pos[:, i]
            m["box_relative_velocity_" + axis + "_mps"] = velocity[:, i]
        for name, value in m.items():
            self.carry_error_sums[name] += value * valid if name in TEMPORAL_METRICS else value
        self.carry_error_steps += 1
        self.carry_motion_samples += valid
        self.carry_motion_metric_valid.copy_(valid)
        self.carry_previous_relative_pos.copy_(relative_pos)
        self.carry_previous_hand_box.copy_(hand_box)
        self.carry_history_valid[:] = True

    def _reward_carry_hand_box_surface(self):
        scale = self.cfg.rewards.carry_hand_surface_violation_scale
        excess = torch.cat((self.carry_side_violation.unsqueeze(-1), self.carry_face_violation), dim=-1)
        return constraint_reward(excess / scale)

    def _reward_carry_hand_slip(self):
        # Invalid post-reset derivatives neither earn a reward nor enter diagnostics.
        return constraint_reward(self.carry_slip_violation / self.cfg.rewards.carry_hand_slip_violation_scale) * self.carry_motion_metric_valid

    def _reward_carry_relative_velocity(self):
        return constraint_reward(self.carry_velocity_violation / self.carry_box_velocity_violation_scale) * self.carry_motion_metric_valid

    def _reward_carry_relative_position(self):
        return constraint_reward(self.carry_position_violation / self.carry_box_position_violation_scale)

    def _reward_carry_box_tilt(self):
        return torch.sum(torch.square(self.projected_gravity_box[:, :2]), dim=-1)

    def _reward_carry_arm_range(self):
        # Sum violations so a single badly twisted joint is not diluted by 13 valid ones.
        return constraint_reward(self.carry_arm_violation / self.cfg.rewards.carry_arm_violation_scale)

    def _reward_zero_command_stillness(self):
        stillness = torch.exp(
            -4.0 * torch.sum(torch.square(self.base_lin_vel_yaw[:, :2]), dim=-1)
            -2.0 * torch.square(self.base_yaw_rate_world)
        )
        return stillness * ~self._command_is_moving()
