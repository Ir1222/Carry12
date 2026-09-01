"""Formal Stage2A external-force fine-tuning for the current CarryBox task."""

import math

import numpy as np
import torch

from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_rotate

from legged_gym.utils.math import wrap_to_pi

from .carrybox import LeggedRobot as CarryBox
from .carrybox_force_utils import (
    compute_admittance_teacher,
    resolve_directional_beta_ranges,
    sample_directional_beta,
    smooth_force_profile,
)


class LeggedRobot(CarryBox):
    """CarryBox with box-COM force events and reward-only admittance targets."""

    FORCE_PHASE_IDLE = 0
    FORCE_PHASE_RAMP_UP = 1
    FORCE_PHASE_HOLD = 2
    FORCE_PHASE_RAMP_DOWN = 3
    FORCE_PHASE_DONE = 4
    _PHASE_NAMES = ("idle", "ramp_up", "hold", "ramp_down", "done")
    _DIRECTION_LOCAL = {
        "+box_x": (1.0, 0.0, 0.0),
        "-box_x": (-1.0, 0.0, 0.0),
        "+box_y": (0.0, 1.0, 0.0),
        "-box_y": (0.0, -1.0, 0.0),
    }

    def _create_envs(self):
        super()._create_envs()
        box_body_count = len(
            self.gym.get_actor_rigid_body_properties(self.envs[0], self.box_handles[0])
        )
        if box_body_count != 1:
            raise AssertionError(f"Expected one box rigid body, got {box_body_count}")
        self.box_rigid_body_index = self.gym.get_actor_rigid_body_handle(
            self.envs[0], self.box_handles[0], 0
        )

    def _init_buffers(self):
        super()._init_buffers()
        n = self.num_envs
        device = self.device

        # Physics-only force state. The force tensor addresses the box body and
        # is never concatenated into actor or critic observations.
        self.external_force_tensor = torch.zeros_like(self.disturbance)
        self.external_force_world = torch.zeros((n, 3), device=device)
        self.external_force_direction_world = torch.zeros((n, 3), device=device)
        self.external_force_active = torch.zeros(n, dtype=torch.bool, device=device)
        self.external_force_scale = torch.zeros(n, device=device)
        self.external_force_peak_N = torch.zeros(n, device=device)
        self.external_force_beta = torch.zeros(n, device=device)
        self.external_force_box_mass = torch.zeros(n, device=device)

        # Independent force-event state machine (physics-substep timing).
        self.force_phase = torch.full((n,), self.FORCE_PHASE_IDLE, dtype=torch.long, device=device)
        self.force_event_count = torch.zeros(n, dtype=torch.long, device=device)
        self.force_event_decision_made = torch.zeros(n, dtype=torch.bool, device=device)
        self.force_stable_carry_streak = torch.zeros(n, dtype=torch.long, device=device)
        self.force_elapsed_physics_steps = torch.zeros(n, dtype=torch.long, device=device)
        self.force_remaining_physics_steps = torch.zeros(n, dtype=torch.long, device=device)
        self.force_ramp_up_steps = torch.zeros(n, dtype=torch.long, device=device)
        self.force_hold_steps = torch.zeros(n, dtype=torch.long, device=device)
        self.force_ramp_down_steps = torch.zeros(n, dtype=torch.long, device=device)
        self.force_hold_duration_s = torch.zeros(n, device=device)

        # Reward-only teacher state. Actor-visible nominal command buffers remain
        # owned and updated by the Stage1 CarryBox implementation.
        self.teacher_vx = torch.zeros(n, device=device)
        self.teacher_heading = torch.zeros(n, device=device)
        self.teacher_heading_error = torch.zeros(n, device=device)
        self.teacher_yaw_rate = torch.zeros(n, device=device)
        self.force_parallel = torch.zeros(n, device=device)
        self.force_perp = torch.zeros(n, device=device)

        self._force_start_pending = torch.zeros(n, dtype=torch.bool, device=device)
        self._force_hold_pending = torch.zeros(n, dtype=torch.bool, device=device)
        self._force_end_pending = torch.zeros(n, dtype=torch.bool, device=device)

        if self.external_force_tensor.shape != self.contact_forces.shape:
            raise AssertionError(
                "External force tensor must match Isaac Gym rigid-body force tensor shape: "
                f"{self.external_force_tensor.shape} != {self.contact_forces.shape}"
            )
        if not 0 <= int(self.box_rigid_body_index) < self.external_force_tensor.shape[1]:
            raise AssertionError(f"Invalid box rigid-body index: {self.box_rigid_body_index}")
        self._validate_external_force_config()
        self.debug_viz = bool(self.cfg.external_force.debug_draw_force)
        self._set_teacher_to_nominal()

    def _validate_external_force_config(self):
        cfg = self.cfg.external_force
        unknown = set(cfg.force_directions) - set(self._DIRECTION_LOCAL)
        if unknown:
            raise ValueError(f"Unsupported Stage2A force directions: {sorted(unknown)}")
        if len(cfg.force_directions) == 0:
            raise ValueError("force_directions must not be empty")
        if not 0.0 <= float(cfg.force_event_probability) <= 1.0:
            raise ValueError("force_event_probability must be in [0, 1]")
        resolve_directional_beta_ranges(
            cfg.curriculum_beta_ranges,
            cfg.curriculum_stage,
            cfg.force_directions,
        )
        if cfg.beta_range is not None and not (
            0.0 <= float(cfg.beta_range[0]) <= float(cfg.beta_range[1])
        ):
            raise ValueError("beta_range override must be non-negative and ordered")
        if not (0.0 <= float(cfg.force_hold_duration_range_s[0]) <= float(cfg.force_hold_duration_range_s[1])):
            raise ValueError("force_hold_duration_range_s must be non-negative and ordered")
        if float(cfg.force_ramp_up_duration_s) < 0.0 or float(cfg.force_ramp_down_duration_s) < 0.0:
            raise ValueError("Force ramp durations must be non-negative")
        if float(cfg.admittance_D_bar) <= 0.0:
            raise ValueError("admittance_D_bar must be positive")
        if int(cfg.max_force_events_per_episode) < 0:
            raise ValueError("max_force_events_per_episode must be non-negative")

    def step(self, actions):
        # This exact parent path is important for disabled-force dynamics parity.
        if not bool(self.cfg.external_force.enable_external_force):
            return super().step(actions)

        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(self.torques)
            )
            self._apply_external_force_substep()
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

        termination_ids, termination_privileged_obs, amp_obs_buf = self.post_physics_step()
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(
                self.privileged_obs_buf, -clip_obs, clip_obs
            )
        return (
            self.obs_buf,
            self.privileged_obs_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
            termination_ids,
            termination_privileged_obs,
            amp_obs_buf,
        )

    def _apply_external_force_substep(self):
        """Apply the current smooth-profile force at box COM before each simulate call."""
        self.external_force_tensor.zero_()
        self.external_force_world.zero_()
        self.external_force_active.zero_()
        self.external_force_scale.zero_()

        active = self.force_remaining_physics_steps > 0
        if not torch.any(active):
            return

        active_ids = torch.nonzero(active, as_tuple=False).flatten()
        elapsed_s = (
            self.force_elapsed_physics_steps[active_ids].float() + 0.5
        ) * float(self.sim_params.dt)
        ramp_up_s = self.force_ramp_up_steps[active_ids].float() * float(self.sim_params.dt)
        hold_s = self.force_hold_steps[active_ids].float() * float(self.sim_params.dt)
        ramp_down_s = self.force_ramp_down_steps[active_ids].float() * float(self.sim_params.dt)
        scale = smooth_force_profile(elapsed_s, ramp_up_s, hold_s, ramp_down_s)
        force = (
            scale.unsqueeze(-1)
            * self.external_force_peak_N[active_ids].unsqueeze(-1)
            * self.external_force_direction_world[active_ids]
        )
        self.external_force_tensor[
            active_ids, int(self.box_rigid_body_index), :
        ] = force
        self.external_force_world[active_ids] = force
        self.external_force_scale[active_ids] = scale
        self.external_force_active[active_ids] = scale > 1.0e-9

        ramp_up_end = self.force_ramp_up_steps[active_ids]
        hold_end = ramp_up_end + self.force_hold_steps[active_ids]
        elapsed_steps = self.force_elapsed_physics_steps[active_ids]
        phase = torch.where(
            elapsed_steps < ramp_up_end,
            torch.full_like(elapsed_steps, self.FORCE_PHASE_RAMP_UP),
            torch.where(
                elapsed_steps < hold_end,
                torch.full_like(elapsed_steps, self.FORCE_PHASE_HOLD),
                torch.full_like(elapsed_steps, self.FORCE_PHASE_RAMP_DOWN),
            ),
        )
        entering_hold = (phase == self.FORCE_PHASE_HOLD) & (
            self.force_phase[active_ids] != self.FORCE_PHASE_HOLD
        )
        self._force_hold_pending[active_ids[entering_hold]] = True
        self.force_phase[active_ids] = phase

        # Isaac Gym's rigid-body force tensor API applies at each body COM and
        # therefore introduces no explicit external torque in this phase.
        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            forceTensor=gymtorch.unwrap_tensor(self.external_force_tensor),
            space=gymapi.CoordinateSpace.GLOBAL_SPACE,
        )

        self.force_elapsed_physics_steps[active_ids] += 1
        self.force_remaining_physics_steps[active_ids] -= 1
        ended = self.force_remaining_physics_steps[active_ids] == 0
        ended_ids = active_ids[ended]
        if len(ended_ids) > 0:
            self.force_phase[ended_ids] = self.FORCE_PHASE_DONE
            self._force_end_pending[ended_ids] = True

    def _update_carry_heading_commands(self):
        if not bool(self.cfg.external_force.enable_external_force):
            super()._update_carry_heading_commands()
            self._set_teacher_to_nominal()
            return

        self._update_nominal_commands_with_force_resample_deferral()
        self._update_force_event_scheduler()
        self._update_teacher_targets()
        self._log_force_debug()

    def _update_nominal_commands_with_force_resample_deferral(self):
        """Stage1 heading controller with its resample timer paused during force."""
        self.carry_policy_commands[:, :3] = self.commands[:, :3]
        self.carry_policy_commands[:, 1] = 0.0
        self.carry_heading_error.zero_()

        entering_carry = self.is_stage_carry & ~self.carry_heading_initialized
        if entering_carry.any():
            if self.cfg.commands.resample_carry_commands:
                entering_env_ids = entering_carry.nonzero(as_tuple=False).flatten()
                self._resample_carry_commands(entering_env_ids)
                self._sample_carry_command_resample_time(entering_env_ids)
                self.carry_policy_commands[entering_env_ids, :3] = self.commands[entering_env_ids, :3]
                self.carry_policy_commands[entering_env_ids, 1] = 0.0
            self.carry_heading_ref[entering_carry] = self.yaw[entering_carry]
            self.carry_heading_initialized[entering_carry] = True

        force_in_progress = (self.force_remaining_physics_steps > 0) | self.external_force_active
        resample_mask = (
            self.is_stage_carry
            & self.carry_heading_initialized
            & ~entering_carry
            & ~force_in_progress
        )
        if self.cfg.commands.resample_carry_commands:
            self.carry_command_resample_time[resample_mask] -= self.dt
            due_env_ids = (
                resample_mask & (self.carry_command_resample_time <= 0.0)
            ).nonzero(as_tuple=False).flatten()
            if len(due_env_ids) > 0:
                self._resample_carry_commands(due_env_ids)
                self._sample_carry_command_resample_time(due_env_ids)
                self.carry_policy_commands[due_env_ids, :3] = self.commands[due_env_ids, :3]
                self.carry_policy_commands[due_env_ids, 1] = 0.0

        carry_mask = self.is_stage_carry & self.carry_heading_initialized
        self.carry_heading_ref[carry_mask] = wrap_to_pi(
            self.carry_heading_ref[carry_mask]
            + self.commands[carry_mask, 2] * self.dt
        )
        self.carry_heading_error[carry_mask] = wrap_to_pi(
            self.carry_heading_ref[carry_mask] - self.yaw[carry_mask]
        )
        self.carry_policy_commands[carry_mask, 2] = torch.clip(
            self.commands[carry_mask, 2]
            + self.cfg.commands.heading_kp * self.carry_heading_error[carry_mask],
            -self.cfg.commands.max_yaw_rate,
            self.cfg.commands.max_yaw_rate,
        )

    def _update_force_event_scheduler(self):
        cfg = self.cfg.external_force
        self.force_stable_carry_streak[:] = torch.where(
            self.is_stage_carry,
            self.force_stable_carry_streak + 1,
            torch.zeros_like(self.force_stable_carry_streak),
        )
        eligible = (
            (self.force_stable_carry_streak >= int(cfg.stable_carry_policy_steps))
            & ~self.force_event_decision_made
            & (self.force_event_count < int(cfg.max_force_events_per_episode))
            & (self.force_remaining_physics_steps == 0)
        )
        eligible_ids = torch.nonzero(eligible, as_tuple=False).flatten()
        if len(eligible_ids) == 0:
            return

        self.force_event_decision_made[eligible_ids] = True
        trigger = torch.rand(len(eligible_ids), device=self.device) < float(
            cfg.force_event_probability
        )
        trigger_ids = eligible_ids[trigger]
        if len(trigger_ids) > 0:
            self._schedule_force_event(trigger_ids)

    def _schedule_force_event(self, env_ids):
        cfg = self.cfg.external_force
        direction_names = tuple(cfg.force_directions)
        direction_ids = torch.randint(
            0, len(direction_names), (len(env_ids),), device=self.device
        )
        direction_local = torch.tensor(
            [self._DIRECTION_LOCAL[direction_names[int(i)]] for i in direction_ids.cpu().tolist()],
            dtype=torch.float,
            device=self.device,
        )
        direction_world = quat_rotate(self.box_states[env_ids, 3:7], direction_local)
        direction_world[:, 2] = 0.0
        direction_norm = torch.linalg.vector_norm(direction_world, dim=-1, keepdim=True)
        if torch.any(direction_norm <= 1.0e-6):
            raise RuntimeError("Box-frame force direction has a degenerate horizontal projection")
        direction_world = direction_world / direction_norm

        beta = sample_directional_beta(
            direction_names,
            direction_ids,
            cfg.curriculum_beta_ranges,
            cfg.curriculum_stage,
            beta_range=cfg.beta_range,
        )
        hold_duration_s = self._uniform_sample(
            cfg.force_hold_duration_range_s, len(env_ids)
        )
        ramp_up_steps = self._seconds_to_physics_steps(
            float(cfg.force_ramp_up_duration_s)
        )
        ramp_down_steps = self._seconds_to_physics_steps(
            float(cfg.force_ramp_down_duration_s)
        )
        hold_steps = torch.clamp(
            torch.round(hold_duration_s / float(self.sim_params.dt)).long(), min=1
        )

        mass = self.box_masses[env_ids]
        self.external_force_direction_world[env_ids] = direction_world
        self.external_force_beta[env_ids] = beta
        self.external_force_box_mass[env_ids] = mass
        self.external_force_peak_N[env_ids] = beta * mass * 9.81
        self.force_hold_duration_s[env_ids] = hold_steps.float() * float(self.sim_params.dt)
        self.force_ramp_up_steps[env_ids] = ramp_up_steps
        self.force_hold_steps[env_ids] = hold_steps
        self.force_ramp_down_steps[env_ids] = ramp_down_steps
        self.force_elapsed_physics_steps[env_ids] = 0
        self.force_remaining_physics_steps[env_ids] = (
            ramp_up_steps + hold_steps + ramp_down_steps
        )
        self.force_phase[env_ids] = self.FORCE_PHASE_RAMP_UP
        self.force_event_count[env_ids] += 1
        self._force_start_pending[env_ids] = True

    def _uniform_sample(self, value_range, count):
        low = float(value_range[0])
        high = float(value_range[1])
        return low + (high - low) * torch.rand(count, device=self.device)

    def _seconds_to_physics_steps(self, duration_s):
        if duration_s <= 0.0:
            return 0
        return max(1, int(round(duration_s / float(self.sim_params.dt))))

    def _set_teacher_to_nominal(self):
        if not hasattr(self, "teacher_vx"):
            return
        self.teacher_vx.copy_(self.carry_policy_commands[:, 0])
        self.teacher_heading.copy_(self.carry_heading_ref)
        self.teacher_heading_error.copy_(self.carry_heading_error)
        self.teacher_yaw_rate.copy_(self.carry_policy_commands[:, 2])
        self.force_parallel.zero_()
        self.force_perp.zero_()

    def _update_teacher_targets(self):
        teacher = compute_admittance_teacher(
            self.external_force_world,
            self.box_masses,
            self.carry_policy_commands[:, 0],
            self.carry_heading_ref,
            self.yaw,
            self.carry_heading_error,
            self.commands[:, 2],
            self.carry_policy_commands[:, 2],
            heading_kp=self.cfg.commands.heading_kp,
            max_yaw_rate=self.cfg.commands.max_yaw_rate,
            admittance_d_bar=self.cfg.external_force.admittance_D_bar,
            max_heading_offset=self.cfg.external_force.max_teacher_heading_offset_rad,
            teacher_vx_min=self.cfg.external_force.teacher_vx_min,
            teacher_vx_max=self.cfg.external_force.teacher_vx_max,
        )
        self.teacher_vx.copy_(teacher[0])
        self.teacher_heading.copy_(teacher[1])
        self.teacher_heading_error.copy_(teacher[2])
        self.teacher_yaw_rate.copy_(teacher[3])
        self.force_parallel.copy_(teacher[4])
        self.force_perp.copy_(teacher[5])

    def _reward_carry_velocity_task(self):
        desired_heading_dir = torch.stack(
            (torch.cos(self.teacher_heading), torch.sin(self.teacher_heading)), dim=-1
        )
        desired_world_lin_vel_xy = self.teacher_vx.unsqueeze(-1) * desired_heading_dir
        actual_world_lin_vel_xy = self.rigid_body_states[:, self.upper_body_index, 7:9]
        lin_vel_error = torch.sum(
            torch.square(desired_world_lin_vel_xy - actual_world_lin_vel_xy), dim=1
        )
        lin_vel_reward = torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

        yaw_vel_error = torch.square(self.teacher_yaw_rate - self.base_ang_vel[:, 2])
        yaw_vel_reward = torch.exp(-yaw_vel_error / self.cfg.rewards.tracking_sigma)
        carry_reward = (
            self.cfg.rewards.carry_lin_vel * lin_vel_reward
            + self.cfg.rewards.carry_yaw_vel * yaw_vel_reward
        )
        carry_reward[~self.is_stage_carry] = 0.0
        return carry_reward

    def _reward_carry_heading_hold(self):
        heading_alignment = torch.exp(
            -torch.square(self.teacher_heading_error)
            / self.cfg.rewards.carry_heading_sigma
        )
        abs_heading_error = torch.abs(self.teacher_heading_error)
        heading_huber_delta = self.cfg.rewards.carry_heading_huber_delta
        heading_huber_error = torch.where(
            abs_heading_error <= heading_huber_delta,
            0.5 * torch.square(self.teacher_heading_error) / heading_huber_delta,
            abs_heading_error - 0.5 * heading_huber_delta,
        )
        heading_huber_max = torch.pi - 0.5 * heading_huber_delta
        heading_huber_error_normalized = heading_huber_error / heading_huber_max
        heading_reward = (
            heading_alignment
            - self.cfg.rewards.carry_heading_huber_weight
            * heading_huber_error_normalized
        )
        heading_reward[~self.is_stage_carry] = 0.0
        return heading_reward

    def _log_force_debug(self):
        cfg = self.cfg.external_force
        if not bool(cfg.debug_logging):
            self._clear_debug_transition_flags()
            return
        env_id = int(cfg.debug_env_id)
        if not 0 <= env_id < self.num_envs:
            self._clear_debug_transition_flags()
            return

        transitions = []
        if bool(self._force_start_pending[env_id]):
            transitions.append("START")
        if bool(self._force_hold_pending[env_id]):
            transitions.append("HOLD")
        if bool(self._force_end_pending[env_id]):
            transitions.append("END")
        interval = max(1, int(cfg.debug_log_interval_policy_steps))
        periodic = (
            (self.external_force_active[env_id] or self.force_remaining_physics_steps[env_id] > 0)
            and self.common_step_counter % interval == 0
        )
        if transitions or periodic:
            event = "/".join(transitions) if transitions else "sample"
            force = self.external_force_world[env_id]
            actual_vel = self.rigid_body_states[env_id, self.upper_body_index, 7:9]
            print(
                "[carrybox_force] "
                f"event={event} phase={self._PHASE_NAMES[int(self.force_phase[env_id])]} "
                f"active={bool(self.external_force_active[env_id])} "
                f"beta={float(self.external_force_beta[env_id]):.4f} "
                f"box_mass={float(self.external_force_box_mass[env_id]):.4f} "
                f"F_world=({float(force[0]):.4f},{float(force[1]):.4f},0.0000) "
                f"F_parallel={float(self.force_parallel[env_id]):.4f} "
                f"F_perp={float(self.force_perp[env_id]):.4f} "
                f"nominal_vx={float(self.carry_policy_commands[env_id, 0]):.4f} "
                f"teacher_vx={float(self.teacher_vx[env_id]):.4f} "
                f"nominal_heading={float(self.carry_heading_ref[env_id]):.4f} "
                f"teacher_heading={float(self.teacher_heading[env_id]):.4f} "
                f"teacher_heading_error={float(self.teacher_heading_error[env_id]):.4f} "
                f"nominal_yaw_rate={float(self.carry_policy_commands[env_id, 2]):.4f} "
                f"teacher_yaw_rate={float(self.teacher_yaw_rate[env_id]):.4f} "
                f"actual_upper_world_v=({float(actual_vel[0]):.4f},{float(actual_vel[1]):.4f}) "
                f"actual_yaw_rate={float(self.base_ang_vel[env_id, 2]):.4f}"
            )
        self._clear_debug_transition_flags()

    def _clear_debug_transition_flags(self):
        self._force_start_pending.zero_()
        self._force_hold_pending.zero_()
        self._force_end_pending.zero_()

    def reset_idx(self, env_ids):
        has_force_buffers = hasattr(self, "force_event_count")
        super().reset_idx(env_ids)
        if not has_force_buffers or len(env_ids) == 0:
            return
        for name in (
            "external_force_tensor",
            "external_force_world",
            "external_force_direction_world",
        ):
            getattr(self, name)[env_ids] = 0.0
        for name in (
            "external_force_scale",
            "external_force_peak_N",
            "external_force_beta",
            "external_force_box_mass",
            "force_hold_duration_s",
            "force_parallel",
            "force_perp",
        ):
            getattr(self, name)[env_ids] = 0.0
        for name in (
            "force_event_count",
            "force_stable_carry_streak",
            "force_elapsed_physics_steps",
            "force_remaining_physics_steps",
            "force_ramp_up_steps",
            "force_hold_steps",
            "force_ramp_down_steps",
        ):
            getattr(self, name)[env_ids] = 0
        for name in (
            "external_force_active",
            "force_event_decision_made",
            "_force_start_pending",
            "_force_hold_pending",
            "_force_end_pending",
        ):
            getattr(self, name)[env_ids] = False
        self.force_phase[env_ids] = self.FORCE_PHASE_IDLE
        self._set_teacher_to_nominal()

    def _draw_debug_vis(self):
        self.gym.clear_lines(self.viewer)
        if not bool(self.cfg.external_force.debug_draw_force):
            return
        scale = float(self.cfg.external_force.debug_force_draw_scale_m_per_N)
        color = np.asarray([[0.9, 0.1, 0.05]], dtype=np.float32)
        for env_id in range(self.num_envs):
            force = self.external_force_world[env_id]
            if float(torch.linalg.vector_norm(force)) <= 1.0e-6:
                continue
            start = self.box_states[env_id, :3]
            end = start + scale * force
            vertices = torch.cat((start, end)).detach().cpu().numpy().reshape(1, 6)
            self.gym.add_lines(
                self.viewer, self.envs[env_id], 1, vertices, color
            )
