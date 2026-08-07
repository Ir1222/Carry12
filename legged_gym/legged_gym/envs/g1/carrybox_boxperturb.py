import math

import numpy as np
import torch

from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_from_angle_axis, quat_rotate, quat_rotate_inverse

from .carrybox import LeggedRobot as CarryBox
from .hand_box_force import (
    decompose_force,
    estimate_box_face_normal_local,
    force_closure_residual,
)


class LeggedRobot(CarryBox):
    """Carry-box task with a single, gated force pulse at the free box COM."""

    _DIRECTION_IDS = {
        "+box_x": 0,
        "-box_x": 1,
        "+box_y": 2,
        "-box_y": 3,
        "-z_world": 4,
        "+z_world": 5,
    }
    _PULSE_PROFILE_IDS = {
        "half_sine": 0,
        "ramp_hold": 1,
        "multi_pulse": 2,
        "jittered_half_sine": 3,
    }
    _PULSE_PROFILE_NAMES = {value: key for key, value in _PULSE_PROFILE_IDS.items()}

    EVALUATION_OBSERVATION_HISTORY_INITIALIZATION = (
        "goal_before_zero_action_single_commit"
    )

    def _init_buffers(self):
        super()._init_buffers()
        n = self.num_envs
        device = self.device

        # The sampled distance is recorded before the reset zero-action step so
        # evaluator metadata does not need to rebuild/commit observations.
        self.evaluation_initial_goal_distance_xy = torch.full(
            (n,), float("nan"), device=device
        )

        # This tensor is intentionally separate from the legacy robot disturbance.
        self.box_perturb_force_tensor = torch.zeros_like(self.disturbance)
        self.box_perturb_force_pos_tensor = torch.zeros_like(self.disturbance)
        self.box_perturb_peak_force_world = torch.zeros((n, 3), device=device)
        self.box_perturb_direction_world = torch.zeros((n, 3), device=device)
        self.box_perturb_direction_local = torch.zeros((n, 3), device=device)
        self.box_perturb_direction_is_world = torch.zeros(n, dtype=torch.bool, device=device)
        self.box_perturb_force_point_box = torch.zeros((n, 3), device=device)
        self.box_perturb_force_point_world = torch.zeros((n, 3), device=device)
        self.box_perturb_external_torque_world = torch.zeros((n, 3), device=device)
        self.box_perturb_debug_draw_direction_local = torch.zeros((n, 3), device=device)
        self.box_perturb_debug_draw_point_box = torch.zeros((n, 3), device=device)
        self.box_perturb_debug_draw_force_N = torch.zeros(n, device=device)
        self.box_perturb_debug_draw_world_z = torch.zeros(n, dtype=torch.bool, device=device)
        self.box_perturb_debug_draw_hold_steps = torch.zeros(n, dtype=torch.long, device=device)
        self.box_perturb_peak_force_N = torch.zeros(n, device=device)
        self.box_perturb_force_peak_cap_N = torch.full((n,), float("nan"), device=device)
        self.box_perturb_beta = torch.zeros(n, device=device)
        self.box_perturb_actual_force_scale = torch.zeros(n, device=device)
        self.box_perturb_pulse_steps = torch.zeros(n, dtype=torch.long, device=device)
        self.box_perturb_pulse_duration_s = torch.zeros(n, device=device)
        self.box_perturb_pulse_profile_id_buf = torch.zeros(n, dtype=torch.long, device=device)
        self.box_perturb_elapsed_physics_steps = torch.zeros(n, dtype=torch.long, device=device)
        self.box_perturb_remaining_physics_steps = torch.zeros(n, dtype=torch.long, device=device)

        self.confirmed_carry_streak = torch.zeros(n, dtype=torch.long, device=device)
        self.box_perturb_decision_made_buf = torch.zeros(n, dtype=torch.bool, device=device)
        self.box_perturb_event_count_buf = torch.zeros(n, dtype=torch.long, device=device)
        self.box_perturb_direction_id_buf = torch.full((n,), -1, dtype=torch.long, device=device)
        self.box_perturb_schedule_confirmed_streak = torch.zeros(
            n, dtype=torch.long, device=device
        )
        self.box_perturb_mass_kg = torch.zeros(n, device=device)
        self.box_perturb_cap_used_buf = torch.zeros(n, dtype=torch.bool, device=device)

        self.box_perturb_recovery_active_buf = torch.zeros(n, dtype=torch.bool, device=device)
        self.box_perturb_recovery_confirmed_streak = torch.zeros(n, dtype=torch.long, device=device)
        self.box_perturb_recovery_elapsed_policy_steps = torch.zeros(n, dtype=torch.long, device=device)
        self.box_perturb_recovery_success_buf = torch.zeros(n, dtype=torch.bool, device=device)
        self.box_perturb_recovery_done_buf = torch.zeros(n, dtype=torch.bool, device=device)
        self.box_perturb_debug_sweep_index = torch.zeros(n, dtype=torch.long, device=device)
        self.box_perturb_debug_sweep_cooldown = torch.zeros(n, dtype=torch.long, device=device)
        self.box_perturb_force_trace = []
        self.box_perturb_trace_enabled = False
        self.box_perturb_trace_verbose = False
        self.box_perturb_trace_phase = "idle"
        self.box_perturb_trace_metadata = {}
        self.box_perturb_last_termination_reason = [""] * n
        self._reset_force_trace_analysis()

        # Run-level counters provide stable rollout metrics even when no episode ends.
        self._perturb_total_decisions = torch.zeros((), device=device)
        self._perturb_total_events = torch.zeros((), device=device)
        self._perturb_total_completed_episodes = torch.zeros((), device=device)
        self._perturb_total_force_peak_N = torch.zeros((), device=device)
        self._perturb_total_beta = torch.zeros((), device=device)
        self._perturb_total_mass_kg = torch.zeros((), device=device)
        self._perturb_total_cap_used = torch.zeros((), device=device)
        self._perturb_total_recoveries = torch.zeros((), device=device)
        self._perturb_total_recovery_successes = torch.zeros((), device=device)

        assert self.box_perturb_force_tensor.shape == self.contact_forces.shape
        assert int(self.box_net_contact_force_index) == 2, (
            "CarryBox actor creation order changed; expected box rigid-body index 2, got "
            f"{int(self.box_net_contact_force_index)}"
        )
        self.debug_viz = bool(self.cfg.box_perturbation.debug_draw_force)

    def step(self, actions):
        """Apply the pulse before every physics simulate call in the decimation loop."""
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        self.render()
        for physics_substep in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(self.torques)
            )
            self._apply_box_perturbation_force()
            self.gym.simulate(self.sim)
            trace_active = self._box_perturb_trace_is_active()
            if self.device == "cpu" or trace_active:
                self.gym.fetch_results(self.sim, True)
            if trace_active:
                self.gym.refresh_actor_root_state_tensor(self.sim)
                self.gym.refresh_rigid_body_state_tensor(self.sim)
                self.gym.refresh_net_contact_force_tensor(self.sim)
                self._record_box_perturb_physics_trace(physics_substep)
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

    def _apply_box_perturbation_force(self):
        """Apply the configured pulse at the box COM or a world force point."""
        self.box_perturb_force_tensor.zero_()
        self.box_perturb_force_pos_tensor.zero_()
        self.box_perturb_actual_force_scale.zero_()
        self.box_perturb_force_point_world[:] = self.box_states[:, 0:3]
        self.box_perturb_external_torque_world.zero_()
        active = (
            bool(self.cfg.box_perturbation.enabled)
            & (self.box_perturb_remaining_physics_steps > 0)
        )
        if torch.any(active):
            direction_world = self._current_perturb_direction_world(active)
            pulse_steps = torch.clamp(self.box_perturb_pulse_steps[active].float(), min=1.0)
            tau_fraction = (
                self.box_perturb_elapsed_physics_steps[active].float() + 0.5
            ) / pulse_steps
            profile = self._pulse_profile_scale(
                tau_fraction, self.box_perturb_pulse_profile_id_buf[active]
            )
            peak_force_world = direction_world * self.box_perturb_peak_force_N[active].unsqueeze(1)
            force = peak_force_world * profile.unsqueeze(1)
            point_offset_world = quat_rotate(
                self.box_states[active, 3:7], self.box_perturb_force_point_box[active]
            )
            point_world = self.box_states[active, 0:3] + point_offset_world
            self.box_perturb_force_tensor[
                active, int(self.box_net_contact_force_index), :
            ] = force
            self.box_perturb_force_pos_tensor[
                active, int(self.box_net_contact_force_index), :
            ] = point_world
            self.box_perturb_force_point_world[active] = point_world
            self.box_perturb_external_torque_world[active] = torch.cross(
                point_offset_world, force, dim=-1
            )
            self.box_perturb_direction_world[active] = direction_world
            self.box_perturb_peak_force_world[active] = peak_force_world
            self.box_perturb_actual_force_scale[active] = profile
            self.box_perturb_debug_draw_point_box[active] = self.box_perturb_force_point_box[active]
            current_force_N = torch.linalg.vector_norm(force, dim=-1)
            self.box_perturb_debug_draw_force_N[active] = torch.maximum(
                self.box_perturb_debug_draw_force_N[active], current_force_N
            )
            self.box_perturb_debug_draw_world_z[active] = self.box_perturb_direction_is_world[active]
            self.box_perturb_debug_draw_direction_local[active] = self.box_perturb_direction_local[active]
            hold_steps = max(
                1,
                int(
                    round(
                        float(getattr(self.cfg.box_perturbation, "debug_force_arrow_hold_s", 1.25))
                        / float(self.dt)
                    )
                ),
            )
            self.box_perturb_debug_draw_hold_steps[active] = hold_steps

        offset_norm = torch.linalg.vector_norm(self.box_perturb_force_point_box, dim=-1)
        use_force_at_pos = torch.any(active & (offset_norm > 1.0e-6))
        # ENV_SPACE has world-aligned axes for these untranslated/unrotated env frames.
        if bool(use_force_at_pos):
            if not hasattr(self.gym, "apply_rigid_body_force_at_pos_tensors"):
                raise RuntimeError(
                    "Isaac Gym force-at-position tensor API is unavailable; "
                    "cannot run off-center box perturbations."
                )
            self.gym.apply_rigid_body_force_at_pos_tensors(
                self.sim,
                forceTensor=gymtorch.unwrap_tensor(self.box_perturb_force_tensor),
                posTensor=gymtorch.unwrap_tensor(self.box_perturb_force_pos_tensor),
                space=gymapi.CoordinateSpace.ENV_SPACE,
            )
        else:
            # The rigid-body force tensor API acts at each body COM and supplies no torque.
            self.gym.apply_rigid_body_force_tensors(
                self.sim,
                forceTensor=gymtorch.unwrap_tensor(self.box_perturb_force_tensor),
                space=gymapi.CoordinateSpace.ENV_SPACE,
            )

        if torch.any(active):
            self.box_perturb_elapsed_physics_steps[active] += 1
            self.box_perturb_remaining_physics_steps[active] -= 1

    def _current_perturb_direction_world(self, env_selector):
        direction = torch.zeros((int(torch.as_tensor(env_selector).sum().item()), 3), device=self.device)
        env_ids = torch.nonzero(env_selector, as_tuple=False).flatten()
        is_world = self.box_perturb_direction_is_world[env_ids]
        if torch.any(is_world):
            direction[is_world] = self.box_perturb_direction_local[env_ids[is_world]]
        if torch.any(~is_world):
            local = self.box_perturb_direction_local[env_ids[~is_world]]
            direction[~is_world] = quat_rotate(self.box_states[env_ids[~is_world], 3:7], local)
        return direction / torch.clamp(
            torch.linalg.vector_norm(direction, dim=-1, keepdim=True), min=1.0e-6
        )

    def _pulse_profile_scale(self, tau_fraction, profile_ids):
        half_sine = torch.sin(math.pi * torch.clamp(tau_fraction, 0.0, 1.0))
        scale = half_sine.clone()
        ramp_mask = profile_ids == self._PULSE_PROFILE_IDS["ramp_hold"]
        if torch.any(ramp_mask):
            tau = torch.clamp(tau_fraction[ramp_mask], 0.0, 1.0)
            scale[ramp_mask] = torch.clamp(tau / 0.25, max=1.0)
        multi_mask = profile_ids == self._PULSE_PROFILE_IDS["multi_pulse"]
        if torch.any(multi_mask):
            tau = torch.clamp(tau_fraction[multi_mask], 0.0, 1.0)
            envelope = torch.sin(math.pi * tau)
            ripple = 0.65 + 0.35 * torch.abs(torch.sin(5.0 * math.pi * tau))
            scale[multi_mask] = envelope * ripple
        jitter_mask = profile_ids == self._PULSE_PROFILE_IDS["jittered_half_sine"]
        if torch.any(jitter_mask):
            amp = float(getattr(self.cfg.box_perturbation, "jittered_half_sine_amplitude", 0.15))
            noise = 1.0 + amp * (2.0 * torch.rand_like(scale[jitter_mask]) - 1.0)
            scale[jitter_mask] = torch.clamp(half_sine[jitter_mask] * noise, min=0.0)
        return scale

    def _update_carry_phase(self):
        # Base post_physics_step calls this after refreshing state/contact tensors.
        super()._update_carry_phase()
        self._update_box_perturbation_state()

    def _update_box_perturbation_state(self):
        cfg = self.cfg.box_perturbation
        if not bool(cfg.enabled):
            self._clear_box_perturbation_state_for_gate()
            self.extras["perturb"] = self._build_perturb_log_info()
            return

        self.confirmed_carry_streak[:] = torch.where(
            self.confirmed_carry_buf,
            self.confirmed_carry_streak + 1,
            torch.zeros_like(self.confirmed_carry_streak),
        )

        self._update_recovery_state()
        self._log_applied_force_debug()

        if bool(cfg.evaluation_mode) and bool(cfg.evaluation_manual_schedule):
            self.extras["perturb"] = self._build_perturb_log_info()
            return

        if bool(cfg.debug_sweep_enabled):
            self._update_debug_force_sweep()
            self.extras["perturb"] = self._build_perturb_log_info()
            return

        eligible = (
            (self.confirmed_carry_streak >= int(cfg.stable_confirmed_carry_policy_steps))
            & ~self.box_perturb_decision_made_buf
            & (self.box_perturb_event_count_buf < int(cfg.max_events_per_episode))
        )
        if torch.any(eligible):
            self.box_perturb_decision_made_buf[eligible] = True
            self._perturb_total_decisions += eligible.float().sum()
            probability = 1.0 if bool(cfg.debug_force_event) else self._stage_probability()
            sampled = torch.rand(self.num_envs, device=self.device) < probability
            trigger_ids = torch.nonzero(eligible & sampled, as_tuple=False).flatten()
            if trigger_ids.numel() > 0:
                self._schedule_box_perturbation(trigger_ids)

        self.extras["perturb"] = self._build_perturb_log_info()

    def _clear_box_perturbation_state_for_gate(self):
        """Make runtime gate-off transitions immediate and free of force leakage."""
        self.box_perturb_force_tensor.zero_()
        self.box_perturb_force_pos_tensor.zero_()
        self.box_perturb_peak_force_world.zero_()
        self.box_perturb_direction_world.zero_()
        self.box_perturb_direction_local.zero_()
        self.box_perturb_direction_is_world.zero_()
        self.box_perturb_force_point_box.zero_()
        self.box_perturb_force_point_world.zero_()
        self.box_perturb_external_torque_world.zero_()
        self.box_perturb_debug_draw_direction_local.zero_()
        self.box_perturb_debug_draw_point_box.zero_()
        self.box_perturb_debug_draw_force_N.zero_()
        self.box_perturb_debug_draw_world_z.zero_()
        self.box_perturb_debug_draw_hold_steps.zero_()
        self.box_perturb_peak_force_N.zero_()
        self.box_perturb_force_peak_cap_N.fill_(float("nan"))
        self.box_perturb_beta.zero_()
        self.box_perturb_mass_kg.zero_()
        self.box_perturb_actual_force_scale.zero_()
        self.box_perturb_pulse_steps.zero_()
        self.box_perturb_pulse_duration_s.zero_()
        self.box_perturb_pulse_profile_id_buf.zero_()
        self.box_perturb_elapsed_physics_steps.zero_()
        self.box_perturb_remaining_physics_steps.zero_()
        self.confirmed_carry_streak.zero_()
        self.box_perturb_decision_made_buf.zero_()
        self.box_perturb_event_count_buf.zero_()
        self.box_perturb_direction_id_buf.fill_(-1)
        self.box_perturb_schedule_confirmed_streak.zero_()
        self.box_perturb_cap_used_buf.zero_()
        self.box_perturb_recovery_active_buf.zero_()
        self.box_perturb_recovery_confirmed_streak.zero_()
        self.box_perturb_recovery_elapsed_policy_steps.zero_()
        self.box_perturb_recovery_success_buf.zero_()
        self.box_perturb_recovery_done_buf.zero_()
        self.box_perturb_debug_sweep_index.zero_()
        self.box_perturb_debug_sweep_cooldown.zero_()

    def _log_applied_force_debug(self):
        cfg = self.cfg.box_perturbation
        # Evaluation traces are persisted to CSV.  Avoid flooding the terminal
        # with policy-step force details during an explicit sweep.
        if bool(cfg.evaluation_mode):
            return
        interval = int(cfg.debug_force_log_interval_policy_steps)
        if not bool(cfg.debug_draw_force) or interval <= 0:
            return
        if self.common_step_counter % interval != 0:
            return
        applied = self.box_perturb_force_tensor[
            :, int(self.box_net_contact_force_index), :
        ]
        active_ids = torch.nonzero(
            torch.linalg.vector_norm(applied, dim=-1) > 1.0e-6,
            as_tuple=False,
        ).flatten()
        if active_ids.numel() == 0:
            return
        env_id = int(active_ids[0].item())
        force = applied[env_id]
        magnitude = torch.linalg.vector_norm(force)
        left_on_box = -self.contact_forces[
            env_id, int(self.left_hand_net_contact_force_index), :
        ]
        right_on_box = -self.contact_forces[
            env_id, int(self.right_hand_net_contact_force_index), :
        ]
        direction = self.box_perturb_direction_world[env_id]
        resistive = torch.dot(left_on_box + right_on_box, -direction)
        print(
            "[Box perturb applied] "
            f"policy_step={self.common_step_counter} env={env_id} "
            f"force_world_N={force.detach().cpu().tolist()} "
            f"magnitude_N={float(magnitude):.6f} "
            f"peak_N={float(self.box_perturb_peak_force_N[env_id]):.6f} "
            f"left_hand_on_box_proxy_N={left_on_box.detach().cpu().tolist()} "
            f"left_norm_N={float(torch.linalg.vector_norm(left_on_box)):.6f} "
            f"right_hand_on_box_proxy_N={right_on_box.detach().cpu().tolist()} "
            f"right_norm_N={float(torch.linalg.vector_norm(right_on_box)):.6f} "
            f"resistive_N={float(resistive):.6f}"
        )

    def begin_box_perturb_trace(self, metadata=None, verbose=False):
        """Start one-env evaluation tracing without changing policy observations."""
        self.box_perturb_force_trace = []
        self.box_perturb_trace_metadata = dict(metadata or {})
        self.box_perturb_trace_verbose = bool(verbose)
        self.box_perturb_trace_phase = "pre"
        self.box_perturb_trace_enabled = True
        self.box_perturb_last_termination_reason[0] = ""
        self._reset_force_trace_analysis()

    def _reset_force_trace_analysis(self):
        """Reset one-env, evaluation-only force analysis state."""
        self._trace_face_locked = torch.zeros((2,), dtype=torch.bool, device=self.device)
        self._trace_face_id = torch.full((2,), -1, dtype=torch.long, device=self.device)
        self._trace_normal_local = torch.zeros((2, 3), device=self.device)
        self._trace_normal_sign = torch.ones((2,), device=self.device)
        self._trace_sign_sum = torch.zeros((2,), device=self.device)
        self._trace_sign_count = torch.zeros((2,), dtype=torch.long, device=self.device)
        self._trace_sign_verified = torch.zeros((2,), dtype=torch.bool, device=self.device)
        self._trace_fn_ema = torch.zeros((2,), device=self.device)
        self._trace_ft_ema = torch.zeros((2,), device=self.device)
        self._trace_ema_initialized = torch.zeros((2,), dtype=torch.bool, device=self.device)
        self._trace_force_baseline = None
        self._trace_force_baseline_count = 0
        self._trace_impulse_Ns = 0.0

    def _freeze_force_trace_baseline(self):
        keys = (
            "left_fn_raw_N", "right_fn_raw_N", "left_ft_raw_N", "right_ft_raw_N",
            "left_fn_ema_N", "right_fn_ema_N", "left_ft_ema_N", "right_ft_ema_N",
        )
        valid = [
            row for row in self.box_perturb_force_trace
            if row.get("phase") == "pre" and row.get("force_decomposition_valid") == 1
        ][-40:]
        self._trace_force_baseline_count = len(valid)
        if len(valid) < 40:
            self._trace_force_baseline = None
            return
        self._trace_force_baseline = {
            key: float(sum(float(row[key]) for row in valid) / len(valid)) for key in keys
        }

    def set_box_perturb_trace_phase(self, phase):
        self.box_perturb_trace_phase = str(phase)

    def end_box_perturb_trace(self):
        self.box_perturb_trace_enabled = False
        self.box_perturb_trace_phase = "idle"
        return list(self.box_perturb_force_trace)

    def _box_perturb_trace_is_active(self):
        return (
            bool(self.cfg.box_perturbation.evaluation_trace_enabled)
            and self.box_perturb_trace_enabled
            and self.num_envs == 1
        )

    def _reset_task(self, env_ids):
        """Install an evaluation goal before BaseTask.reset() takes its first step."""
        super()._reset_task(env_ids)
        self.evaluation_initial_goal_distance_xy[env_ids] = float("nan")
        perturb_cfg = self.cfg.box_perturbation
        if not bool(getattr(perturb_cfg, "evaluation_mode", False)):
            return
        goal_mode = str(getattr(perturb_cfg, "evaluation_goal_mode", "default"))
        if goal_mode == "default":
            return
        if goal_mode != "long_range":
            raise ValueError(f"Unknown evaluation goal mode: {goal_mode}")
        distances = self._set_evaluation_long_range_goals(
            env_ids,
            distance_range=tuple(perturb_cfg.evaluation_goal_distance_range),
            bearing_offset_deg=tuple(perturb_cfg.evaluation_goal_bearing_offset_deg),
        )
        self.evaluation_initial_goal_distance_xy[env_ids] = distances

    def _set_evaluation_long_range_goals(
        self,
        env_ids,
        distance_range=(4.0, 8.0),
        bearing_offset_deg=(15.0, 75.0),
    ):
        """Replace goals without committing an actor observation-history frame."""
        min_distance, max_distance = [float(value) for value in distance_range]
        min_bearing, max_bearing = [float(value) for value in bearing_offset_deg]
        if not (0.0 < min_distance <= max_distance):
            raise ValueError(f"Invalid goal distance range: {distance_range}")
        if not (0.0 <= min_bearing <= max_bearing <= 180.0):
            raise ValueError(f"Invalid goal bearing range: {bearing_offset_deg}")

        box_to_robot = self.root_states[env_ids, 0:2] - self.box_states[env_ids, 0:2]
        base_angle = torch.atan2(box_to_robot[:, 1], box_to_robot[:, 0])
        bearing = min_bearing + (max_bearing - min_bearing) * torch.rand(
            len(env_ids), device=self.device
        )
        sign = torch.where(
            torch.rand(len(env_ids), device=self.device) < 0.5,
            -torch.ones(len(env_ids), device=self.device),
            torch.ones(len(env_ids), device=self.device),
        )
        final_angle = base_angle + sign * bearing * (math.pi / 180.0)
        distance = min_distance + (max_distance - min_distance) * torch.rand(
            len(env_ids), device=self.device
        )
        goal_xy = self.box_states[env_ids, 0:2] + distance.unsqueeze(-1) * torch.stack(
            (torch.cos(final_angle), torch.sin(final_angle)), dim=-1
        )
        self.goal_pos[env_ids, 0:2] = goal_xy
        self.goal_pos[env_ids, 2] = (
            self.env_origins[env_ids, 2] + float(self.cfg.rewards.target_box_height)
        )
        yaw = torch.rand(len(env_ids), device=self.device) * 2.0 * math.pi
        self.goal_rot[env_ids] = quat_from_angle_axis(
            yaw, self.z_axis_unit.expand(len(env_ids), -1)
        )
        self.tar_platform_pos[env_ids, 0:2] = goal_xy
        self.tar_platform_pos[env_ids, 2] = (
            self.env_origins[env_ids, 2] + 0.5 * self._platform_height
        )
        self.robot2goal_dir = self.goal_pos[:, :2] - self.root_states[:, :2]
        self.robot2goal_dist = torch.norm(self.robot2goal_dir, dim=-1)
        self.object2goal_pos = self.box_states[:, :3] - self.goal_pos
        self.object2goal_dist_xy = torch.norm(self.object2goal_pos[:, :2], dim=-1)
        self.object2goal_dist_xyz = torch.norm(self.object2goal_pos, dim=-1)
        return distance

    def set_evaluation_long_range_goal(
        self,
        distance_range=(4.0, 8.0),
        bearing_offset_deg=(15.0, 75.0),
        env_id=0,
    ):
        """Set one evaluation goal without advancing or committing observations.

        The batch evaluator normally configures long-range mode before reset, so
        ``_reset_task`` calls this sampler before the reset zero-action step. This
        public helper remains available for diagnostics, but callers must consume
        the goal on a later physics step instead of calling ``compute_observations``
        immediately.
        """
        env_ids = torch.tensor([env_id], dtype=torch.long, device=self.device)
        distances = self._set_evaluation_long_range_goals(
            env_ids,
            distance_range=distance_range,
            bearing_offset_deg=bearing_offset_deg,
        )
        self.evaluation_initial_goal_distance_xy[env_ids] = distances
        return float(distances[0].item())

    def resolve_force_point_box(self, direction_name, force_point_mode="com", force_point_label="com", env_id=0):
        """Return a box-local force application offset for evaluation sweeps."""
        mode = str(force_point_mode or "com")
        label = str(force_point_label or "com")
        if mode == "com" or label == "com":
            return torch.zeros(3, device=self.device), "com"
        half = 0.5 * self._box_size[env_id].detach()
        direction_local = torch.zeros(3, device=self.device)
        if direction_name.endswith("z_world"):
            world = torch.tensor(
                [0.0, 0.0, 1.0 if direction_name[0] == "+" else -1.0],
                device=self.device,
            ).unsqueeze(0)
            direction_local = quat_rotate_inverse(
                self.box_states[env_id, 3:7].unsqueeze(0), world
            )[0]
        else:
            component = 0 if direction_name[-1] == "x" else 1
            direction_local[component] = 1.0 if direction_name[0] == "+" else -1.0
        normal_axis = int(torch.argmax(torch.abs(direction_local)).item())
        normal_sign = -1.0 if float(direction_local[normal_axis].item()) >= 0.0 else 1.0
        offset = torch.zeros(3, device=self.device)
        offset[normal_axis] = normal_sign * half[normal_axis]
        tangent_axes = [axis for axis in (0, 1, 2) if axis != normal_axis]
        vertical_axis = 2 if 2 in tangent_axes else tangent_axes[1]
        lateral_axis = tangent_axes[0] if tangent_axes[0] != vertical_axis else tangent_axes[1]
        if mode == "box_surface_random":
            for axis in tangent_axes:
                offset[axis] = (2.0 * torch.rand((), device=self.device) - 1.0) * 0.8 * half[axis]
            return offset, "random"
        if mode != "box_surface_grid":
            raise ValueError(f"Unknown force_point_mode: {mode}")
        if label == "face_center":
            pass
        elif label == "face_upper":
            offset[vertical_axis] = 0.5 * half[vertical_axis]
        elif label == "face_lower":
            offset[vertical_axis] = -0.5 * half[vertical_axis]
        elif label == "face_left_edge":
            offset[lateral_axis] = -0.8 * half[lateral_axis]
        elif label == "face_right_edge":
            offset[lateral_axis] = 0.8 * half[lateral_axis]
        else:
            raise ValueError(f"Unknown force_point_label for grid mode: {label}")
        return offset, label

    def schedule_explicit_box_perturbation(
        self,
        direction_name,
        beta,
        env_id=0,
        force_point_box=None,
        pulse_duration_s=None,
        pulse_profile=None,
        force_peak_cap_N=None,
    ):
        """Schedule a deterministic evaluation pulse after the confirmed-carry gate."""
        if direction_name not in self._DIRECTION_IDS:
            raise ValueError(f"Unknown perturbation direction: {direction_name}")
        threshold = int(
            self.cfg.box_perturbation.stable_confirmed_carry_policy_steps
        )
        if int(self.confirmed_carry_streak[env_id].item()) < threshold:
            raise RuntimeError(
                "Perturbation requested before confirmed-carry gate: "
                f"streak={int(self.confirmed_carry_streak[env_id])}, threshold={threshold}"
            )
        if int(self.box_perturb_remaining_physics_steps[env_id].item()) != 0:
            raise RuntimeError("A box perturbation pulse is already active")

        env_ids = torch.tensor([env_id], dtype=torch.long, device=self.device)
        direction_local = torch.zeros((1, 3), device=self.device)
        direction_is_world = torch.zeros(1, dtype=torch.bool, device=self.device)
        if direction_name.endswith("z_world"):
            direction_local[0, 2] = 1.0 if direction_name[0] == "+" else -1.0
            direction_is_world[0] = True
        else:
            component = 0 if direction_name[-1] == "x" else 1
            direction_local[0, component] = 1.0 if direction_name[0] == "+" else -1.0
        beta_tensor = torch.tensor([float(beta)], device=self.device)
        direction_ids = torch.tensor(
            [self._DIRECTION_IDS[direction_name]],
            dtype=torch.long,
            device=self.device,
        )
        self._freeze_force_trace_baseline()
        self.set_box_perturb_trace_phase("pulse")
        self._commit_box_perturbation(
            env_ids,
            direction_local,
            direction_is_world,
            beta_tensor,
            direction_ids,
            f"evaluation:{direction_name}",
            force_point_box=force_point_box,
            pulse_duration_s=pulse_duration_s,
            pulse_profile=pulse_profile,
            force_peak_cap_N=force_peak_cap_N,
        )
        world = self.box_perturb_direction_world[env_id].detach().cpu().tolist()
        local = self.box_perturb_direction_local[env_id].detach().cpu().tolist()
        point = self.box_perturb_force_point_box[env_id].detach().cpu().tolist()
        print(
            "[BoxPerturb] "
            f"direction={direction_name} "
            f"local_direction=({local[0]:+.4f},{local[1]:+.4f},{local[2]:+.4f}) "
            f"world_direction=({world[0]:+.4f},{world[1]:+.4f},{world[2]:+.4f}) "
            f"point_box=({point[0]:+.3f},{point[1]:+.3f},{point[2]:+.3f}) "
            f"profile={self._PULSE_PROFILE_NAMES[int(self.box_perturb_pulse_profile_id_buf[env_id].item())]} "
            f"duration={float(self.box_perturb_pulse_duration_s[env_id].item()):.3f}s "
            f"peak={float(self.box_perturb_peak_force_N[env_id].item()):.4f}N"
        )
        return float(self.box_perturb_peak_force_N[env_id].item())

    def _pairwise_hand_box_contact_audit(self, env_id=0):
        result = {
            "left_pair_count": -1,
            "right_pair_count": -1,
            "left_pair_normal_lambda_N": float("nan"),
            "right_pair_normal_lambda_N": float("nan"),
        }
        try:
            def contact_field(contact, name):
                if hasattr(contact, name):
                    return getattr(contact, name)
                return contact[name]

            contacts = self.gym.get_env_rigid_contacts(self.envs[env_id])
            box_id = int(self.box_net_contact_force_index)
            hand_ids = (
                int(self.left_hand_net_contact_force_index),
                int(self.right_hand_net_contact_force_index),
            )
            counts = [0, 0]
            normal_loads = [0.0, 0.0]
            for contact in contacts:
                pair = {
                    int(contact_field(contact, "body0")),
                    int(contact_field(contact, "body1")),
                }
                for hand_index, hand_id in enumerate(hand_ids):
                    if pair == {hand_id, box_id}:
                        counts[hand_index] += 1
                        normal_loads[hand_index] += float(
                            contact_field(contact, "lambda")
                        )
            result.update(
                left_pair_count=counts[0],
                right_pair_count=counts[1],
                left_pair_normal_lambda_N=normal_loads[0],
                right_pair_normal_lambda_N=normal_loads[1],
            )
        except Exception as exc:
            if not getattr(self, "_pairwise_contact_warning_printed", False):
                print(f"[Box perturb audit] pairwise contacts unavailable: {exc}")
                self._pairwise_contact_warning_printed = True
        return result

    def _record_box_perturb_physics_trace(self, physics_substep):
        env_id = 0
        box_index = int(self.box_net_contact_force_index)
        left_index = int(self.left_hand_net_contact_force_index)
        right_index = int(self.right_hand_net_contact_force_index)
        f_ext = self.box_perturb_force_tensor[env_id, box_index].detach()
        force_point_world = self.box_perturb_force_point_world[env_id].detach()
        force_point_box = self.box_perturb_force_point_box[env_id].detach()
        direction_local = self.box_perturb_direction_local[env_id].detach()
        moment_arm_world = force_point_world - self.box_states[env_id, 0:3].detach()
        external_torque_world = self.box_perturb_external_torque_world[env_id].detach()
        left_on_hand = self.contact_forces[env_id, left_index].detach()
        right_on_hand = self.contact_forces[env_id, right_index].detach()
        left_on_box = -left_on_hand
        right_on_box = -right_on_hand
        combined_on_box = left_on_box + right_on_box
        box_net_contact = self.contact_forces[env_id, box_index].detach()
        direction = self.box_perturb_direction_world[env_id].detach()
        resistive = torch.dot(combined_on_box, -direction)
        left_norm = torch.linalg.vector_norm(left_on_box)
        right_norm = torch.linalg.vector_norm(right_on_box)
        load_asymmetry = torch.abs(left_norm - right_norm) / torch.clamp(
            left_norm + right_norm, min=1.0e-6
        )
        box_vel = self.box_states[env_id, 7:10].detach()
        box_ang_vel = self.box_states[env_id, 10:13].detach()
        left_rel = torch.linalg.vector_norm(
            self.rigid_body_states[env_id, left_index, 7:10] - box_vel
        )
        right_rel = torch.linalg.vector_norm(
            self.rigid_body_states[env_id, right_index, 7:10] - box_vel
        )
        threshold = float(self.cfg.interaction_priv.hand_contact_force_threshold)
        eps = 1.0e-6

        # Lock one box face per hand after confirmed carry.  This is only an
        # estimated contact normal; raw is still the hand body's net force.
        if bool(self.confirmed_carry_buf[env_id].item()) and not bool(torch.all(self._trace_face_locked)):
            hand_pos = self.rigid_body_states[env_id, [left_index, right_index], 0:3]
            relative_world = hand_pos - self.box_states[env_id, 0:3]
            q = self.box_states[env_id, 3:7].unsqueeze(0).expand(2, -1)
            relative_local = quat_rotate_inverse(q, relative_world)
            box_size = self._box_size[env_id].unsqueeze(0).expand(2, -1)
            normal_local, face_id = estimate_box_face_normal_local(relative_local, box_size, eps)
            new_lock = ~self._trace_face_locked
            self._trace_normal_local[new_lock] = normal_local[new_lock]
            self._trace_face_id[new_lock] = face_id[new_lock]
            self._trace_face_locked[new_lock] = True

        q = self.box_states[env_id, 3:7].unsqueeze(0).expand(2, -1)
        normal_world = quat_rotate(q, self._trace_normal_local)
        raw = torch.stack((left_on_hand, right_on_hand), dim=0)
        verify = (
            bool(self.confirmed_carry_buf[env_id].item())
            & self._trace_face_locked
            & ~self._trace_sign_verified
        )
        outward_dot = torch.sum(raw * normal_world, dim=-1)
        self._trace_sign_sum += torch.where(verify, outward_dot, torch.zeros_like(outward_dot))
        self._trace_sign_count += verify.long()
        sign_samples = int(getattr(self.cfg.box_perturbation, "force_sign_verification_samples", 12))
        ready = verify & (self._trace_sign_count >= sign_samples)
        self._trace_normal_sign[ready] = torch.where(
            self._trace_sign_sum[ready] < 0.0, -1.0, 1.0
        )
        self._trace_sign_verified[ready] = True
        normal_world = normal_world * self._trace_normal_sign.unsqueeze(-1)
        fn_signed, fn, _, ft_vector, ft = decompose_force(raw, normal_world)

        alpha = math.exp(-float(self.sim_params.dt) / 0.04)
        analysis_gate = self._trace_face_locked & bool(self.confirmed_carry_buf[env_id].item())
        first = analysis_gate & ~self._trace_ema_initialized
        self._trace_fn_ema[first] = fn[first]
        self._trace_ft_ema[first] = ft[first]
        update = analysis_gate & self._trace_ema_initialized
        self._trace_fn_ema[update] = alpha * self._trace_fn_ema[update] + (1.0 - alpha) * fn[update]
        self._trace_ft_ema[update] = alpha * self._trace_ft_ema[update] + (1.0 - alpha) * ft[update]
        self._trace_ema_initialized |= analysis_gate
        rho = ft / (fn + eps)
        asymmetry = torch.abs(fn[0] - fn[1]) / (fn.sum() + eps)
        closure = force_closure_residual(raw.unsqueeze(0), box_net_contact.unsqueeze(0), eps)[0]
        contacts = torch.linalg.vector_norm(raw, dim=-1) > threshold
        closure_max = float(getattr(self.cfg.box_perturbation, "force_closure_residual_max", 0.2))
        force_valid = bool(
            self.confirmed_carry_buf[env_id].item()
            and torch.all(contacts).item()
            and torch.all(self._trace_sign_verified).item()
            and torch.all(fn_signed > 0.0).item()
            and closure.item() <= closure_max
        )
        self._trace_impulse_Ns += float(torch.linalg.vector_norm(f_ext).item()) * float(self.sim_params.dt)
        if self.box_perturb_trace_phase == "pulse":
            audit = self._pairwise_hand_box_contact_audit(env_id)
        else:
            audit = {
                "left_pair_count": -1,
                "right_pair_count": -1,
                "left_pair_normal_lambda_N": float("nan"),
                "right_pair_normal_lambda_N": float("nan"),
            }

        def vec(prefix, value):
            values = value.cpu().tolist()
            return {
                f"{prefix}_x": values[0],
                f"{prefix}_y": values[1],
                f"{prefix}_z": values[2],
            }

        row = {
            **self.box_perturb_trace_metadata,
            "phase": self.box_perturb_trace_phase,
            "frame": int(self.gym.get_frame_count(self.sim)),
            "policy_step": int(self.common_step_counter),
            "physics_substep": int(physics_substep),
            "elapsed_pulse_physics_steps": int(
                self.box_perturb_elapsed_physics_steps[env_id].item()
            ),
            "beta": float(self.box_perturb_beta[env_id].item()),
            "box_mass_kg": float(self.box_masses[env_id].item()),
            "force_peak_N": float(self.box_perturb_peak_force_N[env_id].item()),
            "force_peak_cap_N": float(self.box_perturb_force_peak_cap_N[env_id].item()),
            "force_uncapped_peak_N": float(
                self.box_perturb_beta[env_id].item() * self.box_masses[env_id].item() * 9.81
            ),
            "force_cap_used": int(self.box_perturb_cap_used_buf[env_id].item()),
            "perturb_direction_is_world": int(self.box_perturb_direction_is_world[env_id].item()),
            "force_impulse_Ns": self._trace_impulse_Ns,
            "pulse_profile": self._PULSE_PROFILE_NAMES[
                int(self.box_perturb_pulse_profile_id_buf[env_id].item())
            ],
            "pulse_duration_s": float(self.box_perturb_pulse_duration_s[env_id].item()),
            "pulse_phase": (
                float(self.box_perturb_elapsed_physics_steps[env_id].item())
                / float(max(int(self.box_perturb_pulse_steps[env_id].item()), 1))
            ),
            "actual_force_scale": float(self.box_perturb_actual_force_scale[env_id].item()),
            "force_baseline_sample_count": self._trace_force_baseline_count,
            "force_baseline_unavailable": int(self._trace_force_baseline is None),
            "confirmed_streak_at_schedule": int(
                self.box_perturb_schedule_confirmed_streak[env_id].item()
            ),
            "f_ext_norm_N": float(torch.linalg.vector_norm(f_ext).item()),
            "external_torque_norm_Nm": float(torch.linalg.vector_norm(external_torque_world).item()),
            "left_hand_on_box_proxy_norm_N": float(left_norm.item()),
            "right_hand_on_box_proxy_norm_N": float(right_norm.item()),
            "combined_hand_on_box_proxy_norm_N": float(
                torch.linalg.vector_norm(combined_on_box).item()
            ),
            "box_net_contact_force_norm_N": float(
                torch.linalg.vector_norm(box_net_contact).item()
            ),
            "resistive_hand_force_N": float(resistive.item()),
            "hand_load_asymmetry": float(load_asymmetry.item()),
            "left_hand_box_rel_speed_mps": float(left_rel.item()),
            "right_hand_box_rel_speed_mps": float(right_rel.item()),
            "left_contact": int(left_norm.item() > threshold),
            "right_contact": int(right_norm.item() > threshold),
            "confirmed_carry": int(self.confirmed_carry_buf[env_id].item()),
            "left_face_id": int(self._trace_face_id[0].item()),
            "right_face_id": int(self._trace_face_id[1].item()),
            "normal_sign_verified": int(torch.all(self._trace_sign_verified).item()),
            "left_fn_signed_N": float(fn_signed[0].item()),
            "right_fn_signed_N": float(fn_signed[1].item()),
            "left_fn_raw_N": float(fn[0].item()),
            "right_fn_raw_N": float(fn[1].item()),
            "left_ft_raw_N": float(ft[0].item()),
            "right_ft_raw_N": float(ft[1].item()),
            "left_fn_ema_N": float(self._trace_fn_ema[0].item()),
            "right_fn_ema_N": float(self._trace_fn_ema[1].item()),
            "left_ft_ema_N": float(self._trace_ft_ema[0].item()),
            "right_ft_ema_N": float(self._trace_ft_ema[1].item()),
            "left_rho_raw": float(rho[0].item()),
            "right_rho_raw": float(rho[1].item()),
            "normal_load_asymmetry": float(asymmetry.item()),
            "force_closure_residual": float(closure.item()),
            "force_decomposition_valid": int(force_valid),
            "box_lin_speed_mps": float(torch.linalg.vector_norm(box_vel).item()),
            "box_ang_speed_radps": float(
                torch.linalg.vector_norm(box_ang_vel).item()
            ),
            **vec("f_ext_world_N", f_ext),
            **vec("force_point_box", force_point_box),
            **vec("force_point_world", force_point_world),
            **vec("moment_arm_world", moment_arm_world),
            **vec("external_torque_world_Nm", external_torque_world),
            **vec("perturb_direction_local", direction_local),
            **vec("left_hand_net_contact_force_world_N", left_on_hand),
            **vec("right_hand_net_contact_force_world_N", right_on_hand),
            **vec("left_hand_on_box_proxy_world_N", left_on_box),
            **vec("right_hand_on_box_proxy_world_N", right_on_box),
            **vec("box_net_contact_force_world_N", box_net_contact),
            **vec("box_lin_vel_world_mps", box_vel),
            **vec("box_ang_vel_world_radps", box_ang_vel),
            **vec("perturb_direction_world", direction),
            **vec("left_face_normal_world", normal_world[0]),
            **vec("right_face_normal_world", normal_world[1]),
            **vec("left_face_normal_box", self._trace_normal_local[0] * self._trace_normal_sign[0]),
            **vec("right_face_normal_box", self._trace_normal_local[1] * self._trace_normal_sign[1]),
            **vec("left_tangential_force_world_N", ft_vector[0]),
            **vec("right_tangential_force_world_N", ft_vector[1]),
            **audit,
        }
        baseline_keys = (
            "left_fn_raw_N", "right_fn_raw_N", "left_ft_raw_N", "right_ft_raw_N",
            "left_fn_ema_N", "right_fn_ema_N", "left_ft_ema_N", "right_ft_ema_N",
        )
        for key in baseline_keys:
            baseline = float("nan") if self._trace_force_baseline is None else self._trace_force_baseline[key]
            row[f"{key}_pre_baseline"] = baseline
            row[f"{key}_delta_from_pre"] = float("nan") if self._trace_force_baseline is None else row[key] - baseline
        self.box_perturb_force_trace.append(row)

    def _schedule_box_perturbation(self, env_ids):
        cfg = self.cfg.box_perturbation
        stage_name = self._stage_name()
        stage_cfg = cfg.stages[stage_name]
        direction_names = stage_cfg["directions"]
        sampled_indices = torch.randint(
            len(direction_names), (env_ids.numel(),), device=self.device
        )

        direction_local = torch.zeros((env_ids.numel(), 3), device=self.device)
        direction_is_world = torch.zeros(env_ids.numel(), dtype=torch.bool, device=self.device)
        beta = torch.zeros(env_ids.numel(), device=self.device)
        direction_ids = torch.full(
            (env_ids.numel(),), -1, dtype=torch.long, device=self.device
        )
        for index, direction_name in enumerate(direction_names):
            mask = sampled_indices == index
            if not torch.any(mask):
                continue
            direction_ids[mask] = self._DIRECTION_IDS[direction_name]
            axis_name = "z" if direction_name.endswith("z_world") else direction_name[-1]
            low, high = stage_cfg["beta"][axis_name]
            beta[mask] = float(low) + (float(high) - float(low)) * torch.rand(
                int(mask.sum().item()), device=self.device
            )
            if direction_name.endswith("z_world"):
                direction_local[mask, 2] = 1.0 if direction_name[0] == "+" else -1.0
                direction_is_world[mask] = True
            else:
                component = 0 if axis_name == "x" else 1
                sign = 1.0 if direction_name[0] == "+" else -1.0
                direction_local[mask, component] = sign

        self._commit_box_perturbation(
            env_ids, direction_local, direction_is_world, beta, direction_ids, stage_name
        )

    def _commit_box_perturbation(
        self,
        env_ids,
        direction_local,
        direction_is_world,
        beta,
        direction_ids,
        label,
        force_point_box=None,
        pulse_duration_s=None,
        pulse_profile=None,
        force_peak_cap_N=None,
    ):
        cfg = self.cfg.box_perturbation
        if bool(cfg.evaluation_mode):
            threshold = int(cfg.stable_confirmed_carry_policy_steps)
            streaks = self.confirmed_carry_streak[env_ids]
            if not torch.all(streaks >= threshold):
                raise AssertionError(
                    "Evaluation force leakage: perturbation scheduled before "
                    f"confirmed-carry threshold {threshold}; streaks={streaks.tolist()}"
                )
        mass = self.box_masses[env_ids]
        uncapped_peak = beta * mass * 9.81
        cap = cfg.force_peak_cap_N if force_peak_cap_N is None else force_peak_cap_N
        if cap is None:
            peak = uncapped_peak
            cap_used = torch.zeros_like(peak, dtype=torch.bool)
            cap_record = torch.full_like(peak, float("nan"))
        else:
            peak = torch.clamp(uncapped_peak, max=float(cap))
            cap_used = uncapped_peak > float(cap)
            cap_record = torch.full_like(peak, float(cap))
        duration = (
            float(cfg.pulse_duration_s)
            if pulse_duration_s is None
            else float(pulse_duration_s)
        )
        profile_name = str(
            getattr(cfg, "pulse_profile", "half_sine")
            if pulse_profile is None
            else pulse_profile
        )
        if profile_name not in self._PULSE_PROFILE_IDS:
            raise ValueError(f"Unknown pulse profile: {profile_name}")
        if force_point_box is None:
            force_point_box_tensor = torch.zeros((env_ids.numel(), 3), device=self.device)
        else:
            force_point_box_tensor = torch.as_tensor(
                force_point_box, dtype=torch.float, device=self.device
            ).reshape(env_ids.numel(), 3)
        pulse_steps = max(1, int(round(duration / float(self.sim_params.dt))))

        direction_local = direction_local / torch.clamp(
            torch.linalg.vector_norm(direction_local, dim=-1, keepdim=True),
            min=1.0e-6,
        )
        self.box_perturb_direction_local[env_ids] = direction_local
        self.box_perturb_direction_is_world[env_ids] = direction_is_world
        direction_world = torch.zeros_like(direction_local)
        if torch.any(direction_is_world):
            direction_world[direction_is_world] = direction_local[direction_is_world]
        if torch.any(~direction_is_world):
            direction_world[~direction_is_world] = quat_rotate(
                self.box_states[env_ids[~direction_is_world], 3:7],
                direction_local[~direction_is_world],
            )
        self.box_perturb_direction_world[env_ids] = direction_world
        self.box_perturb_force_point_box[env_ids] = force_point_box_tensor
        self.box_perturb_force_point_world[env_ids] = self.box_states[env_ids, 0:3] + quat_rotate(
            self.box_states[env_ids, 3:7], force_point_box_tensor
        )
        self.box_perturb_peak_force_N[env_ids] = peak
        self.box_perturb_force_peak_cap_N[env_ids] = cap_record
        self.box_perturb_peak_force_world[env_ids] = direction_world * peak.unsqueeze(1)
        self.box_perturb_beta[env_ids] = beta
        self.box_perturb_mass_kg[env_ids] = mass
        self.box_perturb_actual_force_scale[env_ids] = 0.0
        # Start a fresh viewer-only peak tracker for this perturbation.  The
        # physical force tensor and evaluation trace remain instantaneous.
        self.box_perturb_debug_draw_force_N[env_ids] = 0.0
        self.box_perturb_pulse_steps[env_ids] = pulse_steps
        self.box_perturb_pulse_duration_s[env_ids] = duration
        self.box_perturb_pulse_profile_id_buf[env_ids] = self._PULSE_PROFILE_IDS[profile_name]
        self.box_perturb_cap_used_buf[env_ids] = cap_used
        self.box_perturb_direction_id_buf[env_ids] = direction_ids
        self.box_perturb_schedule_confirmed_streak[env_ids] = (
            self.confirmed_carry_streak[env_ids]
        )
        self.box_perturb_elapsed_physics_steps[env_ids] = 0
        self.box_perturb_remaining_physics_steps[env_ids] = pulse_steps
        self.box_perturb_event_count_buf[env_ids] += 1
        self.box_perturb_recovery_success_buf[env_ids] = False
        self.box_perturb_recovery_done_buf[env_ids] = False

        count = float(env_ids.numel())
        self._perturb_total_events += count
        self._perturb_total_force_peak_N += peak.sum()
        self._perturb_total_beta += beta.sum()
        self._perturb_total_mass_kg += mass.sum()
        self._perturb_total_cap_used += cap_used.float().sum()

        if bool(cfg.debug_force_event) or bool(cfg.debug_sweep_enabled):
            first = int(env_ids[0].item())
            print(
                "[Box perturb debug] "
                f"label={label} env={first} direction_id={int(direction_ids[0])} "
                f"beta={float(beta[0]):.4f} mass_kg={float(mass[0]):.4f} "
                f"peak_N={float(peak[0]):.4f} cap_used={bool(cap_used[0])}"
            )

    def _update_debug_force_sweep(self):
        """Deterministic one-env evaluation sweep; never used by training defaults."""
        cfg = self.cfg.box_perturbation
        self.box_perturb_debug_sweep_cooldown[:] = torch.clamp(
            self.box_perturb_debug_sweep_cooldown - 1, min=0
        )
        directions = tuple(cfg.debug_sweep_directions)
        betas = tuple(cfg.debug_sweep_beta_values)
        total_tests = len(directions) * len(betas)
        eligible = (
            (self.confirmed_carry_streak >= int(cfg.stable_confirmed_carry_policy_steps))
            & (self.box_perturb_remaining_physics_steps == 0)
            & ~self.box_perturb_recovery_active_buf
            & (self.box_perturb_debug_sweep_cooldown == 0)
            & (self.box_perturb_debug_sweep_index < total_tests)
        )
        env_ids = torch.nonzero(eligible, as_tuple=False).flatten()
        for env_id_tensor in env_ids:
            env_id = int(env_id_tensor.item())
            test_index = int(self.box_perturb_debug_sweep_index[env_id].item())
            direction_name = directions[test_index % len(directions)]
            beta_level = test_index // len(directions)
            beta_value = float(betas[beta_level])
            selected_ids = torch.tensor([env_id], dtype=torch.long, device=self.device)
            direction_local = torch.zeros((1, 3), device=self.device)
            direction_is_world = torch.zeros(1, dtype=torch.bool, device=self.device)
            if direction_name.endswith("z_world"):
                direction_local[0, 2] = 1.0 if direction_name[0] == "+" else -1.0
                direction_is_world[0] = True
            else:
                axis_name = direction_name[-1]
                component = 0 if axis_name == "x" else 1
                direction_local[0, component] = 1.0 if direction_name[0] == "+" else -1.0
            beta = torch.tensor([beta_value], device=self.device)
            direction_ids = torch.tensor(
                [self._DIRECTION_IDS[direction_name]],
                dtype=torch.long,
                device=self.device,
            )
            print(
                "[Box perturb sweep] "
                f"test={test_index + 1}/{total_tests} beta_level={beta_level + 1}/{len(betas)} "
                f"direction={direction_name} beta={beta_value:.3f}"
            )
            self._commit_box_perturbation(
                selected_ids,
                direction_local,
                direction_is_world,
                beta,
                direction_ids,
                f"sweep:{direction_name}",
            )
            self.box_perturb_debug_sweep_index[env_id] += 1
            self.box_perturb_debug_sweep_cooldown[env_id] = int(
                cfg.debug_sweep_inter_event_policy_steps
            )

    def _update_recovery_state(self):
        had_event = self.box_perturb_event_count_buf > 0
        pulse_finished = (
            had_event
            & (
                self.box_perturb_elapsed_physics_steps
                >= torch.clamp(self.box_perturb_pulse_steps, min=1)
            )
            & ~self.box_perturb_recovery_active_buf
            & ~self.box_perturb_recovery_done_buf
        )
        self.box_perturb_recovery_active_buf[pulse_finished] = True
        self.box_perturb_recovery_elapsed_policy_steps[pulse_finished] = 0
        self.box_perturb_recovery_confirmed_streak[pulse_finished] = 0

        active = self.box_perturb_recovery_active_buf
        if not torch.any(active):
            return
        self.box_perturb_recovery_elapsed_policy_steps[active] += 1
        self.box_perturb_recovery_confirmed_streak[:] = torch.where(
            active & self.confirmed_carry_buf,
            self.box_perturb_recovery_confirmed_streak + 1,
            torch.zeros_like(self.box_perturb_recovery_confirmed_streak),
        )

        success = active & (
            self.box_perturb_recovery_confirmed_streak
            >= int(self.cfg.box_perturbation.recovery_confirmed_carry_steps)
        )
        if torch.any(success):
            self.box_perturb_recovery_success_buf[success] = True
            self.box_perturb_recovery_done_buf[success] = True
            self.box_perturb_recovery_active_buf[success] = False
            self._perturb_total_recoveries += success.float().sum()
            self._perturb_total_recovery_successes += success.float().sum()

        timeout = self.box_perturb_recovery_active_buf & (
            self.box_perturb_recovery_elapsed_policy_steps
            >= self._recovery_policy_steps()
        )
        if torch.any(timeout):
            self.box_perturb_recovery_active_buf[timeout] = False
            self.box_perturb_recovery_done_buf[timeout] = True
            self._perturb_total_recoveries += timeout.float().sum()

    def check_termination(self):
        """Keep evaluation trials alive on task success while retaining failures."""
        # The current CarryBox base computes success_buf for reporting but does
        # not OR it into reset_buf.  Its older long-range carry_success_buf is
        # intentionally disabled, so evaluation must not access that buffer.
        # All physical failure and timeout termination conditions remain active.
        super().check_termination()

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        has_buffers = hasattr(self, "box_perturb_event_count_buf")
        if has_buffers:
            for env_id_tensor in env_ids:
                env_id = int(env_id_tensor.item())
                if int(self.episode_length_buf[env_id].item()) > 0:
                    reasons = []
                    if hasattr(self, "carry_drop_failure_buf") and bool(
                        self.carry_drop_failure_buf[env_id].item()
                    ):
                        reasons.append("drop")
                    if bool((self.projected_gravity_box[env_id, 2] > -0.05).item()):
                        reasons.append("box_tilt")
                    if bool((self.rigid_body_states[env_id, self.head_index, 2] < 0.6).item()):
                        reasons.append("head_low")
                    if bool((self.root_states[env_id, 2] < 0.2).item()):
                        reasons.append("base_low")
                    if bool((torch.abs(self.roll[env_id]) > 0.5).item()) or bool(
                        (torch.abs(self.pitch[env_id]) > 1.1).item()
                    ):
                        reasons.append("base_tilt")
                    if bool(self.time_out_buf[env_id].item()):
                        reasons.append("timeout")
                    if not reasons and bool(self.reset_buf[env_id].item()):
                        reasons.append("other")
                    self.box_perturb_last_termination_reason[env_id] = "|".join(reasons)
            interrupted_recovery = self.box_perturb_recovery_active_buf[env_ids]
            self._perturb_total_recoveries += interrupted_recovery.float().sum()
            self._perturb_total_completed_episodes += float(len(env_ids))
            episode_events = self.box_perturb_event_count_buf[env_ids].float()
            episode_success = self.box_perturb_recovery_success_buf[env_ids].float()

        super().reset_idx(env_ids)
        if not has_buffers:
            return

        self.extras["episode"]["Perturb/events_per_episode"] = episode_events.mean()
        event_denominator = torch.clamp(episode_events.sum(), min=1.0)
        self.extras["episode"]["Perturb/recovery_success_rate"] = (
            (episode_success * episode_events).sum() / event_denominator
        )

        for name in (
            "box_perturb_force_tensor",
            "box_perturb_force_pos_tensor",
            "box_perturb_peak_force_world",
            "box_perturb_direction_world",
            "box_perturb_direction_local",
            "box_perturb_force_point_box",
            "box_perturb_force_point_world",
            "box_perturb_external_torque_world",
            "box_perturb_debug_draw_direction_local",
            "box_perturb_debug_draw_point_box",
        ):
            getattr(self, name)[env_ids] = 0.0
        for name in (
            "box_perturb_peak_force_N",
            "box_perturb_beta",
            "box_perturb_mass_kg",
            "box_perturb_actual_force_scale",
            "box_perturb_pulse_duration_s",
            "box_perturb_debug_draw_force_N",
        ):
            getattr(self, name)[env_ids] = 0.0
        self.box_perturb_force_peak_cap_N[env_ids] = float("nan")
        for name in (
            "box_perturb_pulse_steps",
            "box_perturb_pulse_profile_id_buf",
            "box_perturb_debug_draw_hold_steps",
            "box_perturb_elapsed_physics_steps",
            "box_perturb_remaining_physics_steps",
            "confirmed_carry_streak",
            "box_perturb_event_count_buf",
            "box_perturb_schedule_confirmed_streak",
            "box_perturb_recovery_confirmed_streak",
            "box_perturb_recovery_elapsed_policy_steps",
            "box_perturb_debug_sweep_cooldown",
        ):
            getattr(self, name)[env_ids] = 0
        for name in (
            "box_perturb_decision_made_buf",
            "box_perturb_direction_is_world",
            "box_perturb_cap_used_buf",
            "box_perturb_recovery_active_buf",
            "box_perturb_recovery_success_buf",
            "box_perturb_recovery_done_buf",
            "box_perturb_debug_draw_world_z",
        ):
            getattr(self, name)[env_ids] = False
        self.box_perturb_direction_id_buf[env_ids] = -1

    def _pulse_physics_steps(self):
        return max(
            1,
            int(
                round(
                    float(self.cfg.box_perturbation.pulse_duration_s)
                    / float(self.sim_params.dt)
                )
            ),
        )

    def _recovery_policy_steps(self):
        policy_dt = float(self.sim_params.dt) * int(self.cfg.control.decimation)
        return max(
            1,
            int(round(float(self.cfg.box_perturbation.recovery_window_s) / policy_dt)),
        )

    def _stage_name(self):
        cfg = self.cfg.box_perturbation
        if cfg.manual_stage_override is not None:
            stage_name = str(cfg.manual_stage_override)
            if stage_name not in cfg.stage_start_policy_steps:
                raise ValueError(f"Unknown box perturbation stage: {stage_name}")
            return stage_name
        if cfg.schedule_mode != "staged_policy_steps":
            raise ValueError(f"Unsupported perturbation schedule mode: {cfg.schedule_mode}")
        stage_name = "C1"
        for candidate, start in sorted(
            cfg.stage_start_policy_steps.items(), key=lambda item: item[1]
        ):
            if self.common_step_counter >= int(start):
                stage_name = candidate
        return stage_name

    def _stage_id(self):
        return int(self._stage_name()[1:])

    def _stage_probability(self):
        return float(self.cfg.box_perturbation.stages[self._stage_name()]["probability"])

    @staticmethod
    def _safe_ratio(numerator, denominator):
        return torch.where(
            denominator > 0.0,
            numerator / torch.clamp(denominator, min=1.0),
            torch.zeros_like(numerator),
        )

    def _build_perturb_log_info(self):
        events = self._perturb_total_events
        recoveries = self._perturb_total_recoveries
        return {
            "enabled": torch.tensor(
                float(bool(self.cfg.box_perturbation.enabled)), device=self.device
            ),
            "stage_id": torch.tensor(float(self._stage_id()), device=self.device),
            "event_trigger_rate": self._safe_ratio(
                events, self._perturb_total_decisions
            ),
            "events_per_episode": self._safe_ratio(
                events, self._perturb_total_completed_episodes
            ),
            "force_peak_N_mean": self._safe_ratio(
                self._perturb_total_force_peak_N, events
            ),
            "beta_mean": self._safe_ratio(self._perturb_total_beta, events),
            "mass_kg_mean": self._safe_ratio(self._perturb_total_mass_kg, events),
            "force_cap_usage_rate": self._safe_ratio(
                self._perturb_total_cap_used, events
            ),
            "recovery_success_rate": self._safe_ratio(
                self._perturb_total_recovery_successes, recoveries
            ),
            "active_pulse_fraction": (
                self.box_perturb_remaining_physics_steps > 0
            ).float().mean(),
            "recovery_window_fraction": self.box_perturb_recovery_active_buf.float().mean(),
        }

    def _draw_debug_vis(self):
        """Draw only the latest commanded external force at its actual force point.

        The CarryBox parent draws red/green/blue box-local XYZ axes.  For
        perturbation debugging those axes are visually misleading, so this
        override clears all viewer lines and draws a FALCON-style red force
        bundle whose tail is the actual force application point.
        """
        self.gym.clear_lines(self.viewer)
        cfg = self.cfg.box_perturbation
        if not bool(cfg.debug_draw_force):
            return

        applied = self.box_perturb_force_tensor[
            :, int(self.box_net_contact_force_index), :
        ]
        live = torch.linalg.vector_norm(applied, dim=-1) > 1.0e-6
        held = (~live) & (self.box_perturb_debug_draw_hold_steps > 0)
        draw_force = torch.zeros_like(applied)
        draw_point = self.box_states[:, 0:3].clone()
        if torch.any(live):
            draw_force[live] = applied[live]
            draw_point[live] = self.box_perturb_force_point_world[live]
        if torch.any(held):
            held_direction = torch.zeros((int(held.sum().item()), 3), device=self.device)
            held_ids = torch.nonzero(held, as_tuple=False).flatten()
            world_z = self.box_perturb_debug_draw_world_z[held_ids]
            if torch.any(world_z):
                held_direction[world_z] = self.box_perturb_debug_draw_direction_local[held_ids[world_z]]
            if torch.any(~world_z):
                local = self.box_perturb_debug_draw_direction_local[held_ids[~world_z]]
                held_direction[~world_z] = quat_rotate(self.box_states[held_ids[~world_z], 3:7], local)
            held_direction = held_direction / torch.clamp(
                torch.linalg.vector_norm(held_direction, dim=-1, keepdim=True),
                min=1.0e-6,
            )
            draw_force[held_ids] = held_direction * self.box_perturb_debug_draw_force_N[held_ids].unsqueeze(-1)
            draw_point[held_ids] = self.box_states[held_ids, 0:3] + quat_rotate(
                self.box_states[held_ids, 3:7], self.box_perturb_debug_draw_point_box[held_ids]
            )
        if torch.any(self.box_perturb_debug_draw_hold_steps > 0):
            self.box_perturb_debug_draw_hold_steps = torch.clamp(
                self.box_perturb_debug_draw_hold_steps - 1, min=0
            )
        scale = float(cfg.debug_force_draw_scale_m_per_N)
        line_count = max(1, int(cfg.debug_force_bundle_line_count))
        jitter = max(0.0, float(cfg.debug_force_bundle_jitter_m))
        max_envs = min(self.num_envs, int(cfg.debug_force_draw_max_envs))
        color = np.asarray([0.851, 0.144, 0.07], dtype=np.float32)

        for env_id in range(max_envs):
            force = draw_force[env_id].detach().cpu().numpy()
            magnitude = float(np.linalg.norm(force))
            if magnitude <= 1.0e-6:
                continue

            start = draw_point[env_id].detach().cpu().numpy()
            end = start + force * scale

            # Match FALCON's force visualization: a small randomly jittered
            # bundle gives Isaac Gym's fixed-width lines a thick, lively look.
            start_jitter = np.random.random((line_count, 3)).astype(np.float32) * jitter
            end_jitter = np.random.random((line_count, 3)).astype(np.float32) * jitter
            starts = np.repeat(start.reshape(1, 3), line_count, axis=0) + start_jitter
            ends = np.repeat(end.reshape(1, 3), line_count, axis=0) + end_jitter
            vertices = np.concatenate((starts, ends), axis=1).astype(np.float32)
            colors = np.repeat(color.reshape(1, 3), line_count, axis=0)
            self.gym.add_lines(
                self.viewer,
                self.envs[env_id],
                line_count,
                vertices,
                colors,
            )
