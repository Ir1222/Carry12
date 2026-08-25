"""Clean CarryBox evaluation env with evaluation-only box force perturbations."""

import hashlib
import json
import math

import numpy as np
import torch
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_rotate

from legged_gym.envs.g1.carrybox import LeggedRobot as CarryBoxBase

from evaluation.force_profiles import half_sine_scale, smooth_hold_scale


class LeggedRobot(CarryBoxBase):
    """Evaluation-only subclass of the clean velocity-command CarryBox env."""

    _PROFILE_IDS = {
        "half_sine": 0,
        "smooth_hold": 1,
    }
    _DIRECTION_IDS = {
        "+box_x": 0,
        "-box_x": 1,
        "+box_y": 2,
        "-box_y": 3,
    }

    def _create_envs(self):
        super()._create_envs()
        self._init_clean_eval_body_indices()

    def _reset_boxes(self, env_ids):
        """Keep the evaluator's start platform below ground from reset onward."""
        super()._reset_boxes(env_ids)
        if len(env_ids) == 0:
            return

        cfg = self.cfg.clean_perturbation
        if not bool(getattr(cfg, "flush_start_platform", False)):
            return

        ground_z = self.env_origins[env_ids, 2]
        box_clearance = float(cfg.box_vertical_clearance)

        # Preserve the base reset's randomized box XY/yaw, but start the box near
        # the ground so removing the raised support cannot cause a long free fall.
        self.box_states[env_ids, 2] = (
            ground_z + self._box_size[env_ids, 2] / 2.0 + box_clearance
        )
        self.box_states[env_ids, 7:13] = 0.0

        # A platform center one full platform height below the ground puts its
        # top face 1 cm underground for the current 2 cm-thick platform. This
        # avoids both a walking obstacle and a coplanar duplicate contact plane.
        self.platform_pos[env_ids, 0:2] = self.box_states[env_ids, 0:2]
        self.platform_pos[env_ids, 2] = ground_z - self._platform_height
        self.platform_states[env_ids, 3:7] = self.default_quat
        self.platform_states[env_ids, 7:13] = 0.0

        self._assert_flush_start_geometry(env_ids, ground_z, box_clearance)

    def _assert_flush_start_geometry(self, env_ids, ground_z, box_clearance):
        """Validate evaluator-only reset geometry before tensors reach Isaac Gym."""
        atol = 1.0e-6
        expected_platform_top = ground_z - self._platform_height / 2.0
        platform_top = self.platform_pos[env_ids, 2] + self._platform_height / 2.0
        if not torch.allclose(
            platform_top, expected_platform_top, atol=atol, rtol=0.0
        ):
            raise AssertionError(
                "Evaluator platform top does not match the below-ground target."
            )
        if torch.any(platform_top >= ground_z).item():
            raise AssertionError("Evaluator platform must remain strictly below ground.")

        box_bottom = (
            self.box_states[env_ids, 2] - self._box_size[env_ids, 2] / 2.0
        )
        expected_box_bottom = ground_z + float(box_clearance)
        if not torch.allclose(box_bottom, expected_box_bottom, atol=atol, rtol=0.0):
            raise AssertionError(
                "Evaluator box bottom does not match its ground clearance."
            )
        if not torch.allclose(
            self.platform_pos[env_ids, 0:2],
            self.box_states[env_ids, 0:2],
            atol=atol,
            rtol=0.0,
        ):
            raise AssertionError("Evaluator platform and box XY positions must align.")

        carryup_height = box_bottom - self.platform_pos[env_ids, 2]
        start_displacement_xy = torch.linalg.vector_norm(
            self.box_states[env_ids, 0:2] - self.platform_pos[env_ids, 0:2],
            dim=-1,
        )
        starts_in_carry = (
            carryup_height > self.cfg.rewards.thresh_carryup_height
        ) | (
            start_displacement_xy
            > self.cfg.rewards.thresh_carry_start_displacement
        )
        if torch.any(starts_in_carry).item():
            raise AssertionError(
                "Evaluator flush-platform reset must not start in the carry stage."
            )

        if torch.any(self.box_states[env_ids, 7:13] != 0.0).item():
            raise AssertionError("Evaluator box reset velocity must be zero.")
        if torch.any(self.platform_states[env_ids, 7:13] != 0.0).item():
            raise AssertionError("Evaluator platform reset velocity must be zero.")

    def _init_clean_eval_body_indices(self):
        body_names = self._get_robot_body_names()
        hand_token = self.cfg.asset.hand_colli_name
        hand_colli_names = [name for name in body_names if hand_token in name]
        left_names = [name for name in hand_colli_names if "left_" in name]
        right_names = [name for name in hand_colli_names if "right_" in name]
        if len(left_names) != 1 or len(right_names) != 1:
            raise AssertionError(
                "Expected exactly one left and one right hand collision body "
                f"matching {hand_token!r}; got left={left_names}, right={right_names}, "
                f"all={hand_colli_names}"
            )

        self.left_hand_contact_proxy_name = left_names[0]
        self.right_hand_contact_proxy_name = right_names[0]
        self.left_hand_contact_proxy_index = self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.actor_handles[0], self.left_hand_contact_proxy_name
        )
        self.right_hand_contact_proxy_index = self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.actor_handles[0], self.right_hand_contact_proxy_name
        )
        if hasattr(self, "hand_colli_indices"):
            expected = {
                int(self.left_hand_contact_proxy_index),
                int(self.right_hand_contact_proxy_index),
            }
            actual = {int(value) for value in self.hand_colli_indices.detach().cpu().tolist()}
            if expected != actual:
                raise AssertionError(
                    "Base hand_colli_indices disagree with name-derived contact proxies: "
                    f"expected={expected}, actual={actual}"
                )

        box_body_count = len(
            self.gym.get_actor_rigid_body_properties(self.envs[0], self.box_handles[0])
        )
        if box_body_count != 1:
            raise AssertionError(f"Expected one box rigid body, got {box_body_count}")
        self.box_rigid_body_index = self.gym.get_actor_rigid_body_handle(
            self.envs[0], self.box_handles[0], 0
        )
        self.box_rigid_body_name = "box"

    def _get_robot_body_names(self):
        if hasattr(self.gym, "get_actor_rigid_body_names"):
            return list(
                self.gym.get_actor_rigid_body_names(self.envs[0], self.actor_handles[0])
            )
        if hasattr(self.gym, "get_actor_rigid_body_dict"):
            return list(
                self.gym.get_actor_rigid_body_dict(self.envs[0], self.actor_handles[0]).keys()
            )
        raise RuntimeError("Isaac Gym API cannot expose actor rigid-body names.")

    def _init_buffers(self):
        super()._init_buffers()
        n = self.num_envs
        device = self.device

        self.clean_eval_force_tensor = torch.zeros_like(self.disturbance)
        self.clean_eval_direction_world = torch.zeros((n, 3), device=device)
        self.clean_eval_peak_force_N = torch.zeros(n, device=device)
        self.clean_eval_beta = torch.zeros(n, device=device)
        self.clean_eval_mass_kg = torch.zeros(n, device=device)
        self.clean_eval_actual_force_scale = torch.zeros(n, device=device)
        self.clean_eval_force_duration_s = torch.zeros(n, device=device)
        self.clean_eval_total_physics_steps = torch.zeros(n, dtype=torch.long, device=device)
        self.clean_eval_elapsed_physics_steps = torch.zeros(n, dtype=torch.long, device=device)
        self.clean_eval_remaining_physics_steps = torch.zeros(n, dtype=torch.long, device=device)
        self.clean_eval_profile_id = torch.zeros(n, dtype=torch.long, device=device)
        self.clean_eval_pulse_duration_s = torch.zeros(n, device=device)
        self.clean_eval_hold_duration_s = torch.zeros(n, device=device)
        self.clean_eval_ramp_up_s = torch.zeros(n, device=device)
        self.clean_eval_ramp_down_s = torch.zeros(n, device=device)
        self.clean_eval_impulse_Ns = torch.zeros(n, device=device)
        self.clean_eval_event_count_buf = torch.zeros(n, dtype=torch.long, device=device)

        self.left_hand_contact_proxy = torch.zeros(n, dtype=torch.bool, device=device)
        self.right_hand_contact_proxy = torch.zeros(n, dtype=torch.bool, device=device)
        self.clean_carry_condition_buf = torch.zeros(n, dtype=torch.bool, device=device)
        self.confirmed_carry_buf = torch.zeros(n, dtype=torch.bool, device=device)
        self.confirmed_carry_streak = torch.zeros(n, dtype=torch.long, device=device)
        self.clean_eval_recovery_active_buf = torch.zeros(n, dtype=torch.bool, device=device)
        self.clean_eval_recovery_success_buf = torch.zeros(n, dtype=torch.bool, device=device)
        self.clean_eval_recovery_done_buf = torch.zeros(n, dtype=torch.bool, device=device)
        self.clean_eval_recovery_confirmed_streak = torch.zeros(n, dtype=torch.long, device=device)
        self.clean_eval_recovery_elapsed_policy_steps = torch.zeros(
            n, dtype=torch.long, device=device
        )
        self.clean_eval_recovery_time_s = torch.full((n,), float("nan"), device=device)

        self.clean_eval_trace = []
        self.clean_eval_trace_enabled = False
        self.clean_eval_trace_phase = "idle"
        self.clean_eval_trace_metadata = {}
        self.clean_eval_last_termination_reason = [""] * n
        self.clean_eval_has_terminal_snapshot = torch.zeros(
            n, dtype=torch.bool, device=device
        )
        self.clean_eval_terminal_peak_force_N = torch.zeros(n, device=device)
        self.clean_eval_terminal_impulse_Ns = torch.zeros(n, device=device)
        self.clean_eval_terminal_force_duration_s = torch.zeros(n, device=device)
        self.clean_eval_terminal_recovery_success_buf = torch.zeros(
            n, dtype=torch.bool, device=device
        )
        self.clean_eval_terminal_recovery_time_s = torch.full(
            (n,), float("nan"), device=device
        )
        self.clean_eval_terminal_confirmed_carry_buf = torch.zeros(
            n, dtype=torch.bool, device=device
        )

        assert self.clean_eval_force_tensor.shape == self.contact_forces.shape
        assert int(self.box_rigid_body_index) < self.contact_forces.shape[1]
        assert int(self.left_hand_contact_proxy_index) < self.contact_forces.shape[1]
        assert int(self.right_hand_contact_proxy_index) < self.contact_forces.shape[1]
        self.debug_viz = bool(self.cfg.clean_perturbation.debug_draw_force)
        print(
            "[CleanCarryBoxPerturb startup] "
            f"contact_forces_shape={tuple(self.contact_forces.shape)}, "
            f"box_rigid_body_index={int(self.box_rigid_body_index)} "
            f"name={self.box_rigid_body_name}, "
            f"left_hand_contact_proxy_index={int(self.left_hand_contact_proxy_index)} "
            f"name={self.left_hand_contact_proxy_name}, "
            f"right_hand_contact_proxy_index={int(self.right_hand_contact_proxy_index)} "
            f"name={self.right_hand_contact_proxy_name}"
        )

    def step(self, actions):
        """Apply a scheduled box force immediately before every physics simulate call."""
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        self.render()
        trace_active = self._trace_is_active()
        for physics_substep in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(self.torques)
            )
            self._apply_box_external_force()
            self.gym.simulate(self.sim)
            if self.device == "cpu" or trace_active:
                self.gym.fetch_results(self.sim, True)
            if trace_active:
                self.gym.refresh_actor_root_state_tensor(self.sim)
                self.gym.refresh_rigid_body_state_tensor(self.sim)
                self.gym.refresh_net_contact_force_tensor(self.sim)
                self._record_trace_row(physics_substep)
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

    def reset_evaluation_trial_state(self, env_ids=None, clear_actor_history=True):
        """Clear evaluator-only state before an independent evaluation reset."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)
        if len(env_ids) == 0:
            return

        if clear_actor_history:
            self.obs_buf[env_ids] = 0.0
            if self.privileged_obs_buf is not None:
                self.privileged_obs_buf[env_ids] = 0.0
            if hasattr(self, "amp_obs_buf"):
                self.amp_obs_buf[env_ids] = 0.0

        if hasattr(self, "clean_eval_event_count_buf"):
            self._reset_clean_eval_buffers(env_ids)
            for env_id_tensor in env_ids:
                env_id = int(env_id_tensor.item())
                self.clean_eval_last_termination_reason[env_id] = ""
                self.clean_eval_has_terminal_snapshot[env_id] = False
                self.clean_eval_terminal_peak_force_N[env_id] = 0.0
                self.clean_eval_terminal_impulse_Ns[env_id] = 0.0
                self.clean_eval_terminal_force_duration_s[env_id] = 0.0
                self.clean_eval_terminal_recovery_success_buf[env_id] = False
                self.clean_eval_terminal_recovery_time_s[env_id] = float("nan")
                self.clean_eval_terminal_confirmed_carry_buf[env_id] = False

    def _apply_box_external_force(self):
        cfg = self.cfg.clean_perturbation
        if not bool(cfg.enabled):
            self._assert_no_force_inactive()
            return

        self.clean_eval_force_tensor.zero_()
        self.clean_eval_actual_force_scale.zero_()
        active = self.clean_eval_remaining_physics_steps > 0
        if torch.any(active):
            active_ids = torch.nonzero(active, as_tuple=False).flatten()
            for env_id_tensor in active_ids:
                env_id = int(env_id_tensor.item())
                scale = self._profile_scale(env_id)
                force = (
                    scale
                    * self.clean_eval_peak_force_N[env_id]
                    * self.clean_eval_direction_world[env_id]
                )
                self.clean_eval_force_tensor[
                    env_id, int(self.box_rigid_body_index), :
                ] = force
                self.clean_eval_actual_force_scale[env_id] = scale
                self.clean_eval_impulse_Ns[env_id] += (
                    torch.linalg.vector_norm(force) * float(self.sim_params.dt)
                )

            self.gym.apply_rigid_body_force_tensors(
                self.sim,
                forceTensor=gymtorch.unwrap_tensor(self.clean_eval_force_tensor),
                space=gymapi.CoordinateSpace.GLOBAL_SPACE,
            )
            self.clean_eval_elapsed_physics_steps[active] += 1
            self.clean_eval_remaining_physics_steps[active] -= 1

    def _assert_no_force_inactive(self):
        if not hasattr(self, "clean_eval_force_tensor"):
            return
        force_norm = float(torch.linalg.vector_norm(self.clean_eval_force_tensor).item())
        actual_scale_norm = float(
            torch.linalg.vector_norm(self.clean_eval_actual_force_scale.float()).item()
        )
        remaining_steps = int(torch.sum(self.clean_eval_remaining_physics_steps).item())
        event_count = int(torch.sum(self.clean_eval_event_count_buf).item())
        if (
            force_norm != 0.0
            or actual_scale_norm != 0.0
            or remaining_steps != 0
            or event_count != 0
        ):
            raise AssertionError(
                "--no_force requires all clean perturbation force state to be zero: "
                f"force_norm={force_norm}, actual_scale_norm={actual_scale_norm}, "
                f"remaining_steps={remaining_steps}, event_count={event_count}"
            )

    def _profile_scale(self, env_id):
        elapsed_s = (
            float(self.clean_eval_elapsed_physics_steps[env_id].item()) + 0.5
        ) * float(self.sim_params.dt)
        profile_id = int(self.clean_eval_profile_id[env_id].item())
        if profile_id == self._PROFILE_IDS["smooth_hold"]:
            return smooth_hold_scale(
                elapsed_s,
                self.clean_eval_ramp_up_s[env_id].item(),
                self.clean_eval_hold_duration_s[env_id].item(),
                self.clean_eval_ramp_down_s[env_id].item(),
            )
        return half_sine_scale(elapsed_s, self.clean_eval_pulse_duration_s[env_id].item())

    def schedule_evaluation_force(
        self,
        direction_name,
        beta,
        profile,
        env_id=0,
        pulse_duration_s=0.10,
        hold_duration_s=1.0,
        ramp_up_s=0.15,
        ramp_down_s=0.15,
    ):
        if direction_name not in self._DIRECTION_IDS:
            raise ValueError(f"Unknown perturbation direction: {direction_name}")
        if profile not in self._PROFILE_IDS:
            raise ValueError(f"Unknown force profile: {profile}")
        if not bool(self.cfg.clean_perturbation.enabled):
            raise RuntimeError("Cannot schedule force while clean_perturbation.enabled is False")
        if not bool(self.confirmed_carry_buf[env_id].item()):
            raise RuntimeError(
                "Perturbation requested before confirmed-carry gate: "
                f"streak={int(self.confirmed_carry_streak[env_id])}"
            )
        if int(self.clean_eval_remaining_physics_steps[env_id].item()) != 0:
            raise RuntimeError("A box perturbation event is already active")
        if int(self.clean_eval_event_count_buf[env_id].item()) >= int(
            self.cfg.clean_perturbation.max_events_per_episode
        ):
            raise RuntimeError("Maximum force events already used for this episode")

        if profile == "smooth_hold":
            duration_s = float(ramp_up_s) + float(hold_duration_s) + float(ramp_down_s)
        else:
            duration_s = float(pulse_duration_s)
        physics_steps = max(1, int(round(duration_s / float(self.sim_params.dt))))
        measured_duration_s = physics_steps * float(self.sim_params.dt)

        env_ids = torch.tensor([env_id], dtype=torch.long, device=self.device)
        direction_local = torch.zeros((1, 3), device=self.device)
        component = 0 if direction_name[-1] == "x" else 1
        direction_local[0, component] = 1.0 if direction_name[0] == "+" else -1.0
        direction_world = quat_rotate(self.box_states[env_ids, 3:7], direction_local)
        direction_world = direction_world / torch.clamp(
            torch.linalg.vector_norm(direction_world, dim=-1, keepdim=True),
            min=1.0e-6,
        )

        beta_t = torch.tensor([float(beta)], dtype=torch.float, device=self.device)
        mass = self.box_masses[env_ids]
        peak_force = beta_t * mass * 9.81

        self.clean_eval_profile_id[env_ids] = self._PROFILE_IDS[profile]
        self.clean_eval_direction_world[env_ids] = direction_world
        self.clean_eval_peak_force_N[env_ids] = peak_force
        self.clean_eval_beta[env_ids] = beta_t
        self.clean_eval_mass_kg[env_ids] = mass
        self.clean_eval_force_duration_s[env_ids] = measured_duration_s
        self.clean_eval_total_physics_steps[env_ids] = physics_steps
        self.clean_eval_elapsed_physics_steps[env_ids] = 0
        self.clean_eval_remaining_physics_steps[env_ids] = physics_steps
        self.clean_eval_pulse_duration_s[env_ids] = float(pulse_duration_s)
        self.clean_eval_hold_duration_s[env_ids] = float(hold_duration_s)
        self.clean_eval_ramp_up_s[env_ids] = float(ramp_up_s)
        self.clean_eval_ramp_down_s[env_ids] = float(ramp_down_s)
        self.clean_eval_actual_force_scale[env_ids] = 0.0
        self.clean_eval_impulse_Ns[env_ids] = 0.0
        self.clean_eval_event_count_buf[env_ids] += 1
        self.clean_eval_recovery_active_buf[env_ids] = False
        self.clean_eval_recovery_success_buf[env_ids] = False
        self.clean_eval_recovery_done_buf[env_ids] = False
        self.clean_eval_recovery_confirmed_streak[env_ids] = 0
        self.clean_eval_recovery_elapsed_policy_steps[env_ids] = 0
        self.clean_eval_recovery_time_s[env_ids] = float("nan")
        self.set_trace_phase("force")

        return {
            "direction_world": self._tensor_list(direction_world[0]),
            "box_mass_kg": float(mass[0].item()),
            "target_force_N": float(peak_force[0].item()),
            "configured_duration_s": float(duration_s),
            "measured_duration_s": float(measured_duration_s),
            "physics_steps": int(physics_steps),
        }

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self._update_confirmed_carry_detector()
        self._update_recovery_state()

    def _update_confirmed_carry_detector(self):
        cfg = self.cfg.clean_perturbation
        left_norm = torch.linalg.vector_norm(
            self.contact_forces[:, int(self.left_hand_contact_proxy_index), :], dim=-1
        )
        right_norm = torch.linalg.vector_norm(
            self.contact_forces[:, int(self.right_hand_contact_proxy_index), :], dim=-1
        )
        self.left_hand_contact_proxy[:] = left_norm > float(cfg.contact_force_threshold)
        self.right_hand_contact_proxy[:] = right_norm > float(cfg.contact_force_threshold)

        rel_lin_vel_norm = torch.linalg.vector_norm(
            self.box_states[:, 7:10] - self.root_states[:, 7:10], dim=-1
        )
        box_ang_vel_norm = torch.linalg.vector_norm(self.box_states[:, 10:13], dim=-1)
        stable_motion = (
            rel_lin_vel_norm < float(cfg.max_box_rel_lin_vel)
        ) & (box_ang_vel_norm < float(cfg.max_box_ang_vel))

        raw_condition = (
            self.is_stage_carry
            & self.left_hand_contact_proxy
            & self.right_hand_contact_proxy
            & stable_motion
        )
        self.clean_carry_condition_buf[:] = raw_condition
        self.confirmed_carry_streak[:] = torch.where(
            raw_condition,
            self.confirmed_carry_streak + 1,
            torch.zeros_like(self.confirmed_carry_streak),
        )
        self.confirmed_carry_buf[:] = (
            self.confirmed_carry_streak
            >= int(cfg.stable_confirmed_carry_policy_steps)
        )
        self.extras["clean_perturb"] = {
            "confirmed_carry_fraction": self.confirmed_carry_buf.float().mean(),
            "left_hand_contact_proxy_fraction": self.left_hand_contact_proxy.float().mean(),
            "right_hand_contact_proxy_fraction": self.right_hand_contact_proxy.float().mean(),
        }

    def _update_recovery_state(self):
        event_finished = (
            (self.clean_eval_event_count_buf > 0)
            & (self.clean_eval_remaining_physics_steps == 0)
            & (self.clean_eval_elapsed_physics_steps >= torch.clamp(self.clean_eval_total_physics_steps, min=1))
            & ~self.clean_eval_recovery_active_buf
            & ~self.clean_eval_recovery_done_buf
        )
        if torch.any(event_finished):
            self.clean_eval_recovery_active_buf[event_finished] = True
            self.clean_eval_recovery_elapsed_policy_steps[event_finished] = 0
            self.clean_eval_recovery_confirmed_streak[event_finished] = 0

        active = self.clean_eval_recovery_active_buf
        if not torch.any(active):
            return
        self.clean_eval_recovery_elapsed_policy_steps[active] += 1
        self.clean_eval_recovery_confirmed_streak[:] = torch.where(
            active & self.clean_carry_condition_buf,
            self.clean_eval_recovery_confirmed_streak + 1,
            torch.zeros_like(self.clean_eval_recovery_confirmed_streak),
        )
        success = active & (
            self.clean_eval_recovery_confirmed_streak
            >= int(self.cfg.clean_perturbation.recovery_confirmed_carry_steps)
        )
        if torch.any(success):
            self.clean_eval_recovery_success_buf[success] = True
            self.clean_eval_recovery_done_buf[success] = True
            self.clean_eval_recovery_active_buf[success] = False
            self.clean_eval_recovery_time_s[success] = (
                self.clean_eval_recovery_elapsed_policy_steps[success].float()
                * float(self.dt)
            )

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        has_eval_buffers = hasattr(self, "clean_eval_event_count_buf")
        if has_eval_buffers:
            self._record_termination_reasons(env_ids)
        super().reset_idx(env_ids)
        if has_eval_buffers:
            self._reset_clean_eval_buffers(env_ids)

    def _record_termination_reasons(self, env_ids):
        for env_id_tensor in env_ids:
            env_id = int(env_id_tensor.item())
            if int(self.episode_length_buf[env_id].item()) <= 0:
                self.clean_eval_last_termination_reason[env_id] = ""
                continue
            reasons = []
            if bool(self.time_out_buf[env_id].item()):
                reasons.append("timeout")
            if bool((self.rigid_body_states[env_id, self.head_index, 2] < 0.6).item()):
                reasons.append("head_low")
            if bool((self.root_states[env_id, 2] < 0.2).item()):
                reasons.append("base_low")
            if bool((torch.abs(self.roll[env_id]) > 0.5).item()) or bool(
                (torch.abs(self.pitch[env_id]) > 1.1).item()
            ):
                reasons.append("base_tilt")
            if hasattr(self, "hip_yaw_indices") and torch.any(
                self.rigid_body_states[env_id, self.hip_yaw_indices, 2] < 0.15
            ).item():
                reasons.append("hip_low")
            if bool(self.reset_buf[env_id].item()) and not reasons:
                reasons.append("other")
            self.clean_eval_last_termination_reason[env_id] = "|".join(reasons)
            self._snapshot_clean_eval_for_summary(env_id)

    def _snapshot_clean_eval_for_summary(self, env_id):
        self.clean_eval_has_terminal_snapshot[env_id] = True
        self.clean_eval_terminal_peak_force_N[env_id] = self.clean_eval_peak_force_N[env_id]
        self.clean_eval_terminal_impulse_Ns[env_id] = self.clean_eval_impulse_Ns[env_id]
        self.clean_eval_terminal_force_duration_s[env_id] = (
            self.clean_eval_force_duration_s[env_id]
        )
        self.clean_eval_terminal_recovery_success_buf[env_id] = (
            self.clean_eval_recovery_success_buf[env_id]
        )
        self.clean_eval_terminal_recovery_time_s[env_id] = (
            self.clean_eval_recovery_time_s[env_id]
        )
        self.clean_eval_terminal_confirmed_carry_buf[env_id] = (
            self.confirmed_carry_buf[env_id]
        )

    def _reset_clean_eval_buffers(self, env_ids):
        for name in (
            "clean_eval_force_tensor",
            "clean_eval_direction_world",
        ):
            getattr(self, name)[env_ids] = 0.0
        for name in (
            "clean_eval_peak_force_N",
            "clean_eval_beta",
            "clean_eval_mass_kg",
            "clean_eval_actual_force_scale",
            "clean_eval_force_duration_s",
            "clean_eval_pulse_duration_s",
            "clean_eval_hold_duration_s",
            "clean_eval_ramp_up_s",
            "clean_eval_ramp_down_s",
            "clean_eval_impulse_Ns",
        ):
            getattr(self, name)[env_ids] = 0.0
        for name in (
            "clean_eval_total_physics_steps",
            "clean_eval_elapsed_physics_steps",
            "clean_eval_remaining_physics_steps",
            "clean_eval_profile_id",
            "clean_eval_event_count_buf",
            "confirmed_carry_streak",
            "clean_eval_recovery_confirmed_streak",
            "clean_eval_recovery_elapsed_policy_steps",
        ):
            getattr(self, name)[env_ids] = 0
        for name in (
            "left_hand_contact_proxy",
            "right_hand_contact_proxy",
            "clean_carry_condition_buf",
            "confirmed_carry_buf",
            "clean_eval_recovery_active_buf",
            "clean_eval_recovery_success_buf",
            "clean_eval_recovery_done_buf",
        ):
            getattr(self, name)[env_ids] = False
        self.clean_eval_recovery_time_s[env_ids] = float("nan")

    def clear_summary_snapshot(self, env_id=0):
        self.clean_eval_has_terminal_snapshot[env_id] = False
        self.clean_eval_terminal_peak_force_N[env_id] = 0.0
        self.clean_eval_terminal_impulse_Ns[env_id] = 0.0
        self.clean_eval_terminal_force_duration_s[env_id] = 0.0
        self.clean_eval_terminal_recovery_success_buf[env_id] = False
        self.clean_eval_terminal_recovery_time_s[env_id] = float("nan")
        self.clean_eval_terminal_confirmed_carry_buf[env_id] = False

    def summary_scalar(self, name, env_id=0):
        current = getattr(self, name)[env_id]
        if not bool(self.clean_eval_has_terminal_snapshot[env_id].item()):
            return current
        terminal_name = name.replace("clean_eval_", "clean_eval_terminal_", 1)
        if hasattr(self, terminal_name):
            return getattr(self, terminal_name)[env_id]
        return current

    def begin_trace(self, metadata=None):
        self.clean_eval_trace = []
        self.clean_eval_trace_metadata = dict(metadata or {})
        self.clean_eval_trace_enabled = True
        self.clean_eval_trace_phase = "wait_carry"
        self.clean_eval_last_termination_reason[0] = ""

    def set_trace_phase(self, phase):
        self.clean_eval_trace_phase = str(phase)

    def end_trace(self):
        self.clean_eval_trace_enabled = False
        self.clean_eval_trace_phase = "idle"
        return list(self.clean_eval_trace)

    def _trace_is_active(self):
        return (
            bool(self.cfg.clean_perturbation.evaluation_trace_enabled)
            and self.clean_eval_trace_enabled
        )

    def _record_trace_row(self, physics_substep):
        env_id = 0
        direction = self.clean_eval_direction_world[env_id].detach()
        force = self.clean_eval_force_tensor[env_id, int(self.box_rigid_body_index)].detach()
        box_delta = self.box_states[env_id, 0:3] - self._trace_force_start_box_pos()
        robot_delta = self.root_states[env_id, 0:3] - self._trace_force_start_robot_pos()
        left_force = self.contact_forces[
            env_id, int(self.left_hand_contact_proxy_index), :
        ].detach()
        right_force = self.contact_forces[
            env_id, int(self.right_hand_contact_proxy_index), :
        ].detach()
        left_rel = torch.linalg.vector_norm(
            self.rigid_body_states[
                env_id, int(self.left_hand_contact_proxy_index), 7:10
            ]
            - self.box_states[env_id, 7:10]
        )
        right_rel = torch.linalg.vector_norm(
            self.rigid_body_states[
                env_id, int(self.right_hand_contact_proxy_index), 7:10
            ]
            - self.box_states[env_id, 7:10]
        )

        def vec(prefix, value):
            values = value.detach().cpu().tolist()
            return {
                f"{prefix}_x": float(values[0]),
                f"{prefix}_y": float(values[1]),
                f"{prefix}_z": float(values[2]),
            }

        row = {
            **self.clean_eval_trace_metadata,
            "phase": self.clean_eval_trace_phase,
            "frame": int(self.gym.get_frame_count(self.sim)),
            "policy_step": int(self.common_step_counter),
            "physics_substep": int(physics_substep),
            "profile_id": int(self.clean_eval_profile_id[env_id].item()),
            "beta": float(self.clean_eval_beta[env_id].item()),
            "box_mass": float(self.box_masses[env_id].item()),
            "peak_force_N": float(self.clean_eval_peak_force_N[env_id].item()),
            "force_duration": float(self.clean_eval_force_duration_s[env_id].item()),
            "force_impulse_Ns": float(self.clean_eval_impulse_Ns[env_id].item()),
            "elapsed_force_physics_steps": int(self.clean_eval_elapsed_physics_steps[env_id].item()),
            "remaining_force_physics_steps": int(self.clean_eval_remaining_physics_steps[env_id].item()),
            "actual_force_scale": float(self.clean_eval_actual_force_scale[env_id].item()),
            "left_hand_contact_proxy": int(self.left_hand_contact_proxy[env_id].item()),
            "right_hand_contact_proxy": int(self.right_hand_contact_proxy[env_id].item()),
            "confirmed_carry": int(self.confirmed_carry_buf[env_id].item()),
            "left_hand_contact_proxy_norm_N": float(torch.linalg.vector_norm(left_force).item()),
            "right_hand_contact_proxy_norm_N": float(torch.linalg.vector_norm(right_force).item()),
            "max_hand_box_relative_speed": float(max(left_rel.item(), right_rel.item())),
            "box_displacement_along_force": float(torch.dot(box_delta, direction).item()),
            "robot_displacement_along_force": float(torch.dot(robot_delta, direction).item()),
            "box_velocity_along_force": float(torch.dot(self.box_states[env_id, 7:10], direction).item()),
            "robot_velocity_along_force": float(torch.dot(self.root_states[env_id, 7:10], direction).item()),
            "vx_tracking_error": float(abs(self.commands[env_id, 0] - self.base_lin_vel[env_id, 0]).item()),
            "vy_tracking_error": float(abs(self.commands[env_id, 1] - self.base_lin_vel[env_id, 1]).item()),
            "yaw_rate_tracking_error": float(abs(self.commands[env_id, 2] - self.base_ang_vel[env_id, 2]).item()),
            **vec("force_world", force),
            **vec("direction_world", direction),
            **vec("box_lin_vel_world", self.box_states[env_id, 7:10]),
            **vec("robot_lin_vel_world", self.root_states[env_id, 7:10]),
            **vec("left_hand_contact_proxy_force_world", left_force),
            **vec("right_hand_contact_proxy_force_world", right_force),
        }
        self.clean_eval_trace.append(row)

    def set_force_start_reference(self, env_id=0):
        self._clean_eval_force_start_box_pos = self.box_states[env_id, 0:3].detach().clone()
        self._clean_eval_force_start_robot_pos = self.root_states[env_id, 0:3].detach().clone()

    def _trace_force_start_box_pos(self):
        if not hasattr(self, "_clean_eval_force_start_box_pos"):
            return self.box_states[0, 0:3].detach()
        return self._clean_eval_force_start_box_pos

    def _trace_force_start_robot_pos(self):
        if not hasattr(self, "_clean_eval_force_start_robot_pos"):
            return self.root_states[0, 0:3].detach()
        return self._clean_eval_force_start_robot_pos

    def evaluation_initial_state_signature(self, env_id=0):
        data = {
            "robot_root_pose": self._tensor_list(
                torch.cat((self.root_states[env_id, 0:3], self.root_states[env_id, 3:7]))
            ),
            "robot_root_linear_velocity": self._tensor_list(self.root_states[env_id, 7:10]),
            "robot_root_angular_velocity": self._tensor_list(self.root_states[env_id, 10:13]),
            "dof_position": self._tensor_list(self.dof_pos[env_id]),
            "dof_velocity": self._tensor_list(self.dof_vel[env_id]),
            "box_pose": self._tensor_list(
                torch.cat((self.box_states[env_id, 0:3], self.box_states[env_id, 3:7]))
            ),
            "box_linear_velocity": self._tensor_list(self.box_states[env_id, 7:10]),
            "box_angular_velocity": self._tensor_list(self.box_states[env_id, 10:13]),
            "command": self._tensor_list(self.commands[env_id, 0:3]),
            "box_mass": float(self.box_masses[env_id].item()),
        }
        payload = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return {
            "data": data,
            "json": payload,
            "sha1": hashlib.sha1(payload.encode("utf-8")).hexdigest(),
        }

    def _draw_debug_vis(self):
        if not self.viewer:
            return
        self.gym.clear_lines(self.viewer)
        cfg = self.cfg.clean_perturbation
        if not bool(cfg.debug_draw_force):
            return

        scale = float(cfg.debug_force_draw_scale_m_per_N)
        line_count = max(1, int(cfg.debug_force_bundle_line_count))
        jitter = max(0.0, float(cfg.debug_force_bundle_jitter_m))
        max_envs = min(self.num_envs, int(cfg.debug_force_draw_max_envs))
        color = np.asarray([0.851, 0.144, 0.07], dtype=np.float32)

        for env_id in range(max_envs):
            force_t = self.clean_eval_force_tensor[
                env_id, int(self.box_rigid_body_index), :
            ]
            if float(torch.linalg.vector_norm(force_t).item()) <= 1.0e-6:
                continue
            start = (
                self.box_states[env_id, 0:3] - self.env_origins[env_id]
            ).detach().cpu().numpy()
            end = start + force_t.detach().cpu().numpy() * scale
            if jitter > 0.0:
                offset = (
                    np.random.random((line_count, 3)).astype(np.float32) - 0.5
                ) * jitter
            else:
                offset = np.zeros((line_count, 3), dtype=np.float32)
            starts = np.repeat(start.reshape(1, 3), line_count, axis=0) + offset
            ends = np.repeat(end.reshape(1, 3), line_count, axis=0) + offset
            vertices = np.concatenate((starts, ends), axis=1).astype(np.float32)
            colors = np.repeat(color.reshape(1, 3), line_count, axis=0)
            self.gym.add_lines(
                self.viewer,
                self.envs[env_id],
                line_count,
                vertices,
                colors,
            )

    @staticmethod
    def _tensor_list(tensor):
        return [
            round(float(value), 9)
            for value in tensor.detach().cpu().reshape(-1).tolist()
        ]
