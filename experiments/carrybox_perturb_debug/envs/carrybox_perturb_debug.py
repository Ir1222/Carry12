import torch

from legged_gym.envs.g1.carrybox_boxperturb import (
    LeggedRobot as CarryBoxPerturb,
)


class LeggedRobot(CarryBoxPerturb):
    """Diagnostics-only wrapper around the baseline carrybox perturbation env."""

    def _init_buffers(self):
        super()._init_buffers()
        self._debug_force_pulse_seen = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def _update_box_perturbation_state(self):
        debug = self._debug_collect_gate_state()
        super()._update_box_perturbation_state()
        self._debug_log_carry_gate(debug)

    def _schedule_box_perturbation(self, env_ids):
        if self._debug_includes_env0(env_ids):
            cfg = self.cfg.box_perturbation
            print(
                "[PerturbSchedule]\n"
                f"step={self.common_step_counter}\n"
                "env=0\n"
                f"eligible=True\n"
                f"stage={self._debug_stage_name()}\n"
                f"probability={self._debug_probability():.6f}\n"
                f"decision_made={bool(self.box_perturb_decision_made_buf[0].item())}\n"
                f"event_count={int(self.box_perturb_event_count_buf[0].item())}\n"
                f"threshold={int(cfg.stable_confirmed_carry_policy_steps)}\n"
                f"debug_force_event={bool(cfg.debug_force_event)}\n"
                "scheduled=True"
            )
        super()._schedule_box_perturbation(env_ids)

    def _commit_box_perturbation(
        self,
        env_ids,
        direction_local,
        direction_is_world,
        beta,
        direction_ids,
        label,
        pulse_duration_s=None,
    ):
        super()._commit_box_perturbation(
            env_ids,
            direction_local,
            direction_is_world,
            beta,
            direction_ids,
            label,
            pulse_duration_s=pulse_duration_s,
        )
        if self._debug_includes_env0(env_ids):
            force = self.box_perturb_force_tensor[
                0, int(self.box_net_contact_force_index), :
            ]
            print(
                "[PerturbCommit]\n"
                f"step={self.common_step_counter}\n"
                "env=0\n"
                f"label={label}\n"
                f"event_count={int(self.box_perturb_event_count_buf[0].item())}\n"
                f"beta={float(self.box_perturb_beta[0].item()):.6f}\n"
                f"peak_force_N={float(self.box_perturb_peak_force_N[0].item()):.6f}\n"
                f"direction_world={self._debug_list(self.box_perturb_direction_world[0])}\n"
                f"applied_force_world={self._debug_list(force)}\n"
                f"applied_magnitude_N={float(torch.linalg.vector_norm(force).item()):.6f}\n"
                f"elapsed_steps={int(self.box_perturb_elapsed_physics_steps[0].item())}\n"
                f"remaining_steps={int(self.box_perturb_remaining_physics_steps[0].item())}"
            )

    def _apply_box_perturbation_force(self):
        before_remaining = self.box_perturb_remaining_physics_steps.clone()
        before_elapsed = self.box_perturb_elapsed_physics_steps.clone()
        super()._apply_box_perturbation_force()

        env_id = 0
        if self.num_envs <= env_id:
            return
        was_active = int(before_remaining[env_id].item()) > 0
        if not was_active:
            self._debug_force_pulse_seen[env_id] = False
            return

        is_start = int(before_elapsed[env_id].item()) == 0
        is_end = int(self.box_perturb_remaining_physics_steps[env_id].item()) == 0
        if is_start:
            self._debug_log_applied_force("start")
            self._debug_force_pulse_seen[env_id] = True
        if is_end:
            self._debug_log_applied_force("end")
            self._debug_force_pulse_seen[env_id] = False

    def _debug_collect_gate_state(self):
        cfg = getattr(self.cfg, "carry_phase", None)
        env_id = 0
        support_height = torch.full_like(
            self.box_states[:, 2],
            float(getattr(cfg, "support_height", 0.0)),
        )
        platform_top_height = self.platform_pos[:, 2] + 0.5 * self._platform_height
        support_height = torch.maximum(support_height, platform_top_height)
        clearance = self.box_states[:, 2] - 0.5 * self._box_size[:, 2] - support_height
        height_mask = clearance > float(getattr(cfg, "clearance_on", 0.05))

        box_rel_lin_vel = self.box_states[:, 7:10] - self.root_states[:, 7:10]
        rel_lin_vel_norm = torch.linalg.vector_norm(box_rel_lin_vel, dim=-1)
        box_ang_vel_norm = torch.linalg.vector_norm(self.box_states[:, 10:13], dim=-1)
        if bool(getattr(cfg, "use_static_check", True)):
            static_mask = (
                rel_lin_vel_norm < float(getattr(cfg, "max_box_rel_lin_vel", 1.0))
            ) & (box_ang_vel_norm < float(getattr(cfg, "max_box_ang_vel", 3.0)))
        else:
            static_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        contact_threshold = float(getattr(cfg, "contact_force_threshold", 1.0))
        left_norm = torch.linalg.vector_norm(
            self.contact_forces[:, self.left_hand_net_contact_force_index, :], dim=-1
        )
        right_norm = torch.linalg.vector_norm(
            self.contact_forces[:, self.right_hand_net_contact_force_index, :], dim=-1
        )
        left_contact = left_norm > contact_threshold
        right_contact = right_norm > contact_threshold
        projected_streak = torch.where(
            self.confirmed_carry_buf,
            self.confirmed_carry_streak + 1,
            torch.zeros_like(self.confirmed_carry_streak),
        )
        perturb_cfg = self.cfg.box_perturbation
        eligible = (
            (projected_streak >= int(perturb_cfg.stable_confirmed_carry_policy_steps))
            & ~self.box_perturb_decision_made_buf
            & (self.box_perturb_event_count_buf < int(perturb_cfg.max_events_per_episode))
        )
        return {
            "carry_phase": bool(self.carry_phase_buf[env_id].item()),
            "confirmed": bool(self.confirmed_carry_buf[env_id].item()),
            "projected_streak": int(projected_streak[env_id].item()),
            "clearance": float(clearance[env_id].item()),
            "height_gate": bool(height_mask[env_id].item()),
            "static_gate": bool(static_mask[env_id].item()),
            "left_contact": bool(left_contact[env_id].item()),
            "right_contact": bool(right_contact[env_id].item()),
            "left_contact_norm_N": float(left_norm[env_id].item()),
            "right_contact_norm_N": float(right_norm[env_id].item()),
            "rel_lin_vel": float(rel_lin_vel_norm[env_id].item()),
            "box_ang_vel": float(box_ang_vel_norm[env_id].item()),
            "eligible": bool(eligible[env_id].item()),
        }

    def _debug_log_carry_gate(self, debug):
        interval = int(
            getattr(
                self.cfg.box_perturbation,
                "debug_carry_gate_log_interval_policy_steps",
                5,
            )
        )
        if interval <= 0 or self.common_step_counter % interval != 0:
            return
        print(
            "[CarryGate]\n"
            f"step={self.common_step_counter}\n"
            "env=0\n"
            f"carry_phase={debug['carry_phase']}\n"
            f"confirmed={debug['confirmed']}\n"
            f"streak={int(self.confirmed_carry_streak[0].item())}\n"
            f"projected_streak_before_update={debug['projected_streak']}\n"
            f"eligible_before_decision={debug['eligible']}\n"
            f"decision_made={bool(self.box_perturb_decision_made_buf[0].item())}\n"
            f"event_count={int(self.box_perturb_event_count_buf[0].item())}\n"
            f"stage={self._debug_stage_name()}\n"
            f"probability={self._debug_probability():.6f}\n"
            f"clearance={debug['clearance']:.6f}\n"
            f"height_gate={debug['height_gate']}\n"
            f"static_gate={debug['static_gate']}\n"
            f"left_contact={debug['left_contact']}\n"
            f"right_contact={debug['right_contact']}\n"
            f"left_contact_norm_N={debug['left_contact_norm_N']:.6f}\n"
            f"right_contact_norm_N={debug['right_contact_norm_N']:.6f}\n"
            f"rel_lin_vel={debug['rel_lin_vel']:.6f}\n"
            f"box_ang_vel={debug['box_ang_vel']:.6f}"
        )

    def _debug_log_applied_force(self, phase):
        force = self.box_perturb_force_tensor[
            0, int(self.box_net_contact_force_index), :
        ]
        print(
            "[PerturbApplied]\n"
            f"phase={phase}\n"
            f"step={self.common_step_counter}\n"
            "env=0\n"
            f"force_world={self._debug_list(force)}\n"
            f"magnitude_N={float(torch.linalg.vector_norm(force).item()):.6f}\n"
            f"peak_N={float(self.box_perturb_peak_force_N[0].item()):.6f}\n"
            f"beta={float(self.box_perturb_beta[0].item()):.6f}\n"
            f"direction_world={self._debug_list(self.box_perturb_direction_world[0])}\n"
            f"elapsed_steps={int(self.box_perturb_elapsed_physics_steps[0].item())}\n"
            f"remaining_steps={int(self.box_perturb_remaining_physics_steps[0].item())}"
        )

    def _debug_stage_name(self):
        try:
            return self._stage_name()
        except Exception as exc:
            return f"unavailable:{exc}"

    def _debug_probability(self):
        cfg = self.cfg.box_perturbation
        if bool(cfg.debug_force_event):
            return 1.0
        try:
            return float(self._stage_probability())
        except Exception:
            return float("nan")

    def _debug_includes_env0(self, env_ids):
        return bool((env_ids == 0).any().item()) if env_ids.numel() > 0 else False

    @staticmethod
    def _debug_list(tensor):
        return [round(float(value), 6) for value in tensor.detach().cpu().tolist()]
