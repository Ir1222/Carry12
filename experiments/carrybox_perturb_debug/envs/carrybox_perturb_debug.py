import math

import numpy as np
import torch
from isaacgym.torch_utils import (
    quat_rotate,
    quat_rotate_inverse,
)

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
        self._debug_force_viewer_check_reported = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._debug_relaxed_carry_streak = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._fixed_scene_previous_snapshot = None
        self._fixed_scene_reset_count = 0

    def _reset_actors(self, env_ids):
        super()._reset_actors(env_ids)
        if self._fixed_scene_enabled():
            self._apply_fixed_robot_state(env_ids)

    def _reset_boxes(self, env_ids):
        super()._reset_boxes(env_ids)
        if self._fixed_scene_enabled():
            self._apply_fixed_box_state(env_ids)

    def _reset_task(self, env_ids):
        super()._reset_task(env_ids)
        if self._fixed_scene_enabled():
            self._apply_fixed_goal_state(env_ids)
            self._update_fixed_scene_derived_state(env_ids)

    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)
        if hasattr(self, "_debug_relaxed_carry_streak"):
            self._debug_relaxed_carry_streak[env_ids] = 0
        if self._fixed_scene_enabled():
            self._debug_log_fixed_scene(env_ids)

    def _fixed_scene_enabled(self):
        return bool(getattr(self._fixed_scene_cfg(), "enabled", False))

    def _fixed_scene_cfg(self):
        return getattr(self.cfg, "fixed_scene", None)

    def _apply_fixed_robot_state(self, env_ids):
        cfg = self._fixed_scene_cfg()
        root_pos_local = self._fixed_tensor(cfg.robot_position)
        root_quat = self._normalized_quat(cfg.robot_orientation)

        self.root_states[env_ids, 0:3] = root_pos_local + self.env_origins[env_ids]
        self.root_states[env_ids, 3:7] = root_quat.expand(len(env_ids), -1)

    def _apply_fixed_box_state(self, env_ids):
        cfg = self._fixed_scene_cfg()
        robot_quat = self.root_states[env_ids, 3:7]
        offset_local = self._fixed_tensor(cfg.box_offset_robot_local).expand(
            len(env_ids), -1
        )
        offset_world = quat_rotate(robot_quat, offset_local)

        box_pos = self.root_states[env_ids, 0:3] + offset_world

        self.box_states[env_ids, 0:3] = box_pos

        self.platform_pos[env_ids, 0:2] = box_pos[:, 0:2]
        self.platform_pos[env_ids, 2] = (
            box_pos[:, 2] - 0.5 * self._box_size[env_ids, 2] - self._platform_height
        )

    def _apply_fixed_goal_state(self, env_ids):
        cfg = self._fixed_scene_cfg()
        robot_quat = self.root_states[env_ids, 3:7]
        bearing = math.radians(float(cfg.goal_bearing_deg))
        direction_local = torch.tensor(
            [math.cos(bearing), math.sin(bearing), 0.0],
            dtype=torch.float,
            device=self.device,
        ).expand(len(env_ids), -1)
        direction_world = quat_rotate(robot_quat, direction_local)
        direction_xy = direction_world[:, 0:2]
        direction_xy = direction_xy / torch.clamp(
            torch.linalg.vector_norm(direction_xy, dim=-1, keepdim=True),
            min=1.0e-9,
        )

        distance = float(cfg.goal_distance_m)
        goal_xy = self.box_states[env_ids, 0:2] + distance * direction_xy
        self.goal_pos[env_ids, 0:2] = goal_xy

        self.tar_platform_pos[env_ids, 0:2] = goal_xy

    def _update_fixed_scene_derived_state(self, env_ids):
        self.robot2object_dir[env_ids] = (
            self.box_states[env_ids, :2] - self.root_states[env_ids, :2]
        )
        self.robot2object_dist[env_ids] = torch.norm(
            self.robot2object_dir[env_ids], dim=-1
        )
        self.robot2goal_dir[env_ids] = (
            self.goal_pos[env_ids, :2] - self.root_states[env_ids, :2]
        )
        self.robot2goal_dist[env_ids] = torch.norm(
            self.robot2goal_dir[env_ids], dim=-1
        )
        self.object2start_pos[env_ids] = (
            self.box_states[env_ids, :3] - self.platform_pos[env_ids, :3]
        )
        self.object2start_dist_xy[env_ids] = torch.norm(
            self.object2start_pos[env_ids, :2], dim=-1
        )
        self.object2start_dist_xyz[env_ids] = torch.norm(
            self.object2start_pos[env_ids], dim=-1
        )
        self.object2goal_pos[env_ids] = (
            self.box_states[env_ids, :3] - self.goal_pos[env_ids]
        )
        self.object2goal_dist_xy[env_ids] = torch.norm(
            self.object2goal_pos[env_ids, :2], dim=-1
        )
        self.object2goal_dist_xyz[env_ids] = torch.norm(
            self.object2goal_pos[env_ids], dim=-1
        )
        self.projected_gravity_box[env_ids] = quat_rotate_inverse(
            self.box_states[env_ids, 3:7], self.gravity_vec[env_ids]
        )
        tag_quat = self.box_states[env_ids, 3:7].repeat_interleave(4, dim=0)
        tag_offset_local = self.tag_pos_local[env_ids].reshape(-1, 3)
        tag_offset_world = quat_rotate(tag_quat, tag_offset_local).view(
            len(env_ids), 4, 3
        )
        self.tag_pos[env_ids] = (
            tag_offset_world + self.box_states[env_ids, :3].unsqueeze(1)
        )

    def _fixed_tensor(self, values):
        return torch.tensor(values, dtype=torch.float, device=self.device)

    def _normalized_quat(self, values):
        quat = self._fixed_tensor(values)
        return quat / torch.clamp(torch.linalg.vector_norm(quat), min=1.0e-9)

    def _debug_log_fixed_scene(self, env_ids):
        if not self._debug_includes_env0(env_ids):
            return

        env_id = 0
        self._fixed_scene_reset_count += 1
        snapshot = {
            "robot_root_pose": torch.cat(
                (
                    self.root_states[env_id, 0:3],
                    self.root_states[env_id, 3:7],
                )
            ).detach().clone(),
            "box_position": self.box_states[env_id, 0:3].detach().clone(),
            "goal_xy": self.goal_pos[env_id, 0:2].detach().clone(),
            "target_platform_xy": self.tar_platform_pos[env_id, 0:2].detach().clone(),
            "parent_sampled_state": torch.cat(
                (
                    self.root_states[env_id, 7:13],
                    self.dof_pos[env_id],
                    self.dof_vel[env_id],
                    self.box_states[env_id, 3:13],
                    self.goal_rot[env_id],
                    self.goal_pos[env_id, 2:3],
                    self.tar_platform_pos[env_id, 2:3],
                )
            ).detach().clone(),
        }
        previous = self._fixed_scene_previous_snapshot
        identical = {}
        max_delta = {}
        for name, value in snapshot.items():
            if previous is None:
                identical[name] = "n/a"
                max_delta[name] = float("nan")
                continue
            delta = torch.max(torch.abs(value - previous[name]))
            identical[name] = bool(
                torch.allclose(value, previous[name], atol=1.0e-6, rtol=0.0)
            )
            max_delta[name] = float(delta.item())
        self._fixed_scene_previous_snapshot = snapshot

        box_goal_distance_xy = float(self.object2goal_dist_xy[env_id].item())
        print(
            "[FixedScene]\n"
            f"reset_count={self._fixed_scene_reset_count}\n"
            f"step={self.common_step_counter}\n"
            "env=0\n"
            f"robot_pos={self._debug_list(self.root_states[env_id, 0:3])}\n"
            f"robot_rot={self._debug_list(self.root_states[env_id, 3:7])}\n"
            f"robot_lin_vel={self._debug_list(self.root_states[env_id, 7:10])}\n"
            f"robot_ang_vel={self._debug_list(self.root_states[env_id, 10:13])}\n"
            f"dof_pos={self._debug_list(self.dof_pos[env_id])}\n"
            f"dof_vel={self._debug_list(self.dof_vel[env_id])}\n"
            f"box_pos={self._debug_list(self.box_states[env_id, 0:3])}\n"
            f"box_rot={self._debug_list(self.box_states[env_id, 3:7])}\n"
            f"box_lin_vel={self._debug_list(self.box_states[env_id, 7:10])}\n"
            f"box_ang_vel={self._debug_list(self.box_states[env_id, 10:13])}\n"
            f"goal_pos={self._debug_list(self.goal_pos[env_id])}\n"
            f"goal_rot={self._debug_list(self.goal_rot[env_id])}\n"
            f"target_platform_pos={self._debug_list(self.tar_platform_pos[env_id])}\n"
            f"box_goal_distance_xy={box_goal_distance_xy:.6f}\n"
            f"fixed_robot_root_pose_identical_to_previous={identical['robot_root_pose']}\n"
            f"fixed_box_position_identical_to_previous={identical['box_position']}\n"
            f"fixed_goal_xy_identical_to_previous={identical['goal_xy']}\n"
            f"fixed_target_platform_xy_identical_to_previous={identical['target_platform_xy']}\n"
            f"parent_sampled_state_identical_to_previous={identical['parent_sampled_state']}\n"
            f"fixed_robot_root_pose_max_abs_delta={max_delta['robot_root_pose']:.9f}\n"
            f"fixed_box_position_max_abs_delta={max_delta['box_position']:.9f}\n"
            f"fixed_goal_xy_max_abs_delta={max_delta['goal_xy']:.9f}\n"
            f"fixed_target_platform_xy_max_abs_delta={max_delta['target_platform_xy']:.9f}\n"
            f"parent_sampled_state_max_abs_delta={max_delta['parent_sampled_state']:.9f}\n"
            f"add_noise={bool(self.add_noise)}\n"
            f"box_pos_noise_scale={float(self.box_cfg.pos_noise_scale):.6f}\n"
            f"box_ang_noise_scale={float(self.box_cfg.ang_noise_scale):.6f}\n"
            f"box_reset_mode={self.box_cfg.reset_mode}"
        )

    def _update_box_perturbation_state(self):
        debug = self._debug_collect_gate_state()
        mode = self._debug_trigger_mode()
        if mode == "confirmed_carry":
            super()._update_box_perturbation_state()
        else:
            self._debug_update_box_perturbation_state(mode, debug)
        self._debug_log_carry_gate(debug)

    def _debug_update_box_perturbation_state(self, mode, debug):
        cfg = self.cfg.box_perturbation
        if not bool(cfg.enabled):
            self._clear_box_perturbation_state_for_gate()
            self._debug_relaxed_carry_streak.zero_()
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

        if mode == "time_after_reset":
            trigger_eligible = self._debug_time_after_reset_eligible()
            self._debug_schedule_trigger(trigger_eligible, mode, debug)
        elif mode == "relaxed_carry":
            trigger_eligible = self._debug_relaxed_carry_eligible(debug)
            self._debug_schedule_trigger(trigger_eligible, mode, debug)
        else:
            raise ValueError(f"Unknown debug_trigger_mode: {mode}")

        self.extras["perturb"] = self._build_perturb_log_info()

    def _debug_time_after_reset_eligible(self):
        trigger_step = int(
            getattr(self.cfg.box_perturbation, "debug_trigger_policy_step", 100)
        )
        return self.episode_length_buf >= trigger_step

    def _debug_relaxed_carry_eligible(self, debug):
        gate = self._debug_compute_gate_tensors()
        cfg = self.cfg.box_perturbation
        relaxed = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        if bool(getattr(cfg, "debug_relaxed_require_height_gate", True)):
            relaxed &= gate["height_gate"]
        if bool(getattr(cfg, "debug_relaxed_require_static_gate", False)):
            relaxed &= gate["static_gate"]

        contact_mode = str(getattr(cfg, "debug_relaxed_contact_mode", "either"))
        if contact_mode == "both":
            relaxed &= gate["left_contact"] & gate["right_contact"]
        elif contact_mode == "either":
            relaxed &= gate["left_contact"] | gate["right_contact"]
        elif contact_mode == "none":
            pass
        else:
            raise ValueError(
                "debug_relaxed_contact_mode must be 'both', 'either', or 'none'; "
                f"got {contact_mode!r}"
            )

        self._debug_relaxed_carry_streak[:] = torch.where(
            relaxed,
            self._debug_relaxed_carry_streak + 1,
            torch.zeros_like(self._debug_relaxed_carry_streak),
        )
        threshold = int(getattr(cfg, "debug_relaxed_stable_policy_steps", 1))
        eligible = self._debug_relaxed_carry_streak >= threshold
        env_id = 0
        debug["relaxed_available"] = True
        debug["relaxed_require_height_gate"] = bool(
            getattr(cfg, "debug_relaxed_require_height_gate", True)
        )
        debug["relaxed_require_static_gate"] = bool(
            getattr(cfg, "debug_relaxed_require_static_gate", False)
        )
        debug["relaxed_contact_mode"] = contact_mode
        debug["relaxed_gate"] = bool(relaxed[env_id].item())
        debug["relaxed_streak"] = int(self._debug_relaxed_carry_streak[env_id].item())
        debug["relaxed_threshold"] = threshold
        debug["relaxed_eligible"] = bool(eligible[env_id].item())
        return eligible

    def _debug_schedule_trigger(self, trigger_eligible, mode, debug):
        cfg = self.cfg.box_perturbation
        eligible = (
            trigger_eligible
            & ~self.box_perturb_decision_made_buf
            & (self.box_perturb_event_count_buf < int(cfg.max_events_per_episode))
            & (self.box_perturb_remaining_physics_steps == 0)
        )
        if not torch.any(eligible):
            return

        self.box_perturb_decision_made_buf[eligible] = True
        self._perturb_total_decisions += eligible.float().sum()
        trigger_ids = torch.nonzero(eligible, as_tuple=False).flatten()
        self._debug_log_trigger_bypass(mode, debug, trigger_ids)
        self._schedule_box_perturbation(trigger_ids)

    def _debug_log_trigger_bypass(self, mode, debug, env_ids):
        if not self._debug_includes_env0(env_ids):
            return

        lines = [
            "[DebugTrigger]",
            f"step={self.common_step_counter}",
            "env=0",
            f"mode={mode}",
            f"episode_policy_step={int(self.episode_length_buf[0].item())}",
            f"original_carry_phase={debug['carry_phase']}",
            f"original_confirmed={debug['confirmed']}",
            f"original_eligible={debug['eligible']}",
            f"height_gate={debug['height_gate']}",
            f"static_gate={debug['static_gate']}",
            f"left_contact={debug['left_contact']}",
            f"right_contact={debug['right_contact']}",
        ]
        if debug.get("relaxed_available", False):
            lines.extend(
                [
                    f"relaxed_require_height_gate={debug['relaxed_require_height_gate']}",
                    f"relaxed_require_static_gate={debug['relaxed_require_static_gate']}",
                    f"relaxed_contact_mode={debug['relaxed_contact_mode']}",
                    f"relaxed_gate={debug['relaxed_gate']}",
                    f"relaxed_streak={debug['relaxed_streak']}",
                    f"relaxed_threshold={debug['relaxed_threshold']}",
                    f"relaxed_eligible={debug['relaxed_eligible']}",
                ]
            )
        print("\n".join(lines))

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
        gate = self._debug_compute_gate_tensors()
        env_id = 0
        perturb_cfg = self.cfg.box_perturbation
        debug = {
            "trigger_mode": self._debug_trigger_mode(),
            "carry_phase": bool(self.carry_phase_buf[env_id].item()),
            "confirmed": bool(self.confirmed_carry_buf[env_id].item()),
            "projected_streak": int(gate["projected_streak"][env_id].item()),
            "clearance": float(gate["clearance"][env_id].item()),
            "height_gate": bool(gate["height_gate"][env_id].item()),
            "static_gate": bool(gate["static_gate"][env_id].item()),
            "left_contact": bool(gate["left_contact"][env_id].item()),
            "right_contact": bool(gate["right_contact"][env_id].item()),
            "left_contact_norm_N": float(gate["left_norm"][env_id].item()),
            "right_contact_norm_N": float(gate["right_norm"][env_id].item()),
            "rel_lin_vel": float(gate["rel_lin_vel_norm"][env_id].item()),
            "box_ang_vel": float(gate["box_ang_vel_norm"][env_id].item()),
            "eligible": bool(gate["eligible"][env_id].item()),
            "time_trigger_policy_step": int(
                getattr(perturb_cfg, "debug_trigger_policy_step", 100)
            ),
            "episode_policy_step": int(self.episode_length_buf[env_id].item()),
            "relaxed_available": False,
        }
        return debug

    def _debug_compute_gate_tensors(self):
        cfg = getattr(self.cfg, "carry_phase", None)
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
            "clearance": clearance,
            "height_gate": height_mask,
            "static_gate": static_mask,
            "left_contact": left_contact,
            "right_contact": right_contact,
            "left_norm": left_norm,
            "right_norm": right_norm,
            "rel_lin_vel_norm": rel_lin_vel_norm,
            "box_ang_vel_norm": box_ang_vel_norm,
            "projected_streak": projected_streak,
            "eligible": eligible,
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
            f"trigger_mode={debug['trigger_mode']}\n"
            f"episode_policy_step={debug['episode_policy_step']}\n"
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
            f"box_ang_vel={debug['box_ang_vel']:.6f}\n"
            f"time_trigger_policy_step={debug['time_trigger_policy_step']}"
        )
        if debug.get("relaxed_available", False):
            print(
                "[RelaxedCarryGate]\n"
                f"step={self.common_step_counter}\n"
                "env=0\n"
                f"require_height_gate={debug['relaxed_require_height_gate']}\n"
                f"require_static_gate={debug['relaxed_require_static_gate']}\n"
                f"contact_mode={debug['relaxed_contact_mode']}\n"
                f"relaxed_gate={debug['relaxed_gate']}\n"
                f"relaxed_streak={debug['relaxed_streak']}\n"
                f"relaxed_threshold={debug['relaxed_threshold']}\n"
                f"relaxed_eligible_before_decision={debug['relaxed_eligible']}"
            )

    def _debug_trigger_mode(self):
        return str(
            getattr(
                self.cfg.box_perturbation,
                "debug_trigger_mode",
                "confirmed_carry",
            )
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

    def _draw_debug_vis(self):
        """Draw the instantaneous physical box perturbation force only."""
        self.gym.clear_lines(self.viewer)
        cfg = self.cfg.box_perturbation
        if not bool(cfg.debug_draw_force):
            return

        scale = float(cfg.debug_force_draw_scale_m_per_N)
        line_count = max(1, int(cfg.debug_force_bundle_line_count))
        jitter = max(0.0, float(cfg.debug_force_bundle_jitter_m))
        max_envs = min(self.num_envs, int(cfg.debug_force_draw_max_envs))
        epsilon = 1.0e-6
        color = np.asarray([0.851, 0.144, 0.07], dtype=np.float32)

        for env_id in range(max_envs):
            force_world_t = self.box_perturb_force_tensor[
                env_id, int(self.box_net_contact_force_index), :
            ]
            magnitude = float(torch.linalg.vector_norm(force_world_t).item())
            if magnitude <= epsilon:
                self._debug_force_viewer_check_reported[env_id] = False
                continue

            box_com_world_t = self.box_states[env_id, 0:3]
            env_origin_t = self.env_origins[env_id]
            start_env_t = box_com_world_t - env_origin_t
            # Tail is the box COM; head points along the external force applied
            # by the environment to the box, not the humanoid reaction force.
            draw_vector_t = force_world_t * scale
            end_env_t = start_env_t + draw_vector_t

            self._debug_log_force_viewer_check_once(
                env_id,
                force_world_t,
                draw_vector_t,
                box_com_world_t,
                start_env_t,
                env_origin_t,
                epsilon,
            )

            start = start_env_t.detach().cpu().numpy()
            end = end_env_t.detach().cpu().numpy()
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

    def _debug_log_force_viewer_check_once(
        self,
        env_id,
        force_world,
        draw_vector,
        box_com_world,
        box_com_draw_frame,
        env_origin,
        epsilon,
    ):
        if bool(self._debug_force_viewer_check_reported[env_id].item()):
            return

        force_norm = torch.linalg.vector_norm(force_world)
        draw_norm = torch.linalg.vector_norm(draw_vector)
        if float(force_norm.item()) <= epsilon or float(draw_norm.item()) <= epsilon:
            return

        alignment = torch.dot(force_world / force_norm, draw_vector / draw_norm)
        prefix = (
            "[ForceViewerCheck]"
            if float(alignment.item()) > 0.999
            else "[ForceViewerCheck WARNING]"
        )
        print(
            f"{prefix}\n"
            f"step={self.common_step_counter}\n"
            f"env={env_id}\n"
            f"force_world={self._debug_list(force_world)}\n"
            f"force_norm_N={float(force_norm.item()):.6f}\n"
            f"draw_vector={self._debug_list(draw_vector)}\n"
            f"draw_norm_m={float(draw_norm.item()):.6f}\n"
            f"direction_alignment={float(alignment.item()):.9f}\n"
            f"box_com_world={self._debug_list(box_com_world)}\n"
            f"box_com_draw_frame={self._debug_list(box_com_draw_frame)}\n"
            f"env_origin={self._debug_list(env_origin)}"
        )
        self._debug_force_viewer_check_reported[env_id] = True

    def _debug_includes_env0(self, env_ids):
        return bool((env_ids == 0).any().item()) if env_ids.numel() > 0 else False

    @staticmethod
    def _debug_list(tensor):
        return [round(float(value), 6) for value in tensor.detach().cpu().tolist()]
