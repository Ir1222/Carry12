"""Real Full CarryBox pickup followed by asynchronous student control.

Locomotion supplies reward/geometry helpers. Reset, physical pickup, observation
construction and pre-carry commands use Full CarryBox. No simulator setters are
called at the handoff.
"""
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from rsl_rl.teacher_handoff import (
    TEACHER_WARMUP, HANDOFF_TRANSITION, STUDENT_CARRY, adapt_action,
    checkpoint_actor, joint_groups, pose_excess, file_sha256,
)
from .carrybox import LeggedRobot as FullCarryBox
from .carrybox_locomotion import LeggedRobot as CarryLocomotion


class LeggedRobot(CarryLocomotion):
    def _init_buffers(self):
        super()._init_buffers()
        h = self.cfg.handoff
        if self.cfg.domain_rand.delay or self.cfg.control.control_type != "P":
            raise ValueError("Teacher handoff requires P control and delay=False")
        if not (0 < h.transition_duration_s < h.transition_timeout_s
                and min(h.warmup_timeout_s, h.student_timeout_s, h.stable_s) > 0
                and h.hysteresis_ratio >= 1 and h.imitation_window_s > 0):
            raise ValueError("Invalid handoff timeouts/hysteresis")
        positive = (h.lift_min_m, h.tilt_max_deg, h.relative_velocity_max,
                    h.box_omega_max, h.posture_penalty_cap)
        if any(not math.isfinite(v) or v <= 0 for v in positive):
            raise ValueError("Handoff physical limits and posture cap must be finite/positive")
        if len(h.fixed_command) != 3 or any(not math.isfinite(v) for v in h.fixed_command):
            raise ValueError("fixed_command must contain three finite values")
        self.teacher, spec = checkpoint_actor(h.teacher_checkpoint, self.device)
        if spec["dims"][0] != self.num_obs or spec["dims"][-1] != self.num_actions:
            raise ValueError("Teacher observation/action contract mismatch")
        if self.num_obs != 738 or self.num_actions != 29:
            raise ValueError("This task requires the audited 738D/29D contract")
        self.groups = joint_groups(self.dof_names)
        from legged_gym import LEGGED_GYM_ROOT_DIR
        canonical = (Path(LEGGED_GYM_ROOT_DIR) / "resources/config/joint_id.txt").read_text().splitlines()
        expected_joints = [line.split()[1] for line in canonical if line.strip()]
        if list(self.dof_names) != expected_joints:
            raise ValueError("Simulator joint order differs from the audited checkpoint contract")
        self.phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.phase_age = self.dof_pos.new_zeros(self.num_envs)
        self.handoff_age = self.phase_age.clone()
        self.stable_time = self.phase_age.clone()
        self.pickup_seen = torch.zeros_like(self.phase, dtype=torch.bool)
        self.handoff_seen = self.pickup_seen.clone()
        self.failure_reason = self.phase.clone()
        self.warmup_elapsed = self.phase_age.clone()
        self.raw_student_action = torch.zeros_like(self.actions)
        self.applied_action = torch.zeros_like(self.actions)
        self.applied_pd_target = torch.zeros_like(self.actions)
        self.rates = self.dof_pos.new_empty(self.num_actions)
        for name, ids in self.groups.items():
            self.rates[ids] = getattr(h, "target_rate_" + name)
        if h.joint_target_rates is not None:
            self.rates = self.dof_pos.new_tensor(h.joint_target_rates)
        if self.rates.shape != (29,) or not torch.all(self.rates > 0):
            raise ValueError("Expected 29 positive PD target rate limits")
        self.pose_reference = None
        if h.posture_reference:
            with open(h.posture_reference, encoding="utf-8") as stream:
                reference = json.load(stream)
            from rsl_rl.teacher_handoff_reference import validate_reference
            validate_reference(reference, self.dof_names)
            if reference["teacher_sha256"] != file_sha256(h.teacher_checkpoint):
                raise ValueError("Posture reference was collected using a different teacher")
            self.pose_ids = [self.dof_names.index(n) for n in reference["joint_names"]]
            self.pose_lower = self.dof_pos.new_tensor(reference["lower"])
            self.pose_upper = self.dof_pos.new_tensor(reference["upper"])
            self.pose_softness = self.dof_pos.new_tensor(reference["softness"])
            self.pose_reference = reference
        if self.cfg.rewards.scales.carry_upper_body_pose and self.pose_reference is None:
            raise ValueError("Collect and validate a real teacher posture reference first")
        self.gait_enabled = any(getattr(self.cfg.rewards.scales, name) != 0 for name in
                                ("gait_foot_slip", "gait_swing_clearance", "gait_foot_scuff", "gait_air_time"))
        if self.gait_enabled:
            self._init_gait_buffers()

    def _init_gait_buffers(self):
        from isaacgym import gymapi
        from legged_gym import LEGGED_GYM_ROOT_DIR
        from legged_gym.scripts.gait_diagnostics import build_gait_geometry
        if self.cfg.terrain.mesh_type != "plane":
            raise ValueError("B2 calibrated clearance requires plane terrain")
        g = self.cfg.gait
        for key in ("contact_threshold_N", "moving_threshold", "horizontal_speed_scale_m_s",
                    "target_height_m", "scuff_height_m", "air_time_range_s",
                    "geometry_fit_tolerance_m", "geometry_validation_tolerance_m"):
            if not math.isfinite(getattr(g, key)) or getattr(g, key) <= 0:
                raise ValueError("Invalid gait parameter: " + key)
        if (not isinstance(g.geometry_max_points, int) or not 4 <= g.geometry_max_points <= 256
                or not math.isfinite(g.slip_tolerance_m_s) or g.slip_tolerance_m_s < 0
                or not math.isfinite(g.air_time_min_s) or g.air_time_min_s < 0):
            raise ValueError("Invalid gait geometry/tolerance settings")
        names = self.gym.get_actor_rigid_body_names(self.envs[0], self.actor_handles[0])
        foot_names = ("left_ankle_roll_link", "right_ankle_roll_link")
        indices = []
        for name in foot_names:
            if name not in names:
                raise ValueError("Missing gait body: " + name)
            resolved = [self.gym.get_actor_rigid_body_index(e, a, names.index(name), gymapi.DOMAIN_ENV)
                        for e, a in zip(self.envs, self.actor_handles)]
            if len(set(resolved)) != 1 or not 0 <= resolved[0] < self.contact_forces.shape[1]:
                raise ValueError("Invalid environment-domain gait body: " + name)
            indices.append(resolved[0])
        if indices[0] == indices[1]:
            raise ValueError("Gait bodies must be distinct")
        self.gait_feet_indices = torch.tensor(indices, device=self.device, dtype=torch.long)
        urdf = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        self.gait_geometry = build_gait_geometry(urdf, g.geometry_max_points,
                                                g.geometry_fit_tolerance_m, g.geometry_validation_tolerance_m)
        bodies = self.gait_geometry["bodies"]
        count = max(bodies[name]["point_count"] for name in foot_names)
        # Pad using an existing point: duplicated vertices do not change min Z.
        points = [bodies[name]["points"] + [bodies[name]["points"][0]] * (count - bodies[name]["point_count"])
                  for name in foot_names]
        self.gait_sole_points = self.dof_pos.new_tensor(points)
        self.gait_geometry["body_indices_domain_env"] = dict(zip(foot_names, indices))
        print("[B2 geometry] " + json.dumps({name: bodies[name]["error"] for name in foot_names}, sort_keys=True))
        self.gait_contact_raw = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
        self.gait_previous_raw = self.gait_contact_raw.clone()
        self.gait_contact_filtered = self.gait_contact_raw.clone()
        self.gait_touchdown = self.gait_contact_raw.clone()
        self.gait_swing = self.gait_contact_raw.clone()
        self.gait_stance_seen = self.gait_contact_raw.clone()
        self.gait_air_time = self.dof_pos.new_zeros(self.num_envs, 2)
        self.gait_preceding_air_time = self.gait_air_time.clone()
        self.gait_horizontal_velocity = self.dof_pos.new_zeros(self.num_envs, 2, 2)
        self.gait_horizontal_speed = self.gait_air_time.clone()
        self.gait_clearance = self.gait_air_time.clone()
        self.gait_previous_owner = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.gait_history_valid = self.gait_previous_owner.clone()
        self.gait_reward_diagnostics = {}

    def _reset_gait_history(self, ids):
        for name in ("contact_raw", "previous_raw", "contact_filtered", "touchdown", "swing", "stance_seen",
                     "air_time", "preceding_air_time", "horizontal_velocity", "horizontal_speed", "clearance",
                     "previous_owner", "history_valid"):
            getattr(self, "gait_" + name)[ids] = 0

    def _gait_moving(self):
        command = self.control_command  # command that produced this physics step
        limit = self.cfg.gait.moving_threshold
        return self.control_mask & ((command[:, :2].norm(dim=-1) > limit) | (command[:, 2].abs() > limit))

    def _update_gait_state(self):
        # Exactly once per physics/reward step, before auto-reset, using PRE-step ownership.
        changed = ~self.gait_history_valid | (self.gait_previous_owner != self.control_mask)
        self._reset_gait_history(changed)
        history = ~changed
        body = self.rigid_body_states[:, self.gait_feet_indices]
        raw = self.contact_forces[:, self.gait_feet_indices].norm(dim=-1) > self.cfg.gait.contact_threshold_N
        filtered = raw | self.gait_previous_raw
        moving = self._gait_moving()
        self.gait_touchdown.copy_(filtered & ~self.gait_contact_filtered & history[:, None]
                                 & self.gait_stance_seen & (self.gait_air_time > 0))
        self.gait_preceding_air_time.copy_(self.gait_air_time)
        self.gait_contact_raw.copy_(raw)
        self.gait_contact_filtered.copy_(filtered)
        self.gait_previous_raw.copy_(raw)
        self.gait_swing.copy_(~filtered)
        # No reward for an airborne episode/owner boundary or time accrued standing.
        self.gait_stance_seen &= moving[:, None]
        self.gait_stance_seen |= filtered & moving[:, None]
        self.gait_air_time.copy_(torch.where(~filtered & moving[:, None], self.gait_air_time + self.dt, 0.0))
        self.gait_horizontal_velocity.copy_(body[:, :, 7:9])
        self.gait_horizontal_speed.copy_(self.gait_horizontal_velocity.norm(dim=-1))
        points = self.gait_sole_points[None].expand(self.num_envs, -1, -1, -1)
        q = F.normalize(body[:, :, None, 3:7], dim=-1)
        xyz = q[..., :3].expand_as(points)
        t = 2 * torch.cross(xyz, points, dim=-1)
        rotated = points + q[..., 3:4] * t + torch.cross(xyz, t, dim=-1)
        # The existing simulator plane is global world Z=0, not env-origin Z.
        self.gait_clearance.copy_(body[:, :, 2] + rotated[..., 2].amin(dim=-1))
        self.gait_previous_owner.copy_(self.control_mask)
        self.gait_history_valid[:] = True

    def _reward_gait_foot_slip(self):
        g = self.cfg.gait
        slip = ((self.gait_horizontal_speed - g.slip_tolerance_m_s) / g.horizontal_speed_scale_m_s).clamp(0, 1)
        return (slip * self.gait_contact_filtered).mean(-1) * self._gait_moving()

    def _reward_gait_swing_clearance(self):
        g = self.cfg.gait
        gate = ((self.gait_horizontal_speed - g.slip_tolerance_m_s) / g.horizontal_speed_scale_m_s).clamp(0, 1)
        score = (self.gait_clearance / g.target_height_m).clamp(0, 1)
        return (self.gait_swing * gate * score).mean(-1) * self._gait_moving()

    def _reward_gait_foot_scuff(self):
        g = self.cfg.gait
        speed = (self.gait_horizontal_speed / g.horizontal_speed_scale_m_s).clamp(0, 1)
        score = torch.exp(-self.gait_clearance.clamp_min(0) / g.scuff_height_m)
        return (self.gait_swing * speed * score).mean(-1) * self._gait_moving()

    def _reward_gait_air_time(self):
        g = self.cfg.gait
        score = ((self.gait_preceding_air_time - g.air_time_min_s) / g.air_time_range_s).clamp(0, 1)
        return (score * self.gait_touchdown).sum(-1).clamp_max(1) * self._gait_moving()

    # Explicitly bypass reference-based locomotion reset and forced carry gates.
    _reset_actors = FullCarryBox._reset_actors
    _reset_boxes = FullCarryBox._reset_boxes
    _compute_is_stage_carry = FullCarryBox._compute_is_stage_carry

    def _compute_carry_velocity_active(self):
        return FullCarryBox._compute_carry_velocity_active(self) | (self.phase != TEACHER_WARMUP)

    def _reset_task(self, env_ids):
        FullCarryBox._reset_task(self, env_ids)
        self.commands[env_ids] = 0.0
        self.commands[env_ids, 0] = self.cfg.handoff.warmup_vx

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        summary = {
            "completed_count": float(len(env_ids)),
            "pickup_success": self.pickup_seen[env_ids].float().mean(),
            "handoff_eligible": self.handoff_seen[env_ids].float().mean(),
            "warmup_seconds": self.warmup_elapsed[env_ids].mean(),
        }
        for code, name in enumerate(("none", "physical", "warmup_timeout", "transition_timeout", "carry_timeout")):
            summary["reason_" + name] = (self.failure_reason[env_ids] == code).float().mean()
        FullCarryBox.reset_idx(self, env_ids)
        self.extras["episode"].update(summary)
        self.phase[env_ids] = TEACHER_WARMUP
        for tensor in (self.phase_age, self.handoff_age, self.stable_time,
                       self.warmup_elapsed, self.grasp_loss_time,
                       self.carry_command_resample_time):
            tensor[env_ids] = 0
        for tensor in (self.pickup_seen, self.handoff_seen, self.carry_history_valid):
            tensor[env_ids] = False
        for tensor in (self.carry_log_sums, self.carry_log_steps,
                       self.carry_lower_log_sums, self.carry_lower_log_steps,
                       self.raw_student_action, self.applied_action, self.applied_pd_target):
            tensor[env_ids] = 0
        self.failure_reason[env_ids] = 0
        self.obs_buf[env_ids] = 0
        self.actions[env_ids] = 0
        if getattr(self, "gait_enabled", False):
            self._reset_gait_history(env_ids)

    def reset(self):
        # Match Full BaseTask's initial zero-action physics step. Teacher must
        # not act on an all-zero history before the first observation exists.
        self._initializing = True
        try:
            obs, critic = super().reset()
        finally:
            self._initializing = False
        self.extras.pop("episode", None)
        return obs, critic

    def _set_student_commands(self, ids):
        if self.cfg.handoff.command_suite == "fixed":
            self.commands[ids] = 0
            self.commands[ids, :3] = self.commands.new_tensor(self.cfg.handoff.fixed_command)
        elif self.cfg.handoff.command_suite == "mixed":
            self._sample_carry_commands(ids)
        else:
            raise ValueError("command_suite must be fixed or mixed")
        self._sample_carry_command_resample_time(ids)
        self.carry_policy_commands[ids, :3] = self.commands[ids, :3]

    def _update_carry_heading_commands(self):
        # Full's command transform is evaluated only on teacher-owned commands.
        # Restore ALL student command fields, including raw lateral/yaw values.
        student = self.phase != TEACHER_WARMUP
        saved = self.commands[student].clone()
        FullCarryBox._update_carry_heading_commands(self)
        self.commands[student] = saved
        self.carry_command_resample_time[student] -= self.dt
        due = (student & (self.carry_command_resample_time <= 0)).nonzero().flatten()
        self._set_student_commands(due)
        self.carry_policy_commands[student, :3] = self.commands[student, :3]

    def readiness(self, ratio=1.0):
        h = self.cfg.handoff
        bottom = self.box_states[:, 2] - self._box_size[:, 2] / 2
        support = torch.maximum(self.platform_pos[:, 2] + self._platform_height / 2,
                                self.env_origins[:, 2])
        lift = bottom - support
        forces = self.contact_forces[:, self.hand_colli_indices].norm(dim=-1)
        bilateral = (forces > self.cfg.rewards.hand_contact_threshold).all(-1)
        tilt = torch.acos((-self.projected_gravity_box[:, 2]).clamp(-1, 1)) * 180 / math.pi
        relative = (self.box_states[:, 7:10] - self.rigid_body_states[:, self.upper_body_index, 7:10]).norm(dim=-1)
        omega = self.box_states[:, 10:13].norm(dim=-1)
        ready = bilateral & (lift > h.lift_min_m / ratio) & (tilt < h.tilt_max_deg * ratio)
        ready &= (relative < h.relative_velocity_max * ratio) & (omega < h.box_omega_max * ratio)
        ready &= self.robot2object_dist < self.cfg.rewards.thresh_robot2object * ratio
        return ready, dict(lift=lift, bilateral=bilateral, tilt=tilt, relative=relative, omega=omega)

    def check_termination(self):
        # Parent's 20s limit is superseded by independent phase budgets.
        FullCarryBox.check_termination(self)
        self._update_hand_contact_state()
        physical = self.reset_buf & ~self.time_out_buf
        warm = self.phase == TEACHER_WARMUP
        transition = self.phase == HANDOFF_TRANSITION
        student = self.phase == STUDENT_CARRY
        controlled = ~warm
        self.phase_age += self.dt
        self.handoff_age[controlled] += self.dt
        self.warmup_elapsed[warm] += self.dt
        strict, metrics = self.readiness()
        loose, _ = self.readiness(self.cfg.handoff.hysteresis_ratio)
        self.pickup_seen |= (metrics["lift"] > self.cfg.handoff.lift_min_m) & metrics["bilateral"]
        self.stable_time = torch.where(warm & strict, self.stable_time + self.dt,
                                       torch.where(warm & loose, self.stable_time, 0.0))
        lost = ~self.hand_contact_filt.any(-1)
        self.grasp_loss_time = torch.where(controlled & lost, self.grasp_loss_time + self.dt, 0.0)
        bottom = self.box_states[:, 2] - self._box_size[:, 2] / 2 - self.env_origins[:, 2]
        physical |= controlled & ((bottom < self.cfg.rewards.box_drop_height)
                      | (metrics["tilt"] > self.cfg.rewards.box_tilt_termination_deg)
                      | (self.robot2object_dist > self.cfg.rewards.robot_box_max_distance)
                      | (self.grasp_loss_time > self.cfg.rewards.grasp_loss_grace_s))
        warm_timeout = warm & (self.phase_age >= self.cfg.handoff.warmup_timeout_s)
        trans_timeout = transition & (self.phase_age >= self.cfg.handoff.transition_timeout_s)
        carry_timeout = student & (self.phase_age >= self.cfg.handoff.student_timeout_s)
        self.failure_reason[:] = 0
        self.failure_reason[carry_timeout] = 4
        self.failure_reason[trans_timeout] = 3
        self.failure_reason[warm_timeout] = 2
        self.failure_reason[physical] = 1
        # Only successful carry horizon is a time truncation. Supervisor failures
        # are genuine terminal failures and must not bootstrap.
        self.time_out_buf = carry_timeout & ~physical
        self.reset_buf = physical | warm_timeout | trans_timeout | carry_timeout
        entering = warm & (self.stable_time >= self.cfg.handoff.stable_s) & ~self.reset_buf
        entered = entering.nonzero().flatten()
        self.phase[entered] = HANDOFF_TRANSITION
        self.phase_age[entered] = 0
        self.handoff_age[entered] = 0
        self.handoff_seen[entered] = True
        self._set_student_commands(entered)
        completed = transition & loose & (self.phase_age >= self.cfg.handoff.transition_duration_s) & ~self.reset_buf
        self.phase[completed] = STUDENT_CARRY
        self.phase_age[completed] = 0
        self.extras["time_outs"] = self.time_out_buf.clone()

    def _critic_context(self):
        return torch.cat((F.one_hot(self.phase, 3).float(), self.phase_age[:, None],
                          self.handoff_age[:, None], self.actions), dim=-1)

    def compute_observations(self):
        FullCarryBox.compute_observations(self)
        self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, self._critic_context()), -1)

    def compute_termination_observations(self, env_ids):
        original = FullCarryBox.compute_termination_observations(self, env_ids)
        return torch.cat((original, self._critic_context()[env_ids]), -1)

    def _prepare_reward_function(self):
        super()._prepare_reward_function()
        # Mask BEFORE Full's episode_sums accumulation as well as before PPO.
        # Late lookup of control_mask uses action ownership at the current step.
        self.reward_functions = [
            lambda function=function: function() * self.control_mask
            for function in self.reward_functions
        ]
        if getattr(self, "gait_enabled", False):
            self.reward_functions = [self._gait_logged_reward(name, function)
                                     for name, function in zip(self.reward_names, self.reward_functions)]

    def _gait_logged_reward(self, name, function):
        def measured():
            value = function()
            self.gait_reward_diagnostics["reward_raw/" + name] = value.detach()
            # reward_scales already includes dt, exactly as in the parent sum.
            self.gait_reward_diagnostics["reward_weighted/" + name] = value.detach() * self.reward_scales[name]
            return value
        return measured

    def _reward_termination(self):
        return FullCarryBox._reward_termination(self) * self.control_mask

    def compute_reward(self):
        if getattr(self, "gait_enabled", False):
            self._update_gait_state()
        super().compute_reward()
        self.rew_buf *= self.control_mask  # pre-step ownership, not post-step phase
        ready, metrics = self.readiness()
        pose_error = torch.zeros_like(self.phase_age)
        if self.pose_reference is not None:
            pose_error = pose_excess(self.dof_pos[:, self.pose_ids], self.pose_lower,
                                      self.pose_upper, self.pose_softness).square().mean(-1).sqrt()
        self.step_diagnostics = dict(metrics)
        if getattr(self, "gait_enabled", False):
            self.step_diagnostics.update(self.gait_reward_diagnostics)
        if self.pose_reference is not None:
            self.step_diagnostics["posture_error"] = pose_error
        self.last_step_snapshot = {
            "q": self.dof_pos.clone(), "dq": self.dof_vel.clone(),
            "raw_action": self.raw_student_action.clone(),
            "applied_action": self.applied_action.clone(),
            "pd_target": self.joint_pos_target.clone(),
            "ready": ready.clone(), "phase": self.phase.clone(),
            "control_phase": self.control_phase.clone(),
            "handoff_age": self.handoff_age.clone(), "phase_age": self.phase_age.clone(),
            "warmup_elapsed": self.warmup_elapsed.clone(),
            "pickup": self.pickup_seen.clone(), "eligible": self.handoff_seen.clone(),
            "failure_reason": self.failure_reason.clone(),
            "command": self.control_command.clone(),
            "actual": torch.stack((self.base_lin_vel_yaw[:, 0], self.base_lin_vel_yaw[:, 1],
                                     self.base_yaw_rate_world), -1),
            "box_size": self._box_size.clone(),
            "posture_error": pose_error.clone(),
            **{k: v.clone() for k, v in self.step_diagnostics.items()},
        }
        if getattr(self, "gait_diagnostics", None) is not None:
            self.gait_diagnostics.capture_state(self)

    def _reward_carry_upper_body_pose(self):
        if self.pose_reference is None:
            return torch.zeros_like(self.rew_buf)
        reference = self.pose_reference
        size = self._box_size
        low = size.new_tensor(reference["box_size_min"]) - self.cfg.handoff.posture_box_size_tolerance_m
        high = size.new_tensor(reference["box_size_max"]) + self.cfg.handoff.posture_box_size_tolerance_m
        matched = ((size >= low) & (size <= high)).all(-1)
        excess = pose_excess(self.dof_pos[:, self.pose_ids], self.pose_lower,
                             self.pose_upper, self.pose_softness)
        # Penalty (scale positive). Zero inside the data-derived feasible family.
        return -excess.square().mean(-1).clamp_max(self.cfg.handoff.posture_penalty_cap) * matched * self.control_mask

    def step(self, actions):
        self.extras.pop("episode", None)
        control_phase = self.phase.clone()
        self.control_phase = control_phase
        self.control_command = self.carry_policy_commands[:, :3].clone()
        self.control_mask = (self.phase != TEACHER_WARMUP).clone()
        self.raw_student_action.copy_(actions)
        with torch.no_grad():
            teacher = self.teacher(self.obs_buf).detach()
        h = self.cfg.handoff
        command = self.carry_policy_commands[:, :3]
        safe_command = (command[:, 1].abs() <= h.imitation_max_vy) & (command[:, 2].abs() <= h.imitation_max_yaw)
        safe_command &= (command[:, 0] >= 0) & (command[:, 0] <= h.imitation_max_vx)
        ready, _ = self.readiness()
        imitation_weight = (1 - self.handoff_age / h.imitation_window_s).clamp(0, 1)
        imitation_weight *= self.control_mask & safe_command & ready
        desired = torch.where(self.control_mask[:, None], actions, teacher)
        if getattr(self, "_initializing", False):
            desired = torch.zeros_like(desired)
        applied = adapt_action(desired, self.actions, self.rates, self.dt,
                               self.cfg.control.action_scale, self.cfg.normalization.clip_actions,
                               self.control_mask, h.adapter)
        if getattr(self, "gait_diagnostics", None) is not None:
            self.gait_diagnostics.capture_action(self, desired, applied)
        # Capture diagnostics before FullCarryBox auto-resets done environments.
        self.applied_action.copy_(applied)
        self.applied_pd_target.copy_(self.default_dof_pos + applied * self.cfg.control.action_scale)
        valid = self.control_mask.clone()
        result = super().step(applied)
        result[4]["student_valid"] = valid
        result[4]["teacher_action"] = teacher
        result[4]["imitation_weight"] = imitation_weight.detach()
        result[4]["applied_actions"] = self.last_step_snapshot["applied_action"]
        result[4]["pd_targets"] = self.last_step_snapshot["pd_target"]
        result[4]["control_phase"] = control_phase
        # Parent updates last_actions after reset; restore reset-only histories.
        ids = result[5]
        self.actions[ids] = 0
        self.last_actions[ids] = 0
        self.last_last_actions[ids] = 0
        return result
