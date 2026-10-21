"""Command-only specialization. Physics, teacher, A5 adapter and B2 rewards are inherited."""
import math
import torch

from rsl_rl.teacher_handoff import TEACHER_WARMUP, STUDENT_CARRY
from .carrybox import LeggedRobot as FullCarryBox
from .carrybox_teacher_warmup import LeggedRobot as Warmup


class LeggedRobot(Warmup):
    def _init_buffers(self):
        super()._init_buffers()
        h = self.cfg.handoff
        if not math.isfinite(h.mixed_command_hold_s) or h.mixed_command_hold_s < 0:
            raise ValueError("mixed_command_hold_s must be finite and nonnegative")
        rates = h.mixed_command_slew_rates
        if len(rates) != 3 or any(not math.isfinite(v) or v <= 0 for v in rates):
            raise ValueError("Expected three finite positive velocity-command slew rates")
        self._init_mixed_commands()

    def _init_mixed_commands(self):
        self.target_command = self.commands.new_zeros(self.num_envs, 3)
        self.applied_command = torch.zeros_like(self.target_command)
        self.student_control_elapsed = self.commands.new_zeros(self.num_envs)
        # Integer ticks prevent a 2.0s hold drifting by one step in float32.
        self.student_control_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.mixed_released = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.mixed_ramp_active = self.mixed_released.clone()
        self.mixed_target_samples = self.student_control_steps.clone()
        self.mixed_target_changed = self.mixed_released.clone()
        self._reset_mixed_commands(torch.arange(self.num_envs, device=self.device))

    def _reset_mixed_commands(self, ids):
        self.target_command[ids] = self.target_command.new_tensor([self.cfg.handoff.warmup_vx, 0., 0.])
        self.applied_command[ids] = self.target_command[ids]
        self.student_control_elapsed[ids] = 0
        self.student_control_steps[ids] = 0
        self.carry_command_resample_time[ids] = 0
        self.carry_command_mode[ids] = -1  # held initial command, no sampled target mode
        self.mixed_released[ids] = False
        self.mixed_ramp_active[ids] = False
        self.mixed_target_samples[ids] = 0
        self.mixed_target_changed[ids] = False

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if hasattr(self, "target_command"):
            self._reset_mixed_commands(env_ids)

    def _set_student_commands(self, ids):
        # Called by the unchanged supervisor at successful teacher eligibility.
        # This is prepared for the NEXT (first student-owned) physical action.
        self._reset_mixed_commands(ids)
        self.commands[ids] = 0
        self.commands[ids, :3] = self.applied_command[ids]
        self.carry_policy_commands[ids] = self.commands[ids]

    def _update_carry_heading_commands(self):
        student = self.phase != TEACHER_WARMUP
        saved = self.commands[student].clone()
        policy = self.carry_policy_commands[student].clone()
        timers = self.carry_command_resample_time[student].clone()
        FullCarryBox._update_carry_heading_commands(self)
        self.commands[student] = saved
        self.carry_policy_commands[student] = policy
        self.carry_command_resample_time[student] = timers
        # Only physical steps actually owned by the student accrue hold time.
        self.student_control_steps += self.control_mask.long()
        self.student_control_elapsed.copy_(self.student_control_steps * self.dt)

    def _sample_mixed_targets(self, ids):
        if not len(ids):
            return
        saved = self.commands[ids].clone()
        # Reuse the actual velocity-tracking sampler, including all dead zones.
        self._sample_carry_commands(ids)
        self.target_command[ids] = self.commands[ids, :3]
        self.commands[ids] = saved
        self._sample_carry_command_resample_time(ids)
        self.mixed_target_samples[ids] += 1
        self.mixed_target_changed[ids] = True

    def _advance_mixed_commands(self):
        self.mixed_target_changed[:] = False
        student = self.phase != TEACHER_WARMUP
        released_before = self.mixed_released.clone()
        if self.cfg.handoff.command_suite == "mixed":
            release = (self.phase == STUDENT_CARRY) & ~released_before
            release &= self.student_control_elapsed + 1e-7 >= self.cfg.handoff.mixed_command_hold_s
            self.mixed_released |= release
            # Timers run from target sampling, irrespective of ramp completion.
            self.carry_command_resample_time[released_before] -= self.dt
            due = release | (released_before & (self.carry_command_resample_time <= 1e-7))
            self._sample_mixed_targets(due.nonzero().flatten())
        elif self.cfg.handoff.command_suite == "fixed":
            # Explicit evaluation override; retain the same initial hold and adapter.
            release = (self.phase == STUDENT_CARRY) & ~released_before
            release &= self.student_control_elapsed + 1e-7 >= self.cfg.handoff.mixed_command_hold_s
            self.mixed_released |= release
            self.target_command[release] = self.commands.new_tensor(self.cfg.handoff.fixed_command)
            axes = self.target_command.abs() > 1e-6
            axis_count = axes.sum(-1)
            mode = torch.where(axis_count == 0, 0, torch.where(axis_count == 1, axes.long().argmax(-1) + 1, 4))
            self.carry_command_mode[release] = mode[release]
            self.mixed_target_changed[release] = True
            self.mixed_target_samples[release] += 1
        else:
            raise ValueError("command_suite must be fixed or mixed")
        active = student & self.mixed_released
        delta = self.target_command - self.applied_command
        limit = self.commands.new_tensor(self.cfg.handoff.mixed_command_slew_rates) * self.dt
        self.applied_command[active] += delta.clamp(-limit, limit)[active]
        self.mixed_ramp_active.copy_(active & ((self.target_command - self.applied_command).abs() > 1e-6).any(-1))
        self.commands[student] = 0
        self.commands[student, :3] = self.applied_command[student]
        self.carry_policy_commands[student] = self.commands[student]

    def compute_observations(self):
        # Called AFTER reward/snapshot and per-env reset. The actor now observes
        # exactly the command that step() will snapshot for its physical action.
        self._advance_mixed_commands()
        super().compute_observations()

    def step(self, actions):
        # Preserve metadata for THIS action before any future target is sampled.
        self._control_target = self.target_command.clone()
        self._control_mode = self.carry_command_mode.clone()
        self._control_ramp = self.mixed_ramp_active.clone()
        self._control_hold = ~self.mixed_released.clone()
        self._control_target_changed = self.mixed_target_changed.clone()
        return super().step(actions)

    def compute_reward(self):
        # The supervisor may have just prepared the first student command while
        # this last physical action still belonged to the teacher. Restore the
        # physical reference only around reward/snapshot construction.
        future = self.carry_policy_commands.clone()
        self.carry_policy_commands[self.control_mask, :3] = self.control_command[self.control_mask]
        history = self.carry_history_valid.clone()
        super().compute_reward()
        self.carry_policy_commands.copy_(future)
        self.last_step_snapshot.update(target_command=self._control_target,
            target_mode=self._control_mode, command_ramp=self._control_ramp,
            command_hold=self._control_hold, target_sampled=self._control_target_changed)
        self.last_step_snapshot.update(box_relative_pos=self.box_relative_pos.clone(),
            box_relative_speed=torch.where(history, self.box_relative_velocity.norm(dim=-1), float("nan")),
            hand_box_slip=torch.where(history[:, None], self.hand_slip, float("nan")))
        if self.gait_enabled:
            self.last_step_snapshot.update(
                gait_slip=torch.where(self.gait_contact_filtered, self.gait_horizontal_speed, float("nan")),
                gait_clearance=torch.where(self.gait_swing, self.gait_clearance, float("nan")),
                gait_scuff=(self.gait_swing & (self.gait_clearance < self.cfg.gait.scuff_height_m)).float(),
                gait_air_time=self.gait_air_time.clone(),
                gait_touchdown_air_time=torch.where(self.gait_touchdown, self.gait_preceding_air_time, float("nan")))
