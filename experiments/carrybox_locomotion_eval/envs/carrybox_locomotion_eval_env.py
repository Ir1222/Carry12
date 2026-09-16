"""Deterministic evaluator subclass of the carry-only specialist task."""

import math

import torch

from legged_gym.envs.g1 import carrybox_locomotion


class CarryBoxLocomotionEvalEnv(carrybox_locomotion.LeggedRobot):
    """Carry-only environment with a trial-constant externally set command."""

    def _init_buffers(self):
        super()._init_buffers()
        self._eval_command_active = False
        self._eval_requested_command = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device
        )
        self.eval_last_termination_reason = [""] * self.num_envs

    def set_evaluation_command(self, command):
        requested = torch.as_tensor(
            command, dtype=self.commands.dtype, device=self.device
        )
        if requested.shape != (3,):
            raise ValueError("Evaluation command must contain [vx, vy, yaw_rate]")
        self._eval_requested_command[:] = requested
        self._eval_command_active = True
        self._apply_evaluation_command()

    def _apply_evaluation_command(self, env_ids=None):
        if not self._eval_command_active:
            return
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.commands[env_ids, :3] = self._eval_requested_command[env_ids]
        self.carry_policy_commands[env_ids, :3] = (
            self._eval_requested_command[env_ids]
        )

    def _sample_carry_commands(self, env_ids):
        if self._eval_command_active:
            self._apply_evaluation_command(env_ids)
            return
        super()._sample_carry_commands(env_ids)

    def _sample_carry_command_resample_time(self, env_ids):
        if self._eval_command_active:
            self.carry_command_resample_time[env_ids] = float("inf")
            return
        super()._sample_carry_command_resample_time(env_ids)

    def _update_carry_heading_commands(self):
        if not self._eval_command_active:
            super()._update_carry_heading_commands()
            return
        # The native implementation retains raw yaw rate for the actor while
        # integrating carry_heading_ref. An infinite timer disables only its
        # random-resampling branch; heading integration remains untouched.
        self._apply_evaluation_command()
        self.carry_command_resample_time[:] = float("inf")
        super()._update_carry_heading_commands()
        self._assert_command_tensors()

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if len(env_ids) == 0:
            return
        self._apply_evaluation_command(env_ids)
        self.carry_command_resample_time[env_ids] = float("inf")
        # A new benchmark trial must not inherit history from the prior command.
        # BaseTask.reset() will append exactly one fresh frame via its mandatory
        # zero-action simulation step.
        self.obs_buf[env_ids] = 0.0
        if hasattr(self, "amp_obs_buf"):
            self.amp_obs_buf[env_ids] = 0.0

    def clear_evaluation_outcome(self):
        self.eval_last_termination_reason = [""] * self.num_envs

    def _assert_command_tensors(self, atol=1.0e-6):
        expected = self._eval_requested_command
        if not torch.allclose(
            self.commands[:, :3], expected, atol=atol, rtol=0.0
        ):
            raise AssertionError("Raw evaluation command changed during rollout")
        if not torch.allclose(
            self.carry_policy_commands[:, :3], expected, atol=atol, rtol=0.0
        ):
            raise AssertionError("Policy evaluation command changed during rollout")

    def assert_evaluation_command(self, obs=None, atol=1.0e-6):
        self._assert_command_tensors(atol=atol)
        if obs is not None:
            current_command_obs = obs[:, -self.num_task_obs :][:, -3:]
            if not torch.allclose(
                current_command_obs,
                self._eval_requested_command,
                atol=atol,
                rtol=0.0,
            ):
                raise AssertionError(
                    "Current task-observation command differs from requested command"
                )

    def _termination_masks(self):
        box_bottom = self.box_states[:, 2] - 0.5 * self._box_size[:, 2]
        tilt_cos = math.cos(
            math.radians(self.cfg.rewards.box_tilt_termination_deg)
        )
        return (
            ("timeout", self.time_out_buf),
            ("box_drop", box_bottom < self.cfg.rewards.box_drop_height),
            (
                "robot_box_separation",
                self.robot2object_dist
                > self.cfg.rewards.robot_box_max_distance,
            ),
            (
                "excessive_box_tilt",
                self.projected_gravity_box[:, 2] > -tilt_cos,
            ),
            (
                "grasp_loss",
                self.grasp_loss_time > self.cfg.rewards.grasp_loss_grace_s,
            ),
            (
                "humanoid_head_low",
                self.rigid_body_states[:, self.head_index, 2] < 0.6,
            ),
            ("humanoid_root_low", self.root_states[:, 2] < 0.2),
            (
                "humanoid_tilt",
                torch.logical_or(
                    torch.abs(self.roll) > 0.5,
                    torch.abs(self.pitch) > 1.1,
                ),
            ),
            (
                "humanoid_excess_speed",
                torch.norm(self.rigid_body_states[:, 2, 7:9], dim=-1) > 3.0,
            ),
            (
                "humanoid_hip_low",
                torch.any(
                    self.rigid_body_states[:, self.hip_yaw_indices, 2] < 0.15,
                    dim=1,
                ),
            ),
            (
                "humanoid_forbidden_contact",
                torch.any(
                    torch.norm(
                        self.contact_forces[
                            :, self.termination_contact_indices, :
                        ],
                        dim=-1,
                    )
                    > 10.0,
                    dim=1,
                ),
            ),
        )

    def check_termination(self):
        super().check_termination()
        terminated = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(terminated) == 0:
            return
        masks = self._termination_masks()
        for env_id in terminated.detach().cpu().tolist():
            reason = "termination"
            for label, mask in masks:
                if bool(mask[env_id].item()):
                    reason = label
                    break
            self.eval_last_termination_reason[env_id] = reason


def configure_deterministic_evaluation(
    cfg, *, carry_motion_id: int, carry_phase: float, episode_length_s: float
):
    """Apply the benchmark's one-env, nominal-physics configuration."""
    cfg.env.num_envs = 1
    cfg.env.episode_length_s = float(episode_length_s)
    cfg.env.test = True
    cfg.terrain.curriculum = False
    cfg.commands.curriculum = False
    cfg.commands.resampling_time = 0.0
    cfg.commands.resample_carry_commands = False
    cfg.noise.add_noise = False

    box = cfg.asset.box
    box.fixed_carry_reset = True
    box.fixed_carry_motion_id = int(carry_motion_id)
    box.fixed_carry_phase = float(carry_phase)
    box.use_random = False
    box.use_mass_size_mixture = False
    box.random_size = False
    box.random_density = False
    box.random_props = False

    domain = cfg.domain_rand
    domain.use_random = False
    for name in (
        "randomize_actuation_offset",
        "randomize_motor_strength",
        "randomize_payload_mass",
        "randomize_com_displacement",
        "randomize_link_mass",
        "randomize_friction",
        "randomize_restitution",
        "randomize_kp",
        "randomize_kd",
        "randomize_initial_joint_pos",
        "disturbance",
        "push_robots",
        "delay",
    ):
        setattr(domain, name, False)
    return cfg


def assert_nominal_configuration(cfg):
    if cfg.env.num_envs != 1 or cfg.noise.add_noise:
        raise AssertionError("Evaluation requires one environment and no noise")
    if not cfg.asset.box.fixed_carry_reset:
        raise AssertionError("Evaluation requires fixed carry-reference reset")
    if any(
        getattr(cfg.domain_rand, name)
        for name in (
            "randomize_actuation_offset",
            "randomize_motor_strength",
            "randomize_payload_mass",
            "randomize_com_displacement",
            "randomize_link_mass",
            "randomize_friction",
            "randomize_restitution",
            "randomize_kp",
            "randomize_kd",
            "randomize_initial_joint_pos",
            "disturbance",
            "push_robots",
            "delay",
        )
    ):
        raise AssertionError("Evaluation domain randomization is not fully disabled")
    if (
        cfg.asset.box.use_mass_size_mixture
        or cfg.asset.box.random_size
        or cfg.asset.box.random_density
        or cfg.asset.box.random_props
    ):
        raise AssertionError("Evaluation box properties must be deterministic")
