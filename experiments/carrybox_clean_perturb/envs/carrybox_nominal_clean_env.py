"""Deterministic nominal-clean CarryBox evaluator env."""

import torch

from envs.carrybox_perturb_env import LeggedRobot as CarryBoxPerturbEnv


class LeggedRobot(CarryBoxPerturbEnv):
    """CarryBox perturb env with deterministic nominal reset surfaces."""

    def _init_buffers(self):
        super()._init_buffers()
        self._apply_nominal_clean_sensor_state()

    def _reset_default(self, env_ids):
        super()._reset_default(env_ids)
        self.root_states[env_ids, 7:13] = 0.0
        self.dof_pos[env_ids] = self.default_dof_pos
        self.dof_vel[env_ids] = 0.0
        self._reset_default_env_ids = env_ids

    def _reset_boxes(self, env_ids):
        if len(env_ids) == 0:
            return

        cfg = self.cfg.nominal_clean
        box_pos = self.env_origins[env_ids].clone()
        box_pos[:, 0] += float(cfg.fixed_box_x)
        box_pos[:, 1] += float(cfg.fixed_box_y)
        raw_box_z = self.env_origins[env_ids, 2] + self._box_size[env_ids, 2] / 2.0
        box_pos[:, 2] = raw_box_z + float(cfg.box_vertical_clearance)

        self.box_states[env_ids, 0:3] = box_pos
        self.box_states[env_ids, 3:7] = self.default_quat
        self.box_states[env_ids, 7:13] = 0.0

        self.platform_pos[env_ids, 0:2] = box_pos[:, 0:2]
        self.platform_pos[env_ids, 2] = (
            raw_box_z - self._box_size[env_ids, 2] / 2.0 - self._platform_height
        )
        self.platform_states[env_ids, 3:7] = self.default_quat
        self.platform_states[env_ids, 7:13] = 0.0

        self.thresh_tag[env_ids] = float(cfg.fixed_thresh_tag)
        self.far_pos_offset[env_ids] = 0.0
        self.can_see_tag[env_ids] = False
        self.has_seen_tag[env_ids] = False

    def _reset_task(self, env_ids):
        super()._reset_task(env_ids)
        self._set_nominal_clean_command(env_ids)

    def _resample_carry_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        self.commands[env_ids, :] = 0.0
        self._set_nominal_clean_command(env_ids)

    def reset_evaluation_trial_state(self, env_ids=None, clear_actor_history=True):
        super().reset_evaluation_trial_state(
            env_ids=env_ids,
            clear_actor_history=clear_actor_history,
        )
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)
        self._set_nominal_clean_command(env_ids)
        self._apply_nominal_clean_sensor_state(env_ids)

    def _set_nominal_clean_command(self, env_ids):
        command = torch.tensor(
            self.cfg.nominal_clean.command,
            dtype=self.commands.dtype,
            device=self.device,
        )
        self.commands[env_ids, 0:3] = command.unsqueeze(0)

    def _apply_nominal_clean_sensor_state(self, env_ids=None):
        if not hasattr(self.cfg, "nominal_clean"):
            return
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        cfg = self.cfg.nominal_clean
        self.hfov_rad[env_ids] = float(cfg.fixed_camera_hfov_rad)
        self.vfov_rad[env_ids] = float(cfg.fixed_camera_vfov_rad)
        self.facing_angle[env_ids] = float(cfg.fixed_camera_facing_angle)
        self.thresh_tag[env_ids] = float(cfg.fixed_thresh_tag)
        self.far_pos_offset[env_ids] = 0.0
