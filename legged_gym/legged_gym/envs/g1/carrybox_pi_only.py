"""Current carry-box task with critic-only interaction privileged information."""

import torch
from isaacgym.torch_utils import quat_rotate_inverse

from .carrybox import LeggedRobot as CarryBoxBase


class LeggedRobot(CarryBoxBase):
    """Append configurable interaction PI to the carry-box critic only."""

    def _create_envs(self):
        super()._create_envs()
        self._init_interaction_privileged_indices()

    def _init_interaction_privileged_indices(self):
        """Resolve environment-domain rigid-body indices used by the PI proxy."""
        include_force_vectors = self.cfg.interaction_priv.include_force_vectors
        expected_interaction_dim = (
            self.cfg.env.num_interaction_kinematic_obs
            + self.cfg.env.num_interaction_contact_flag_obs
            + (
                self.cfg.env.num_interaction_force_vector_obs
                if include_force_vectors
                else 0
            )
        )
        expected_privileged_dim = (
            self.cfg.env.num_current_frame_critic_obs + expected_interaction_dim
        )

        assert self.cfg.env.num_actor_obs == 738
        assert self.cfg.env.num_current_frame_critic_obs == 126
        assert self.cfg.env.num_interaction_priv_obs == expected_interaction_dim
        assert self.cfg.env.num_privileged_obs == expected_privileged_dim
        assert self.cfg.env.num_actions == 29

        env = self.envs[0]
        robot_handle = self.actor_handles[0]
        robot_body_names = self.gym.get_actor_rigid_body_names(env, robot_handle)

        hand_collision_names = [
            name
            for name in robot_body_names
            if self.cfg.asset.hand_colli_name in name
        ]
        left_hand_names = [name for name in hand_collision_names if "left_" in name]
        right_hand_names = [name for name in hand_collision_names if "right_" in name]
        assert len(left_hand_names) == 1, left_hand_names
        assert len(right_hand_names) == 1, right_hand_names
        assert self.hand_colli_indices.numel() == 2, self.hand_colli_indices

        self.left_hand_net_contact_force_index = (
            self.gym.find_actor_rigid_body_handle(env, robot_handle, left_hand_names[0])
        )
        self.right_hand_net_contact_force_index = (
            self.gym.find_actor_rigid_body_handle(env, robot_handle, right_hand_names[0])
        )
        assert self.left_hand_net_contact_force_index >= 0
        assert self.right_hand_net_contact_force_index >= 0
        assert {
            int(self.left_hand_net_contact_force_index),
            int(self.right_hand_net_contact_force_index),
        } == {int(index) for index in self.hand_colli_indices}

        if include_force_vectors:
            self.box_actor_body_count = len(
                self.gym.get_actor_rigid_body_properties(env, self.box_handles[0])
            )
            assert self.box_actor_body_count == 1, self.box_actor_body_count
            self.box_net_contact_force_index = self.gym.get_actor_rigid_body_handle(
                env, self.box_handles[0], 0
            )
            assert self.box_net_contact_force_index >= 0

    def _compute_interaction_privileged_proxy(self):
        """Build the critic-only interaction PI in the policy frame."""
        if not self.cfg.interaction_priv.enabled:
            return torch.zeros(
                self.num_envs,
                self.cfg.env.num_interaction_priv_obs,
                device=self.device,
                dtype=torch.float,
            )

        body_count = self.contact_forces.shape[1]
        rigid_body_count = self.rigid_body_states.shape[1]
        force_indices = [
            self.left_hand_net_contact_force_index,
            self.right_hand_net_contact_force_index,
        ]
        if self.cfg.interaction_priv.include_force_vectors:
            force_indices.append(self.box_net_contact_force_index)
        for index in force_indices:
            assert 0 <= int(index) < body_count
            assert 0 <= int(index) < rigid_body_count
        if self.cfg.interaction_priv.include_force_vectors:
            assert self.box_actor_body_count == 1

        policy_quat = self.rigid_body_states[:, self.upper_body_index, 3:7]
        box_lin_vel_local = quat_rotate_inverse(
            policy_quat, self.box_states[:, 7:10]
        )
        box_ang_vel_local = quat_rotate_inverse(
            policy_quat, self.box_states[:, 10:13]
        )

        left_hand_force_world = self.contact_forces[
            :, self.left_hand_net_contact_force_index, :
        ]
        right_hand_force_world = self.contact_forces[
            :, self.right_hand_net_contact_force_index, :
        ]

        threshold = self.cfg.interaction_priv.hand_contact_force_threshold
        left_contact_flag = (
            torch.linalg.vector_norm(left_hand_force_world, dim=-1, keepdim=True)
            > threshold
        ).float()
        right_contact_flag = (
            torch.linalg.vector_norm(right_hand_force_world, dim=-1, keepdim=True)
            > threshold
        ).float()

        interaction_components = [
            box_lin_vel_local * self.cfg.interaction_priv.box_lin_vel_scale,
            box_ang_vel_local * self.cfg.interaction_priv.box_ang_vel_scale,
        ]
        if self.cfg.interaction_priv.include_force_vectors:
            box_force_world = self.contact_forces[
                :, self.box_net_contact_force_index, :
            ]
            interaction_components.extend(
                (
                    quat_rotate_inverse(policy_quat, left_hand_force_world)
                    * self.cfg.interaction_priv.net_contact_force_scale,
                    quat_rotate_inverse(policy_quat, right_hand_force_world)
                    * self.cfg.interaction_priv.net_contact_force_scale,
                    quat_rotate_inverse(policy_quat, box_force_world)
                    * self.cfg.interaction_priv.net_contact_force_scale,
                )
            )
        interaction_components.extend((left_contact_flag, right_contact_flag))
        interaction_priv = torch.cat(interaction_components, dim=-1)
        interaction_priv = torch.clamp(
            interaction_priv,
            -self.cfg.interaction_priv.clip_value,
            self.cfg.interaction_priv.clip_value,
        )
        assert interaction_priv.shape == (
            self.num_envs,
            self.cfg.env.num_interaction_priv_obs,
        )
        return interaction_priv

    def compute_observations(self):
        super().compute_observations()
        assert self.obs_buf.shape == (self.num_envs, self.cfg.env.num_actor_obs)
        assert self.obs_buf.shape == (self.num_envs, 738)
        assert self.privileged_obs_buf.shape == (
            self.num_envs,
            self.cfg.env.num_current_frame_critic_obs,
        )
        assert self.privileged_obs_buf.shape == (self.num_envs, 126)

        interaction_priv = self._compute_interaction_privileged_proxy()
        self.privileged_obs_buf = torch.cat(
            (self.privileged_obs_buf, interaction_priv), dim=-1
        )
        assert self.privileged_obs_buf.shape == (
            self.num_envs,
            self.cfg.env.num_privileged_obs,
        )

    def compute_termination_observations(self, env_ids):
        base_terminal_obs = super().compute_termination_observations(env_ids)
        assert base_terminal_obs.shape == (
            env_ids.numel(),
            self.cfg.env.num_current_frame_critic_obs,
        )
        assert base_terminal_obs.shape == (env_ids.numel(), 126)

        interaction_priv = self._compute_interaction_privileged_proxy()[env_ids]
        terminal_obs = torch.cat((base_terminal_obs, interaction_priv), dim=-1)
        assert terminal_obs.shape == (
            env_ids.numel(),
            self.cfg.env.num_privileged_obs,
        )
        return terminal_obs
