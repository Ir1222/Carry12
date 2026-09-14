# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch


class CarryLocomotionRolloutStorage:
    class Transition:
        def __init__(self):
            self.observations = None
            self.critic_observations = None
            self.next_critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None

        def clear(self):
            self.__init__()

    def __init__(
        self,
        num_envs,
        num_transitions_per_env,
        obs_shape,
        privileged_obs_shape,
        actions_shape,
        device="cpu",
    ):
        self.device = device
        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.actions_shape = actions_shape

        self.observations = torch.zeros(
            num_transitions_per_env, num_envs, *obs_shape, device=self.device
        )
        if privileged_obs_shape[0] is not None:
            self.privileged_observations = torch.zeros(
                num_transitions_per_env,
                num_envs,
                *privileged_obs_shape,
                device=self.device,
            )
            self.next_privileged_observations = torch.zeros(
                num_transitions_per_env,
                num_envs,
                *privileged_obs_shape,
                device=self.device,
            )
        else:
            self.privileged_observations = None
            self.next_privileged_observations = None

        self.rewards = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.actions = torch.zeros(
            num_transitions_per_env,
            num_envs,
            *actions_shape,
            device=self.device,
        )
        self.dones = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        ).byte()

        self.actions_log_prob = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.values = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.returns = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.advantages = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.mu = torch.zeros(
            num_transitions_per_env,
            num_envs,
            *actions_shape,
            device=self.device,
        )
        self.sigma = torch.zeros(
            num_transitions_per_env,
            num_envs,
            *actions_shape,
            device=self.device,
        )

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.step = 0

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")

        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None:
            self.privileged_observations[self.step].copy_(
                transition.critic_observations
            )
            self.next_privileged_observations[self.step].copy_(
                transition.next_critic_observations
            )
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(
            transition.actions_log_prob.view(-1, 1)
        )
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = (
                self.rewards[step]
                + next_is_not_terminal * gamma * next_values
                - self.values[step]
            )
            advantage = (
                delta + next_is_not_terminal * gamma * lam * advantage
            )
            self.returns[step] = advantage + self.values[step]

        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (
            self.advantages.std() + 1e-8
        )

    def get_statistics(self):
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat(
            (
                flat_dones.new_tensor([-1], dtype=torch.int64),
                flat_dones.nonzero(as_tuple=False)[:, 0],
            )
        )
        trajectory_lengths = done_indices[1:] - done_indices[:-1]
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * (self.num_transitions_per_env - 1)
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(
            num_mini_batches * mini_batch_size,
            requires_grad=False,
            device=self.device,
        )

        observations = self.observations[:-1].flatten(0, 1)
        if self.privileged_observations is not None:
            critic_observations = self.privileged_observations[:-1].flatten(0, 1)
            next_critic_observations = (
                self.next_privileged_observations[:-1].flatten(0, 1)
            )
        else:
            critic_observations = observations
            next_critic_observations = observations

        next_observations = self.observations[1:].flatten(0, 1)
        actions = self.actions[:-1].flatten(0, 1)
        values = self.values[:-1].flatten(0, 1)
        returns = self.returns[:-1].flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob[:-1].flatten(0, 1)
        advantages = self.advantages[:-1].flatten(0, 1)
        old_mu = self.mu[:-1].flatten(0, 1)
        old_sigma = self.sigma[:-1].flatten(0, 1)
        not_dones = 1 - self.dones[:-1].float().flatten(0, 1)

        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                yield (
                    observations[batch_idx],
                    next_observations[batch_idx],
                    critic_observations[batch_idx],
                    actions[batch_idx],
                    next_critic_observations[batch_idx],
                    not_dones[batch_idx],
                    values[batch_idx],
                    advantages[batch_idx],
                    returns[batch_idx],
                    old_actions_log_prob[batch_idx],
                    old_mu[batch_idx],
                    old_sigma[batch_idx],
                )
