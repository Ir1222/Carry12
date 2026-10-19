"""Masked asynchronous student trajectories; teacher slots never reach minibatches."""
import torch

from .carry_locomotion_rollout_storage import CarryLocomotionRolloutStorage


class TeacherWarmupRolloutStorage(CarryLocomotionRolloutStorage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.valid = torch.zeros_like(self.dones, dtype=torch.bool)
        self.timeout = torch.zeros_like(self.valid)
        self.next_values = torch.zeros_like(self.values)
        self.next_observations = torch.zeros_like(self.observations)
        self.teacher_actions = torch.zeros_like(self.actions)
        self.imitation_weights = torch.zeros_like(self.values)
        self.applied_actions = torch.zeros_like(self.actions)
        self.pd_targets = torch.zeros_like(self.actions)
        self.control_phases = torch.zeros_like(self.dones, dtype=torch.long)

    def add_student_transition(self, transition, valid, timeout, next_values,
                               next_obs, teacher_actions, imitation_weights,
                               applied_actions=None, pd_targets=None, control_phases=None):
        t = self.step
        self.valid[t].copy_(valid.reshape(-1, 1))
        self.timeout[t].copy_(timeout.reshape(-1, 1))
        self.next_values[t].copy_(next_values)
        self.next_observations[t].copy_(next_obs)
        self.teacher_actions[t].copy_(teacher_actions)
        self.imitation_weights[t].copy_(imitation_weights.reshape(-1, 1))
        if applied_actions is not None:
            self.applied_actions[t].copy_(applied_actions)
        if pd_targets is not None:
            self.pd_targets[t].copy_(pd_targets)
        if control_phases is not None:
            self.control_phases[t].copy_(control_phases.reshape(-1, 1))
        super().add_transitions(transition)

    def compute_returns(self, last_values, gamma, lam):
        del last_values  # explicit post-action values, including terminal obs
        advantage = torch.zeros_like(self.values[0])
        for t in reversed(range(self.step)):
            valid = self.valid[t]
            done = self.dones[t].bool()
            bootstrap = (~done | self.timeout[t]).float()
            # A reset or teacher slot breaks recursion. A rollout boundary uses
            # next_values for bootstrap, but cannot import uncollected rewards.
            next_valid = self.valid[t + 1] if t + 1 < self.step else torch.zeros_like(valid)
            continuation = (~done & next_valid).float()
            delta = self.rewards[t] + gamma * bootstrap * self.next_values[t] - self.values[t]
            advantage = torch.where(valid, delta + gamma * lam * continuation * advantage, 0.0)
            self.returns[t] = torch.where(valid, advantage + self.values[t], 0.0)
            self.advantages[t] = advantage
        selected = self.advantages[:self.step][self.valid[:self.step]]
        if selected.numel():
            normalized = (selected - selected.mean()) / (selected.std(unbiased=False) + 1e-8)
            self.advantages[:self.step][self.valid[:self.step]] = normalized

    @property
    def valid_count(self):
        return int(self.valid[:self.step].sum().item())

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        ids = self.valid[:self.step].flatten().nonzero().flatten()
        if len(ids) < num_mini_batches:
            raise ValueError("Not enough student samples for nonempty minibatches")

        def flat(tensor):
            return tensor[:self.step].flatten(0, 1)

        cont = (~self.dones[:self.step].bool()).float().flatten(0, 1)
        for _ in range(num_epochs):
            shuffled = ids[torch.randperm(len(ids), device=ids.device)]
            for batch_ids in torch.tensor_split(shuffled, num_mini_batches):
                self.batch_teacher_actions = flat(self.teacher_actions)[batch_ids]
                self.batch_imitation_weights = flat(self.imitation_weights)[batch_ids]
                yield (
                    flat(self.observations)[batch_ids],
                    flat(self.next_observations)[batch_ids],
                    flat(self.privileged_observations)[batch_ids],
                    flat(self.actions)[batch_ids],
                    flat(self.next_privileged_observations)[batch_ids],
                    cont[batch_ids],
                    flat(self.values)[batch_ids],
                    flat(self.advantages)[batch_ids],
                    flat(self.returns)[batch_ids],
                    flat(self.actions_log_prob)[batch_ids],
                    flat(self.mu)[batch_ids],
                    flat(self.sigma)[batch_ids],
                )
