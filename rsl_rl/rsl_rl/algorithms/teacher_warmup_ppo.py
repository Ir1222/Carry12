"""CarryLocomotionPPO with student-only data and handoff supervision."""
import torch

from .carry_locomotion_ppo import CarryLocomotionPPO
from rsl_rl.storage.teacher_warmup_rollout_storage import TeacherWarmupRolloutStorage


class TeacherWarmupPPO(CarryLocomotionPPO):
    def __init__(self, actor_critic, min_valid_samples=64,
                 handoff_imitation_coef=0.5, imitation_arms_weight=2.0,
                 imitation_waist_weight=2.0, imitation_legs_weight=0.25,
                 imitation_joint_weights=None, groups=None, **kwargs):
        super().__init__(actor_critic, **kwargs)
        self.min_valid_samples = max(min_valid_samples, self.num_mini_batches)
        self.handoff_imitation_coef = handoff_imitation_coef
        self.groups = groups
        self.weights = torch.ones(actor_critic.num_actions, device=self.device)
        if groups:
            for name, weight in (("arms", imitation_arms_weight), ("waist", imitation_waist_weight),
                                 ("legs", imitation_legs_weight)):
                self.weights[groups[name]] = weight
        if imitation_joint_weights is not None:
            self.weights = torch.tensor(imitation_joint_weights, device=self.device, dtype=torch.float)
        if (self.weights.shape != (actor_critic.num_actions,) or not torch.isfinite(self.weights).all()
                or not torch.all(self.weights >= 0) or self.weights.sum() <= 0):
            raise ValueError("Expected nonnegative action weights with positive sum")
        if min_valid_samples < 1 or handoff_imitation_coef < 0:
            raise ValueError("Invalid sample minimum or imitation coefficient")
        self.diagnostics = {}

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape,
                     critic_obs_shape, action_shape):
        self.storage = TeacherWarmupRolloutStorage(num_envs, num_transitions_per_env,
                                                  actor_obs_shape, critic_obs_shape,
                                                  action_shape, self.device)

    def act(self, obs, critic_obs):
        # The environment rotates/replaces observation tensors after stepping.
        # Clone here to make ownership explicit and immune to in-place reset.
        if not torch.isfinite(obs).all() or not torch.isfinite(critic_obs).all():
            raise ValueError("Nonfinite observations: refusing to fabricate PPO samples")
        return super().act(obs.clone(), critic_obs.clone())

    def process_env_step(self, rewards, dones, infos, next_critic_obs, next_obs):
        tr = self.transition
        tr.rewards = rewards.detach().clone()
        tr.dones = dones.detach().clone()
        tr.next_critic_observations = next_critic_obs.detach().clone()
        next_values = self.actor_critic.evaluate(next_critic_obs).detach()
        self.storage.add_student_transition(
            tr, infos["student_valid"], infos["time_outs"], next_values,
            next_obs, infos["teacher_action"].detach(), infos["imitation_weight"].detach(),
            infos.get("applied_actions"), infos.get("pd_targets"), infos.get("control_phase"))
        self.transition.clear()
        self.actor_critic.reset(dones)

    def _auxiliary_loss(self, observations, action_mean):
        del observations
        gap = (action_mean - self.storage.batch_teacher_actions.detach()).square()
        per_sample = (gap * self.weights).sum(-1) / self.weights.sum()
        temporal = self.storage.batch_imitation_weights.squeeze(-1)
        # Mean over valid PPO samples: dilution as the local window expires is
        # intentional; weights are NOT renormalized to erase temporal decay.
        loss = (per_sample * temporal).mean()
        self._diag_sums["handoff_imitation_loss"] += float(loss.detach())
        self._diag_sums["imitation_active_ratio"] += float((temporal > 0).float().mean())
        if self.groups:
            for name, ids in self.groups.items():
                self._diag_sums["gap_" + name] += float(gap[:, ids].mean().sqrt().detach())
        return self.handoff_imitation_coef * loss

    def update(self):
        valid = self.storage.valid_count
        self.diagnostics = {"valid_samples": valid,
                            "valid_ratio": valid / max(1, self.storage.step * self.storage.num_envs),
                            "skipped_update": float(valid < self.min_valid_samples)}
        if valid < self.min_valid_samples:
            self.storage.clear()
            return (0.0,) * 4
        self._diag_sums = dict.fromkeys(("handoff_imitation_loss", "imitation_active_ratio",
                                        "gap_arms", "gap_waist", "gap_legs"), 0.0)
        result = super().update()
        divisor = self.num_learning_epochs * self.num_mini_batches
        self.diagnostics.update({k: v / divisor for k, v in self._diag_sums.items()})
        # std is a directly parameterized Normal scale in the existing actor.
        with torch.no_grad():
            self.actor_critic.std.clamp_(min=1e-4)
        return result
