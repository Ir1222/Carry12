"""CarryLocomotion runner specialization with masked rollout collection."""
import copy
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from .carry_locomotion_on_policy_runner import CarryLocomotionOnPolicyRunner
from rsl_rl.algorithms.teacher_warmup_ppo import TeacherWarmupPPO
from rsl_rl.teacher_handoff import checkpoint_actor, file_sha256, read_checkpoint


class TeacherWarmupOnPolicyRunner(CarryLocomotionOnPolicyRunner):
    def __init__(self, env, train_cfg, student_checkpoint, log_dir, device="cpu"):
        if torch.device(env.device) != torch.device(device):
            raise ValueError("Use matching RL and simulator tensor devices")
        cfg = copy.deepcopy(train_cfg)
        _, spec = checkpoint_actor(student_checkpoint)
        if spec["dims"][0] != env.num_obs or spec["dims"][-1] != env.num_actions:
            raise ValueError("Student checkpoint observation/action contract mismatch")
        cfg["policy"]["actor_hidden_dims"] = spec["dims"][1:-1]
        cfg["policy"]["activation"] = "elu"
        algorithm = cfg["algorithm"]
        extra_names = ("min_valid_samples", "handoff_imitation_coef", "imitation_arms_weight",
                       "imitation_waist_weight", "imitation_legs_weight", "imitation_joint_weights")
        extras = {name: algorithm.pop(name) for name in extra_names}
        super().__init__(env, cfg, log_dir, device)
        self.alg = TeacherWarmupPPO(self.alg.actor_critic, groups=env.groups,
                                    device=device, **algorithm, **extras)
        self.alg.init_storage(env.num_envs, self.num_steps_per_env,
                              [env.num_obs], [env.num_privileged_obs], [env.num_actions])
        self.contract = {
            "schema": 1, "teacher_sha256": file_sha256(env.cfg.handoff.teacher_checkpoint),
            "dof_names": list(env.dof_names), "actor_obs": env.num_obs,
            "critic_obs": env.num_privileged_obs, "dt": env.dt,
            "action_scale": env.cfg.control.action_scale,
            "handoff": train_cfg["handoff_contract"],
            "algorithm": train_cfg["algorithm"],
            "policy": cfg["policy"],
            "reference_sha256": file_sha256(env.cfg.handoff.posture_reference)
            if env.cfg.handoff.posture_reference else None,
            "rewards": train_cfg.get("reward_contract"),
            "environment": train_cfg.get("environment_contract"),
        }
        self.writer = SummaryWriter(log_dir) if log_dir else None
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            with open(os.path.join(log_dir, "config.json"), "w", encoding="utf-8") as stream:
                json.dump({"train": train_cfg, "contract": self.contract}, stream, indent=2)

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if init_at_random_ep_len:
            raise ValueError("Random episode ages invalidate teacher warmup timeouts")
        obs = self.env.get_observations()
        critic = self.env.get_privileged_observations()
        self.alg.train_mode()
        for iteration in range(self.current_learning_iteration,
                               self.current_learning_iteration + num_learning_iterations):
            started = time.time()
            episode_logs = []
            phase_counts = torch.zeros(3, device=self.env.device)
            tracking_sum = torch.zeros(3, device=self.env.device)
            tracking_count = 0
            physics_sums = {}
            physics_count = 0
            with torch.no_grad():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic)
                    obs, critic, rewards, dones, infos, ids, terminal_critic, _ = self.env.step(actions)
                    next_critic = critic.clone()
                    next_critic[ids] = terminal_critic.clamp(
                        -self.env.cfg.normalization.clip_observations,
                        self.env.cfg.normalization.clip_observations)
                    self.alg.process_env_step(rewards, dones, infos, next_critic, obs)
                    phase_counts += torch.bincount(self.env.phase, minlength=3)
                    valid = infos["student_valid"] & ~dones.bool()
                    actual = torch.stack((self.env.base_lin_vel_yaw[:, 0], self.env.base_lin_vel_yaw[:, 1],
                                          self.env.base_yaw_rate_world), -1)
                    command = getattr(self.env, "last_step_snapshot", {}).get(
                        "command", self.env.carry_policy_commands[:, :3])
                    tracking_sum += ((command - actual)[valid] ** 2).sum(0)
                    tracking_count += int(valid.sum())
                    physics_count += int(valid.sum())
                    for name, value in self.env.step_diagnostics.items():
                        physics_sums[name] = physics_sums.get(name, 0.0) + value[valid].float().sum()
                    if "episode" in infos:
                        episode_logs.append({k: float(v) for k, v in infos["episode"].items()})
                self.alg.compute_returns(critic)
            losses = self.alg.update()
            self.current_learning_iteration = iteration + 1
            elapsed = time.time() - started
            self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
            if self.writer:
                for name, value in zip(("value", "surrogate", "policy_smoothness", "value_smoothness"), losses):
                    self.writer.add_scalar("Loss/" + name, value, iteration)
                for name, value in self.alg.diagnostics.items():
                    self.writer.add_scalar("Handoff/" + name, value, iteration)
                for phase, name in enumerate(("teacher", "transition", "student")):
                    self.writer.add_scalar("Phase/" + name, phase_counts[phase] / phase_counts.sum(), iteration)
                if tracking_count:
                    for i, axis in enumerate(("vx", "vy", "yaw")):
                        self.writer.add_scalar("TrackingRMSE/" + axis, (tracking_sum[i] / tracking_count).sqrt(), iteration)
                for name in {k for row in episode_logs for k in row}:
                    if name == "completed_count":
                        continue
                    rows = [row for row in episode_logs if name in row]
                    self.writer.add_scalar("Episode/" + name,
                                           sum(row[name] * row.get("completed_count", 1) for row in rows)
                                           / sum(row.get("completed_count", 1) for row in rows), iteration)
                if physics_count:
                    for name, value in physics_sums.items():
                        self.writer.add_scalar("Physics/" + name, value / physics_count, iteration)
                self.writer.add_scalar("Perf/fps", self.num_steps_per_env * self.env.num_envs / elapsed, iteration)
            print(f"[TeacherWarmup] iteration={iteration + 1} valid={self.alg.diagnostics['valid_ratio']:.3f} "
                  f"skipped={self.alg.diagnostics['skipped_update']:.0f} loss={losses[0]:.4f}")
            if (iteration + 1) % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{iteration + 1}.pt"))
        if self.log_dir:
            final_path = os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt")
            if not os.path.exists(final_path):
                self.save(final_path)
        if self.writer:
            self.writer.flush()

    def save(self, path, infos=None):
        if os.path.exists(path):
            raise FileExistsError("Refusing to overwrite checkpoint: " + path)
        payload = {"model_state_dict": self.alg.actor_critic.state_dict(),
                   "optimizer_state_dict": self.alg.optimizer.state_dict(),
                   "iter": self.current_learning_iteration, "infos": infos,
                   "handoff_contract": self.contract, "timesteps": self.tot_timesteps,
                   "rng_torch": torch.get_rng_state(), "rng_numpy": np.random.get_state(),
                   "rng_python": random.getstate(),
                   "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
        # Atomic replacement is only inside this new run's output directory.
        torch.save(payload, path + ".tmp")
        os.replace(path + ".tmp", path)

    def load(self, path):
        checkpoint = read_checkpoint(path)
        if checkpoint.get("handoff_contract") != self.contract:
            raise ValueError("Resume contract differs (teacher, joints, adapter, loss or reference)")
        self.alg.actor_critic.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.alg.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.alg.learning_rate = self.alg.optimizer.param_groups[0]["lr"]
        self.current_learning_iteration = checkpoint["iter"]
        self.tot_timesteps = checkpoint["timesteps"]
        torch.set_rng_state(checkpoint["rng_torch"])
        np.random.set_state(checkpoint["rng_numpy"])
        random.setstate(checkpoint["rng_python"])
        if checkpoint["rng_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint["rng_cuda"])
        print("[Resume] Model/optimizer/RNG restored; simulator starts fresh teacher episodes")

    def load_init_policy(self, path, load_action_std=True):
        state = read_checkpoint(path)["model_state_dict"]
        actor = {k[6:]: v for k, v in state.items() if k.startswith("actor.")}
        self.alg.actor_critic.actor.load_state_dict(actor, strict=True)
        if load_action_std:
            std = state["std"]
            if std.shape != self.alg.actor_critic.std.shape or not torch.all(std > 0):
                raise ValueError("Invalid checkpoint action standard deviations")
            with torch.no_grad():
                self.alg.actor_critic.std.copy_(std.to(self.device))
        self.current_learning_iteration = 0
        print("[Init] Loaded actor + std; critic and Adam are fresh; AMP is not loaded")
