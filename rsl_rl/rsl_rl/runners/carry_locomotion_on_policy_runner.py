# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import os
import statistics
import time
from collections import deque

import torch
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

import rsl_rl
from rsl_rl.algorithms import CarryLocomotionPPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCritic
from rsl_rl.utils import store_code_state


class CarryLocomotionOnPolicyRunner:
    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs
        else:
            num_critic_obs = self.env.num_one_step_obs

        self.num_actor_obs = self.env.num_obs
        self.num_critic_obs = num_critic_obs
        self.actor_history_length = self.env.actor_history_length
        actor_critic_class = eval(self.cfg["policy_class_name"])
        actor_critic: ActorCritic = actor_critic_class(
            self.num_actor_obs,
            self.num_critic_obs,
            self.actor_history_length,
            self.env.num_actions,
            **self.policy_cfg,
        ).to(self.device)

        algorithm_cfg = dict(self.alg_cfg)
        use_muon_optim = (
            self.cfg.get("use_muon_optim", False)
            or algorithm_cfg.pop("use_muon_optim", False)
        )
        if use_muon_optim:
            raise NotImplementedError(
                "CarryLocomotionPPO V1 supports Adam only; set use_muon_optim=False."
            )
        self.alg = CarryLocomotionPPO(
            actor_critic,
            use_muon_optim=use_muon_optim,
            device=self.device,
            **algorithm_cfg,
        )
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.env.num_obs],
            [self.env.num_privileged_obs],
            [self.env.num_actions],
        )

        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

        _, _ = self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.log_dir is not None and self.writer is None:
            self.logger_type = self.cfg.get("logger", "wandb").lower()

            if self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(
                    log_dir=self.log_dir, flush_secs=10, cfg=self.cfg
                )
                self.writer.log_config(
                    self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg
                )
            elif self.logger_type == "tensorboard":
                self.writer = TensorboardSummaryWriter(
                    log_dir=self.log_dir, flush_secs=10
                )
            else:
                raise AssertionError("logger type not found")

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf,
                high=int(self.env.max_episode_length),
            )
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )
        cur_episode_length = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            tracking_sq_error_sum = torch.zeros(
                3, dtype=torch.float, device=self.device
            )
            tracking_valid_count = torch.zeros(
                (), dtype=torch.float, device=self.device
            )
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    (
                        obs,
                        privileged_obs,
                        rewards,
                        dones,
                        infos,
                        termination_ids,
                        termination_privileged_obs,
                        _,
                    ) = self.env.step(actions)

                    critic_obs = (
                        privileged_obs
                        if privileged_obs is not None
                        else obs
                    )
                    obs = obs.to(self.device)
                    critic_obs = critic_obs.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)
                    termination_ids = termination_ids.to(self.device)
                    termination_privileged_obs = (
                        termination_privileged_obs.to(self.device)
                    )

                    command = self.env.carry_policy_commands[:, :3]
                    actual = torch.stack(
                        (
                            self.env.base_lin_vel_yaw[:, 0],
                            self.env.base_lin_vel_yaw[:, 1],
                            self.env.base_yaw_rate_world,
                        ),
                        dim=-1,
                    )
                    squared_error = (command - actual) ** 2
                    valid_mask = ~(dones.reshape(-1) > 0)
                    tracking_sq_error_sum += squared_error[valid_mask].sum(
                        dim=0
                    )
                    tracking_valid_count += valid_mask.sum()

                    next_critic_obs = critic_obs.clone().detach()
                    next_critic_obs[termination_ids] = (
                        termination_privileged_obs.clone().detach()
                    )

                    self.alg.process_env_step(
                        rewards, dones, infos, next_critic_obs
                    )

                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(
                            cur_reward_sum[new_ids][:, 0]
                            .cpu()
                            .numpy()
                            .tolist()
                        )
                        lenbuffer.extend(
                            cur_episode_length[new_ids][:, 0]
                            .cpu()
                            .numpy()
                            .tolist()
                        )
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                safe_tracking_count = tracking_valid_count.clamp_min(1.0)
                tracking_rmse = torch.where(
                    tracking_valid_count > 0,
                    torch.sqrt(
                        tracking_sq_error_sum / safe_tracking_count
                    ),
                    torch.zeros_like(tracking_sq_error_sum),
                )

                stop = time.time()
                collection_time = stop - start

                start = stop
                self.alg.compute_returns(critic_obs)

            (
                mean_value_loss,
                mean_surrogate_loss,
                mean_policy_smooth_loss,
                mean_value_smooth_loss,
            ) = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            self.current_learning_iteration = it
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
            ep_infos.clear()
            if it == start_iter:
                git_file_paths = store_code_state(
                    self.log_dir, self.git_status_repos
                )
                if self.logger_type == "wandb" and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        self.save(
            os.path.join(
                self.log_dir,
                f"model_{self.current_learning_iteration}.pt",
            )
        )

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat(
                        (infotensor, ep_info[key].to(self.device))
                    )
                value = torch.mean(infotensor)
                self.writer.add_scalar(
                    "Episode/" + key, value, locs["it"]
                )
                ep_string += (
                    f"{f'Mean episode {key}:':>{pad}} {value:.4f}\n"
                )

        mean_std = self.alg.actor_critic.std[0:10].mean()
        fps = int(
            self.num_steps_per_env
            * self.env.num_envs
            / (locs["collection_time"] + locs["learn_time"])
        )

        self.writer.add_scalar(
            "Loss/value_function", locs["mean_value_loss"], locs["it"]
        )
        self.writer.add_scalar(
            "Loss/surrogate", locs["mean_surrogate_loss"], locs["it"]
        )
        self.writer.add_scalar(
            "Loss/policy_smoothness",
            locs["mean_policy_smooth_loss"],
            locs["it"],
        )
        self.writer.add_scalar(
            "Loss/value_smoothness",
            locs["mean_value_smooth_loss"],
            locs["it"],
        )
        self.writer.add_scalar(
            "Loss/learning_rate", self.alg.learning_rate, locs["it"]
        )
        self.writer.add_scalar(
            "Policy/mean_noise_std", mean_std.item(), locs["it"]
        )
        self.writer.add_scalar(
            "TrackingRMSE/vx", locs["tracking_rmse"][0], locs["it"]
        )
        self.writer.add_scalar(
            "TrackingRMSE/vy", locs["tracking_rmse"][1], locs["it"]
        )
        self.writer.add_scalar(
            "TrackingRMSE/yaw_rate",
            locs["tracking_rmse"][2],
            locs["it"],
        )
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar(
            "Perf/collection time", locs["collection_time"], locs["it"]
        )
        self.writer.add_scalar(
            "Perf/learning_time", locs["learn_time"], locs["it"]
        )
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar(
                "Train/mean_reward",
                statistics.mean(locs["rewbuffer"]),
                locs["it"],
            )
            self.writer.add_scalar(
                "Train/mean_episode_length",
                statistics.mean(locs["lenbuffer"]),
                locs["it"],
            )
            if self.logger_type != "wandb":
                self.writer.add_scalar(
                    "Train/mean_reward/time",
                    statistics.mean(locs["rewbuffer"]),
                    self.tot_time,
                )
                self.writer.add_scalar(
                    "Train/mean_episode_length/time",
                    statistics.mean(locs["lenbuffer"]),
                    self.tot_time,
                )

        heading = (
            f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} "
            "\033[0m "
        )
        log_string = (
            f"{'#' * width}\n"
            f"{heading.center(width, ' ')}\n\n"
            f"{'Computation:':>{pad}} {fps:.0f} steps/s "
            f"(collection: {locs['collection_time']:.3f}s, "
            f"learning {locs['learn_time']:.3f}s)\n"
            f"{'Value function loss:':>{pad}} "
            f"{locs['mean_value_loss']:.4f}\n"
            f"{'Surrogate loss:':>{pad}} "
            f"{locs['mean_surrogate_loss']:.4f}\n"
            f"{'Policy smoothness loss:':>{pad}} "
            f"{locs['mean_policy_smooth_loss']:.4f}\n"
            f"{'Value smoothness loss:':>{pad}} "
            f"{locs['mean_value_smooth_loss']:.4f}\n"
            f"{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"
            f"{'Tracking RMSE [vx, vy, yaw]:':>{pad}} "
            f"{locs['tracking_rmse'][0]:.4f}, "
            f"{locs['tracking_rmse'][1]:.4f}, "
            f"{locs['tracking_rmse'][2]:.4f}\n"
        )
        if len(locs["rewbuffer"]) > 0:
            log_string += (
                f"{'Mean reward:':>{pad}} "
                f"{statistics.mean(locs['rewbuffer']):.2f}\n"
                f"{'Mean episode length:':>{pad}} "
                f"{statistics.mean(locs['lenbuffer']):.2f}\n"
            )

        log_string += ep_string
        log_string += (
            f"{'-' * width}\n"
            f"{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"
            f"{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"
            f"{'Total time:':>{pad}} {self.tot_time:.2f}s\n"
            f"{'ETA:':>{pad}} "
            f"{self.tot_time / (locs['it'] + 1) * (locs['num_learning_iterations'] - locs['it']):.1f}s\n"
        )
        print(log_string)

    def save(self, path, infos=None):
        torch.save(
            {
                "model_state_dict": self.alg.actor_critic.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "iter": self.current_learning_iteration + 1,
                "infos": infos,
            },
            path,
        )

    def load_init_policy(self, path, load_action_std=True):
        checkpoint = torch.load(path, map_location=self.device)
        if "model_state_dict" not in checkpoint:
            raise KeyError(
                "Initialization checkpoint is missing 'model_state_dict'."
            )
        source = checkpoint["model_state_dict"]
        actor_state = {
            key[len("actor.") :]: value
            for key, value in source.items()
            if key.startswith("actor.")
        }
        if not actor_state:
            raise KeyError(
                "Initialization checkpoint contains no 'actor.*' parameters."
            )

        target_actor_state = self.alg.actor_critic.actor.state_dict()
        missing_actor_keys = target_actor_state.keys() - actor_state.keys()
        unexpected_actor_keys = actor_state.keys() - target_actor_state.keys()
        if missing_actor_keys or unexpected_actor_keys:
            raise RuntimeError(
                "Actor parameter keys do not exactly match. Missing: "
                f"{sorted(missing_actor_keys)}; unexpected: "
                f"{sorted(unexpected_actor_keys)}."
            )
        shape_mismatches = [
            (
                key,
                tuple(actor_state[key].shape),
                tuple(target_actor_state[key].shape),
            )
            for key in target_actor_state
            if actor_state[key].shape != target_actor_state[key].shape
        ]
        if shape_mismatches:
            raise RuntimeError(
                "Actor parameter shapes do not exactly match "
                f"(key, checkpoint, current): {shape_mismatches}."
            )

        source_std = None
        if load_action_std:
            if "std" not in source:
                raise KeyError(
                    "Initialization checkpoint is missing action parameter 'std'."
                )
            source_std = source["std"]
            target_std = self.alg.actor_critic.std
            if source_std.shape != target_std.shape:
                raise ValueError(
                    "Action std shape mismatch: checkpoint has "
                    f"{tuple(source_std.shape)}, current policy expects "
                    f"{tuple(target_std.shape)}."
                )

        self.alg.actor_critic.actor.load_state_dict(actor_state, strict=True)
        print(f"[InitPolicy] Loaded actor from: {path}")
        if load_action_std:
            with torch.no_grad():
                target_std.copy_(
                    source_std.to(
                        device=target_std.device, dtype=target_std.dtype
                    )
                )
            print("[InitPolicy] Loaded action std")

        self.current_learning_iteration = 0
        print("[InitPolicy] Critic remains freshly initialized")
        print("[InitPolicy] Optimizer remains freshly initialized")
        print("[InitPolicy] Starting from iteration 0")
        return checkpoint.get("infos")

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        required_keys = {
            "model_state_dict",
            "optimizer_state_dict",
            "iter",
        }
        missing_keys = required_keys.difference(checkpoint)
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise KeyError(
                f"Carry-locomotion checkpoint is missing required state: {missing}"
            )

        self.alg.actor_critic.load_state_dict(
            checkpoint["model_state_dict"], strict=True
        )
        self.alg.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.alg.learning_rate = self.alg.optimizer.param_groups[0]["lr"]
        self.current_learning_iteration = checkpoint["iter"]
        return checkpoint.get("infos")

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
