# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import os
from datetime import datetime

import isaacgym

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import task_registry
from legged_gym.utils.helpers import (
    class_to_dict,
    get_args,
    get_load_path,
    update_cfg_from_args,
)
from rsl_rl.runners import CarryLocomotionOnPolicyRunner


def _format_checkpoint_path(path):
    return path.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)


def train(args):
    if args.resume and args.init_policy_path:
        raise ValueError("--resume and --init_policy_path are mutually exclusive.")
    if args.finetune_path is not None:
        raise ValueError(
            "train_carry_locomotion.py uses --init_policy_path instead of "
            "--finetune_path"
        )

    env, _ = task_registry.make_env(name=args.task, args=args)
    _, train_cfg = task_registry.get_cfgs(args.task)
    _, train_cfg = update_cfg_from_args(None, train_cfg, args)

    if train_cfg.runner.finetune_path is not None:
        raise ValueError(
            "train_carry_locomotion.py uses --init_policy_path instead of "
            "--finetune_path"
        )

    log_root = os.path.join(
        LEGGED_GYM_ROOT_DIR,
        "logs",
        train_cfg.runner.experiment_name,
    )
    log_dir = os.path.join(
        log_root,
        datetime.now().strftime("%b%d_%H-%M-%S")
        + "_"
        + train_cfg.runner.run_name,
    )

    runner = CarryLocomotionOnPolicyRunner(
        env,
        class_to_dict(train_cfg),
        log_dir,
        device=args.rl_device,
    )

    if args.resume:
        if train_cfg.runner.resume_path is not None:
            resume_path = _format_checkpoint_path(
                train_cfg.runner.resume_path
            )
        else:
            resume_path = get_load_path(
                log_root,
                train_cfg.runner.load_run,
                train_cfg.runner.checkpoint,
            )
        print(f"Loading model from: {resume_path}")
        runner.load(resume_path)
    elif args.init_policy_path:
        init_policy_path = _format_checkpoint_path(args.init_policy_path)
        runner.load_init_policy(init_policy_path)

    runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True,
    )


if __name__ == "__main__":
    train(get_args())
