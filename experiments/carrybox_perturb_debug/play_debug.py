import os
import sys


EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(EXPERIMENT_DIR, "..", ".."))
for path in (
    REPO_ROOT,
    os.path.join(REPO_ROOT, "legged_gym"),
    os.path.join(REPO_ROOT, "rsl_rl"),
):
    if path not in sys.path:
        sys.path.insert(0, path)

import isaacgym  # noqa: F401,E402
import torch  # noqa: E402

from legged_gym.envs import *  # noqa: F401,F403,E402
from legged_gym.scripts.play_ActorOnly import (  # noqa: E402
    load_actor_only_for_inference,
)
from legged_gym.utils import get_args, task_registry  # noqa: E402

from configs.debug_config import apply_debug_config  # noqa: E402
from envs.carrybox_perturb_debug import LeggedRobot as CarryBoxPerturbDebug  # noqa: E402


BASE_TASK = "carrybox_perturb"
DEBUG_TASK = "carrybox_perturb_debug"


def _base_task_from_args(task_name):
    if task_name in (BASE_TASK, DEBUG_TASK):
        return BASE_TASK
    raise ValueError(
        "play_debug.py only supports --task carrybox_perturb "
        "or --task carrybox_perturb_debug."
    )


def _register_debug_task(env_cfg, train_cfg):
    task_registry.register(DEBUG_TASK, CarryBoxPerturbDebug, env_cfg, train_cfg)


def play(args):
    requested_task = args.task
    base_task = _base_task_from_args(requested_task)

    env_cfg, train_cfg = task_registry.get_cfgs(name=base_task)
    env_cfg = apply_debug_config(env_cfg)

    train_cfg.runner.resume = False
    args.resume = False

    _register_debug_task(env_cfg, train_cfg)

    print(
        "Using experiment task carrybox_perturb_debug "
        f"with baseline config from {base_task}; requested task was {requested_task}."
    )
    env, _ = task_registry.make_env(name=DEBUG_TASK, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env,
        name=DEBUG_TASK,
        args=args,
        train_cfg=train_cfg,
        log_root=None,
    )
    if not args.play_dataset:
        load_actor_only_for_inference(ppo_runner, args.resume_path, device=env.device)
    policy = ppo_runner.get_inference_policy(device=env.device)

    for i in range(10 * int(env.max_episode_length)):
        env.commands[:, 0] = 0.8
        env.commands[:, 1] = 0.0
        env.commands[:, 2] = 0.0
        env.gym.fetch_results(env.sim, True)
        actions = policy(obs.detach())
        if args.play_dataset:
            env.play_dataset_step(i)
        else:
            obs, _, _, _, _, _, _, _ = env.step(actions.detach())


if __name__ == "__main__":
    args = get_args()
    play(args)
