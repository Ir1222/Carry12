import os

from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym  # noqa: F401
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import (
    export_jit_to_onnx,
    export_policy_as_jit,
    get_args,
    load_onnx_policy,
    task_registry,
)
from legged_gym.utils.helpers import update_cfg_from_args

import torch


def _resolve_checkpoint_path(checkpoint_path):
    if checkpoint_path is None:
        raise ValueError("play_ActorOnly.py requires --resume_path.")

    resolved_path = os.path.expanduser(
        checkpoint_path.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
    )
    if not os.path.isfile(resolved_path):
        raise FileNotFoundError(f"Checkpoint not found: {resolved_path}")
    return resolved_path


def load_actor_only_for_inference(ppo_runner, checkpoint_path, device):
    """Load only actor parameters so critic-observation changes do not affect play."""
    checkpoint_path = _resolve_checkpoint_path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise KeyError(
            f"Checkpoint does not contain 'model_state_dict': {checkpoint_path}"
        )

    checkpoint_state = checkpoint["model_state_dict"]
    actor_state = {
        key: value
        for key, value in checkpoint_state.items()
        if key.startswith("actor.") or key == "std"
    }
    if not any(key.startswith("actor.") for key in actor_state):
        raise RuntimeError(
            f"Checkpoint has no actor parameters under 'actor.*': {checkpoint_path}"
        )

    actor_critic = ppo_runner.alg.actor_critic
    current_state = actor_critic.state_dict()
    shape_mismatches = {}
    for key, value in actor_state.items():
        if key not in current_state:
            shape_mismatches[key] = (tuple(value.shape), None)
        elif value.shape != current_state[key].shape:
            shape_mismatches[key] = (
                tuple(value.shape),
                tuple(current_state[key].shape),
            )
    if shape_mismatches:
        raise RuntimeError(
            "Checkpoint actor is incompatible with the current policy network: "
            f"{shape_mismatches}"
        )

    incompatible = actor_critic.load_state_dict(actor_state, strict=False)
    missing_required = [
        key
        for key in incompatible.missing_keys
        if not key.startswith("critic.") and key != "std"
    ]
    if missing_required or incompatible.unexpected_keys:
        raise RuntimeError(
            "Actor-only checkpoint load was incomplete: "
            f"missing={missing_required}, unexpected={incompatible.unexpected_keys}"
        )

    actor_input_dim = checkpoint_state["actor.0.weight"].shape[1]
    current_actor_input_dim = current_state["actor.0.weight"].shape[1]
    checkpoint_critic_dim = (
        checkpoint_state["critic.0.weight"].shape[1]
        if "critic.0.weight" in checkpoint_state
        else "absent"
    )
    current_critic_dim = (
        current_state["critic.0.weight"].shape[1]
        if "critic.0.weight" in current_state
        else "absent"
    )
    print(
        f"Loaded actor-only policy from: {checkpoint_path} "
        f"(actor_input={actor_input_dim}, current_actor_input={current_actor_input_dim}, "
        f"checkpoint_critic={checkpoint_critic_dim}, current_critic={current_critic_dim}; "
        "critic intentionally skipped)"
    )


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # Apply the existing CLI config overrides before selecting playback-only
    # episode and command settings. task_registry.make_env applies these again
    # immediately before construction; the operation is idempotent.
    env_cfg, _ = update_cfg_from_args(env_cfg, None, args)

    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 10)
    env_cfg.env.test = True
    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.delay = False
    env_cfg.domain_rand.push_robots = False

    # This script always loads actor-only manually after runner construction.
    train_cfg.runner.resume = False
    args.resume = False

    # carrybox
    if args.task in ("carrybox", "carrybox_force", "carrybox_perturb"):
        env_cfg.asset.box.random_props = False
        env_cfg.asset.box.reset_mode = "default"
        force_enabled = (
            args.task == "carrybox_force"
            and env_cfg.external_force.enable_external_force
        )
        if not force_enabled:
            env_cfg.env.episode_length_s = 10
        if args.task == "carrybox" or (
            args.task == "carrybox_force" and not force_enabled
        ):
            env_cfg.commands.resample_carry_commands = False

    if args.play_dataset:
        env_cfg.viewer.pos = [-5, -5, 4]
        env_cfg.viewer.lookat = [0, 0, 2.0]

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    # load policy
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    if not args.play_dataset:
        load_actor_only_for_inference(ppo_runner, args.resume_path, device=env.device)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # export policy as a jit & onnx module (used to run it from C++)
    if EXPORT_POLICY:
        policy_name = "policy_name"
        path = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "exported", "policies")
        export_policy_as_jit(ppo_runner.alg.actor_critic, path, policy_name)
        print("Exported policy as jit script to: ", path)

        jit_path = os.path.join(path, f"{policy_name}.pt")
        jit_model = torch.jit.load(jit_path)
        dummy_input = torch.randn(1, obs.shape[1], device="cpu")
        onnx_path = os.path.join(path, f"{policy_name}.onnx")
        export_jit_to_onnx(jit_model, onnx_path, dummy_input)
        policy = load_onnx_policy(onnx_path)

    for i in range(10 * int(env.max_episode_length)):
        env.commands[:, 0] = 0.8
        env.commands[:, 1] = 0.0
        env.commands[:, 2] = 0.0
        env.gym.fetch_results(env.sim, True)
        actions = policy(obs.detach())
        if args.play_dataset:
            env.play_dataset_step(i)
        else:
            obs, _, rews, dones, infos, _, _, amp_state = env.step(actions.detach())


if __name__ == "__main__":
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
