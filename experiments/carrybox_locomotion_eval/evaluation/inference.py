"""Actor-only checkpoint loading with explicit architecture validation."""

import os
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
LEGGED_GYM_ROOT_DIR = REPO_ROOT / "legged_gym"


def resolve_checkpoint_path(checkpoint_path):
    if not checkpoint_path:
        raise ValueError("Carry-locomotion evaluation requires --resume_path")
    expanded = str(checkpoint_path).format(
        LEGGED_GYM_ROOT_DIR=str(LEGGED_GYM_ROOT_DIR)
    )
    resolved = os.path.abspath(os.path.expanduser(expanded))
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"Checkpoint not found: {resolved}")
    return resolved


def build_actor_only_policy(env, train_cfg, checkpoint_path, device):
    """Build the configured Actor and load no critic/optimizer state."""
    from legged_gym.utils.helpers import class_to_dict
    from rsl_rl.modules import ActorCritic

    actor_input_dim = int(env.actor_obs_length)
    if actor_input_dim != int(env.num_obs):
        raise AssertionError(
            f"actor_obs_length={actor_input_dim} but num_obs={env.num_obs}"
        )
    history_length = int(env.actor_history_length)
    one_step_dim = int(env.num_one_step_actor_obs)
    if actor_input_dim != history_length * one_step_dim:
        raise AssertionError(
            "Actor observation/history mismatch: "
            f"{actor_input_dim} != {history_length} * {one_step_dim}"
        )
    if one_step_dim - int(env.num_task_obs) != int(env.num_one_step_proprio_obs):
        raise AssertionError("One-step task/proprio observation dimensions disagree")

    actor_critic = ActorCritic(
        actor_input_dim,
        int(env.num_privileged_obs),
        history_length,
        int(env.num_actions),
        **class_to_dict(train_cfg.policy),
    ).to(device)

    resolved = resolve_checkpoint_path(checkpoint_path)
    checkpoint = torch.load(resolved, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint has no model_state_dict: {resolved}")
    source = checkpoint["model_state_dict"]
    source_actor = {
        key[len("actor.") :]: value
        for key, value in source.items()
        if key.startswith("actor.")
    }
    if not source_actor:
        raise KeyError(f"Checkpoint has no actor.* parameters: {resolved}")

    target_actor = actor_critic.actor.state_dict()
    missing = sorted(target_actor.keys() - source_actor.keys())
    unexpected = sorted(source_actor.keys() - target_actor.keys())
    mismatches = {
        key: (tuple(source_actor[key].shape), tuple(target_actor[key].shape))
        for key in target_actor.keys() & source_actor.keys()
        if source_actor[key].shape != target_actor[key].shape
    }
    if missing or unexpected or mismatches:
        raise RuntimeError(
            "Checkpoint Actor is incompatible with the carry-locomotion task: "
            f"missing={missing}, unexpected={unexpected}, shapes={mismatches}"
        )

    first_linear = next(
        layer for layer in actor_critic.actor if isinstance(layer, torch.nn.Linear)
    )
    last_linear = [
        layer for layer in actor_critic.actor if isinstance(layer, torch.nn.Linear)
    ][-1]
    checkpoint_input = int(source_actor["0.weight"].shape[1])
    if checkpoint_input != actor_input_dim or first_linear.in_features != actor_input_dim:
        raise AssertionError(
            f"Actor input mismatch: checkpoint={checkpoint_input}, env={actor_input_dim}"
        )
    if last_linear.out_features != int(env.num_actions):
        raise AssertionError(
            f"Actor action mismatch: actor={last_linear.out_features}, env={env.num_actions}"
        )

    actor_critic.actor.load_state_dict(source_actor, strict=True)
    actor_critic.eval()
    print(
        "[ACTOR] actor-only load complete: "
        f"input={actor_input_dim}, history={history_length}, "
        f"one_step={one_step_dim}, task={env.num_task_obs}, "
        f"actions={env.num_actions}"
    )
    return actor_critic.act_inference, resolved


def assert_observation_compatibility(env, obs, requested_command):
    expected_shape = (int(env.num_envs), int(env.actor_obs_length))
    if tuple(obs.shape) != expected_shape:
        raise AssertionError(
            f"Observation shape {tuple(obs.shape)} != {expected_shape}"
        )
    if int(env.num_task_obs) != 15:
        raise AssertionError(f"Expected 15 task observations, got {env.num_task_obs}")
    if int(env.actor_history_length) != 6:
        raise AssertionError(
            f"Expected actor history length 6, got {env.actor_history_length}"
        )
    if int(env.num_actions) != 29:
        raise AssertionError(f"Expected 29 actions, got {env.num_actions}")
    history_prefix = obs[:, : -int(env.num_one_step_actor_obs)]
    if torch.count_nonzero(history_prefix).item() != 0:
        raise AssertionError(
            "Reset observation contains stale actor-history frames from a prior trial"
        )
    env.assert_evaluation_command(obs)
    observed = obs[:, -env.num_task_obs :][:, -3:]
    expected = observed.new_tensor(requested_command).expand_as(observed)
    if not torch.allclose(observed, expected, atol=1.0e-6, rtol=0.0):
        raise AssertionError(
            f"Task command observation {observed.tolist()} != {list(requested_command)}"
        )
