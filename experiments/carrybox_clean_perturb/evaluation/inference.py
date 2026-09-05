"""Shared CarryBox actor loading and observation checks (no simulator imports)."""

import os
from pathlib import Path

import torch

from configs.evaluation_config import FIXED_COMMAND

LEGGED_GYM_ROOT_DIR = str(Path(__file__).resolve().parents[3] / "legged_gym")
EXPECTED_ACTOR_INPUT_DIM = 738
EXPECTED_TASK_OBS_DIM = 15
EXPECTED_ACTION_DIM = 29


def _resolve_checkpoint_path(checkpoint_path):
    if checkpoint_path is None:
        raise ValueError("CarryBox evaluation requires --resume_path.")
    resolved_path = os.path.expanduser(
        checkpoint_path.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
    )
    if not os.path.isfile(resolved_path):
        raise FileNotFoundError(f"Checkpoint not found: {resolved_path}")
    return resolved_path


def load_actor_only_for_inference(ppo_runner, checkpoint_path, device):
    checkpoint_path = _resolve_checkpoint_path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint does not contain 'model_state_dict': {checkpoint_path}")

    checkpoint_state = checkpoint["model_state_dict"]
    if "actor.0.weight" not in checkpoint_state:
        raise RuntimeError("Checkpoint actor is missing actor.0.weight")
    first_weight = checkpoint_state["actor.0.weight"]
    if first_weight.ndim != 2:
        raise RuntimeError("Checkpoint actor.0.weight must be a matrix")
    actor_input_dim = first_weight.shape[1]
    if int(actor_input_dim) != EXPECTED_ACTOR_INPUT_DIM:
        raise AssertionError(
            f"Expected checkpoint actor input {EXPECTED_ACTOR_INPUT_DIM}, got {actor_input_dim}"
        )

    actor_state = {
        key: value
        for key, value in checkpoint_state.items()
        if key.startswith("actor.") or key == "std"
    }
    if not any(key.startswith("actor.") for key in actor_state):
        raise RuntimeError(f"Checkpoint has no actor parameters: {checkpoint_path}")

    actor_critic = ppo_runner.alg.actor_critic
    current_state = actor_critic.state_dict()
    linear_layers = [layer for layer in actor_critic.actor.modules()
                     if isinstance(layer, torch.nn.Linear)]
    if not linear_layers or linear_layers[-1].out_features != EXPECTED_ACTION_DIM:
        raise AssertionError("Current CarryBox actor must output 29 actions")
    current_actor_input_dim = current_state["actor.0.weight"].shape[1]
    if int(current_actor_input_dim) != EXPECTED_ACTOR_INPUT_DIM:
        raise AssertionError(
            "Current policy actor input mismatch: "
            f"expected {EXPECTED_ACTOR_INPUT_DIM}, got {current_actor_input_dim}"
        )

    missing_actor = sorted(
        key for key in current_state if key.startswith("actor.") and key not in actor_state
    )
    if missing_actor:
        raise RuntimeError(f"Checkpoint actor is missing parameters: {missing_actor}")
    shape_mismatches = {}
    for key, value in actor_state.items():
        if key not in current_state:
            shape_mismatches[key] = (tuple(value.shape), None)
        elif value.shape != current_state[key].shape:
            shape_mismatches[key] = (tuple(value.shape), tuple(current_state[key].shape))
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

    print(
        "[ASSERT] checkpoint Actor loads successfully "
        f"(actor_input_dim={actor_input_dim}, current_actor_input_dim={current_actor_input_dim})"
    )
    return checkpoint_path


def _current_task_obs(env, obs):
    return obs[0, -int(env.num_task_obs):]


def assert_startup_compatibility(env, obs, expected_command=FIXED_COMMAND):
    if int(env.num_actions) != EXPECTED_ACTION_DIM:
        raise AssertionError(f"Expected 29 actions, got {env.num_actions}")
    if int(env.num_task_obs) != EXPECTED_TASK_OBS_DIM:
        raise AssertionError(f"Expected task obs dim 15, got {env.num_task_obs}")
    if int(env.actor_obs_length) != EXPECTED_ACTOR_INPUT_DIM:
        raise AssertionError(
            f"Expected actor obs dim 738, got {env.actor_obs_length}"
        )
    if tuple(obs.shape) != (env.num_envs, EXPECTED_ACTOR_INPUT_DIM):
        raise AssertionError(f"Unexpected obs shape: {tuple(obs.shape)}")
    task_obs = _current_task_obs(env, obs)
    if task_obs.numel() != EXPECTED_TASK_OBS_DIM:
        raise AssertionError(f"Expected current task obs length 15, got {task_obs.numel()}")
    command = task_obs[-3:]
    if expected_command is not None:
        expected = torch.tensor(expected_command, dtype=command.dtype, device=command.device)
        if not torch.allclose(command, expected, atol=1.0e-6, rtol=0.0):
            raise AssertionError(
                "Current task observation command mismatch: "
                f"expected={expected_command}, got={command.detach().cpu().tolist()}"
            )
    forbidden_goal_attrs = (
        "goal_pos",
        "goal_rot",
        "robot2goal_dir",
        "robot2goal_dist",
        "object2goal_pos",
        "object2goal_dist_xy",
        "object2goal_dist_xyz",
    )
    present = [name for name in forbidden_goal_attrs if hasattr(env, name)]
    if present:
        raise AssertionError(f"Goal/relocation evaluator attributes present: {present}")
    print(f"[ASSERT] Actor input dimension = {env.actor_obs_length}")
    print(f"[ASSERT] task obs dimension = {env.num_task_obs}")
    print(
        "[ASSERT] current task obs env0 = "
        f"{[round(float(v), 6) for v in task_obs.detach().cpu().tolist()]}"
    )
    print(
        "[ASSERT] last 3 task-obs values = "
        f"{[round(float(v), 6) for v in command.detach().cpu().tolist()]}"
    )
    print("[ASSERT] no goal-conditioned task observation")


