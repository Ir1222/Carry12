"""Simulator-independent checkpoint and control contracts for teacher handoff."""
import hashlib
import math
import re

import torch
from torch import nn

TEACHER_WARMUP, HANDOFF_TRANSITION, STUDENT_CARRY = range(3)


def matching_control_dt(actual, expected):
    """Allow <=1 ns of SimParams float32 representation error, not a changed timestep."""
    return (isinstance(actual, (int, float)) and not isinstance(actual, bool)
            and isinstance(expected, (int, float)) and not isinstance(expected, bool)
            and math.isfinite(actual) and math.isfinite(expected) and actual > 0 and expected > 0
            and math.isclose(actual, expected, rel_tol=0., abs_tol=1e-9))


def read_checkpoint(path):
    # These are explicitly supplied, trusted training checkpoints.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def checkpoint_actor(path, device="cpu", frozen=True):
    checkpoint = read_checkpoint(path)
    state = checkpoint["model_state_dict"]
    actor_state = {k[6:]: v for k, v in state.items() if k.startswith("actor.")}
    ids = sorted(int(k.split(".")[0]) for k in actor_state
                 if re.fullmatch(r"\d+\.weight", k))
    if not ids or ids != list(range(0, 2 * len(ids), 2)):
        raise ValueError("Expected unnormalized Sequential ELU MLP actor")
    layers, dims = [], []
    for i, index in enumerate(ids):
        weight = actor_state[f"{index}.weight"]
        if weight.ndim != 2 or (dims and dims[-1] != weight.shape[1]):
            raise ValueError("Invalid MLP dimensions")
        if not dims:
            dims.append(weight.shape[1])
        dims.append(weight.shape[0])
        layers.append(nn.Linear(weight.shape[1], weight.shape[0]))
        if i + 1 < len(ids):
            layers.append(nn.ELU())
    actor = nn.Sequential(*layers).to(device)
    actor.load_state_dict(actor_state, strict=True)
    actor.requires_grad_(not frozen)
    actor.eval()
    return actor, {"dims": dims, "critic_input": state["critic.0.weight"].shape[1],
                   "std_shape": list(state["std"].shape),
                   "amp": "amp_state_dict" in checkpoint,
                   "activation": "ELU (source-code contract; absent from state_dict)"}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def joint_groups(names):
    groups = {"arms": [], "waist": [], "legs": []}
    for i, name in enumerate(names):
        group = "waist" if name.startswith("waist_") else (
            "arms" if any(part in name for part in ("shoulder", "elbow", "wrist")) else "legs")
        groups[group].append(i)
    if [len(groups[k]) for k in ("arms", "waist", "legs")] != [14, 3, 12]:
        raise ValueError("Expected G1 29 DOFs: 14 arms, 3 waist, 12 legs")
    return groups


def adapt_action(raw, previous, rates, dt, scale, clip, enabled, mode):
    """Rate limit PD targets in rad/s; previous is the last EXECUTED action.

    No hidden filter state: previous applied action is already in the 738D
    observation. PPO stores the pre-transform sample and its Normal log density.
    """
    if mode not in ("hard", "rate_limited") or dt <= 0 or scale <= 0:
        raise ValueError("Invalid adapter mode, timestep or action scale")
    clipped = raw.clamp(-clip, clip)
    if mode == "hard":
        return clipped
    if torch.any(rates <= 0):
        raise ValueError("Target rates must be positive")
    bound = rates * dt / scale
    limited = previous + (clipped - previous).clamp(-bound, bound)
    return torch.where(enabled[:, None], limited, clipped)


def pose_excess(q, lower, upper, softness):
    return ((lower - q).clamp_min(0) + (q - upper).clamp_min(0)) / softness
