"""Diagnostic only: one CarryBox PhysX environment, two frozen MLP actors.

Tested for Python syntax only. Requires Ir1222/Carry12 velocity-tracking checkout,
Isaac Gym, both trusted checkpoints, and compatible observation/action contracts.
Does NOT retrain, blend actions, or reset at the handoff.

Usage:
python legged_gym/legged_gym/scripts/test_teacher_handoff.py \
  --task carrybox --full_ckpt /path/full.pt --loco_ckpt /path/loco.pt \
  --vx 0.4 --hold_s 3.0 --headless
"""
import argparse
import csv
import math
import os
import re
import sys
from pathlib import Path

import isaacgym  # noqa: F401; import before torch
import torch
from torch import nn

from legged_gym.envs import *  # noqa: F401,F403; registers tasks
from legged_gym.utils import get_args, task_registry


def parse_args():
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    p.add_argument("--full_ckpt", required=True)
    p.add_argument("--loco_ckpt", required=True)
    p.add_argument("--vx", type=float, default=0.4)
    p.add_argument("--stable_s", type=float, default=0.3)
    p.add_argument("--hold_s", type=float, default=3.0)
    p.add_argument("--tilt_limit_deg", type=float, default=20.0)
    p.add_argument("--rel_vel_limit", type=float, default=0.3)
    p.add_argument("--box_omega_limit", type=float, default=1.0)
    p.add_argument("--max_episode_s", type=float, default=35.0)
    p.add_argument("--log_csv", default="handoff_diagnostic.csv")
    own, other = p.parse_known_args()
    saved = sys.argv
    try:
        sys.argv = [saved[0], *other]
        gym_args = get_args()
    finally:
        sys.argv = saved
    if gym_args.task != "carrybox":
        raise ValueError("This diagnostic must use --task carrybox")
    gym_args.num_envs = 1
    if not 0.0 < own.vx <= 0.8:
        raise ValueError("For first pass use common nominal vx in (0, 0.8], e.g. 0.4")
    return gym_args, own


def load_sequential_mlp_actor(ckpt_path, device):
    """Load current-repo ELU Sequential Actor; infer hidden widths from checkpoint.

    Handles different MLP hidden widths. Intentional fail-fast for different
    architectures, normalization wrappers, recurrent nets, or non-MLP checkpoints.
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    # Only load checkpoints trained by you / sources you trust.
    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:  # legacy PyTorch
        state = torch.load(ckpt_path, map_location="cpu")
    sd = state["model_state_dict"]
    actor_sd = {k[len("actor."):]: v for k, v in sd.items() if k.startswith("actor.")}
    linear_ids = sorted(int(k.split(".")[0]) for k in actor_sd if re.fullmatch(r"\d+\.weight", k))
    if not linear_ids:
        raise RuntimeError(f"No Sequential actor linear layers found: {ckpt_path}")
    modules = []
    for idx, layer_id in enumerate(linear_ids):
        if layer_id != 2 * idx:
            raise RuntimeError("Non-standard architecture: inspect the original Actor class")
        weight = actor_sd[f"{layer_id}.weight"]
        if weight.ndim != 2:
            raise RuntimeError("Non-MLP actor: inspect checkpoint architecture")
        modules.append(nn.Linear(weight.shape[1], weight.shape[0]))
        if idx != len(linear_ids) - 1:
            modules.append(nn.ELU())  # Current Carry12 ActorCritic default
    actor = nn.Sequential(*modules)
    actor.load_state_dict(actor_sd, strict=True)
    actor.to(device).eval()
    for p in actor.parameters():
        p.requires_grad_(False)
    print(f"[ACTOR] {ckpt_path}")
    print("        MLP dims:", [int(actor_sd[f"{i}.weight"].shape[1]) for i in linear_ids[:1]]
          + [int(actor_sd[f"{i}.weight"].shape[0]) for i in linear_ids])
    critic_first = sd.get("critic.0.weight")
    print("        critic input:", None if critic_first is None else critic_first.shape[1],
          " AMP:", "amp_state_dict" in state)
    return actor


def update_initial_command_without_history_shift(env, vx):
    """Change latest command fields once; do NOT call compute_observations twice."""
    env.commands[:, :] = 0.0
    env.commands[:, 0] = vx
    env.carry_policy_commands[:, :] = 0.0
    env.carry_policy_commands[:, 0] = vx
    # Current Carry12 frame layout = 108 proprio + 12 box + 3 command.
    env.obs_buf[:, -3:] = env.carry_policy_commands[:, :3]


def physical_readiness(env, p):
    # All data here is simulator-privileged: diagnostic supervisor ONLY.
    box_bottom = env.box_states[:, 2] - 0.5 * env._box_size[:, 2]
    support = torch.maximum(
        env.platform_pos[:, 2] + 0.5 * env._platform_height,
        env.env_origins[:, 2],
    )
    lift_m = box_bottom - support
    contact_force = torch.linalg.vector_norm(
        env.contact_forces[:, env.hand_colli_indices, :], dim=-1
    )
    hands_ok = (contact_force > float(env.cfg.rewards.hand_contact_threshold)).all(-1)
    tilt_deg = torch.acos((-env.projected_gravity_box[:, 2]).clamp(-1.0, 1.0)) * (180 / math.pi)
    box_omega = torch.linalg.vector_norm(env.box_states[:, 10:13], dim=-1)
    # Robot torso linear velocity and box velocity are both WORLD frame here.
    torso_vel_world = env.rigid_body_states[:, env.upper_body_index, 7:10]
    rel_v = torch.linalg.vector_norm(env.box_states[:, 7:10] - torso_vel_world, dim=-1)
    ready = (lift_m > 0.10) & hands_ok & (tilt_deg < p.tilt_limit_deg) & (
        rel_v < p.rel_vel_limit) & (box_omega < p.box_omega_limit)
    return ready, lift_m, contact_force, tilt_deg, rel_v, box_omega


def main():
    args, own = parse_args()
    env_cfg, _ = task_registry.get_cfgs(name="carrybox")
    env_cfg.env.num_envs = 1
    env_cfg.env.test = True
    env_cfg.env.episode_length_s = own.max_episode_s
    env_cfg.asset.box.reset_mode = "default"
    env_cfg.asset.box.random_props = False
    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.delay = False
    env_cfg.commands.resample_carry_commands = False

    # Exactly ONE simulator. Do not construct either training runner here.
    env, _ = task_registry.make_env(name="carrybox", args=args, env_cfg=env_cfg)
    full = load_sequential_mlp_actor(own.full_ckpt, env.device)
    loco = load_sequential_mlp_actor(own.loco_ckpt, env.device)
    if full[0].in_features != env.num_obs or loco[0].in_features != env.num_obs:
        raise RuntimeError(
            f"Observation contract mismatch: full={full[0].in_features}, "
            f"loco={loco[0].in_features}, env={env.num_obs}. "
            "Build explicit per-teacher observation adapters; do NOT pad/truncate."
        )
    if full[-1].out_features != env.num_actions or loco[-1].out_features != env.num_actions:
        raise RuntimeError("Action mismatch: need explicit joint mapping/actuation adapter")

    obs, _ = env.reset()  # ONE initial reset; no resets on handoff
    update_initial_command_without_history_shift(env, own.vx)
    obs = env.get_observations()

    dt = float(env.dt)
    streak = 0
    required = max(1, math.ceil(own.stable_s / dt))
    after = 0
    post_bilateral_steps = 0
    post_max_tilt = 0.0
    post_max_rel_v = 0.0
    post_contact_loss_streak = 0
    required_after = max(1, math.ceil(own.hold_s / dt))
    switched = False
    prev_action = torch.zeros((1, env.num_actions), device=env.device)
    rows = []
    reason = "episode_limit"

    print(f"[INFO] dt={dt:.4f}s, stable_steps={required}; hold_steps={required_after}; "
          "keep vx constant through handoff")
    with torch.inference_mode():
        for t in range(int(own.max_episode_s / dt) - 2):
            applied_mode = "loco" if switched else "full"
            a_full = full(obs)
            a_loco = loco(obs)  # shadow forward even while Full is active
            a = a_loco if switched else a_full
            # Difference is logged, not blended or clamped (to measure true gap).
            actor_delta_max = (a_loco - a_full).abs().max().item()
            action_step_max = (a - prev_action).abs().max().item()
            prev_action = a.detach().clone()
            obs, _, _, dones, _, _, _, _ = env.step(a.detach())
            if bool(dones[0].item()):
                reason = "terminated_after_handoff" if switched else "terminated_before_handoff"
                print("[DONE]", reason, "at t", round(t * dt, 2))
                break

            ready, lift_m, hand_forces, tilt_deg, rel_v, box_omega = physical_readiness(env, own)
            ok = bool(ready[0].item())
            if not switched:
                streak = streak + 1 if ok else 0
                if streak >= required:
                    switched = True  # next decision uses Locomotion; no sim reset
                    print(f"[HANDOFF] t={t * dt:.3f}s; "
                          f"angle={tilt_deg[0].item():.1f}deg, "
                          f"rel_v={rel_v[0].item():.3f}m/s; "
                          f"teacher_delta_max={actor_delta_max:.3f}")
            else:
                after += 1
                post_bilateral_steps += int((hand_forces[0] > env.cfg.rewards.hand_contact_threshold).all().item())
                post_max_tilt = max(post_max_tilt, float(tilt_deg[0].item()))
                post_max_rel_v = max(post_max_rel_v, float(rel_v[0].item()))
                post_contact_loss_streak = (post_contact_loss_streak + 1 if not
                    bool((hand_forces[0] > env.cfg.rewards.hand_contact_threshold).any().item()) else 0)
                # Full CarryBox config does not enforce Locomotion's box-tilt/drop termination.
                box_bottom = env.box_states[0, 2] - 0.5 * env._box_size[0, 2]
                if (float(tilt_deg[0].item()) > 50.0 or
                    float(box_bottom.item()) < 0.15 or
                    post_contact_loss_streak * dt > 0.20):
                    reason = "handoff_physical_failure"
                    print("[FAIL] Contact/tilt/drop safety criterion after handoff")
                    break
                if after >= required_after:
                    reason = "handoff_hold_completed"
                    print("[SUCCESS] Loco policy maintained physical carry for", own.hold_s, "seconds")
                    break
            rows.append({
                "time_s": round((t + 1) * dt, 5),
                "mode": applied_mode,
                "ready": int(ok),
                "lift_m": float(lift_m[0].item()),
                "hand_L_N": float(hand_forces[0, 0].item()),
                "hand_R_N": float(hand_forces[0, 1].item()),
                "tilt_deg": float(tilt_deg[0].item()),
                "box_omega_rad_s": float(box_omega[0].item()),
                "robot_box_rel_v_m_s": float(rel_v[0].item()),
                "actor_delta_max": actor_delta_max,
                "applied_action_step_max": action_step_max,
                "carry_velocity_active": int(env.carry_velocity_active[0].item()),
                "can_see_tag": int(env.can_see_tag[0].item()),
                "has_seen_tag": int(env.has_seen_tag[0].item()),
                "command_vx": float(env.carry_policy_commands[0, 0].item()),
                "command_vy": float(env.carry_policy_commands[0, 1].item()),
                "command_yaw": float(env.carry_policy_commands[0, 2].item()),
            })
    if rows:
        file_path = Path(own.log_csv)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print("[LOG]", file_path.resolve())
    print("[RESULT]", reason, "switched=", switched,
          "post_handoff_steps=", after,
          "bilateral_fraction=", round(post_bilateral_steps / max(after, 1), 3),
          "max_tilt_deg=", round(post_max_tilt, 2),
          "max_rel_v=", round(post_max_rel_v, 3))


if __name__ == "__main__":
    main()
