"""Runtime parity diagnostics for the clean CarryBox evaluator."""

import copy

import torch


def short_values(tensor, count=10):
    values = tensor.detach().cpu().reshape(-1).tolist()
    return [round(float(value), 6) for value in values[:count]]


def tensor_norm(tensor):
    return float(torch.linalg.vector_norm(tensor.detach()).item())


def tensor_min_max(tensor):
    flat = tensor.detach().reshape(-1)
    if flat.numel() == 0:
        return float("nan"), float("nan")
    return float(torch.min(flat).item()), float(torch.max(flat).item())


def tensor_digest(tensor):
    flat = tensor.detach().reshape(-1)
    weights = torch.arange(1, flat.numel() + 1, device=flat.device, dtype=flat.dtype)
    return float(torch.sum(flat * weights).item())


def _attr_tensor(env, name, env_id=0):
    if not hasattr(env, name):
        return None
    value = getattr(env, name)
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value
        return value[env_id]
    return None


def force_norm(env, env_id=0):
    if not hasattr(env, "clean_eval_force_tensor"):
        return 0.0
    if not hasattr(env, "box_rigid_body_index"):
        return tensor_norm(env.clean_eval_force_tensor[env_id])
    return tensor_norm(env.clean_eval_force_tensor[env_id, int(env.box_rigid_body_index)])


def print_actor_history(label, env, obs, env_id=0):
    one_step_dim = int(env.num_one_step_actor_obs)
    history_len = int(env.actor_history_length)
    history = obs[env_id].detach().reshape(history_len, one_step_dim)
    task_dim = int(env.num_task_obs)

    print(f"[PARITY][{label}] actor_history shape={tuple(history.shape)}")
    for frame_id in range(history_len):
        frame = history[frame_id]
        task_obs = frame[-task_dim:]
        command = task_obs[-3:]
        nonzero = int(torch.count_nonzero(frame).item())
        print(
            f"[HISTORY][{label}] h={frame_id} "
            f"norm={tensor_norm(frame):.9f} "
            f"nonzero={nonzero} "
            f"task_obs={short_values(task_obs, count=task_dim)} "
            f"command={short_values(command, count=3)}"
        )


def print_initial_state(label, env, obs, env_id=0):
    print(f"[PARITY][{label}] initial_state")
    print(f"seed={getattr(env.cfg, 'seed', 'unknown')}")
    print(f"num_envs={env.num_envs}")
    print(f"episode_length={int(env.max_episode_length)}")
    print(f"env.test={bool(getattr(env.cfg.env, 'test', False))}")
    print(f"command={short_values(env.commands[env_id, 0:3], count=3)}")
    print(f"root_pose={short_values(torch.cat((env.root_states[env_id, 0:3], env.root_states[env_id, 3:7])), count=7)}")
    print(f"root_lin_vel={short_values(env.root_states[env_id, 7:10], count=3)}")
    print(f"root_ang_vel={short_values(env.root_states[env_id, 10:13], count=3)}")
    print(f"dof_pos_norm={tensor_norm(env.dof_pos[env_id]):.9f}")
    print(f"dof_pos_first10={short_values(env.dof_pos[env_id], count=10)}")
    print(f"dof_vel_norm={tensor_norm(env.dof_vel[env_id]):.9f}")
    print(f"dof_vel_first10={short_values(env.dof_vel[env_id], count=10)}")
    print(f"Kp_factors_first10={short_values(_attr_tensor(env, 'Kp_factors', env_id), count=10)}")
    print(f"Kd_factors_first10={short_values(_attr_tensor(env, 'Kd_factors', env_id), count=10)}")
    print(f"motor_strength_first10={short_values(_attr_tensor(env, 'motor_strength', env_id), count=10)}")
    print(f"actuation_offset_first10={short_values(_attr_tensor(env, 'actuation_offset', env_id), count=10)}")
    payload = _attr_tensor(env, "payload", env_id)
    com_displacement = _attr_tensor(env, "com_displacement", env_id)
    friction = _attr_tensor(env, "friction_coeffs", env_id)
    restitution = _attr_tensor(env, "restitution_coeffs", env_id)
    print(f"payload={short_values(payload, count=3) if payload is not None else 'missing'}")
    print(f"com_displacement={short_values(com_displacement, count=3) if com_displacement is not None else 'missing'}")
    print(f"friction={short_values(friction, count=3) if friction is not None else 'missing'}")
    print(f"restitution={short_values(restitution, count=3) if restitution is not None else 'missing'}")
    print(f"box_size={short_values(env._box_size[env_id], count=3)}")
    print(f"box_mass={float(env.box_masses[env_id].item()):.9f}")
    print(f"box_pose={short_values(torch.cat((env.box_states[env_id, 0:3], env.box_states[env_id, 3:7])), count=7)}")
    print(f"box_lin_vel={short_values(env.box_states[env_id, 7:10], count=3)}")
    print(f"box_ang_vel={short_values(env.box_states[env_id, 10:13], count=3)}")
    print(f"obs_norm={tensor_norm(obs[env_id]):.9f}")
    print(f"obs_digest={tensor_digest(obs[env_id]):.9f}")
    print_actor_history(label, env, obs, env_id=env_id)


def termination_reason(env, env_id=0, termination_ids=None):
    if hasattr(env, "clean_eval_last_termination_reason"):
        reason = env.clean_eval_last_termination_reason[env_id]
        if reason:
            return reason
    reasons = []
    if hasattr(env, "time_out_buf") and bool(env.time_out_buf[env_id].item()):
        reasons.append("timeout")
    if hasattr(env, "head_index") and bool((env.rigid_body_states[env_id, env.head_index, 2] < 0.6).item()):
        reasons.append("head_low")
    if bool((env.root_states[env_id, 2] < 0.2).item()):
        reasons.append("base_low")
    if bool((torch.abs(env.roll[env_id]) > 0.5).item()) or bool(
        (torch.abs(env.pitch[env_id]) > 1.1).item()
    ):
        reasons.append("base_tilt")
    if termination_ids is not None and termination_ids.numel() > 0:
        ids = {int(value) for value in termination_ids.detach().cpu().reshape(-1).tolist()}
        if env_id in ids and not reasons:
            reasons.append("termination")
    return "|".join(reasons)


def print_policy_step_trace(
    label,
    step_id,
    env,
    actor_obs,
    actions,
    dones,
    termination_ids=None,
    env_id=0,
):
    action_min, action_max = tensor_min_max(actions[env_id])
    torque_min, torque_max = tensor_min_max(env.torques[env_id])
    finite_obs = bool(torch.isfinite(actor_obs[env_id]).all().item())
    finite_action = bool(torch.isfinite(actions[env_id]).all().item())
    finite_torque = bool(torch.isfinite(env.torques[env_id]).all().item())
    reason = termination_reason(env, env_id=env_id, termination_ids=termination_ids)
    actor_input_command = actor_obs[env_id, -3:]
    print(
        f"[TRACE][{label}] step={step_id} "
        f"root_z={float(env.root_states[env_id, 2].item()):.6f} "
        f"roll={float(env.roll[env_id].item()):.6f} "
        f"pitch={float(env.pitch[env_id].item()):.6f} "
        f"base_lin_vel={short_values(env.base_lin_vel[env_id], count=3)} "
        f"base_ang_vel={short_values(env.base_ang_vel[env_id], count=3)} "
        f"env_command={short_values(env.commands[env_id, 0:3], count=3)} "
        f"actor_input_command={short_values(actor_input_command, count=3)} "
        f"actor_obs_norm={tensor_norm(actor_obs[env_id]):.9f} "
        f"action_min={action_min:.9f} action_max={action_max:.9f} "
        f"action_norm={tensor_norm(actions[env_id]):.9f} "
        f"torque_min={torque_min:.9f} torque_max={torque_max:.9f} "
        f"torque_norm={tensor_norm(env.torques[env_id]):.9f} "
        f"projected_gravity={short_values(env.projected_gravity[env_id], count=3)} "
        f"reset_buf={int(dones[env_id].item())} "
        f"termination_reason={reason} "
        f"external_force_norm={force_norm(env, env_id=env_id):.9f} "
        f"finite_obs={finite_obs} finite_action={finite_action} finite_torque={finite_torque}"
    )


def compare_actor_only_to_full_checkpoint(ppo_runner, checkpoint_path, obs, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_state = checkpoint["model_state_dict"]
    actor_critic = ppo_runner.alg.actor_critic

    full_loaded = copy.deepcopy(actor_critic).to(device)
    full_loaded.load_state_dict(checkpoint_state)
    full_loaded.eval()

    actor_state = actor_critic.state_dict()
    max_actor_diff = 0.0
    max_actor_key = ""
    for key, full_value in full_loaded.state_dict().items():
        if not (key.startswith("actor.") or key == "std"):
            continue
        diff = torch.max(torch.abs(actor_state[key] - full_value)).item()
        if float(diff) > max_actor_diff:
            max_actor_diff = float(diff)
            max_actor_key = key

    with torch.no_grad():
        action_current = actor_critic.act_inference(obs.detach())
        action_full = full_loaded.act_inference(obs.detach())
    max_action_diff = float(torch.max(torch.abs(action_current - action_full)).item())
    print(
        "[PARITY][checkpoint] "
        f"max_actor_parameter_difference={max_actor_diff:.12g} "
        f"key={max_actor_key or 'none'} "
        f"max_action_difference={max_action_diff:.12g}"
    )
    if max_actor_diff != 0.0:
        raise AssertionError(
            "Actor-only loader does not match full checkpoint actor parameters: "
            f"max_diff={max_actor_diff} key={max_actor_key}"
        )
    if max_action_diff > 1.0e-6:
        raise AssertionError(
            "Actor-only and full checkpoint policies disagree on the same observation: "
            f"max_action_diff={max_action_diff}"
        )
