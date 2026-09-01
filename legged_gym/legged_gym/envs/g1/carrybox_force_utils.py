"""Pure tensor helpers for Stage2A CarryBox force compliance.

This module deliberately has no Isaac Gym dependency so the force profile and
teacher mathematics can be unit-tested on a CPU-only development machine.
"""

import math

import torch


def resolve_directional_beta_ranges(
    curriculum_beta_ranges,
    curriculum_stage,
    direction_names,
):
    """Return validated beta ranges in the same order as ``direction_names``."""
    allowed_stages = tuple(sorted(curriculum_beta_ranges))
    if curriculum_stage not in curriculum_beta_ranges:
        raise ValueError(
            f"force curriculum stage must be one of {allowed_stages}, "
            f"got {curriculum_stage!r}"
        )

    stage_ranges = curriculum_beta_ranges[curriculum_stage]
    missing = [name for name in direction_names if name not in stage_ranges]
    if missing:
        raise ValueError(
            f"force curriculum stage {curriculum_stage} has no beta range for {missing}"
        )

    resolved = []
    for name in direction_names:
        value_range = stage_ranges[name]
        if len(value_range) != 2:
            raise ValueError(f"beta range for {name} must contain exactly two values")
        low, high = float(value_range[0]), float(value_range[1])
        if not 0.0 <= low <= high:
            raise ValueError(
                f"beta range for {name} must be non-negative and ordered, got {value_range}"
            )
        resolved.append((low, high))
    return tuple(resolved)


def sample_directional_beta(
    direction_names,
    direction_ids,
    curriculum_beta_ranges,
    curriculum_stage,
    beta_range=None,
):
    """Sample beta per environment from its selected direction's range.

    A non-None ``beta_range`` is a global override, including the fixed
    ``(X, X)`` range installed by ``--force_beta X``.
    """
    count = int(direction_ids.numel())
    if beta_range is not None:
        low, high = float(beta_range[0]), float(beta_range[1])
        if not 0.0 <= low <= high:
            raise ValueError("beta_range override must be non-negative and ordered")
        return low + (high - low) * torch.rand(count, device=direction_ids.device)

    ranges = resolve_directional_beta_ranges(
        curriculum_beta_ranges,
        curriculum_stage,
        direction_names,
    )
    range_tensor = torch.tensor(ranges, dtype=torch.float, device=direction_ids.device)
    if torch.any((direction_ids < 0) | (direction_ids >= len(direction_names))):
        raise ValueError("direction_ids contains an index outside direction_names")
    selected_ranges = range_tensor[direction_ids]
    return selected_ranges[:, 0] + (
        selected_ranges[:, 1] - selected_ranges[:, 0]
    ) * torch.rand(count, device=direction_ids.device)


def smooth_force_profile(elapsed_s, ramp_up_s, hold_s, ramp_down_s):
    """Cosine ramp-up, constant hold, and cosine ramp-down force scale."""
    ramp_up_s = torch.clamp(ramp_up_s, min=0.0)
    hold_s = torch.clamp(hold_s, min=0.0)
    ramp_down_s = torch.clamp(ramp_down_s, min=0.0)
    hold_start = ramp_up_s
    ramp_down_start = ramp_up_s + hold_s
    total_s = ramp_down_start + ramp_down_s

    ramp_up_tau = elapsed_s / torch.clamp(ramp_up_s, min=1.0e-9)
    ramp_up_scale = 0.5 * (1.0 - torch.cos(math.pi * torch.clamp(ramp_up_tau, 0.0, 1.0)))

    ramp_down_tau = (elapsed_s - ramp_down_start) / torch.clamp(ramp_down_s, min=1.0e-9)
    ramp_down_scale = 0.5 * (1.0 + torch.cos(math.pi * torch.clamp(ramp_down_tau, 0.0, 1.0)))

    scale = torch.where(elapsed_s < hold_start, ramp_up_scale, torch.ones_like(elapsed_s))
    scale = torch.where(elapsed_s >= ramp_down_start, ramp_down_scale, scale)
    return torch.where((elapsed_s >= 0.0) & (elapsed_s <= total_s), scale, torch.zeros_like(scale))


def wrap_to_pi(angles):
    """Non-mutating angle wrap used by the pure teacher helper."""
    return torch.atan2(torch.sin(angles), torch.cos(angles))


def compute_admittance_teacher(
    external_force_world,
    box_mass,
    nominal_vx,
    nominal_heading,
    robot_yaw,
    nominal_heading_error,
    nominal_raw_yaw_rate,
    nominal_yaw_target,
    heading_kp,
    max_yaw_rate,
    admittance_d_bar,
    max_heading_offset,
    teacher_vx_min,
    teacher_vx_max,
):
    """Compute reward-only Stage2A teacher targets.

    The zero-horizontal-force branch returns the supplied Stage1 targets
    directly, avoiding a second subtly different heading controller.
    """
    forward = torch.stack((torch.cos(nominal_heading), torch.sin(nominal_heading)), dim=-1)
    lateral = torch.stack((-torch.sin(nominal_heading), torch.cos(nominal_heading)), dim=-1)
    force_xy = external_force_world[..., :2]
    force_parallel = torch.sum(force_xy * forward, dim=-1)
    force_perp = torch.sum(force_xy * lateral, dim=-1)

    damping = torch.clamp(box_mass * float(admittance_d_bar), min=1.0e-9)
    u_parallel = nominal_vx + force_parallel / damping
    u_perp = force_perp / damping
    u_parallel_for_heading = torch.clamp(u_parallel, min=0.0)
    raw_heading_offset = torch.atan2(u_perp, u_parallel_for_heading)
    heading_offset = torch.clamp(
        raw_heading_offset,
        min=-float(max_heading_offset),
        max=float(max_heading_offset),
    )
    teacher_heading = wrap_to_pi(nominal_heading + heading_offset)

    resultant_velocity_world = nominal_vx.unsqueeze(-1) * forward + force_xy / damping.unsqueeze(-1)
    teacher_direction = torch.stack((torch.cos(teacher_heading), torch.sin(teacher_heading)), dim=-1)
    teacher_vx = torch.sum(resultant_velocity_world * teacher_direction, dim=-1)
    teacher_vx = torch.clamp(teacher_vx, min=float(teacher_vx_min), max=float(teacher_vx_max))

    teacher_heading_error = wrap_to_pi(teacher_heading - robot_yaw)
    teacher_yaw_rate = torch.clamp(
        nominal_raw_yaw_rate + float(heading_kp) * teacher_heading_error,
        min=-float(max_yaw_rate),
        max=float(max_yaw_rate),
    )

    zero_force = torch.linalg.vector_norm(force_xy, dim=-1) <= 1.0e-9
    teacher_vx = torch.where(zero_force, nominal_vx, teacher_vx)
    teacher_heading = torch.where(zero_force, nominal_heading, teacher_heading)
    teacher_heading_error = torch.where(zero_force, nominal_heading_error, teacher_heading_error)
    teacher_yaw_rate = torch.where(zero_force, nominal_yaw_target, teacher_yaw_rate)

    return (
        teacher_vx,
        teacher_heading,
        teacher_heading_error,
        teacher_yaw_rate,
        force_parallel,
        force_perp,
    )
