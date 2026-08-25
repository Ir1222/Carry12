"""Pure outcome-state helpers for CarryBox evaluator metrics."""

import torch


def humanoid_failure_masks(head_z, base_z, roll, pitch, hip_z):
    """Return the evaluator's named humanoid-failure masks."""
    return {
        "head_low": head_z < 0.6,
        "base_low": base_z < 0.2,
        "base_tilt": (torch.abs(roll) > 0.5) | (torch.abs(pitch) > 1.1),
        "hip_low": torch.any(hip_z < 0.15, dim=1),
    }


def box_instability_masks(box_velocity_xy, projected_gravity_z, box_termination):
    """Return named box conditions that already terminate the base environment."""
    masks = {
        "box_unstable_speed": torch.linalg.vector_norm(
            box_velocity_xy, dim=-1
        )
        > 3.0,
    }
    if bool(box_termination):
        masks["box_tilt"] = projected_gravity_z > -0.05
    return masks


def update_box_drop_state(
    carry_achieved,
    box_failure,
    drop_streak,
    confirmed_carry,
    box_bottom_z,
    ground_z,
    ground_clearance,
    confirm_steps,
):
    """Latch a drop only after confirmed carry and sustained near-ground contact."""
    confirm_steps = int(confirm_steps)
    if confirm_steps < 1:
        raise ValueError(f"confirm_steps must be positive, got {confirm_steps}")

    carry_achieved = carry_achieved | confirmed_carry
    grounded_after_carry = carry_achieved & (
        box_bottom_z <= ground_z + float(ground_clearance)
    )
    drop_streak = torch.where(
        grounded_after_carry,
        drop_streak + 1,
        torch.zeros_like(drop_streak),
    )
    newly_failed = ~box_failure & (drop_streak >= confirm_steps)
    box_failure = box_failure | newly_failed
    return carry_achieved, box_failure, drop_streak, newly_failed
