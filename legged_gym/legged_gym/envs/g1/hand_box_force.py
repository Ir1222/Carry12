"""Pure Torch helpers for estimating and decomposing hand--box contact force.

The input force is a rigid body's net contact force.  The projection is exact for
the supplied normal, but it is not a pairwise hand--box contact measurement.
"""

import torch


def estimate_box_face_normal_local(relative_position_local, box_size, eps=1.0e-6):
    """Estimate the closest box face normal from a point in the box frame.

    Returns ``(normal_local, face_id)`` where face ids are ``2 * axis + sign``
    and ``sign`` is one for the positive face.
    """
    normalized_distance = relative_position_local.abs() / (
        0.5 * box_size + float(eps)
    )
    face_axis = normalized_distance.argmax(dim=-1)
    face_coordinate = relative_position_local.gather(
        -1, face_axis.unsqueeze(-1)
    ).squeeze(-1)
    face_sign = torch.where(face_coordinate >= 0.0, 1.0, -1.0)
    normal_local = torch.zeros_like(relative_position_local)
    normal_local.scatter_(-1, face_axis.unsqueeze(-1), face_sign.unsqueeze(-1))
    face_id = 2 * face_axis + (face_sign > 0.0).long()
    return normal_local, face_id


def decompose_force(force, unit_normal):
    """Project force into compressive normal and residual tangential parts."""
    fn_signed = torch.sum(force * unit_normal, dim=-1)
    fn = torch.relu(fn_signed)
    fn_vector = fn.unsqueeze(-1) * unit_normal
    ft_vector = force - fn_vector
    ft = torch.linalg.vector_norm(ft_vector, dim=-1)
    return fn_signed, fn, fn_vector, ft_vector, ft


def force_closure_residual(hand_forces, box_force, eps=1.0e-6):
    """Normalized residual for hand resultants plus the box net contact force."""
    numerator = torch.linalg.vector_norm(hand_forces.sum(dim=-2) + box_force, dim=-1)
    denominator = (
        torch.linalg.vector_norm(hand_forces, dim=-1).sum(dim=-1)
        + torch.linalg.vector_norm(box_force, dim=-1)
        + float(eps)
    )
    return numerator / denominator
