"""Offline carry calibration, analysis, and regression reference kernels.

Quaternions are XYZW, positions are link origins in world coordinates, and
hands are ordered [left_palm_link, right_palm_link]. The training environment
implements its rewards directly in envs/g1/carrybox_locomotion.py and does not
import this module. Keep these standalone kernels for CPU analysis and the
historical 7cb664d regression oracle. Neither training nor evaluation imports
this module; current constraints and metrics are independent of this calibration.
"""

import json
import math

import torch


METRIC_NAMES = (
    "left_hand_side_error_m", "right_hand_side_error_m",
    "left_hand_direction_error_rad", "right_hand_direction_error_rad",
    "hand_midpoint_error_m", "box_relative_position_error_m",
    "box_relative_orientation_error_rad", "box_relative_motion_error_mps",
    "arm_reference_error_rad",
)
MOTION_METRIC_INDEX = METRIC_NAMES.index("box_relative_motion_error_mps")
REWARD_NAMES = (
    "carry_hand_box_surface", "carry_arm_pose", "carry_relative_position",
    "carry_relative_orientation", "carry_relative_velocity",
)


def quat_conjugate(q):
    return torch.cat((-q[..., :3], q[..., 3:4]), dim=-1)


def quat_multiply(a, b):
    a, b = torch.broadcast_tensors(a, b)
    xyz = (a[..., 3:4] * b[..., :3] + b[..., 3:4] * a[..., :3]
           + torch.cross(a[..., :3], b[..., :3], dim=-1))
    w = a[..., 3:4] * b[..., 3:4] - (a[..., :3] * b[..., :3]).sum(-1, keepdim=True)
    return torch.cat((xyz, w), dim=-1)


def quat_rotate(q, v):
    xyz, v = torch.broadcast_tensors(q[..., :3], v)
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + q[..., 3:4] * t + torch.cross(xyz, t, dim=-1)


def quat_rotate_inverse(q, v):
    return quat_rotate(quat_conjugate(q), v)


def quaternion_log(q):
    """Shortest SO(3) rotation vector, invariant to the quaternion sign."""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    q = torch.where(q[..., 3:4] < 0.0, -q, q)
    length = q[..., :3].norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(length, q[..., 3:4].clamp_min(0.0))
    factor = torch.where(length > 1.0e-7, angle / length.clamp_min(1.0e-12),
                         torch.full_like(length, 2.0))
    return q[..., :3] * factor


class CarryCalibration:
    """Named targets and widths loaded once onto the simulation device."""

    def __init__(self, data, device="cpu", dtype=torch.float32):
        if data.get("schema_version") != 1:
            raise ValueError("Unsupported carry-preservation calibration schema")
        self.torso_link = data["torso_link"]
        self.hand_links = tuple(data["hand_links"])
        self.arm_joint_names = tuple(data["arm_joint_names"])
        if self.hand_links != ("left_palm_link", "right_palm_link"):
            raise ValueError("Carry calibration must preserve left/right palm order")
        suffixes = ("shoulder_pitch_joint", "shoulder_roll_joint", "shoulder_yaw_joint",
                    "elbow_joint", "wrist_roll_joint", "wrist_pitch_joint", "wrist_yaw_joint")
        expected = tuple(side + "_" + suffix for side in ("left", "right") for suffix in suffixes)
        if self.arm_joint_names != expected:
            raise ValueError("Carry calibration requires the 14 named arm joints, without waist/legs")
        self.policy_dt = float(data["policy_dt"])
        if not math.isfinite(self.policy_dt) or self.policy_dt <= 0.0:
            raise ValueError("Invalid calibration policy_dt")
        shapes = {
            "hand_directions": (2, 3), "hand_direction_sigma": (2,),
            "hand_normal_sigma": (2,), "arm_target": (14,), "arm_sigma": (14,),
            "position_target": (3,), "position_sigma": (3,),
            "orientation_target_xyzw": (4,), "orientation_sigma": (3,),
            "motion_sigma": (3,),
        }
        for name, shape in shapes.items():
            value = torch.as_tensor(data[name], device=device, dtype=dtype)
            if tuple(value.shape) != shape or not bool(torch.isfinite(value).all()):
                raise ValueError("Invalid carry calibration tensor: " + name)
            if "sigma" in name and not bool((value > 0.0).all()):
                raise ValueError("Carry calibration widths must be positive: " + name)
            setattr(self, name, value)
        for name in ("hand_directions", "orientation_target_xyzw"):
            value = getattr(self, name)
            if not bool(torch.allclose(value.norm(dim=-1), torch.ones_like(value.norm(dim=-1)), atol=1e-5)):
                raise ValueError("Carry calibration directions/quaternions must be unit length")
        self.side_sign = torch.tensor([1.0, -1.0], device=device, dtype=dtype)

    @classmethod
    def load(cls, path, device="cpu", dtype=torch.float32):
        with open(path, encoding="utf-8") as stream:
            return cls(json.load(stream), device=device, dtype=dtype)


def compute_preservation(torso_pos, torso_quat, box_pos, box_quat, hand_pos,
                         box_size, arm_pos, previous_relative_pos, history_valid,
                         dt, calibration):
    """Return rewards, nine raw metrics, and current torso-relative position.

    No history mutation occurs here. Invalid history yields zero motion reward
    and zero placeholder motion error; callers must use history_valid when
    accumulating metrics. This avoids rewarding a fabricated zero derivative.
    """
    c = calibration
    relative_pos = quat_rotate_inverse(torso_quat, box_pos - torso_pos)
    relative_quat = quat_multiply(quat_conjugate(torso_quat), box_quat)
    rotation_error = quaternion_log(quat_multiply(
        quat_conjugate(c.orientation_target_xyzw), relative_quat))

    hand_from_box = hand_pos - box_pos[:, None, :]
    hand_box = quat_rotate_inverse(box_quat[:, None, :], hand_from_box)
    hand_torso = quat_rotate_inverse(torso_quat[:, None, :], hand_from_box)
    half = 0.5 * box_size[:, None, :]
    normal_error = c.side_sign * hand_box[..., 1] - half[..., 1]
    overflow = (hand_box[..., [0, 2]].abs() - half[..., [0, 2]]).clamp_min(0.0)

    length = hand_torso.norm(dim=-1)
    direction = hand_torso / length.unsqueeze(-1).clamp_min(1.0e-9)
    desired = c.hand_directions.unsqueeze(0).expand_as(direction)
    cross = torch.cross(direction, desired, dim=-1).norm(dim=-1)
    dot = (direction * desired).sum(dim=-1)
    direction_error = torch.atan2(cross, dot)
    direction_error = torch.where(length > 1.0e-9, direction_error,
                                  torch.full_like(direction_error, math.pi))
    hand_energy = (normal_error / c.hand_normal_sigma).square()
    hand_energy += (overflow / c.hand_normal_sigma[None, :, None]).square().sum(-1)
    hand_energy += (direction_error / c.hand_direction_sigma).square()

    arm_error = arm_pos - c.arm_target
    position_error = relative_pos - c.position_target
    velocity = (relative_pos - previous_relative_pos) / dt
    velocity = torch.where(history_valid[:, None], velocity, torch.zeros_like(velocity))
    rewards = {
        "carry_hand_box_surface": torch.exp(-0.25 * hand_energy.sum(-1)),
        "carry_arm_pose": torch.exp(-0.5 * (arm_error / c.arm_sigma).square().mean(-1)),
        "carry_relative_position": torch.exp(-0.5 * (position_error / c.position_sigma).square().sum(-1)),
        "carry_relative_orientation": torch.exp(-0.5 * (rotation_error / c.orientation_sigma).square().sum(-1)),
        "carry_relative_velocity": torch.exp(-0.5 * (velocity / c.motion_sigma).square().sum(-1)) * history_valid,
    }
    metrics = torch.stack((
        normal_error[:, 0].abs(), normal_error[:, 1].abs(),
        direction_error[:, 0], direction_error[:, 1],
        hand_from_box.mean(1).norm(dim=-1), position_error.norm(dim=-1),
        rotation_error.norm(dim=-1), velocity.norm(dim=-1),
        arm_error.square().mean(-1).sqrt(),
    ), dim=-1)
    return rewards, metrics, relative_pos


class CarryMetricAccumulator:
    """Episode metrics counted from actual samples, independent of episode age."""

    def __init__(self, num_envs, device, dtype=torch.float32):
        self.sums = torch.zeros(num_envs, len(METRIC_NAMES), device=device, dtype=dtype)
        self.counts = torch.zeros_like(self.sums)

    def add(self, metrics, motion_valid):
        valid = torch.ones_like(metrics)
        valid[:, MOTION_METRIC_INDEX] = motion_valid.to(metrics.dtype)
        self.sums += metrics * valid
        self.counts += valid

    def pop(self, env_ids):
        sums = self.sums[env_ids].sum(0)
        counts = self.counts[env_ids].sum(0)
        means = torch.where(counts > 0, sums / counts.clamp_min(1),
                            torch.full_like(sums, float("nan")))
        self.sums[env_ids] = 0.0
        self.counts[env_ids] = 0.0
        return {"carry/" + name: means[i] for i, name in enumerate(METRIC_NAMES)}
