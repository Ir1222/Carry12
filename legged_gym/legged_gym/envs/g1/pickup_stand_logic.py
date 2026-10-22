"""Tensor-only task logic, also usable in CPU tests without Isaac Gym."""
import torch
import math
import xml.etree.ElementTree as ET

APPROACH, PICKUP, HOLD = 0, 1, 2


def advance_phase(phase, contact_time, completion_time, near, bilateral, complete,
                  dt, contact_seconds, completion_seconds):
    contact_time = torch.where(near & bilateral, contact_time + dt, 0.)
    phase = torch.where((phase == APPROACH) & (contact_time >= contact_seconds), PICKUP, phase)
    completion_time = torch.where((phase == PICKUP) & complete, completion_time + dt, 0.)
    entered = (phase == PICKUP) & (completion_time >= completion_seconds)
    return torch.where(entered, HOLD, phase), contact_time, completion_time, entered


def completion_conditions(base_height, base_tilt, box_height, box_tilt,
                          clearance, bilateral, feet_down, relative_speed,
                          base_vertical_speed, box_vertical_speed, cfg):
    # Deliberately no XY stillness test: completion immediately commands zero,
    # and stillness is the *subsequent* hold objective, not a braking phase.
    return ((base_height >= cfg.min_base_height) & (base_tilt <= cfg.max_base_tilt)
            & (box_height >= cfg.min_box_height) & (box_height <= cfg.max_box_height)
            & (box_tilt <= cfg.max_box_tilt) & (clearance >= cfg.min_clearance)
            & bilateral & feet_down & (relative_speed <= cfg.max_relative_speed)
            & (base_vertical_speed.abs() <= cfg.max_vertical_speed)
            & (box_vertical_speed.abs() <= cfg.max_vertical_speed))


def palm_box_contact(palms, box_position, box_quaternion, size, forces,
                     force_threshold, surface_tolerance):
    """Net forces alone cannot identify the contacted body; require geometry too.

    Palms must be close to opposing box Y faces in box coordinates. This rejects
    contacts with the floor/platform and two palms on the same side.
    """
    relative = palms - box_position[:, None]
    q = box_quaternion[:, None, :3].expand_as(relative)
    w = box_quaternion[:, None, 3:4]
    local = relative - 2 * w * torch.cross(q, relative, dim=-1) + 2 * torch.cross(q, torch.cross(q, relative, dim=-1), dim=-1)
    half = size[:, None] / 2
    outside = (local.abs() - half).clamp_min(0).norm(dim=-1)
    side = (local[..., 1].abs() - half[..., 1]).abs() <= surface_tolerance
    each = (forces.norm(dim=-1) > force_threshold) & (outside <= surface_tolerance) & side
    opposing = local[:, 0, 1] * local[:, 1, 1] < 0
    return each.all(-1) & opposing, each


class ResetKinematics:
    """URDF FK for reset observations before Gym refreshes rigid-body states.

    Positions use the same pelvis origin and torso rotation as carrybox's
    end-effector observation. No simulator state or control target is changed.
    """
    def __init__(self, urdf, dof_names):
        self.joints = []
        for joint in ET.parse(urdf).findall('joint'):
            origin = joint.find('origin')
            xyz = [float(x) for x in origin.get('xyz', '0 0 0').split()] if origin is not None else [0., 0., 0.]
            r, p, y = [float(x) for x in origin.get('rpy', '0 0 0').split()] if origin is not None else [0., 0., 0.]
            cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
            rotation = [[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                        [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]]
            axis = joint.find('axis')
            axis = [float(x) for x in axis.get('xyz').split()] if axis is not None else [1., 0., 0.]
            index = dof_names.index(joint.get('name')) if joint.get('type') != 'fixed' else None
            self.joints.append((joint.find('parent').get('link'), joint.find('child').get('link'), xyz, rotation, axis, index))
        self.targets = ['left_palm_link', 'right_palm_link', 'left_ankle_pitch_link', 'right_ankle_pitch_link', 'mid360_link']

    def end_effectors(self, positions, return_torso=False):
        batch = len(positions)
        identity = torch.eye(3, device=positions.device, dtype=positions.dtype).expand(batch, -1, -1)
        transforms = {'pelvis': (torch.zeros(batch, 3, device=positions.device, dtype=positions.dtype), identity)}
        remaining = list(self.joints)
        while remaining:
            progress = []
            for entry in remaining:
                parent, child, xyz, fixed, axis, index = entry
                if parent not in transforms:
                    progress.append(entry)
                    continue
                translation, rotation = transforms[parent]
                offset, fixed = positions.new_tensor(xyz), positions.new_tensor(fixed)
                child_rotation = rotation @ fixed
                if index is not None:
                    u = positions.new_tensor(axis)
                    u = u / u.norm()
                    x, y, z = u
                    skew = positions.new_tensor([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
                    theta = positions[:, index, None, None]
                    joint_rotation = identity + theta.sin() * skew + (1 - theta.cos()) * (skew @ skew)
                    child_rotation = child_rotation @ joint_rotation
                transforms[child] = (translation + (rotation @ offset).squeeze(-1), child_rotation)
            if len(progress) == len(remaining):
                raise ValueError('Unresolved G1 URDF kinematic tree')
            remaining = progress
        torso_inverse = transforms['torso_link'][1].transpose(1, 2)
        ee = torch.cat([(torso_inverse @ transforms[name][0].unsqueeze(-1)).squeeze(-1)
                        for name in self.targets], dim=-1)
        return (ee, torso_inverse) if return_torso else ee
