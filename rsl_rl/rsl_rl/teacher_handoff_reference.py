"""Validated data-derived upper-body intervals; no synthetic teacher statistics."""
import math

from rsl_rl.teacher_handoff import joint_groups


def validate_reference(reference, dof_names):
    groups = joint_groups(dof_names)
    expected = {dof_names[i] for i in groups["arms"] + groups["waist"]}
    names = reference.get("joint_names", [])
    if reference.get("schema") != 1 or len(names) != 17 or set(names) != expected:
        raise ValueError("Reference must contain all 17 named waist/shoulder/elbow/wrist joints")
    if reference.get("sample_count", 0) < 100 or reference.get("successful_episodes", 0) < 3:
        raise ValueError("Reference requires >=100 stable samples from >=3 successful teacher episodes")
    if reference.get("source_kind") != "full_teacher_physical_pickup" or not reference.get("teacher_sha256"):
        raise ValueError("Missing real teacher rollout provenance")
    arrays = [reference.get(k, []) for k in ("lower", "upper", "softness")]
    if any(len(x) != 17 for x in arrays):
        raise ValueError("Posture interval dimensions do not match joints")
    if any(not math.isfinite(float(v)) for x in arrays for v in x):
        raise ValueError("Nonfinite posture reference")
    if any(lo >= hi or soft <= 0 for lo, hi, soft in zip(*arrays)):
        raise ValueError("Unordered intervals or nonpositive softness")
    for key in ("box_size_min", "box_size_max"):
        values = reference.get(key, [])
        if len(values) != 3 or any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError("Invalid box size provenance")
    if any(lo > hi for lo, hi in zip(reference["box_size_min"], reference["box_size_max"])):
        raise ValueError("Unordered box size interval")
