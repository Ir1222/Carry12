"""Reproduce the historical CarryWith target-matching audit (offline only).

For the current physical-constraint scale audit, use analyze_constraints instead.

Run from the repository root:
    python -m experiments.carry_preservation.analyze

Requires CPU PyTorch, NumPy, SciPy, PyYAML and tqdm, but not Isaac Gym.
The real MotionLib is loaded with only its pure torch_utils dependency; no
simulator methods or MotionLib calculations are replaced.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
GYM_ROOT = ROOT / "legged_gym"
sys.path.insert(0, str(GYM_ROOT))
from legged_gym.carry_preservation import (  # noqa: E402
    CarryCalibration, METRIC_NAMES, compute_preservation,
)

DEFAULT_CALIBRATION = GYM_ROOT / "resources/config/carry_preservation.json"
DEFAULT_REPORT = ROOT / "analysis/carry_preservation"
MOTION_CONFIG = GYM_ROOT / "resources/config/carrybox_locomotion.yaml"
MAPPING = GYM_ROOT / "resources/config/joint_id.txt"
URDF = GYM_ROOT / "resources/robots/g1/urdf/g1_29dof.urdf"
REFERENCE_FPS = 60.0
POLICY_DT = 0.02
NOMINAL_BOX_SIZE = np.array([0.35, 0.35, 0.30])
FLOORS = {"hand_normal_m": 0.02, "hand_direction_rad": np.deg2rad(5),
          "arm_rad": 0.15, "position_m": 0.03,
          "orientation_rad": np.deg2rad(5), "motion_mps": 0.10}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def motionlib_class():
    utils_name = "legged_gym.utils"
    previous = sys.modules.get(utils_name)
    # utils/__init__.py imports the simulator. Supply only the real standalone
    # dependency to the offline loader, and restore the package afterwards.
    package = types.ModuleType(utils_name)
    package.torch_utils = load_module(
        "_carry_audit_torch_utils", GYM_ROOT / "legged_gym/utils/torch_utils.py")
    sys.modules[utils_name] = package
    try:
        module = load_module("_carry_audit_motionlib", GYM_ROOT / "legged_gym/envs/motionlib/motionlib_carrybox.py")
    finally:
        if previous is None:
            del sys.modules[utils_name]
        else:
            sys.modules[utils_name] = previous
    return module.MotionLib


def vector(element, key="xyz"):
    text = element.get(key, "0 0 0") if element is not None else "0 0 0"
    return np.fromstring(text, sep=" ")


class Kinematics:
    def __init__(self):
        self.mapping = {line.split()[1]: int(line.split()[0]) for line in MAPPING.read_text().splitlines()}
        self.names = list(self.mapping)
        self.joints = list(ET.parse(URDF).getroot().findall("joint"))

    def forward(self, position, rotation, dofs):
        poses = {"pelvis": (position, rotation)}
        pending = self.joints.copy()
        while pending:
            advanced = False
            for joint in pending.copy():
                parent = joint.find("parent").get("link")
                if parent not in poses:
                    continue
                pp, pr = poses[parent]
                origin = joint.find("origin")
                jp = pp + np.einsum("nij,j->ni", pr, vector(origin))
                jr = pr @ Rotation.from_euler("xyz", vector(origin, "rpy")).as_matrix()
                if joint.get("type") != "fixed":
                    if joint.get("type") != "revolute":
                        raise ValueError("Only this robot's fixed/revolute joints are supported")
                    angle = dofs[:, self.mapping[joint.get("name")]]
                    jr = jr @ Rotation.from_rotvec(angle[:, None] * vector(joint.find("axis"))).as_matrix()
                poses[joint.find("child").get("link")] = (jp, jr)
                pending.remove(joint)
                advanced = True
            if not advanced:
                raise ValueError("URDF does not form a connected tree rooted at pelvis")
        return poses


def local(rotation, vectors):
    return np.einsum("nji,nj->ni", rotation, vectors)


def derivative(value, dt=1.0 / REFERENCE_FPS):
    return np.gradient(value, dt, axis=0, edge_order=2)


def angular_velocity(rotation, dt=1.0 / REFERENCE_FPS):
    out = np.empty((len(rotation), 3))
    out[1:-1] = Rotation.from_matrix(rotation[2:] @ rotation[:-2].transpose(0, 2, 1)).as_rotvec() / (2 * dt)
    out[0] = Rotation.from_matrix(rotation[1] @ rotation[0].T).as_rotvec() / dt
    out[-1] = Rotation.from_matrix(rotation[-1] @ rotation[-2].T).as_rotvec() / dt
    return out


def summary(values):
    values = np.asarray(values, dtype=float)
    median = np.median(values, axis=0)
    result = {
        "count": len(values), "mean": values.mean(0), "median": median,
        "std": values.std(0), "mad": np.median(abs(values - median), axis=0),
        "percentiles_5_25_50_75_95": np.percentile(values, [5, 25, 50, 75, 95], axis=0),
        "min": values.min(0), "max": values.max(0),
    }
    if values.ndim == 2:
        result["covariance"] = np.cov(values, rowvar=False)
    flat = values.reshape(len(values), -1)
    result["histogram_12_bins"] = [
        {"counts": np.histogram(flat[:, i], bins=12)[0],
         "edges": np.histogram(flat[:, i], bins=12)[1]}
        for i in range(flat.shape[1])
    ]
    return result


def correlation(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.ndim == 1:
        a = a[:, None]
    if b.ndim == 1:
        b = b[:, None]
    a, b = a - a.mean(0), b - b.mean(0)
    denom = np.sqrt((a * a).sum(0)[:, None] * (b * b).sum(0)[None, :])
    return np.divide(a.T @ b, denom, out=np.zeros_like(denom), where=denom > 1e-12)


def source_hash(path):
    content = path.read_bytes()
    if path.suffix != ".pt":
        content = content.replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def sample_motion(lib, kinematics, motion_id, times):
    times = torch.as_tensor(times, dtype=torch.float32)
    ids = torch.full((len(times),), motion_id, dtype=torch.long)
    rp, rq, _, _, dofs, _, _ = lib.get_motion_state("carryWith", ids, times)
    bp, bq, _, _ = lib.get_obj_motion_state("carryWith", ids, times)
    rp, rq, bp, bq, dofs = [value.numpy() for value in (rp, rq, bp, bq, dofs)]
    rr, br = Rotation.from_quat(rq).as_matrix(), Rotation.from_quat(bq).as_matrix()
    poses = kinematics.forward(rp, rr, dofs)
    return dict(rp=rp, rq=rq, rr=rr, bp=bp, bq=bq, br=br, dofs=dofs, poses=poses)


def load_dataset():
    kinematics = Kinematics()
    configs = [MOTION_CONFIG, GYM_ROOT / "resources/config/carrybox_no_relocation.yaml"]
    configured_paths = []
    for config in configs:
        entries = yaml.safe_load(config.read_text())["motions"]["carryWith"]
        configured_paths.append([(config.parent / entry["file"]).resolve() for entry in entries])
    available = set((GYM_ROOT / "resources/dataset/dataset_carrybox/carryWith").glob("*.pt"))
    if any(set(paths) != available for paths in configured_paths):
        raise ValueError("Audit must cover every available CarryWith motion in both configurations")
    lib = motionlib_class()(
        str(MOTION_CONFIG), str(MAPPING), kinematics.names,
        REFERENCE_FPS, "cpu", 10, [0.95, 1.05], 0.7,
    )
    data = []
    for i, raw in enumerate(lib.motion_data["carryWith"]):
        n = len(raw["base_height"])
        d = sample_motion(lib, kinematics, i, np.arange(n))
        d.update(name=configured_paths[0][i].name, raw=raw, n=n)
        d["quantities"] = {}
        for frame in ("pelvis", "torso_link"):
            p, r = d["poses"][frame]
            rel = local(r, d["bp"] - p)
            relrot = r.transpose(0, 2, 1) @ d["br"]
            vdiff = derivative(d["bp"]) - derivative(p)
            vcomp = local(r, vdiff - np.cross(angular_velocity(r), d["bp"] - p))
            quantities = {
                "position": rel, "rotation_vector": Rotation.from_matrix(relrot).as_rotvec(),
                "world_velocity_difference": vdiff, "compensated_velocity": vcomp,
                "local_position_derivative": derivative(rel),
                "relative_angular_velocity": local(r, angular_velocity(d["br"]) - angular_velocity(r)),
                "relative_rotation_derivative": angular_velocity(relrot),
            }
            mean_rotation = Rotation.from_matrix(relrot).mean()
            quantities["rotation_angle_about_clip_mean"] = (
                mean_rotation.inv() * Rotation.from_matrix(relrot)
            ).magnitude()
            for key, value in list(quantities.items()):
                if "velocity" in key or "derivative" in key:
                    quantities[key + "_norm"] = np.linalg.norm(value, axis=-1)
            d[frame] = dict(position=rel, rotation=relrot)
            d["quantities"].update({frame + "/" + key: value for key, value in quantities.items()})
        torso_rot = d["poses"]["torso_link"][1]
        d["hand_box"], d["hand_rays"] = [], []
        for side, raw_index in (("left", 0), ("right", 1)):
            palm = d["poses"][side + "_palm_link"][0]
            offset = palm - d["bp"]
            hand_box = local(d["br"], offset)
            ray = local(torso_rot, offset)
            ray /= np.linalg.norm(ray, axis=1, keepdims=True)
            d["hand_box"].append(hand_box)
            d["hand_rays"].append(ray)
            q = d["quantities"]
            q[side + "/world_position"] = palm
            q[side + "/box_position"] = hand_box
            q[side + "/nominal_box_normalized_position"] = hand_box / (NOMINAL_BOX_SIZE / 2)
            q[side + "/rubber_hand_box_position"] = local(d["br"], d["poses"][side + "_rubber_hand"][0] - d["bp"])
            q[side + "/raw_palm_box_position"] = local(d["br"], raw["link_position"][:, raw_index].numpy() - raw["box_pos_local"].numpy())
            q[side + "/torso_grasp_direction"] = ray
            q[side + "/fk_minus_stored_palm"] = palm - d["rp"] - raw["link_position"][:, raw_index].numpy()
        d["hand_box"] = np.stack(d["hand_box"], axis=1)
        d["hand_rays"] = np.stack(d["hand_rays"], axis=1)
        d["quantities"]["fk_hand_midpoint_error"] = d["hand_box"].mean(1)
        d["quantities"]["stored_hand_midpoint_error"] = (
            (raw["link_position"][:, 0] + raw["link_position"][:, 1]) / 2 - raw["box_pos_local"]
        ).numpy()
        d["quantities"]["arm_positions"] = d["dofs"][:, 15:]
        d["quantities"]["waist_positions_diagnostic_only"] = d["dofs"][:, 12:15]
        sampled = sample_motion(lib, kinematics, i, np.arange(0, n - 1 + 1e-4, REFERENCE_FPS * POLICY_DT))
        p, r = sampled["poses"]["torso_link"]
        d["policy_velocity"] = np.diff(local(r, sampled["bp"] - p), axis=0) / POLICY_DT
        d["policy_sampled"] = sampled
        d["quantities"]["torso_causal_velocity_50hz"] = d["policy_velocity"]
        data.append(d)
    sources = configs + configured_paths[0] + [MAPPING, URDF,
        GYM_ROOT / "legged_gym/envs/motionlib/motionlib_carrybox.py",
        GYM_ROOT / "legged_gym/envs/g1/carrybox_config.py"]
    provenance = {str(path.relative_to(ROOT)).replace("\\", "/"): source_hash(path) for path in sources}
    return data, kinematics, provenance


def center(samples):
    return np.mean([np.median(sample, axis=0) for sample in samples], axis=0)


def widths(samples, target, floor):
    bound = np.max([np.percentile(abs(sample - target), 95, axis=0) for sample in samples], axis=0)
    return np.maximum(1.5 * bound, floor)


def calibrate(data, kinematics, provenance):
    arms = [d["dofs"][:, 15:] for d in data]
    positions = [d["torso_link"]["position"] for d in data]
    arm_target, position_target = center(arms), center(positions)
    quats = np.concatenate([Rotation.from_matrix(d["torso_link"]["rotation"]).as_quat() for d in data])
    weights = np.concatenate([np.full(d["n"], 1.0 / (len(data) * d["n"])) for d in data])
    orientation_target = Rotation.from_quat(quats).mean(weights=weights)
    orientation_errors = [(orientation_target.inv() * Rotation.from_matrix(d["torso_link"]["rotation"])).as_rotvec() for d in data]
    direction_target = np.mean([d["hand_rays"].mean(0) for d in data], axis=0)
    direction_target /= np.linalg.norm(direction_target, axis=-1, keepdims=True)
    direction_errors = [np.arccos(np.clip((d["hand_rays"] * direction_target).sum(-1), -1, 1)) for d in data]
    normal_residuals = [d["hand_box"][..., 1] - np.median(d["hand_box"][..., 1], axis=0) for d in data]
    return {
        "schema_version": 1, "source_sha256": provenance,
        "source_hash_convention": "SHA256; CRLF normalized to LF for text, raw bytes for .pt",
        "motion_frames": {d["name"]: d["n"] for d in data},
        "reference_fps": REFERENCE_FPS, "policy_dt": POLICY_DT,
        "torso_link": "torso_link", "hand_links": ["left_palm_link", "right_palm_link"],
        "arm_joint_names": kinematics.names[15:],
        "hand_directions": direction_target,
        "hand_direction_sigma": widths(direction_errors, 0, FLOORS["hand_direction_rad"]),
        "hand_normal_sigma": widths(normal_residuals, 0, FLOORS["hand_normal_m"]),
        "arm_target": arm_target, "arm_sigma": widths(arms, arm_target, FLOORS["arm_rad"]),
        "position_target": position_target, "position_sigma": widths(positions, position_target, FLOORS["position_m"]),
        "orientation_target_xyzw": orientation_target.as_quat(),
        "orientation_sigma": widths(orientation_errors, 0, FLOORS["orientation_rad"]),
        "motion_sigma": widths([d["policy_velocity"] for d in data], 0, FLOORS["motion_mps"]),
        "calibration_rule": "Equal-motion centers; widths=max(1.5*max_motion_P95(abs residual), floor)",
        "floors": FLOORS,
        "limitations": [
            "No measured box dimensions; nominal box-normalized statistics are a scenario, not metadata.",
            "Stored box center equals stored palm midpoint; box orientation is synthesized from pelvis heading.",
            "Reference hands must be surface-retargeted to test physical nominal-box grasp geometry.",
            "Two clips have constant arms. Width floors deliberately allow adaptation to randomized boxes.",
        ],
    }


def tensor(value):
    return torch.as_tensor(np.asarray(value), dtype=torch.float64)


def evaluate_sample(sample, calibration, *, surface_retarget=False, box_size=NOMINAL_BOX_SIZE, dt=1 / REFERENCE_FPS):
    tp, tr = sample["poses"]["torso_link"]
    palms = np.stack([sample["poses"][s + "_palm_link"][0] for s in ("left", "right")], axis=1)
    sizes = np.broadcast_to(box_size, (len(tp), 3)).copy()
    if surface_retarget:
        # Geometric ray intersection only: this is NOT an IK solution, a new
        # reset, or a claim that the resulting posture is dynamically valid.
        offset = palms - sample["bp"][:, None, :]
        box_offset = np.einsum("nji,nhj->nhi", sample["br"], offset)
        expected_sign = np.array([1, -1])
        if not np.all(expected_sign * box_offset[..., 1] > 0):
            raise ValueError("Reference lost designated left/right side semantics")
        scale = sizes[:, None, 1] / (2 * abs(box_offset[..., 1]))
        palms = sample["bp"][:, None, :] + offset * scale[..., None]
    relative = local(tr, sample["bp"] - tp)
    previous = np.concatenate((relative[:1], relative[:-1]))
    valid = torch.ones(len(tp), dtype=torch.bool)
    valid[0] = False
    rewards, metrics, _ = compute_preservation(
        tensor(tp), tensor(Rotation.from_matrix(tr).as_quat()),
        tensor(sample["bp"]), tensor(sample["bq"]), tensor(palms), tensor(sizes),
        tensor(sample["dofs"][:, 15:]), tensor(previous), valid, dt, calibration,
    )
    return rewards, metrics, valid


def build_statistics(data, kinematics, calibration):
    stats = {"frame_notes": {
        "root_base": "Same URDF frame as pelvis; not an independent candidate.",
        "box_orientation": "MotionLib heading-only reconstruction, not measured orientation.",
        "box_size_correlation": "Not identifiable: no size metadata or observed size diversity.",
        "nominal_box_size_for_scenarios": NOMINAL_BOX_SIZE,
        "arm_joint_order": kinematics.names[15:],
        "derivatives": "Within each clip only; SO(3) angular derivatives avoid stored Euler wrap spikes.",
    }, "pooled": {}, "motions": {}}
    for key in data[0]["quantities"]:
        stats["pooled"][key] = summary(np.concatenate([d["quantities"][key] for d in data]))
    c = CarryCalibration(calibration, dtype=torch.float64)
    validation = {}
    for d in data:
        motion = {"quantities": {key: summary(value) for key, value in d["quantities"].items()}}
        speed = local(d["rr"], derivative(d["rp"]))
        yaw_rate = angular_velocity(d["rr"])[:, 2]
        features = np.column_stack((np.linspace(0, 1, d["n"]), speed[:, :2], yaw_rate))
        motion["condition_columns"] = ["phase", "body_vx", "body_vy", "world_yaw_rate"]
        motion["conditional_correlations"] = {
            key: correlation(value, features) for key, value in d["quantities"].items()
            if len(value) == d["n"] and value.ndim == 2
        }
        motion["hand_x_vs_relative_yaw"] = correlation(
            d["hand_box"][..., 0], d["quantities"]["torso_link/rotation_vector"][:, 2])
        motion["arm_fraction_at_median"] = np.isclose(
            d["dofs"][:, 15:], np.median(d["dofs"][:, 15:], axis=0), atol=1e-6).mean(0)
        motion["phase_thirds"] = {
            key: [chunk.mean(0) for chunk in np.array_split(d["quantities"][key], 3)]
            for key in ("torso_link/position", "torso_link/rotation_vector", "arm_positions")
        }
        motion["base_velocity"] = summary(speed)
        motion["yaw_rate"] = summary(yaw_rate)
        motion["stored_angular_velocity_warning"] = summary(d["raw"]["base_angular_velocity"].numpy())
        stats["motions"][d["name"]] = motion
        raw_rewards, raw_metrics, _ = evaluate_sample(d, c)
        target_rewards, _, _ = evaluate_sample(d, c, surface_retarget=True)
        policy_rewards, _, valid = evaluate_sample(d["policy_sampled"], c, dt=POLICY_DT)
        validation[d["name"]] = {
            "raw_rewards": {key: summary(value.numpy()) for key, value in raw_rewards.items() if key != "carry_relative_velocity"},
            "surface_retargeted_grasp": summary(target_rewards["carry_hand_box_surface"].numpy()),
            "motion_reward_50hz": summary(policy_rewards["carry_relative_velocity"][valid].numpy()),
            "raw_metrics": {name: summary(raw_metrics[:, i].numpy()) for i, name in enumerate(METRIC_NAMES) if "motion" not in name},
        }
    stats["reward_validation"] = validation
    return stats


def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_summary(path, data, calibration, stats):
    rows = []
    for d in data:
        v = stats["reward_validation"][d["name"]]
        values = [v["raw_rewards"][key]["mean"] for key in (
            "carry_hand_box_surface", "carry_arm_pose", "carry_relative_position", "carry_relative_orientation")]
        values += [v["motion_reward_50hz"]["mean"], v["surface_retargeted_grasp"]["mean"]]
        rows.append("| " + d["name"] + " | " + " | ".join("%.4f" % x for x in values) + " |")
    text = """# Historical CarryWith target-matching audit

This is an offline historical audit. See REPORT.md and constraint_statistics.json for current physical constraints.

Reproduce with `python -m experiments.carry_preservation.analyze` from the repository root.
Full distributions, histograms, covariances and conditional correlations are in `statistics.json`.
Source hashes and historical calibration parameters are in `legged_gym/resources/config/carry_preservation.json`.

All three clips are used (773 frames at 60 Hz); differentiation never crosses clip boundaries.
Root/base and pelvis are the same frame. Torso transforms are reconstructed using the current URDF.
Stored palm positions agree with FK to within a few millimetres; that discrepancy is retained in the statistics.

The files do not contain reference box dimensions or measured box rotations. Box center is exactly
the stored hand midpoint; MotionLib reconstructs box heading from the pelvis. Nominal-size normalized
coordinates describe a 0.35 x 0.35 x 0.30 m scenario, not measured reference-object metadata.

| Motion | Raw grasp | Arm | Position | Orientation | Motion (50 Hz) | Surface-retargeted grasp |
|---|---:|---:|---:|---:|---:|---:|
""" + "\n".join(rows) + "\n\n"
    text += "Raw grasp scores are low because the raw palms are inside the nominal box. Surface retargeting\n"
    text += "intersects each measured hand ray with its designated side face. It preserves direction but is\n"
    text += "only a geometry test, not a feasible-pose, contact, IK, or simulator validation. No reset is retargeted.\n\n"
    text += "Torso-relative position standard deviation (m): " + str(np.round(stats["pooled"]["torso_link/position"]["std"], 6).tolist()) + ".\n\n"
    text += "Pelvis-relative position standard deviation (m): " + str(np.round(stats["pooled"]["pelvis/position"]["std"], 6).tolist()) + ".\n\n"
    text += "Calibration widths use the largest per-motion P95 residual, multiplied by 1.5 and floored.\n"
    text += "Centers give equal influence to each motion. A zero pooled MAD is never used as a reward width.\n"
    text += "The full technical interpretation, tests and training instructions are in REPORT.md.\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    data, kinematics, provenance = load_dataset()
    calibration = calibrate(data, kinematics, provenance)
    statistics = build_statistics(data, kinematics, calibration)
    write_json(args.calibration, calibration)
    write_json(args.report_dir / "statistics.json", statistics)
    write_summary(args.report_dir / "reference_summary.md", data, calibration, statistics)
    print("Calibrated", sum(d["n"] for d in data), "frames from", len(data), "CarryWith clips")
    print("Calibration:", args.calibration)
    print("Reference summary:", args.report_dir / "reference_summary.md")


if __name__ == "__main__":
    main()
