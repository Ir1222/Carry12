"""Optional evaluation-only measurement. No simulator writes, RNG or policy calls.

Contact means net body force magnitude > 1 N (not a ground-pair sensor).
Each source is filtered by current OR previous raw contact; combined contact is
the OR of pitch/roll sources. History is independent of the reward buffers.
"""
import math
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import numpy as np

SIDES = ("left", "right")
SOURCES = ("pitch", "roll")
WINDOWS = ("teacher_warmup", "student_transition", "student_0_1s",
           "student_1_3s", "student_3_15s", "student_15s_onward")
SETTINGS = dict(contact_threshold_N=1.0, contact_filter="current OR previous raw; reset on episode/owner change",
                combined_contact="pitch OR roll", min_valid_swing_s=0.04,
                near_ground_m=0.02, scuff_horizontal_speed_m_s=0.10,
                clearance="minimum rotated collision-mesh vertex Z above plane Z=0",
                slip="integral of sole-centroid horizontal speed during combined filtered contact; rolling is included",
                time_windows="pre-step control_phase; carry windows use time since first student control, including transition")


def cpu(value):
    return value.detach().cpu().numpy().copy()


def rotate(q, v):
    """XYZW quaternion rotation with NumPy broadcasting; never edits inputs."""
    q = np.asarray(q)
    if not np.isfinite(q).all() or np.any(np.linalg.norm(q, axis=-1) < 1e-8):
        raise ValueError("Invalid body quaternion")
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    t = 2 * np.cross(q[..., :3], v)
    return v + q[..., 3:4] * t + np.cross(q[..., :3], t)


def mesh_vertices(path):
    data = Path(path).read_bytes()
    count = struct.unpack_from("<I", data, 80)[0] if len(data) >= 84 else 0
    if len(data) == 84 + 50 * count:
        triangles = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])
        vertices = np.frombuffer(data, dtype=triangles, count=count, offset=84)["vertices"].reshape(-1, 3)
    else:
        vertices = np.array([[float(x) for x in line.split()[1:]]
                             for line in data.decode("ascii").splitlines()
                             if line.strip().startswith("vertex ")])
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("Invalid STL: " + str(path))
    return np.unique(vertices, axis=0).astype(float)


def foot_geometry(urdf):
    """Use active collision meshes, including URDF scale and collision transform.

    Commented-out spheres are deliberately excluded. Min support over vertices
    equals min support over their convex hull, but PhysX inflation/contact offsets
    and collision cooking are not represented. This is a geometric approximation.
    """
    urdf = Path(urdf)
    tree = ET.parse(str(urdf))
    result = {}
    for side in SIDES:
        for source in SOURCES:
            name = f"{side}_ankle_{source}_link"
            link = tree.find(f"link[@name='{name}']")
            points = []
            files = []
            for collision in link.findall("collision"):
                mesh = collision.find("geometry/mesh")
                if mesh is None:
                    raise ValueError("Expected mesh collision for " + name)
                path = (urdf.parent / mesh.get("filename")).resolve()
                vertices = mesh_vertices(path) * np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
                origin = collision.find("origin")
                xyz = np.fromstring(origin.get("xyz", "0 0 0") if origin is not None else "0 0 0", sep=" ")
                r, p, y = np.fromstring(origin.get("rpy", "0 0 0") if origin is not None else "0 0 0", sep=" ")
                cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
                matrix = np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                                   [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]])
                points.append(vertices @ matrix.T + xyz)
                files.append(str(path))
            vertices = np.concatenate(points)
            sole = vertices[vertices[:, 2] <= vertices[:, 2].min() + .001].mean(0)
            result[name] = dict(vertices=vertices, sole=sole, meshes=files,
                                bounds=[vertices.min(0).tolist(), vertices.max(0).tolist()])
    return result


def window(row):
    phase = int(row["control_phase"])
    if phase == 0:
        return WINDOWS[0]
    if phase == 1:
        return WINDOWS[1]
    if phase != 2:
        raise ValueError("Invalid control phase")
    age = float(row["handoff_relative_s"])
    if age < 0:
        raise ValueError("Student sample without handoff age")
    return WINDOWS[2 + (age >= 1) + (age >= 3) + (age >= 15)]


class GaitDiagnostics:
    """Capture at compute_reward, before FullCarryBox auto-reset mutates state."""
    def __init__(self, env, urdf, gymapi):
        if env.cfg.terrain.mesh_type != "plane":
            raise ValueError("Sole clearance currently requires the existing plane terrain")
        self.geometry = foot_geometry(urdf)
        self.indices = {}
        self.legacy_indices = cpu(env.feet_indices).tolist()
        if len(self.legacy_indices) != 2:
            raise ValueError("Expected two legacy feet for source comparison")
        # Explicit DOMAIN_ENV is essential: two platforms + box precede robot.
        names = env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0])
        for name in self.geometry:
            if name not in names:
                raise ValueError("Missing diagnostic body: " + name)
            local = names.index(name)
            indices = [env.gym.get_actor_rigid_body_index(e, a, local, gymapi.DOMAIN_ENV)
                       for e, a in zip(env.envs, env.actor_handles)]
            if len(set(indices)) != 1 or not 0 <= indices[0] < env.contact_forces.shape[1]:
                raise ValueError("Invalid environment-domain rigid body index: " + name)
            self.indices[name] = indices[0]
        if len(set(self.indices.values())) != 4:
            raise ValueError("Diagnostic foot bodies must be distinct")
        self.hand_indices = [env.gym.get_actor_rigid_body_index(env.envs[0], env.actor_handles[0], names.index(name),
                             gymapi.DOMAIN_ENV) for name in names if env.cfg.asset.hand_colli_name in name]
        if len(self.hand_indices) != 2:
            raise ValueError("Expected two physical hand contact bodies")
        self.previous = {}

    def metadata(self):
        geometry = {}
        for name, g in self.geometry.items():
            geometry[name] = {k: v for k, v in g.items() if k not in ("vertices", "sole")}
            geometry[name].update(sole_centroid_local=g["sole"].tolist(), vertex_count=len(g["vertices"]))
        return dict(settings=SETTINGS, body_indices_domain_env=self.indices,
                    legacy_feet_indices=self.legacy_indices, hand_indices_domain_env=self.hand_indices,
                    geometry=geometry,
                    limitations=["Net forces include all collision partners; ground pair is not identified.",
                                 "Mesh clearance excludes PhysX contact offset, shape inflation and cooking differences.",
                                 "Sole-centroid slip is a proxy: foot rolling can contribute.",
                                 "50 Hz policy-step sampling can miss brief contacts between samples."])

    def capture_action(self, env, desired, applied):
        self.action = dict(raw=cpu(env.raw_student_action), desired=cpu(desired),
                           previous=cpu(env.actions), applied=cpu(applied), rates=cpu(env.rates))
        if any(not np.isfinite(v).all() for v in self.action.values()):
            raise ValueError("Nonfinite action diagnostic")

    def capture_state(self, env):
        self.state = dict(bodies=cpu(env.rigid_body_states), forces=cpu(env.contact_forces),
                          legacy_contact=cpu(env.contact_filt), legacy_first=cpu(env.first_contacts),
                          legacy_air=cpu(env.feet_air_time))
        if any(not np.isfinite(v).all() for v in self.state.values()):
            raise ValueError("Nonfinite body/contact diagnostic")
        # Preserve the predicates actually used by the existing termination
        # code, including its current body indices, without changing decisions.
        root = cpu(env.root_states)
        bodies, forces = self.state["bodies"], self.state["forces"]
        controlled = cpu(env.control_mask).astype(bool)
        r = env.cfg.rewards
        metrics = env.last_step_snapshot
        bottom = cpu(env.box_states)[:, 2] - cpu(env._box_size)[:, 2] / 2 - cpu(env.env_origins)[:, 2]
        self.causes = dict(
            contact=(np.linalg.norm(forces[:, cpu(env.termination_contact_indices).astype(int)], axis=-1) > 10).any(-1),
            head_low=bodies[:, env.head_index, 2] < .6,
            base_low=root[:, 2] < .2,
            roll_pitch=(np.abs(cpu(env.roll)) > .5) | (np.abs(cpu(env.pitch)) > 1.1),
            body2_speed=np.linalg.norm(bodies[:, 2, 7:9], axis=-1) > 3.,
            hip_low=(bodies[:, cpu(env.hip_yaw_indices).astype(int), 2] < .15).any(-1),
            box_orientation=(cpu(env.projected_gravity_box)[:, 2] > -.05) & env.box_cfg.box_termination,
            box_drop=controlled & (bottom < r.box_drop_height),
            box_tilt=controlled & (cpu(metrics["tilt"]) > r.box_tilt_termination_deg),
            box_distance=controlled & (cpu(env.robot2object_dist) > r.robot_box_max_distance),
            grasp_loss=controlled & (cpu(env.grasp_loss_time) > r.grasp_loss_grace_s))

    @staticmethod
    def fields(names):
        fields = ["gait_window", "support_state", "leg_limiter_any", "leg_limiter_fraction",
                  "physical_bilateral"]
        for side in SIDES:
            fields += [f"{side}_{key}" for key in ("contact", "touchdown", "liftoff", "swing_age_s", "stance_age_s",
                       "valid_swing", "clearance_m", "vx", "vy", "vz", "horizontal_speed", "sole_horizontal_speed",
                       "contact_speed", "slip_step_m", "near_ground_motion", "scuff_onset",
                       "legacy_contact", "legacy_first_contact", "legacy_air_time_s")]
            for source in SOURCES:
                fields += [f"{side}_{source}_{key}" for key in ("fx", "fy", "fz", "force_norm", "contact_raw", "contact_filtered")]
        fields += [f"{kind}:{name}" for kind in ("action_clipped", "action_rate_limited", "previous_applied_action") for name in names]
        fields += ["termination:" + k for k in ("contact", "head_low", "base_low", "roll_pitch", "body2_speed",
                   "hip_low", "box_orientation", "box_drop", "box_tilt", "box_distance", "grasp_loss")]
        return fields

    def row(self, row, e, env):
        owner = bool(int(row["student_valid"]))
        identity = (int(row["episode"]), owner)
        old = self.previous.get(e)
        if old is None or old["identity"] != identity:
            old = dict(identity=identity, raw={}, contact={}, age={}, scuff={})
        result = dict(gait_window=window(row))
        contacts = []
        state = self.state
        for si, side in enumerate(SIDES):
            flags = []
            for source in SOURCES:
                name = f"{side}_ankle_{source}_link"
                force = state["forces"][e, self.indices[name]]
                norm = float(np.linalg.norm(force))
                raw = norm > SETTINGS["contact_threshold_N"]
                filtered = raw or old["raw"].get(name, False)
                flags.append(filtered)
                result.update({f"{side}_{source}_{k}": v for k, v in zip(
                    ("fx", "fy", "fz", "force_norm", "contact_raw", "contact_filtered"),
                    (*force.tolist(), norm, int(raw), int(filtered)))})
                old["raw"][name] = raw
            contact = any(flags)
            contacts.append(contact)
            prior = old["contact"].get(side)
            changed = prior is not None and contact != prior
            age = env.dt if prior is None or changed else old["age"][side] + env.dt
            name = f"{side}_ankle_roll_link"
            body = state["bodies"][e, self.indices[name]]
            g = self.geometry[name]
            rotated = rotate(body[3:7], g["vertices"])
            clearance = float(np.min(body[2] + rotated[:, 2]))
            sole_offset = rotate(body[3:7], g["sole"])
            velocity = body[7:10]
            sole_velocity = velocity + np.cross(body[10:13], sole_offset)
            horizontal = float(np.linalg.norm(velocity[:2]))
            sole_speed = float(np.linalg.norm(sole_velocity[:2]))
            near = clearance < SETTINGS["near_ground_m"] and sole_speed > SETTINGS["scuff_horizontal_speed_m_s"]
            vals = dict(contact=int(contact), touchdown=int(changed and contact), liftoff=int(changed and not contact),
                        swing_age_s=age if not contact else 0., stance_age_s=age if contact else 0.,
                        valid_swing=int(not contact and age + 1e-8 >= SETTINGS["min_valid_swing_s"]),
                        clearance_m=clearance, vx=float(velocity[0]), vy=float(velocity[1]), vz=float(velocity[2]),
                        horizontal_speed=horizontal, sole_horizontal_speed=sole_speed,
                        contact_speed=sole_speed if contact else "", slip_step_m=sole_speed * env.dt if contact else 0.,
                        near_ground_motion=int(near), scuff_onset=int(near and prior is not None and not old["scuff"].get(side, False)),
                        legacy_contact=int(state["legacy_contact"][e, si]),
                        legacy_first_contact=int(state["legacy_first"][e, si]),
                        legacy_air_time_s=float(state["legacy_air"][e, si]))
            result.update({f"{side}_{k}": v for k, v in vals.items()})
            old["contact"][side], old["age"][side], old["scuff"][side] = contact, age, near
        self.previous[e] = old
        result["support_state"] = ("double_support" if all(contacts) else "left_only" if contacts[0]
                                   else "right_only" if contacts[1] else "no_contact")
        hand_force = state["forces"][e, self.hand_indices]
        result["physical_bilateral"] = int((np.linalg.norm(hand_force, axis=-1) > env.cfg.rewards.hand_contact_threshold).all())
        action = self.action
        clip = env.cfg.normalization.clip_actions
        desired = action["desired"][e]
        clipped = np.clip(desired, -clip, clip)
        limited = np.abs(action["applied"][e] - clipped) > 1e-6
        # Report raw student clipping only on student-owned samples.
        clipped_flags = owner & (np.abs(action["raw"][e] - np.clip(action["raw"][e], -clip, clip)) > 1e-6)
        limited = limited & owner
        leg_ids = env.groups["legs"]
        result["leg_limiter_any"] = int(limited[leg_ids].any())
        result["leg_limiter_fraction"] = float(limited[leg_ids].mean())
        for j in leg_ids:
            name = env.dof_names[j]
            result["action_clipped:" + name] = int(clipped_flags[j])
            result["action_rate_limited:" + name] = int(limited[j])
            result["previous_applied_action:" + name] = float(action["previous"][e, j])
        result.update({"termination:" + k: int(v[e]) for k, v in self.causes.items()})
        for key, value in result.items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError("Nonfinite gait diagnostic: " + key)
        return result


def evaluation_handoff_overrides(saved, evaluated):
    """Record deliberate evaluation conditions; never alter the saved contract."""
    keys = ("student_timeout_s", "warmup_vx", "fixed_command", "command_suite")
    return {key: {"checkpoint": saved.get(key), "evaluation": evaluated.get(key)}
            for key in keys if saved.get(key) != evaluated.get(key)}


def validate_a5_contract(checkpoint, spec, teacher_hash, reference_hash, handoff, names, scale, dt, rewards=None):
    """Fail closed; an original locomotion checkpoint is never an A5 substitute."""
    contract = checkpoint.get("handoff_contract")
    if not contract or checkpoint.get("iter", 0) <= 0:
        raise ValueError("Diagnostics require a trained A5 checkpoint with handoff_contract and positive iter")
    if spec["dims"][0] != 738 or spec["dims"][-1] != 29:
        raise ValueError("A5 actor must be 738D/29D")
    if contract.get("teacher_sha256") != teacher_hash or contract.get("reference_sha256") != reference_hash:
        raise ValueError("A5 teacher/reference hash mismatch")
    if not reference_hash:
        raise ValueError("A5 posture reference is required")
    saved = contract["handoff"]
    required = ("ablation", "adapter", "target_rate_arms", "target_rate_waist", "target_rate_legs", "joint_target_rates")
    if any(key not in saved for key in required) or saved["ablation"] != "A5":
        raise ValueError("Incomplete or non-A5 saved action-adapter contract")
    for key, expected in (("ablation", "A5"), ("warmup_vx", .8), ("fixed_command", [.8, 0., 0.]),
                          ("command_suite", "fixed"), ("adapter", "rate_limited")):
        if handoff.get(key) != expected:
            raise ValueError("Diagnostic configuration requires " + key + "=" + str(expected))
    # This experiment deliberately evaluates vx=0.8, even if the checkpoint
    # records another training command. The fixed vx=0.8 guards above still
    # apply. These are evaluation conditions, not action-adapter settings.
    evaluation_keys = ("student_timeout_s", "warmup_vx", "fixed_command", "command_suite")
    for key, value in saved.items():
        if key not in evaluation_keys and handoff.get(key) != value:
            raise ValueError("Diagnostic handoff differs from trained A5: " + key
                             + " (checkpoint=" + repr(value) + ", evaluation=" + repr(handoff.get(key)) + ")")
    for key, expected in (("dof_names", list(names)), ("action_scale", scale), ("dt", dt), ("actor_obs", 738)):
        if contract.get(key) != expected:
            raise ValueError("A5 action/observation contract mismatch: " + key)
    if not contract.get("rewards") or (rewards is not None and contract["rewards"] != rewards):
        raise ValueError("A5 reward/posture contract unavailable or differs from evaluator")
    return contract
