"""Controlled fixed-command visualization for Stage2A force compliance.

This script is intentionally only a diagnostic wrapper.  The registered
``carrybox_force`` environment remains the sole owner of force scheduling,
force application, beta sampling, force profiles, and teacher computation.
"""

import argparse
import math
import os
import sys
from collections import deque

import isaacgym  # noqa: F401 -- Isaac Gym must be imported before torch.
from isaacgym.torch_utils import quat_rotate

import numpy as np
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import (
    G1CarryBoxForce,
    G1CarryBoxForceCfg,
    G1CarryBoxForceCfgPPO,
)
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import update_cfg_from_args


TASK_NAME = "carrybox_force"


def _parse_args():
    """Remove play-only flags before delegating to the shared Isaac Gym parser."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--vx", type=float, default=0.4)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw_rate", type=float, default=0.0)
    parser.add_argument("--no_force", action="store_true")
    parser.add_argument("--diagnostic_interval", type=int, default=10)
    parser.add_argument("--pre_force_ready_s", type=float, default=2.0)
    parser.add_argument("--baseline_window_s", type=float, default=1.0)
    parser.add_argument("--baseline_v_tol", type=float, default=0.15)
    parser.add_argument("--baseline_yaw_tol", type=float, default=0.15)
    parser.add_argument("--diagnostic_viz", action="store_true")
    parser.add_argument("--velocity_draw_scale", type=float, default=1.0)
    parser.add_argument("--force_direction_draw_length", type=float, default=0.75)
    local_args, remaining = parser.parse_known_args(sys.argv[1:])

    original_argv = sys.argv
    sys.argv = [original_argv[0], *remaining]
    try:
        args = get_args()
    finally:
        sys.argv = original_argv

    for name, value in vars(local_args).items():
        setattr(args, name, value)
    return args


def _resolve_checkpoint(path):
    if path is None:
        raise ValueError("play_force.py requires --resume_path <STAGE2_CHECKPOINT>.")
    resolved = os.path.expanduser(
        path.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
    )
    resolved = os.path.abspath(resolved)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"Checkpoint not found: {resolved}")
    return resolved


def _validate_fixed_command(args, env_cfg):
    requested = {
        "vx": (args.vx, "lin_vel_x"),
        "vy": (args.vy, "lin_vel_y"),
        "yaw_rate": (args.yaw_rate, "ang_vel_yaw"),
    }
    for label, (value, range_name) in requested.items():
        if not math.isfinite(value):
            raise ValueError(f"{label} must be finite, got {value!r}.")
        low, high = getattr(env_cfg.commands.ranges, range_name)
        if value < float(low) - 1.0e-9 or value > float(high) + 1.0e-9:
            raise ValueError(
                "Requested fixed command is outside Stage1 training support. "
                f"{label}={value} is outside [{float(low)}, {float(high)}]."
            )


def _validate_force_request(args, env_cfg, random_direction_requested):
    force_cfg = env_cfg.external_force
    stage = int(force_cfg.curriculum_stage)
    if stage not in force_cfg.curriculum_beta_ranges:
        allowed = tuple(sorted(force_cfg.curriculum_beta_ranges))
        raise ValueError(
            f"force curriculum stage must be one of {allowed}, got {stage}."
        )

    stage_ranges = force_cfg.curriculum_beta_ranges[stage]
    directions = tuple(force_cfg.force_directions)
    unknown = [name for name in directions if name not in stage_ranges]
    if unknown:
        raise ValueError(f"Unsupported force direction(s): {unknown}.")

    if args.force_beta is None:
        return
    if not math.isfinite(args.force_beta):
        raise ValueError(f"force_beta must be finite, got {args.force_beta!r}.")

    outside = []
    for direction in directions:
        low, high = stage_ranges[direction]
        if args.force_beta < float(low) - 1.0e-9 or args.force_beta > float(high) + 1.0e-9:
            outside.append((direction, (float(low), float(high))))
    if outside:
        detail = ", ".join(
            f"{direction}=[{value_range[0]}, {value_range[1]}]"
            for direction, value_range in outside
        )
        suffix = (
            " Specify --force_direction together with --force_beta if the beta "
            "is in-distribution for only one direction."
            if random_direction_requested or len(directions) > 1
            else ""
        )
        raise ValueError(
            "Requested beta is outside the selected curriculum-stage training "
            f"range. beta={args.force_beta}; {detail}.{suffix}"
        )


def _validate_diagnostic_args(args):
    finite_nonnegative = (
        ("pre_force_ready_s", args.pre_force_ready_s),
        ("baseline_v_tol", args.baseline_v_tol),
        ("baseline_yaw_tol", args.baseline_yaw_tol),
    )
    for name, value in finite_nonnegative:
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"--{name} must be finite and nonnegative, got {value!r}.")

    finite_positive = (
        ("baseline_window_s", args.baseline_window_s),
        ("velocity_draw_scale", args.velocity_draw_scale),
        ("force_direction_draw_length", args.force_direction_draw_length),
    )
    for name, value in finite_positive:
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name} must be finite and positive, got {value!r}.")

    if args.diagnostic_interval < 1:
        raise ValueError("--diagnostic_interval must be at least 1 policy step.")


def _format_range(value_range):
    return f"[{float(value_range[0]):.3f}, {float(value_range[1]):.3f}]"


def _print_configuration(
    args,
    env_cfg,
    checkpoint,
    random_direction_requested,
    policy_dt,
    training_stable_steps,
    diagnostic_stable_steps,
):
    force_cfg = env_cfg.external_force
    command_ranges = env_cfg.commands.ranges
    direction_label = (
        "training-distribution random"
        if random_direction_requested
        else force_cfg.force_directions[0]
    )
    beta_label = (
        "training-distribution random"
        if args.force_beta is None
        else f"{args.force_beta:.3f}"
    )

    print("\n================ play_force ================\n")
    print(f"checkpoint:\n  {checkpoint}\n")
    print(f"task:\n  {TASK_NAME}\n")
    print(f"seed:\n  {env_cfg.seed}\n")
    print("fixed nominal command:")
    print(f"  vx       = {args.vx:.3f} m/s")
    print(f"  vy       = {args.vy:.3f} m/s")
    print(f"  yaw_rate = {args.yaw_rate:.3f} rad/s\n")
    print("training command support:")
    print(f"  vx       = {_format_range(command_ranges.lin_vel_x)}")
    print(f"  vy       = {_format_range(command_ranges.lin_vel_y)}")
    print(f"  yaw_rate = {_format_range(command_ranges.ang_vel_yaw)}\n")
    print("carry command resampling:\n  OFF\n")
    print("force trigger:")
    print(f"  policy dt = {policy_dt:.4f} s")
    print(f"  training stable-carry steps = {training_stable_steps}")
    print(f"  diagnostic stable-carry requirement = {args.pre_force_ready_s:.3f} s")
    print(f"  diagnostic stable-carry steps = {diagnostic_stable_steps}")
    print(
        "  effective stable-carry steps = "
        f"{int(force_cfg.stable_carry_policy_steps)}\n"
    )
    print("  NOTE: force magnitude, profile, scheduler, and teacher are unchanged.")
    if int(force_cfg.stable_carry_policy_steps) > training_stable_steps:
        print(
            "  This visual diagnostic intentionally extends only the scheduler's\n"
            "  stable-carry waiting duration for a cleaner locomotion baseline.\n"
        )
    else:
        print(
            "  The training minimum dominates, so no additional diagnostic "
            "delay is applied.\n"
        )
    print("force:")
    print(f"  enabled = {bool(force_cfg.enable_external_force)}")
    print(f"  curriculum stage = {int(force_cfg.curriculum_stage)}")
    print(f"  direction = {direction_label}")
    print(f"  beta = {beta_label}\n")
    print("training beta range:")
    stage_ranges = force_cfg.curriculum_beta_ranges[int(force_cfg.curriculum_stage)]
    for direction in force_cfg.force_directions:
        print(f"  {direction} = {_format_range(stage_ranges[direction])}")
    print("\nforce profile:")
    print(f"  ramp_up = {float(force_cfg.force_ramp_up_duration_s):.3f} s")
    print(f"  hold = {_format_range(force_cfg.force_hold_duration_range_s)} s")
    print(f"  ramp_down = {float(force_cfg.force_ramp_down_duration_s):.3f} s\n")
    print("teacher:")
    print(
        "  vx range = "
        f"[{float(force_cfg.teacher_vx_min):.3f}, "
        f"{float(force_cfg.teacher_vx_max):.3f}] m/s"
    )
    print(
        "  max heading offset = "
        f"{float(force_cfg.max_teacher_heading_offset_rad):.3f} rad\n"
    )
    print("policy interface:")
    print(f"  Actor observation = {int(env_cfg.env.num_actor_obs)}")
    print(f"  Critic observation = {int(env_cfg.env.num_privileged_obs)}\n")
    print(f"num_envs:\n  {int(env_cfg.env.num_envs)}\n")
    print("baseline diagnostic:")
    print(f"  window = {args.baseline_window_s:.3f} s")
    print(f"  forward-speed tolerance = {args.baseline_v_tol:.3f} m/s")
    print(f"  world-yaw-rate tolerance = {args.baseline_yaw_tol:.3f} rad/s\n")
    if args.diagnostic_viz:
        state = "disabled because --headless was set" if args.headless else "enabled"
        print("viewer diagnostic vectors:")
        print(f"  state = {state}")
        print("  NOMINAL world velocity = white")
        print("  TEACHER desired world velocity = green")
        print("  ACTUAL upper-body world velocity = blue")
        print("  FORCE normalized world direction = red")
        print(f"  velocity draw scale = {args.velocity_draw_scale:.3f} m per (m/s)")
        print(
            "  force-direction draw length = "
            f"{args.force_direction_draw_length:.3f} m\n"
        )
    print(
        "grasp metric note:\n"
        "  summaries use carry-task hand_contact_filt. The force scheduler uses\n"
        "  its separate force_last_hand_contacts one-step memory; the two filters\n"
        "  are not claimed to be identical.\n"
    )
    print("============================================\n")

    selected_directions = set(force_cfg.force_directions)
    teacher_span = float(force_cfg.teacher_vx_max) - float(force_cfg.teacher_vx_min)
    close_margin = 0.05 * teacher_span
    if (
        force_cfg.enable_external_force
        and "+box_x" in selected_directions
        and float(force_cfg.teacher_vx_max) - args.vx <= close_margin
    ):
        print("[play_force warning]\n")
        print("Nominal vx is close to teacher_vx_max.")
        print("Forward +box_x compliance may be clipped by the teacher velocity limit.")
        print("Recommended diagnostic value: --vx 0.4\n")


def _set_fixed_command(env, args, synchronize_policy_buffer=False):
    env.commands[:, 0] = args.vx
    env.commands[:, 1] = args.vy
    env.commands[:, 2] = args.yaw_rate
    if synchronize_policy_buffer:
        # One initial sync prevents the first action from seeing the reset-time
        # command.  Later derived heading/yaw targets remain environment-owned.
        env.carry_policy_commands[:, :3] = env.commands[:, :3]
        env.compute_observations()


def _mean(values):
    return sum(values) / len(values) if values else float("nan")


def _std(values):
    if not values:
        return float("nan")
    mean = _mean(values)
    return math.sqrt(_mean([(value - mean) ** 2 for value in values]))


def _mean_xy(samples, key):
    return (_mean([sample[key][0] for sample in samples]),
            _mean([sample[key][1] for sample in samples]))


def _circular_mean(values):
    if not values:
        return float("nan")
    return math.atan2(
        _mean([math.sin(value) for value in values]),
        _mean([math.cos(value) for value in values]),
    )


def _wrap_to_pi(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def _heading_axes(heading):
    return (
        (math.cos(heading), math.sin(heading)),
        (-math.sin(heading), math.cos(heading)),
    )


def _dot_xy(vector, axis):
    return vector[0] * axis[0] + vector[1] * axis[1]


class ForceDiagnostic:
    """Read-only env-0 force, tracking, and viewer diagnostics."""

    _VECTOR_COLORS = np.asarray(
        (
            (1.0, 1.0, 1.0),
            (0.1, 0.9, 0.1),
            (0.1, 0.4, 1.0),
            (0.95, 0.1, 0.05),
        ),
        dtype=np.float32,
    )

    def __init__(self, env, args, effective_stable_steps):
        if args.diagnostic_interval < 1:
            raise ValueError("--diagnostic_interval must be at least 1 policy step.")
        if int(env.hand_colli_indices.numel()) != 2:
            raise RuntimeError(
                "play_force.py expects exactly two hand collision bodies, got "
                f"{int(env.hand_colli_indices.numel())}."
            )
        self.env = env
        self.args = args
        self.interval = args.diagnostic_interval
        self.effective_stable_steps = effective_stable_steps
        window_steps = max(
            1, int(math.ceil(args.baseline_window_s / float(env.dt)))
        )
        self.pre_window = deque(maxlen=window_steps)
        self.event = None
        self.hold_samples = []
        self.event_policy_steps = 0
        self.physics_active_printed = False
        self.summary_printed = False
        self.no_force_ready_streak = 0
        self.no_force_samples = deque(maxlen=window_steps)

    @property
    def event_started(self):
        return self.event is not None

    def _read_state(self):
        env = self.env
        upper_state = env.rigid_body_states[0, env.upper_body_index]
        packed = torch.cat(
            (
                upper_state[7:9],
                env.box_states[0, 7:9],
                env.external_force_world[0],
                torch.stack(
                    (
                        upper_state[12],
                        env.base_ang_vel[0, 2],
                        env.yaw[0],
                        env.teacher_vx[0],
                        env.teacher_heading[0],
                        env.teacher_heading_error[0],
                        env.teacher_yaw_rate[0],
                        env.commands[0, 0],
                        env.commands[0, 1],
                        env.commands[0, 2],
                        env.carry_heading_ref[0],
                        env.external_force_scale[0],
                        env.force_phase[0].float(),
                        env.is_stage_carry[0].float(),
                    )
                ),
                env.hand_contact_filt[0].float(),
                env.force_last_hand_contacts[0].float(),
            )
        ).tolist()
        return {
            "upper_world_xy": (packed[0], packed[1]),
            "box_world_xy": (packed[2], packed[3]),
            "force_world": (packed[4], packed[5], packed[6]),
            "upper_world_yaw_rate": packed[7],
            "controller_body_yaw_rate": packed[8],
            "robot_yaw_world": packed[9],
            "teacher_vx": packed[10],
            "teacher_heading_world": packed[11],
            "teacher_heading_error": packed[12],
            "teacher_yaw_rate": packed[13],
            "command_vx": packed[14],
            "command_vy": packed[15],
            "command_yaw_rate": packed[16],
            "nominal_heading_world": packed[17],
            "force_scale": packed[18],
            "phase": int(packed[19]),
            "is_stage_carry": bool(packed[20]),
            "left_contact": bool(packed[21]),
            "right_contact": bool(packed[22]),
            "force_scheduler_left_contact": bool(packed[23]),
            "force_scheduler_right_contact": bool(packed[24]),
        }

    def record_initial_state(self):
        self.pre_window.append(self._read_state())

    def _infer_direction_name(self, direction_world):
        names = tuple(self.env.cfg.external_force.force_directions)
        local = torch.tensor(
            [self.env._DIRECTION_LOCAL[name] for name in names],
            dtype=torch.float,
            device=self.env.device,
        )
        box_quat = self.env.box_states[0, 3:7].unsqueeze(0).expand(len(names), -1)
        candidates = quat_rotate(box_quat, local)
        candidates[:, 2] = 0.0
        candidates /= torch.clamp(
            torch.linalg.vector_norm(candidates, dim=-1, keepdim=True), min=1.0e-9
        )
        direction = torch.tensor(direction_world, device=self.env.device)
        scores = torch.sum(candidates * direction.unsqueeze(0), dim=-1)
        return names[int(torch.argmax(scores))]

    def _schedule_event(self, sample):
        env = self.env
        direction_world = tuple(
            float(value) for value in env.external_force_direction_world[0].tolist()
        )
        self.event = {
            "direction": self._infer_direction_name(direction_world),
            "direction_world": direction_world,
            "beta": float(env.external_force_beta[0]),
            "box_mass": float(env.external_force_box_mass[0]),
            "peak_force": float(env.external_force_peak_N[0]),
            "hold_duration": float(env.force_hold_duration_s[0]),
            "nominal_heading_world": sample["nominal_heading_world"],
            "pre_samples": list(self.pre_window),
            "force_scheduler_left_contact": sample[
                "force_scheduler_left_contact"
            ],
            "force_scheduler_right_contact": sample[
                "force_scheduler_right_contact"
            ],
        }
        forward, lateral = _heading_axes(self.event["nominal_heading_world"])
        peak_world = tuple(
            self.event["peak_force"] * value
            for value in self.event["direction_world"][:2]
        )
        force_parallel = _dot_xy(peak_world, forward)
        force_perp = _dot_xy(peak_world, lateral)
        force_angle = math.atan2(force_perp, force_parallel)
        self.event.update(
            force_parallel=force_parallel,
            force_perp=force_perp,
            force_angle=force_angle,
        )

        print("\n========== FORCE SCHEDULED ==========\n")
        print(f"box-frame direction: {self.event['direction']}")
        print(f"beta: {self.event['beta']:.4f}")
        print(f"box mass: {self.event['box_mass']:.4f} kg")
        print(f"F_peak: {self.event['peak_force']:.4f} N")
        print(f"hold duration: {self.event['hold_duration']:.4f} s")
        print(
            "external_force_direction_world (frozen at schedule): "
            f"({direction_world[0]:.4f}, {direction_world[1]:.4f}, "
            f"{direction_world[2]:.4f})\n"
        )
        heading = self.event["nominal_heading_world"]
        print("force relative to nominal heading:")
        print(f"  nominal_heading_world = {heading:.4f} rad / {math.degrees(heading):.2f} deg")
        print(f"  F_parallel_to_nominal = {force_parallel:.4f} N")
        print(f"  F_perp_to_nominal = {force_perp:.4f} N")
        print(
            "  force_angle_relative_to_nominal = "
            f"{force_angle:.4f} rad / {math.degrees(force_angle):.2f} deg\n"
        )
        print(
            "The event is scheduled, but no force was applied in the policy step "
            "that created it."
        )
        print("=====================================\n")

    def _add_relative_metrics(self, sample, include_force=True):
        forward, lateral = _heading_axes(sample["nominal_heading_world"])
        sample["upper_nominal_forward_speed"] = _dot_xy(
            sample["upper_world_xy"], forward
        )
        sample["upper_nominal_lateral_speed"] = _dot_xy(
            sample["upper_world_xy"], lateral
        )
        sample["box_nominal_forward_speed"] = _dot_xy(
            sample["box_world_xy"], forward
        )
        sample["box_nominal_lateral_speed"] = _dot_xy(
            sample["box_world_xy"], lateral
        )
        teacher_forward, _ = _heading_axes(sample["teacher_heading_world"])
        sample["upper_speed_along_teacher_heading"] = _dot_xy(
            sample["upper_world_xy"], teacher_forward
        )
        teacher_world_xy = (
            sample["teacher_vx"] * teacher_forward[0],
            sample["teacher_vx"] * teacher_forward[1],
        )
        sample["teacher_planar_velocity_error"] = math.hypot(
            sample["upper_world_xy"][0] - teacher_world_xy[0],
            sample["upper_world_xy"][1] - teacher_world_xy[1],
        )
        sample["teacher_heading_offset"] = _wrap_to_pi(
            sample["teacher_heading_world"] - sample["nominal_heading_world"]
        )
        if include_force and self.event is not None:
            direction = self.event["direction_world"]
            sample["upper_along_force"] = _dot_xy(
                sample["upper_world_xy"], direction
            )
            sample["box_along_force"] = _dot_xy(
                sample["box_world_xy"], direction
            )
        return sample

    def _print_physics_active(self, sample):
        phase_name = self.env._PHASE_NAMES[sample["phase"]]
        force = sample["force_world"]
        print("\n========== FORCE PHYSICS ACTIVE ==========\n")
        print(f"phase: {phase_name}")
        print(
            "current external_force_world: "
            f"({force[0]:.4f}, {force[1]:.4f}, {force[2]:.4f}) N"
        )
        print(f"current force scale: {sample['force_scale']:.6f}")
        print(
            "detected after the policy step containing the first active "
            "physics substep"
        )
        print("==========================================\n")
        self.physics_active_printed = True

    def _print_periodic(self, sample):
        env = self.env
        phase_name = env._PHASE_NAMES[sample["phase"]]
        sample = self._add_relative_metrics(sample)
        force = sample["force_world"]
        print("\n[play_force]\n")
        print(f"phase: {phase_name}")
        print(f"beta: {self.event['beta']:.4f}")
        print(f"force_world_xyz: ({force[0]:.4f}, {force[1]:.4f}, {force[2]:.4f}) N")
        print(f"force_scale: {sample['force_scale']:.6f}")
        print(f"F_peak: {self.event['peak_force']:.4f} N\n")
        print(f"command_vx: {sample['command_vx']:.4f}")
        print(f"teacher_vx: {sample['teacher_vx']:.4f}\n")
        nominal = sample["nominal_heading_world"]
        teacher = sample["teacher_heading_world"]
        offset = sample["teacher_heading_offset"]
        print(f"nominal_heading_world: {nominal:.4f} rad / {math.degrees(nominal):.2f} deg")
        print(f"teacher_heading_world: {teacher:.4f} rad / {math.degrees(teacher):.2f} deg")
        print(f"teacher_heading_offset: {offset:.4f} rad / {math.degrees(offset):.2f} deg")
        print(f"teacher_yaw_rate: {sample['teacher_yaw_rate']:.4f} rad/s\n")
        print(f"actual_robot_yaw_world: {sample['robot_yaw_world']:.4f} rad")
        print(f"upper_world_yaw_rate: {sample['upper_world_yaw_rate']:.4f} rad/s")
        print(f"controller_body_yaw_rate: {sample['controller_body_yaw_rate']:.4f} rad/s\n")
        print(f"upper_world_vx: {sample['upper_world_xy'][0]:.4f}")
        print(f"upper_world_vy: {sample['upper_world_xy'][1]:.4f}")
        print(f"upper_nominal_forward_speed: {sample['upper_nominal_forward_speed']:.4f}")
        print(f"upper_nominal_lateral_speed: {sample['upper_nominal_lateral_speed']:.4f}\n")
        print(f"box_world_vx: {sample['box_world_xy'][0]:.4f}")
        print(f"box_world_vy: {sample['box_world_xy'][1]:.4f}")
        print(f"box_nominal_forward_speed: {sample['box_nominal_forward_speed']:.4f}")
        print(f"box_nominal_lateral_speed: {sample['box_nominal_lateral_speed']:.4f}\n")
        print(f"upper_v_along_force: {sample['upper_along_force']:.4f}")
        print(f"box_v_along_force: {sample['box_along_force']:.4f}\n")
        print("positive along-force velocity means motion along the applied world-force direction.\n")
        print(f"left_hand_contact: {sample['left_contact']}")
        print(f"right_hand_contact: {sample['right_contact']}")

    def _baseline_quality(self, samples):
        forward = [sample["upper_nominal_forward_speed"] for sample in samples]
        world_yaw_rate = [sample["upper_world_yaw_rate"] for sample in samples]
        forward_error = _mean(forward) - self.args.vx
        yaw_error = _mean(world_yaw_rate) - self.args.yaw_rate
        acceptable = (
            abs(forward_error) <= self.args.baseline_v_tol
            and abs(yaw_error) <= self.args.baseline_yaw_tol
        )
        return {
            "status": "ACCEPTABLE" if acceptable else "POOR",
            "forward_mean": _mean(forward),
            "forward_std": _std(forward),
            "lateral_mean": _mean(
                [sample["upper_nominal_lateral_speed"] for sample in samples]
            ),
            "lateral_std": _std(
                [sample["upper_nominal_lateral_speed"] for sample in samples]
            ),
            "world_yaw_mean": _mean(world_yaw_rate),
            "world_yaw_std": _std(world_yaw_rate),
            "forward_error": forward_error,
            "world_yaw_error": yaw_error,
        }

    def _print_baseline_quality(self, quality):
        print(f"  BASELINE QUALITY: {quality['status']}")
        print(f"  mean upper nominal-forward speed: {quality['forward_mean']:.4f} m/s")
        print(f"  std upper nominal-forward speed: {quality['forward_std']:.4f} m/s")
        print(f"  forward speed tracking error: {quality['forward_error']:.4f} m/s")
        print(f"  mean upper nominal-lateral speed: {quality['lateral_mean']:.4f} m/s")
        print(f"  std upper nominal-lateral speed: {quality['lateral_std']:.4f} m/s")
        print(f"  mean upper world yaw rate: {quality['world_yaw_mean']:.4f} rad/s")
        print(f"  std upper world yaw rate: {quality['world_yaw_std']:.4f} rad/s")
        print(f"  world yaw-rate tracking error: {quality['world_yaw_error']:.4f} rad/s")
        print(f"  commanded vx: {self.args.vx:.4f} m/s")
        print(f"  commanded yaw_rate: {self.args.yaw_rate:.4f} rad/s")
        if quality["status"] == "POOR":
            print("\n  Warning:")
            print("  The robot was not tracking the nominal locomotion command well before force onset.")
            print("  Force-compliance interpretation is confounded.")

    def _print_summary(self, interrupted=False):
        if self.summary_printed:
            return
        pre = [
            self._add_relative_metrics(dict(sample))
            for sample in self.event["pre_samples"]
        ]
        hold = self.hold_samples
        pre_upper = _mean_xy(pre, "upper_world_xy")
        pre_box = _mean_xy(pre, "box_world_xy")
        hold_upper = _mean_xy(hold, "upper_world_xy")
        hold_box = _mean_xy(hold, "box_world_xy")
        pre_upper_projected = _mean([sample["upper_along_force"] for sample in pre])
        pre_box_projected = _mean([sample["box_along_force"] for sample in pre])
        hold_upper_projected = _mean(
            [sample["upper_along_force"] for sample in hold]
        )
        hold_box_projected = _mean([sample["box_along_force"] for sample in hold])

        print("\n========== FORCE EVENT SUMMARY ==========\n")
        quality = self._baseline_quality(pre)
        pre_upper_forward = _mean([s["upper_nominal_forward_speed"] for s in pre])
        pre_upper_lateral = _mean([s["upper_nominal_lateral_speed"] for s in pre])
        pre_box_forward = _mean([s["box_nominal_forward_speed"] for s in pre])
        pre_box_lateral = _mean([s["box_nominal_lateral_speed"] for s in pre])
        hold_upper_forward = _mean([s["upper_nominal_forward_speed"] for s in hold])
        hold_upper_lateral = _mean([s["upper_nominal_lateral_speed"] for s in hold])
        hold_box_forward = _mean([s["box_nominal_forward_speed"] for s in hold])
        hold_box_lateral = _mean([s["box_nominal_lateral_speed"] for s in hold])
        nominal_heading = _circular_mean([s["nominal_heading_world"] for s in hold])
        teacher_heading = _circular_mean([s["teacher_heading_world"] for s in hold])
        teacher_offset = _circular_mean([s["teacher_heading_offset"] for s in hold])
        pre_yaw = _circular_mean([s["robot_yaw_world"] for s in pre])
        hold_yaw = _circular_mean([s["robot_yaw_world"] for s in hold])
        actual_heading_change = _wrap_to_pi(hold_yaw - pre_yaw)

        print("force condition:")
        print(f"  box-frame direction: {self.event['direction']}")
        print(f"  beta: {self.event['beta']:.4f}")
        print(f"  box mass: {self.event['box_mass']:.4f} kg")
        print(f"  F_peak: {self.event['peak_force']:.4f} N\n")
        direction = self.event["direction_world"]
        angle = self.event["force_angle"]
        print("force geometry:")
        print(f"  world direction: ({direction[0]:.4f}, {direction[1]:.4f})")
        print(f"  F_parallel to nominal: {self.event['force_parallel']:.4f} N")
        print(f"  F_perp to nominal: {self.event['force_perp']:.4f} N")
        print(f"  angle relative to nominal: {angle:.4f} rad / {math.degrees(angle):.2f} deg\n")
        print("nominal command:")
        print(f"  vx: {self.args.vx:.4f} m/s")
        print(f"  vy: {self.args.vy:.4f} m/s")
        print(f"  yaw_rate: {self.args.yaw_rate:.4f} rad/s\n")
        print("baseline quality:")
        self._print_baseline_quality(quality)
        print("\nPRE:")
        print(f"  upper world velocity: ({pre_upper[0]:.4f}, {pre_upper[1]:.4f}) m/s")
        print(f"  upper nominal-forward speed: {pre_upper_forward:.4f} m/s")
        print(f"  upper nominal-lateral speed: {pre_upper_lateral:.4f} m/s")
        print(f"  upper along-force speed: {pre_upper_projected:.4f} m/s")
        print(f"  box world velocity: ({pre_box[0]:.4f}, {pre_box[1]:.4f}) m/s")
        print(f"  box nominal-forward speed: {pre_box_forward:.4f} m/s")
        print(f"  box nominal-lateral speed: {pre_box_lateral:.4f} m/s")
        print(f"  box along-force speed: {pre_box_projected:.4f} m/s\n")
        print("HOLD:")
        print(f"  upper world velocity: ({hold_upper[0]:.4f}, {hold_upper[1]:.4f}) m/s")
        print(f"  upper nominal-forward speed: {hold_upper_forward:.4f} m/s")
        print(f"  upper nominal-lateral speed: {hold_upper_lateral:.4f} m/s")
        print(f"  upper along-force speed: {hold_upper_projected:.4f} m/s")
        print(f"  box world velocity: ({hold_box[0]:.4f}, {hold_box[1]:.4f}) m/s")
        print(f"  box nominal-forward speed: {hold_box_forward:.4f} m/s")
        print(f"  box nominal-lateral speed: {hold_box_lateral:.4f} m/s")
        print(f"  box along-force speed: {hold_box_projected:.4f} m/s\n")
        print("response delta (HOLD - PRE):")
        print(f"  upper nominal-forward speed: {hold_upper_forward - pre_upper_forward:.4f} m/s")
        print(f"  upper nominal-lateral speed: {hold_upper_lateral - pre_upper_lateral:.4f} m/s")
        print(f"  upper along-force speed: {hold_upper_projected - pre_upper_projected:.4f} m/s")
        print(f"  box nominal-forward speed: {hold_box_forward - pre_box_forward:.4f} m/s")
        print(f"  box nominal-lateral speed: {hold_box_lateral - pre_box_lateral:.4f} m/s")
        print(f"  box along-force speed: {hold_box_projected - pre_box_projected:.4f} m/s\n")
        print("  positive along-force speed means motion along the normalized applied world-force direction.")
        print("  It is one diagnostic dimension, not an automatic compliance verdict.\n")
        print("teacher:")
        print(f"  mean teacher vx: {_mean([s['teacher_vx'] for s in hold]):.4f}")
        print(f"  mean nominal heading world: {nominal_heading:.4f} rad / {math.degrees(nominal_heading):.2f} deg")
        print(f"  mean teacher heading world: {teacher_heading:.4f} rad / {math.degrees(teacher_heading):.2f} deg")
        print(f"  mean teacher heading offset: {teacher_offset:.4f} rad / {math.degrees(teacher_offset):.2f} deg")
        print(f"  mean teacher yaw rate: {_mean([s['teacher_yaw_rate'] for s in hold]):.4f} rad/s\n")
        print("observed:")
        print(f"  mean PRE robot yaw world: {pre_yaw:.4f} rad / {math.degrees(pre_yaw):.2f} deg")
        print(f"  mean HOLD robot yaw world: {hold_yaw:.4f} rad / {math.degrees(hold_yaw):.2f} deg")
        print(f"  actual heading change: {actual_heading_change:.4f} rad / {math.degrees(actual_heading_change):.2f} deg")
        print(f"  mean upper world yaw rate: {_mean([s['upper_world_yaw_rate'] for s in hold]):.4f} rad/s")
        print(f"  mean controller body yaw rate: {_mean([s['controller_body_yaw_rate'] for s in hold]):.4f} rad/s")
        print(f"  mean upper nominal-forward speed: {hold_upper_forward:.4f} m/s")
        print(f"  mean upper nominal-lateral speed: {hold_upper_lateral:.4f} m/s\n")
        mean_teacher_vx = _mean([s["teacher_vx"] for s in hold])
        mean_actual_teacher_speed = _mean(
            [s["upper_speed_along_teacher_heading"] for s in hold]
        )
        mean_teacher_yaw = _mean([s["teacher_yaw_rate"] for s in hold])
        mean_controller_yaw = _mean([s["controller_body_yaw_rate"] for s in hold])
        print("teacher vs actual:")
        print(f"  mean teacher speed along teacher heading: {mean_teacher_vx:.4f} m/s")
        print(f"  mean actual speed along teacher heading: {mean_actual_teacher_speed:.4f} m/s")
        print(f"  signed speed tracking difference (actual - teacher): {mean_actual_teacher_speed - mean_teacher_vx:.4f} m/s")
        print(f"  mean planar velocity-error norm used before reward shaping: {_mean([s['teacher_planar_velocity_error'] for s in hold]):.4f} m/s")
        print(f"  mean teacher yaw-rate target: {mean_teacher_yaw:.4f} rad/s")
        print(f"  mean reward-comparator body yaw rate: {mean_controller_yaw:.4f} rad/s")
        print(f"  yaw-rate tracking difference (body actual - teacher): {mean_controller_yaw - mean_teacher_yaw:.4f} rad/s")
        print(f"  mean teacher heading offset: {math.degrees(teacher_offset):.2f} deg")
        print(f"  actual PRE-to-HOLD heading change: {math.degrees(actual_heading_change):.2f} deg")
        print(f"  mean formal teacher heading error (teacher - robot yaw): {math.degrees(_circular_mean([s['teacher_heading_error'] for s in hold])):.2f} deg\n")
        pre_bilateral_preserved = bool(pre) and all(
            sample["left_contact"] and sample["right_contact"] for sample in pre
        )
        hold_bilateral_preserved = bool(hold) and all(
            sample["left_contact"] and sample["right_contact"] for sample in hold
        )
        print("grasp:")
        print(
            "  bilateral hand contact preserved during PRE = "
            f"{pre_bilateral_preserved}"
        )
        print(
            "  bilateral hand contact preserved during HOLD = "
            f"{hold_bilateral_preserved}"
        )
        print("  metric source: carry-task hand_contact_filt")
        print(
            "  force_last_hand_contacts buffer after scheduling = "
            f"({self.event['force_scheduler_left_contact']}, "
            f"{self.event['force_scheduler_right_contact']})"
        )
        print(
            "  this stores current contact for the scheduler's next step and is "
            "not the carry-task filter"
        )
        print("\ninterpretation warning:")
        if quality["status"] == "POOR":
            print("  baseline confounded; do not attribute the full PRE-to-HOLD delta to force.")
        else:
            print("  baseline acceptable for descriptive interpretation; no success/fail verdict is assigned.")
        if interrupted:
            print("\n  note: episode reset before the formal force event completed.")
        print("\n=========================================\n")
        self.summary_printed = True

    def _update_no_force_tracking(self, sample):
        ready = (
            sample["is_stage_carry"]
            and sample["left_contact"]
            and sample["right_contact"]
        )
        self.no_force_ready_streak = self.no_force_ready_streak + 1 if ready else 0
        if self.no_force_ready_streak >= self.effective_stable_steps:
            self.no_force_samples.append(
                self._add_relative_metrics(sample, include_force=False)
            )

    def _print_no_force_summary(self):
        if self.summary_printed:
            return
        samples = list(self.no_force_samples)
        print("\n========== NO-FORCE COMMAND TRACKING ==========\n")
        print("command:")
        print(f"  vx = {self.args.vx:.4f} m/s")
        print(f"  vy = {self.args.vy:.4f} m/s")
        print(f"  yaw_rate = {self.args.yaw_rate:.4f} rad/s\n")
        if not samples:
            print("actual:\n  no stable-carry tracking samples were collected.\n")
            print("tracking:\n  unavailable\n")
            print("heading:\n  unavailable\n")
            print("grasp:\n  bilateral contact preserved = False\n")
            print(
                "Warning: the robot did not sustain is_stage_carry plus the "
                "carry-task bilateral hand_contact_filt for the requested "
                f"{self.args.pre_force_ready_s:.3f} s readiness duration."
            )
            print("\n================================================\n")
            self.summary_printed = True
            return

        quality = self._baseline_quality(samples)
        first_yaw = samples[0]["robot_yaw_world"]
        last_yaw = samples[-1]["robot_yaw_world"]
        heading_change = _wrap_to_pi(last_yaw - first_yaw)
        nominal_heading = _circular_mean(
            [sample["nominal_heading_world"] for sample in samples]
        )
        print("actual:")
        print(f"  mean upper nominal-forward speed = {quality['forward_mean']:.4f} m/s")
        print(f"  std upper nominal-forward speed = {quality['forward_std']:.4f} m/s")
        print(f"  mean upper nominal-lateral speed = {quality['lateral_mean']:.4f} m/s")
        print(f"  std upper nominal-lateral speed = {quality['lateral_std']:.4f} m/s")
        print(f"  mean upper world yaw rate = {quality['world_yaw_mean']:.4f} rad/s")
        print(f"  std upper world yaw rate = {quality['world_yaw_std']:.4f} rad/s")
        print(f"  mean controller body yaw rate = {_mean([s['controller_body_yaw_rate'] for s in samples]):.4f} rad/s\n")
        print("tracking:")
        print(f"  forward speed error = {quality['forward_error']:.4f} m/s")
        print(f"  world yaw-rate error = {quality['world_yaw_error']:.4f} rad/s")
        print(f"  BASELINE QUALITY: {quality['status']}\n")
        if quality["status"] == "POOR":
            print(
                "Warning: fixed-command tracking was outside the configured "
                "tolerances even without external force.\n"
            )
        print("heading:")
        print(f"  mean nominal heading world = {nominal_heading:.4f} rad / {math.degrees(nominal_heading):.2f} deg")
        print(f"  actual robot heading change = {heading_change:.4f} rad / {math.degrees(heading_change):.2f} deg\n")
        print("grasp:")
        print(
            "  bilateral contact preserved = "
            f"{all(s['left_contact'] and s['right_contact'] for s in samples)}"
        )
        print("  metric source: carry-task hand_contact_filt")
        print("\n================================================\n")
        self.summary_printed = True

    def draw_viewer_vectors(self):
        if (
            not self.args.diagnostic_viz
            or self.env.viewer is None
            or not self.env.enable_viewer_sync
        ):
            return
        env = self.env
        nominal_heading = env.carry_heading_ref[0]
        teacher_heading = env.teacher_heading[0]
        nominal = torch.stack(
            (
                env.commands[0, 0] * torch.cos(nominal_heading),
                env.commands[0, 0] * torch.sin(nominal_heading),
                torch.zeros((), device=env.device),
            )
        )
        teacher = torch.stack(
            (
                env.teacher_vx[0] * torch.cos(teacher_heading),
                env.teacher_vx[0] * torch.sin(teacher_heading),
                torch.zeros((), device=env.device),
            )
        )
        actual = env.rigid_body_states[0, env.upper_body_index, 7:10].clone()
        actual[2] = 0.0
        force_direction = env.external_force_direction_world[0]
        vectors = torch.stack((nominal, teacher, actual, force_direction))
        vectors[:3] *= float(self.args.velocity_draw_scale)
        vectors[3] *= float(self.args.force_direction_draw_length)

        anchor = env.rigid_body_states[0, env.upper_body_index, :3].clone()
        anchor[2] += 0.35
        starts = anchor.unsqueeze(0).expand(4, -1)
        vertices = torch.cat((starts, starts + vectors), dim=-1)
        env.gym.add_lines(
            env.viewer,
            env.envs[0],
            4,
            vertices.detach().cpu().numpy(),
            self._VECTOR_COLORS,
        )

    def after_step(self, done):
        sample = self._read_state()
        if self.args.no_force:
            if done:
                self._print_no_force_summary()
            else:
                self._update_no_force_tracking(sample)
            return

        if self.event is None:
            if not done:
                self.pre_window.append(sample)
            scheduled = (
                not done
                and int(self.env.force_event_count[0]) > 0
                and float(self.env.external_force_peak_N[0]) > 0.0
            )
            if not scheduled:
                return
            self._schedule_event(sample)
            return

        if done:
            self._print_summary(interrupted=True)
            return

        self.event_policy_steps += 1
        if bool(self.env.external_force_active[0]) and not self.physics_active_printed:
            self._print_physics_active(sample)
        if sample["phase"] == self.env.FORCE_PHASE_HOLD:
            self.hold_samples.append(self._add_relative_metrics(sample))

        force_in_progress = (
            bool(self.env.external_force_active[0])
            or int(self.env.force_remaining_physics_steps[0]) > 0
        )
        if force_in_progress and self.event_policy_steps % self.interval == 0:
            self._print_periodic(sample)

        if sample["phase"] == self.env.FORCE_PHASE_DONE:
            self._print_summary()

    def finish(self):
        if self.args.no_force:
            self._print_no_force_summary()
        elif self.event is not None and not self.summary_printed:
            self._print_summary(interrupted=True)


def play(args):
    if args.task != TASK_NAME:
        raise ValueError(
            f"play_force.py only supports --task {TASK_NAME}; got {args.task!r}."
        )
    if args.play_dataset:
        raise ValueError("play_force.py does not support --play_dataset.")
    if args.finetune_path is not None:
        raise ValueError(
            "play_force.py requires full --resume_path loading; --finetune_path is not supported."
        )
    if args.num_envs not in (None, 1):
        raise ValueError("play_force.py is a single-environment diagnostic; use --num_envs 1.")
    if args.no_force and (
        args.enable_external_force
        or args.force_direction
        or args.force_beta is not None
    ):
        raise ValueError(
            "--no_force cannot be combined with force-enabling, direction, or beta overrides."
        )

    checkpoint = _resolve_checkpoint(args.resume_path)
    env_cfg, train_cfg = task_registry.get_cfgs(name=TASK_NAME)
    if not isinstance(env_cfg, G1CarryBoxForceCfg) or not isinstance(
        train_cfg, G1CarryBoxForceCfgPPO
    ):
        raise RuntimeError("carrybox_force is not registered with the formal Stage2A configs.")

    random_direction_requested = args.force_direction in (None, "random")
    if args.force_direction == "random":
        # ``None`` is the formal environment's training-distribution behavior.
        args.force_direction = None

    env_cfg, train_cfg = update_cfg_from_args(env_cfg, train_cfg, args)
    env_cfg.env.num_envs = 1
    env_cfg.env.test = True
    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.delay = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.asset.box.random_props = False
    env_cfg.asset.box.reset_mode = "default"
    env_cfg.commands.resample_carry_commands = False
    env_cfg.external_force.enable_external_force = not args.no_force

    _validate_diagnostic_args(args)
    policy_dt = float(env_cfg.sim.dt) * int(env_cfg.control.decimation)
    if not math.isfinite(policy_dt) or policy_dt <= 0.0:
        raise ValueError(f"Invalid policy timestep derived from config: {policy_dt!r}.")
    training_stable_steps = int(
        env_cfg.external_force.stable_carry_policy_steps
    )
    diagnostic_stable_steps = int(
        math.ceil(args.pre_force_ready_s / policy_dt)
    )
    effective_stable_steps = max(
        training_stable_steps, diagnostic_stable_steps
    )
    env_cfg.external_force.stable_carry_policy_steps = effective_stable_steps

    args.num_envs = 1
    args.resume = True
    args.resume_path = checkpoint
    train_cfg.runner.resume = True
    train_cfg.runner.resume_path = checkpoint

    _validate_fixed_command(args, env_cfg)
    _validate_force_request(args, env_cfg, random_direction_requested)
    _print_configuration(
        args,
        env_cfg,
        checkpoint,
        random_direction_requested,
        policy_dt,
        training_stable_steps,
        diagnostic_stable_steps,
    )

    env, _ = task_registry.make_env(
        name=TASK_NAME, args=args, env_cfg=env_cfg
    )
    if not isinstance(env, G1CarryBoxForce):
        raise RuntimeError("task_registry did not construct the formal carrybox_force environment.")
    if not math.isclose(float(env.dt), policy_dt, rel_tol=0.0, abs_tol=1.0e-12):
        raise RuntimeError(
            "Environment policy timestep differs from sim.dt * control.decimation: "
            f"env.dt={float(env.dt)}, derived={policy_dt}."
        )
    if args.diagnostic_viz and env.viewer is not None:
        # carrybox_force owns the per-policy-step clear_lines call. Enabling its
        # debug-viz lifecycle prevents these play-only vectors from accumulating.
        env.debug_viz = True

    runner, _ = task_registry.make_alg_runner(
        env=env,
        name=TASK_NAME,
        args=args,
        train_cfg=train_cfg,
        log_root=None,
    )
    policy = runner.get_inference_policy(device=env.device)

    _set_fixed_command(env, args, synchronize_policy_buffer=True)
    obs = env.get_observations()
    diagnostic = ForceDiagnostic(env, args, effective_stable_steps)
    diagnostic.record_initial_state()

    with torch.inference_mode():
        for _ in range(int(env.max_episode_length)):
            # Reassert only the user-owned raw command.  The environment remains
            # free to update carry_heading_ref and carry_policy_commands[:, 2].
            _set_fixed_command(env, args)
            env.gym.fetch_results(env.sim, True)
            actions = policy(obs.detach())
            obs, _, _, dones, _, _, _, _ = env.step(actions.detach())
            done = bool(dones[0])
            diagnostic.after_step(done)
            if not done:
                diagnostic.draw_viewer_vectors()
            if done:
                break

    diagnostic.finish()

    if env_cfg.external_force.enable_external_force and not diagnostic.event_started:
        print(
            "[play_force] No force event was scheduled before the episode ended. "
            "The formal lifted-box, bilateral-contact, stable-carry trigger was "
            "not satisfied."
        )


if __name__ == "__main__":
    play(_parse_args())
