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
PRE_FORCE_WINDOW_S = 0.5


def _parse_args():
    """Remove play-only flags before delegating to the shared Isaac Gym parser."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--vx", type=float, default=0.4)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw_rate", type=float, default=0.0)
    parser.add_argument("--no_force", action="store_true")
    parser.add_argument("--diagnostic_interval", type=int, default=10)
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


def _format_range(value_range):
    return f"[{float(value_range[0]):.3f}, {float(value_range[1]):.3f}]"


def _print_configuration(args, env_cfg, checkpoint, random_direction_requested):
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


class ForceDiagnostic:
    """Read-only, env-0 event logging with a small script-local buffer."""

    def __init__(self, env, interval):
        if interval < 1:
            raise ValueError("--diagnostic_interval must be at least 1 policy step.")
        if int(env.hand_colli_indices.numel()) != 2:
            raise RuntimeError(
                "play_force.py expects exactly two hand collision bodies, got "
                f"{int(env.hand_colli_indices.numel())}."
            )
        self.env = env
        self.interval = interval
        window_steps = max(1, int(math.ceil(PRE_FORCE_WINDOW_S / float(env.dt))))
        self.pre_window = deque(maxlen=window_steps)
        self.event = None
        self.hold_samples = []
        self.event_policy_steps = 0
        self.summary_printed = False

    @property
    def event_started(self):
        return self.event is not None

    def _read_state(self):
        env = self.env
        upper_xy = env.rigid_body_states[0, env.upper_body_index, 7:9].tolist()
        box_xy = env.box_states[0, 7:9].tolist()
        force_world = env.external_force_world[0].tolist()
        contacts = env.hand_contact_filt[0].tolist()
        return {
            "upper_xy": (float(upper_xy[0]), float(upper_xy[1])),
            "box_xy": (float(box_xy[0]), float(box_xy[1])),
            "actual_yaw_rate": float(env.base_ang_vel[0, 2]),
            "teacher_vx": float(env.teacher_vx[0]),
            "teacher_heading": float(env.teacher_heading[0]),
            "teacher_yaw_rate": float(env.teacher_yaw_rate[0]),
            "nominal_vx": float(env.commands[0, 0]),
            "nominal_heading": float(env.carry_heading_ref[0]),
            "force_world": tuple(float(value) for value in force_world),
            "left_contact": bool(contacts[0]),
            "right_contact": bool(contacts[1]),
            "phase": int(env.force_phase[0]),
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

    def _start_event(self, sample):
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
            "pre_samples": list(self.pre_window),
        }
        print("\n========== FORCE START ==========\n")
        print(f"direction: {self.event['direction']}")
        print(f"beta: {self.event['beta']:.4f}")
        print(f"box mass: {self.event['box_mass']:.4f} kg")
        print(f"peak force N: {self.event['peak_force']:.4f}")
        print(f"hold duration: {self.event['hold_duration']:.4f} s")
        print(
            "world force direction: "
            f"({direction_world[0]:.4f}, {direction_world[1]:.4f}, "
            f"{direction_world[2]:.4f})\n"
        )
        print("nominal command:")
        print(f"  vx: {sample['nominal_vx']:.4f}")
        print(f"  vy: {float(env.commands[0, 1]):.4f}")
        print(f"  yaw_rate: {float(env.commands[0, 2]):.4f}\n")
        print(f"teacher vx: {sample['teacher_vx']:.4f}")
        print(f"teacher heading: {sample['teacher_heading']:.4f}")
        print(f"teacher yaw rate: {sample['teacher_yaw_rate']:.4f}\n")
        print("=================================\n")

    def _add_projection(self, sample):
        direction = self.event["direction_world"]
        sample["upper_along_force"] = (
            sample["upper_xy"][0] * direction[0]
            + sample["upper_xy"][1] * direction[1]
        )
        sample["box_along_force"] = (
            sample["box_xy"][0] * direction[0]
            + sample["box_xy"][1] * direction[1]
        )
        return sample

    def _print_periodic(self, sample):
        env = self.env
        phase_name = env._PHASE_NAMES[sample["phase"]]
        sample = self._add_projection(sample)
        force = sample["force_world"]
        print("\n[play_force]\n")
        print(f"phase: {phase_name}")
        print(f"beta: {self.event['beta']:.4f}")
        print(f"F_world: ({force[0]:.4f}, {force[1]:.4f}, {force[2]:.4f}) N")
        print(f"F_peak: {self.event['peak_force']:.4f} N\n")
        print(f"nominal_vx: {sample['nominal_vx']:.4f}")
        print(f"teacher_vx: {sample['teacher_vx']:.4f}\n")
        print(f"nominal_heading: {sample['nominal_heading']:.4f}")
        print(f"teacher_heading: {sample['teacher_heading']:.4f}")
        print(f"teacher_yaw_rate: {sample['teacher_yaw_rate']:.4f}\n")
        print(f"upper_body_world_vx: {sample['upper_xy'][0]:.4f}")
        print(f"upper_body_world_vy: {sample['upper_xy'][1]:.4f}")
        print(f"actual_yaw_rate: {sample['actual_yaw_rate']:.4f}\n")
        print(f"box_world_vx: {sample['box_xy'][0]:.4f}")
        print(f"box_world_vy: {sample['box_xy'][1]:.4f}\n")
        print(f"upper_v_along_force: {sample['upper_along_force']:.4f}")
        print(f"box_v_along_force: {sample['box_along_force']:.4f}\n")
        print(f"left_hand_contact: {sample['left_contact']}")
        print(f"right_hand_contact: {sample['right_contact']}")

    def _print_summary(self, interrupted=False):
        if self.summary_printed:
            return
        pre = [self._add_projection(dict(sample))
               for sample in self.event["pre_samples"]]
        hold = self.hold_samples
        pre_upper = _mean_xy(pre, "upper_xy")
        pre_box = _mean_xy(pre, "box_xy")
        hold_upper = _mean_xy(hold, "upper_xy")
        hold_box = _mean_xy(hold, "box_xy")
        pre_upper_projected = _mean([sample["upper_along_force"] for sample in pre])
        pre_box_projected = _mean([sample["box_along_force"] for sample in pre])
        hold_upper_projected = _mean(
            [sample["upper_along_force"] for sample in hold]
        )
        hold_box_projected = _mean([sample["box_along_force"] for sample in hold])

        print("\n========== FORCE EVENT SUMMARY ==========\n")
        print("force:")
        print(f"  direction: {self.event['direction']}")
        print(f"  beta: {self.event['beta']:.4f}")
        print(f"  box_mass: {self.event['box_mass']:.4f} kg")
        print(f"  F_peak: {self.event['peak_force']:.4f} N\n")
        print("nominal:")
        print(f"  vx: {self.event['pre_samples'][-1]['nominal_vx']:.4f}")
        print(f"  vy: {float(self.env.commands[0, 1]):.4f}")
        print(f"  yaw_rate: {float(self.env.commands[0, 2]):.4f}\n")
        print("pre-force:")
        print(f"  mean upper world v = ({pre_upper[0]:.4f}, {pre_upper[1]:.4f})")
        print(f"  mean box world v   = ({pre_box[0]:.4f}, {pre_box[1]:.4f})")
        print(f"  projected upper v along force = {pre_upper_projected:.4f}")
        print(f"  projected box v along force   = {pre_box_projected:.4f}\n")
        print("during HOLD:")
        print(f"  mean upper world v = ({hold_upper[0]:.4f}, {hold_upper[1]:.4f})")
        print(f"  mean box world v   = ({hold_box[0]:.4f}, {hold_box[1]:.4f})")
        print(f"  mean projected upper v along force = {hold_upper_projected:.4f}")
        print(f"  mean projected box v along force   = {hold_box_projected:.4f}\n")
        print("response:")
        print(
            "  delta upper projected v = HOLD - PRE = "
            f"{hold_upper_projected - pre_upper_projected:.4f}"
        )
        print(
            "  delta box projected v   = HOLD - PRE = "
            f"{hold_box_projected - pre_box_projected:.4f}\n"
        )
        print("teacher:")
        print(f"  mean teacher vx: {_mean([s['teacher_vx'] for s in hold]):.4f}")
        print(
            "  mean teacher heading: "
            f"{_circular_mean([s['teacher_heading'] for s in hold]):.4f}"
        )
        print(
            "  mean teacher yaw rate: "
            f"{_mean([s['teacher_yaw_rate'] for s in hold]):.4f}\n"
        )
        print("observed:")
        print(
            "  mean actual yaw rate: "
            f"{_mean([s['actual_yaw_rate'] for s in hold]):.4f}\n"
        )
        bilateral_preserved = bool(hold) and all(
            sample["left_contact"] and sample["right_contact"] for sample in hold
        )
        print("grasp:")
        print(
            "  bilateral hand contact preserved during HOLD = "
            f"{bilateral_preserved}"
        )
        if interrupted:
            print("\n  note: episode reset before the formal force event completed.")
        print("\n=========================================\n")
        self.summary_printed = True

    def after_step(self, done):
        sample = self._read_state()
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
            self._start_event(sample)
            return

        if done:
            self._print_summary(interrupted=True)
            return

        self.event_policy_steps += 1
        if sample["phase"] == self.env.FORCE_PHASE_HOLD:
            self.hold_samples.append(self._add_projection(sample))

        force_in_progress = (
            bool(self.env.external_force_active[0])
            or int(self.env.force_remaining_physics_steps[0]) > 0
        )
        if force_in_progress and self.event_policy_steps % self.interval == 0:
            self._print_periodic(sample)

        if sample["phase"] == self.env.FORCE_PHASE_DONE:
            self._print_summary()


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

    args.num_envs = 1
    args.resume = True
    args.resume_path = checkpoint
    train_cfg.runner.resume = True
    train_cfg.runner.resume_path = checkpoint

    _validate_fixed_command(args, env_cfg)
    _validate_force_request(args, env_cfg, random_direction_requested)
    if args.diagnostic_interval < 1:
        raise ValueError("--diagnostic_interval must be at least 1 policy step.")
    _print_configuration(
        args, env_cfg, checkpoint, random_direction_requested
    )

    env, _ = task_registry.make_env(
        name=TASK_NAME, args=args, env_cfg=env_cfg
    )
    if not isinstance(env, G1CarryBoxForce):
        raise RuntimeError("task_registry did not construct the formal carrybox_force environment.")

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
    diagnostic = ForceDiagnostic(env, args.diagnostic_interval)
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
            if done:
                break

    if env_cfg.external_force.enable_external_force and not diagnostic.event_started:
        print(
            "[play_force] No force event was scheduled before the episode ended. "
            "The formal lifted-box, bilateral-contact, stable-carry trigger was "
            "not satisfied."
        )


if __name__ == "__main__":
    play(_parse_args())
