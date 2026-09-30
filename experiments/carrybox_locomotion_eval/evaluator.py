"""Deterministic, contact-qualified no-force CarryBox evaluation."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
for path in (EXPERIMENT_DIR, REPO_ROOT, REPO_ROOT / "legged_gym", REPO_ROOT / "rsl_rl"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluation.command_suite import build_evaluation_suite, modes_from_cli  # noqa: E402
from evaluation.metrics import aggregate_protocols, write_csv  # noqa: E402


TASK_NAME = "carrybox_locomotion_deterministic_eval"


def parse_evaluator_args(argv=None):
    command_line = [sys.argv[0], *(sys.argv[1:] if argv is None else argv)]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("train_distribution",), default="train_distribution")
    parser.add_argument("--protocol", choices=("constant", "step", "both"), default="constant")
    parser.add_argument("--mode", choices=("all", "stand", "vx", "vy", "yaw", "mixed"), default="all")
    for singular, plural, kind in (("seed", "seeds", int),
                                   ("carry_motion_id", "carry_motion_ids", int),
                                   ("carry_phase", "carry_phases", float)):
        group = parser.add_mutually_exclusive_group()
        group.add_argument("--" + singular, type=kind)
        group.add_argument("--" + plural, type=kind, nargs="+")
    parser.add_argument("--stable_hold", type=float, default=.20)
    parser.add_argument("--settle_timeout", type=float, default=2.)
    parser.add_argument("--warmup", type=float, default=.20)
    parser.add_argument("--duration", type=float, default=3.)
    parser.add_argument("--save_csv", action="store_true")
    parser.add_argument("--output_dir", type=str)
    args, remaining = parser.parse_known_args(argv)
    args.seeds = args.seeds if args.seeds is not None else [1 if args.seed is None else args.seed]
    args.carry_motion_ids = (args.carry_motion_ids if args.carry_motion_ids is not None
                             else [0 if args.carry_motion_id is None else args.carry_motion_id])
    args.carry_phases = (args.carry_phases if args.carry_phases is not None
                          else [.5 if args.carry_phase is None else args.carry_phase])
    if not (math.isfinite(args.warmup) and args.warmup >= 0 and
            math.isfinite(args.duration) and args.duration > 0 and
            math.isfinite(args.stable_hold) and args.stable_hold > 0 and
            math.isfinite(args.settle_timeout) and args.settle_timeout >= args.stable_hold):
        parser.error("warmup >= 0, duration > 0, and 0 < stable_hold <= settle_timeout must be finite")
    if args.protocol == "step" and args.mode == "stand":
        parser.error("stand has no step protocol; use constant or both")
    if args.output_dir and not args.save_csv:
        parser.error("--output_dir requires --save_csv")
    args.remaining_isaac_args = remaining
    args.command_line = command_line
    if argv is None:
        sys.argv = [sys.argv[0], *remaining]
    return args


def _checkpoint_label(checkpoint):
    path = Path(checkpoint)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{path.parent.name}_{path.stem}").strip("_")


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision():
    def run(*args):
        return subprocess.run(("git", *args), cwd=REPO_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    try:
        return {"commit": run("rev-parse", "HEAD"),
                "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": None}


def _write_json(path, value):
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def _validate_commands(conditions, cfg, motion_count):
    for c in conditions:
        if not 0 <= c.carry_motion_id < motion_count:
            raise ValueError(f"Motion {c.carry_motion_id} outside carryWith [0,{motion_count-1}]")
        for command in c.commands:
            if c.mode == "stand" and any(command):
                raise ValueError("Stand trial has a moving command")
            for i, value in enumerate(command):
                bounds = (cfg.commands.carry_mixed_ranges[i] if c.mode == "mixed" else
                          (cfg.commands.carry_vx_range, cfg.commands.carry_vy_range,
                           cfg.commands.carry_yaw_rate_range)[i])
                if not math.isfinite(value) or not bounds[0] - 1e-8 <= value <= bounds[1] + 1e-8:
                    raise ValueError(f"{c.trial_id} command {command} exceeds specialist range")
            if c.mode in ("vx", "vy", "yaw") and any(
                    abs(value) > 1e-10 for i, value in enumerate(command)
                    if i != {"vx": 0, "vy": 1, "yaw": 2}[c.mode]):
                raise ValueError(f"{c.trial_id} contains an inactive-axis command")


def evaluate(eval_args, legged_args):
    if not legged_args.resume_path:
        raise ValueError("evaluator.py requires --resume_path")
    if getattr(legged_args, "finetune_path", None):
        raise ValueError("Inference accepts --resume_path, not --finetune_path")
    conditions = build_evaluation_suite(
        seeds=eval_args.seeds, carry_motion_ids=eval_args.carry_motion_ids,
        carry_phases=eval_args.carry_phases, modes=modes_from_cli(eval_args.mode),
        protocol=eval_args.protocol)
    import isaacgym  # noqa: F401  Isaac Gym must precede torch imports.
    from evaluation.inference import resolve_checkpoint_path
    checkpoint_path = resolve_checkpoint_path(legged_args.resume_path)
    output_dir = None
    if eval_args.save_csv:
        if eval_args.output_dir:
            output_dir = Path(eval_args.output_dir).expanduser().resolve()
        else:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            output_dir = EXPERIMENT_DIR / "results" / _checkpoint_label(checkpoint_path) / stamp
        if output_dir.exists():
            raise FileExistsError(f"Evaluation output directory already exists: {output_dir}")

    from envs.carrybox_locomotion_eval_env import (
        CarryBoxLocomotionEvalEnv, assert_nominal_configuration,
        configure_deterministic_evaluation)
    from evaluation.inference import build_actor_only_policy
    from evaluation.trial import TRACE_FIELDS, run_suite, duration_steps
    from evaluation.readiness import ReadinessConfig
    from legged_gym.envs.g1.carrybox_locomotion_config import G1Cfg, G1CfgPPO
    from legged_gym.utils import task_registry
    from legged_gym.utils.helpers import set_seed

    max_segments = 5 if eval_args.protocol in ("step", "both") else 1
    episode_length_s = max(20., eval_args.settle_timeout +
                           max_segments * (eval_args.warmup + eval_args.duration) + 2.)
    env_cfg = configure_deterministic_evaluation(
        G1Cfg(), carry_motion_id=eval_args.carry_motion_ids[0],
        carry_phase=eval_args.carry_phases[0], episode_length_s=episode_length_s)
    train_cfg = G1CfgPPO()
    env_cfg.seed = train_cfg.seed = eval_args.seeds[0]
    train_cfg.runner.resume = False
    train_cfg.runner.finetune_path = None
    assert_nominal_configuration(env_cfg)
    legged_args.task = TASK_NAME
    legged_args.num_envs = 1
    legged_args.seed = eval_args.seeds[0]
    legged_args.resume = False
    task_registry.register(TASK_NAME, CarryBoxLocomotionEvalEnv, env_cfg, train_cfg)
    env, _ = task_registry.make_env(name=TASK_NAME, args=legged_args, env_cfg=env_cfg)
    _validate_commands(conditions, env_cfg, env.motionlib.num_motion[env._CARRY_SKILL])
    policy, checkpoint_path = build_actor_only_policy(env, train_cfg, checkpoint_path, env.device)
    print("[CONFIG] carry-only specialist; nominal physics and fixed CarryWith resets")
    print("[FRAME] planar=pelvis yaw; yaw=world z; box planar=pelvis yaw")
    print(f"[SUITE] {eval_args.suite}/{eval_args.protocol} trials={len(conditions)}")

    metadata = None
    trace_writer = None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=False)
        config = ReadinessConfig.from_env(env, eval_args.stable_hold, eval_args.settle_timeout)
        metadata = {
            "protocol_version": "contact_gated_v1", "status": "running",
            "checkpoint": checkpoint_path, "checkpoint_sha256": _sha256(checkpoint_path),
            "git": _git_revision(), "suite": eval_args.suite, "protocol": eval_args.protocol,
            "mode": eval_args.mode, "seeds": eval_args.seeds,
            "carry_motion_ids": eval_args.carry_motion_ids,
            "carry_phases": eval_args.carry_phases,
            "policy_dt": float(env.dt),
            "stable_required_steps": max(1, math.ceil(config.stable_hold_s / env.dt - 1e-9)),
            "warmup_steps": duration_steps(eval_args.warmup, env.dt, allow_zero=True),
            "measure_steps": duration_steps(eval_args.duration, env.dt),
            "readiness_config": asdict(config),
            "frames": {"planar": "pelvis_yaw", "yaw_rate": "pelvis_world_z",
                       "box_planar": "pelvis_yaw"},
            "command_ranges": {"vx": env_cfg.commands.carry_vx_range,
                               "vy": env_cfg.commands.carry_vy_range,
                               "yaw_rate": env_cfg.commands.carry_yaw_rate_range,
                               "mixed": env_cfg.commands.carry_mixed_ranges},
            "command_case_note": "deterministic training-range coverage, not an unbiased distribution estimate",
            "command_line": eval_args.command_line,
        }
        _write_json(output_dir / "run_metadata.json", metadata)
        manifest = []
        for condition in conditions:
            for index, command in enumerate(condition.commands):
                manifest.append({**condition.as_row(), "segment_index": index,
                                 "segment_vx": command[0], "segment_vy": command[1],
                                 "segment_yaw_rate": command[2]})
        write_csv(str(output_dir / "command_manifest.csv"), manifest)

        def trace_writer(trial_id, samples):
            write_csv(str(output_dir / "traces" / f"{trial_id}.csv"),
                      samples, fieldnames=TRACE_FIELDS)

    try:
        summaries, segments = run_suite(
            env, policy, conditions, warmup_s=eval_args.warmup,
            duration_s=eval_args.duration, seed_fn=set_seed,
            trace_writer=trace_writer, stable_hold_s=eval_args.stable_hold,
            settle_timeout_s=eval_args.settle_timeout)
        summaries = [{"checkpoint": checkpoint_path, "suite": eval_args.suite,
                      "warmup_s": eval_args.warmup,
                      "requested_duration_s": eval_args.duration, **row} for row in summaries]
        segments = [{"checkpoint": checkpoint_path, "suite": eval_args.suite,
                     "warmup_s": eval_args.warmup,
                     "requested_duration_s": eval_args.duration, **row} for row in segments]
        mode_summaries = aggregate_protocols(summaries, segments)
        if output_dir is not None:
            write_csv(str(output_dir / "summary.csv"), summaries)
            write_csv(str(output_dir / "segment_summary.csv"), segments)
            write_csv(str(output_dir / "mode_summary.csv"), mode_summaries)
            metadata["status"] = "completed"
            metadata["number_of_attempts"] = len(summaries)
            _write_json(output_dir / "run_metadata.json", metadata)
            print(f"[OUTPUT] {output_dir}")
    except Exception as exc:
        if metadata is not None:
            metadata["status"] = "failed"
            metadata["error"] = type(exc).__name__ + ": " + str(exc)
            _write_json(output_dir / "run_metadata.json", metadata)
        raise
    for row in mode_summaries:
        print(f"[MODE] {row['protocol']}/{row['mode']}/{row['case_type']}: "
              f"attempts={row['number_of_trials']} "
              f"init={row['initialization_success_rate']:.3f} "
              f"end_to_end={row['end_to_end_completion_rate']:.3f}")
    return summaries, mode_summaries


if __name__ == "__main__":
    evaluator_args = parse_evaluator_args()
    import isaacgym  # noqa: F401
    from legged_gym.utils import get_args
    evaluate(evaluator_args, get_args())
