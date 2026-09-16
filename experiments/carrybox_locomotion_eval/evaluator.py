"""Entry point for deterministic, no-force carry-locomotion evaluation."""

import argparse
import os
import re
import sys
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
for path in (REPO_ROOT, REPO_ROOT / "legged_gym", REPO_ROOT / "rsl_rl"):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

import isaacgym  # noqa: E402,F401

from envs.carrybox_locomotion_eval_env import (  # noqa: E402
    CarryBoxLocomotionEvalEnv,
    assert_nominal_configuration,
    configure_deterministic_evaluation,
)
from evaluation.command_suite import (  # noqa: E402
    build_command_suite,
    modes_from_cli,
)
from evaluation.inference import build_actor_only_policy  # noqa: E402
from evaluation.metrics import aggregate_by_mode, write_csv  # noqa: E402
from evaluation.trial import TRACE_FIELDS, run_suite  # noqa: E402
from legged_gym.envs.g1.carrybox_locomotion_config import (  # noqa: E402
    G1Cfg,
    G1CfgPPO,
)
from legged_gym.utils import get_args, task_registry  # noqa: E402
from legged_gym.utils.helpers import set_seed  # noqa: E402


TASK_NAME = "carrybox_locomotion_deterministic_eval"


def parse_evaluator_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--suite", choices=("train_distribution",),
        default="train_distribution",
    )
    parser.add_argument(
        "--mode", choices=("all", "stand", "vx", "vy", "yaw", "mixed"),
        default="all",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--carry_motion_id", type=int, default=0)
    parser.add_argument("--carry_phase", type=float, default=0.5)
    parser.add_argument("--warmup", type=float, default=0.20)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--save_csv", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)
    args, remaining = parser.parse_known_args()
    if args.warmup < 0.0 or args.duration <= 0.0:
        raise ValueError("--warmup must be >= 0 and --duration must be > 0")
    if not 0.0 <= args.carry_phase <= 1.0:
        raise ValueError("--carry_phase must be in [0, 1]")
    sys.argv = [sys.argv[0], *remaining]
    return args


def _checkpoint_label(checkpoint):
    checkpoint = Path(checkpoint)
    raw = f"{checkpoint.parent.name}_{checkpoint.stem}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_")


def _default_output_dir(checkpoint):
    return EXPERIMENT_DIR / "results" / _checkpoint_label(checkpoint)


def evaluate(eval_args, legged_args):
    if not legged_args.resume_path:
        raise ValueError("evaluator.py requires --resume_path")
    if getattr(legged_args, "finetune_path", None):
        raise ValueError("Inference accepts --resume_path, not --finetune_path")

    conditions = build_command_suite(
        seed=eval_args.seed,
        carry_motion_id=eval_args.carry_motion_id,
        carry_phase=eval_args.carry_phase,
        modes=modes_from_cli(eval_args.mode),
    )
    episode_length_s = max(20.0, eval_args.warmup + eval_args.duration + 1.0)
    env_cfg = configure_deterministic_evaluation(
        G1Cfg(),
        carry_motion_id=eval_args.carry_motion_id,
        carry_phase=eval_args.carry_phase,
        episode_length_s=episode_length_s,
    )
    train_cfg = G1CfgPPO()
    env_cfg.seed = eval_args.seed
    train_cfg.seed = eval_args.seed
    train_cfg.runner.resume = False
    train_cfg.runner.finetune_path = None
    assert_nominal_configuration(env_cfg)

    legged_args.task = TASK_NAME
    legged_args.num_envs = 1
    legged_args.seed = eval_args.seed
    legged_args.resume = False
    task_registry.register(
        TASK_NAME, CarryBoxLocomotionEvalEnv, env_cfg, train_cfg
    )
    env, _ = task_registry.make_env(
        name=TASK_NAME, args=legged_args, env_cfg=env_cfg
    )
    policy, checkpoint = build_actor_only_policy(
        env, train_cfg, legged_args.resume_path, env.device
    )

    print("[CONFIG] carry-only specialist environment")
    print(
        "[CONFIG] fixed carry reset: "
        f"motion_id={eval_args.carry_motion_id}, phase={eval_args.carry_phase}"
    )
    print("[CONFIG] noise/domain-randomization/disturbance/push/delay: OFF")
    print("[CONFIG] box property randomization: OFF")
    print("[FRAME] robot planar velocity: pelvis yaw frame (reward/runner aligned)")
    print("[FRAME] yaw rate: pelvis world-z angular velocity")
    print("[FRAME] box planar velocity: same pelvis yaw frame")
    print(f"[SUITE] {eval_args.suite}, mode={eval_args.mode}, trials={len(conditions)}")

    output_dir = None
    trace_writer = None
    if eval_args.save_csv:
        output_dir = Path(eval_args.output_dir or _default_output_dir(checkpoint))
        trace_dir = output_dir / "traces"

        def trace_writer(trial_id, samples):
            write_csv(
                str(trace_dir / f"{trial_id}.csv"),
                samples,
                fieldnames=TRACE_FIELDS,
            )

        write_csv(
            str(output_dir / "command_manifest.csv"),
            [condition.as_row() for condition in conditions],
        )

    summaries = run_suite(
        env,
        policy,
        conditions,
        warmup_s=eval_args.warmup,
        duration_s=eval_args.duration,
        seed_fn=set_seed,
        trace_writer=trace_writer,
    )
    summaries = [
        {
            "checkpoint": checkpoint,
            "suite": eval_args.suite,
            "warmup_s": eval_args.warmup,
            "requested_duration_s": eval_args.duration,
            **summary,
        }
        for summary in summaries
    ]
    mode_summaries = aggregate_by_mode(summaries)

    if output_dir is not None:
        write_csv(str(output_dir / "summary.csv"), summaries)
        write_csv(str(output_dir / "mode_summary.csv"), mode_summaries)
        print(f"[OUTPUT] {output_dir}")

    for row in mode_summaries:
        print(
            f"[MODE] {row['mode']}: trials={row['number_of_trials']} "
            f"completion={row['completion_rate']:.3f} "
            f"carry={row['final_confirmed_carry_rate']:.3f}"
        )
    return summaries, mode_summaries


if __name__ == "__main__":
    evaluator_args = parse_evaluator_args()
    evaluate(evaluator_args, get_args())
