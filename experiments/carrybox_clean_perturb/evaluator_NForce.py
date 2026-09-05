"""No-force CarryBox locomotion velocity evaluator.

Velocity statistics are collected once per policy step, only after the existing
confirmed-carry gate has remained true for the requested additional warmup.
"""

import argparse
import math
import os
import sys


EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(EXPERIMENT_DIR, "..", ".."))
for path in (
    EXPERIMENT_DIR,
    REPO_ROOT,
    os.path.join(REPO_ROOT, "legged_gym"),
    os.path.join(REPO_ROOT, "rsl_rl"),
):
    if path not in sys.path:
        sys.path.insert(0, path)

import isaacgym  # noqa: F401,E402
import torch  # noqa: E402

from configs.evaluation_config import FIXED_COMMAND  # noqa: E402
from configs.nominal_clean_config import apply_nominal_clean_config  # noqa: E402
from envs.carrybox_nominal_clean_env import (  # noqa: E402
    LeggedRobot as NominalCleanCarryBoxEnv,
)
from evaluation.inference import (  # noqa: E402
    load_actor_only_for_inference,
)
from evaluation.nforce_trial import run_trial  # noqa: E402
from evaluation.nforce_velocity import (  # noqa: E402
    NForceVelocityCsvLogger,
)
from legged_gym.envs.g1.carrybox_config import G1Cfg, G1CfgPPO  # noqa: E402
from legged_gym.utils import get_args, task_registry  # noqa: E402
from legged_gym.utils.helpers import set_seed  # noqa: E402


NFORCE_TASK = "carrybox_nforce_velocity_eval"
DEFAULT_STEADY_CARRY_WARMUP_S = 0.20
DEFAULT_STEADY_DURATION_S = 5.0
DEFAULT_EPISODE_LENGTH_S = 30.0


class NForceCarryBoxEnv(NominalCleanCarryBoxEnv):
    """Nominal evaluator env with a terminal-yaw snapshot for CSV diagnostics."""

    def _init_buffers(self):
        super()._init_buffers()
        self.nforce_terminal_base_yaw_rad = torch.full(
            (self.num_envs,), float("nan"), device=self.device
        )

    def _snapshot_clean_eval_for_summary(self, env_id):
        self.nforce_terminal_base_yaw_rad[env_id] = self.yaw[env_id].reshape(-1)[0]
        super()._snapshot_clean_eval_for_summary(env_id)


def _parse_command(text):
    parts = [part.strip() for part in str(text).split(",")]
    if len(parts) != 3 or any(not part for part in parts):
        raise argparse.ArgumentTypeError(
            "--command must contain exactly three values: VX,VY,YAW_RATE"
        )
    try:
        command = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--command must contain exactly three floating-point values"
        ) from exc
    if not all(math.isfinite(value) for value in command):
        raise argparse.ArgumentTypeError("--command values must be finite")
    return command


def parse_evaluator_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--command",
        type=_parse_command,
        default=FIXED_COMMAND,
        metavar="VX,VY,YAW_RATE",
    )
    parser.add_argument(
        "--steady_carry_warmup",
        type=float,
        default=DEFAULT_STEADY_CARRY_WARMUP_S,
    )
    parser.add_argument(
        "--steady_duration",
        type=float,
        default=DEFAULT_STEADY_DURATION_S,
    )
    parser.add_argument("--save_csv", action="store_true", default=False)
    parser.add_argument("--output_dir", type=str, default=None)
    eval_args, remaining = parser.parse_known_args()
    if not math.isfinite(eval_args.steady_carry_warmup):
        parser.error("--steady_carry_warmup must be finite")
    if eval_args.steady_carry_warmup < 0.0:
        parser.error("--steady_carry_warmup must be non-negative")
    if not math.isfinite(eval_args.steady_duration):
        parser.error("--steady_duration must be finite")
    if eval_args.steady_duration <= 0.0:
        parser.error("--steady_duration must be positive")
    sys.argv = [sys.argv[0], *remaining]
    return eval_args


def _register_task(env_cfg, train_cfg):
    task_registry.register(NFORCE_TASK, NForceCarryBoxEnv, env_cfg, train_cfg)


def _checkpoint_label(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    parent = os.path.basename(os.path.dirname(path))
    return f"{parent}_{stem}" if parent else stem


def _default_output_dir(checkpoint):
    return os.path.join(
        EXPERIMENT_DIR,
        "results",
        _checkpoint_label(checkpoint),
        "nforce_velocity",
    )


def play(eval_args, legged_args):
    if legged_args.resume_path is None:
        raise ValueError("evaluator_NForce.py requires --resume_path.")
    if getattr(legged_args, "finetune_path", None) is not None:
        raise ValueError("NForce inference uses --resume_path; --finetune_path is not supported.")

    command = tuple(float(value) for value in eval_args.command)
    seed = 1 if legged_args.seed is None else int(legged_args.seed)
    episode_length_s = max(
        DEFAULT_EPISODE_LENGTH_S,
        20.0 + float(eval_args.steady_carry_warmup) + float(eval_args.steady_duration),
    )

    env_cfg = G1Cfg()
    train_cfg = G1CfgPPO()
    env_cfg = apply_nominal_clean_config(
        env_cfg,
        trace_enabled=False,
        command=command,
        episode_length_s=episode_length_s,
    )
    env_cfg.clean_perturbation.enabled = False
    env_cfg.clean_perturbation.evaluation_trace_enabled = False
    env_cfg.clean_perturbation.debug_draw_force = False
    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.push_robots = False

    train_cfg.runner.resume = False
    train_cfg.runner.finetune_path = None
    legged_args.resume = False
    legged_args.num_envs = 1
    legged_args.task = NFORCE_TASK
    _register_task(env_cfg, train_cfg)

    env, _ = task_registry.make_env(
        name=NFORCE_TASK,
        args=legged_args,
        env_cfg=env_cfg,
    )
    print("[CONFIG] deterministic nominal reset with one environment")
    print("[CONFIG] clean perturbation force disabled")
    print("[CONFIG] domain_rand.disturbance=False")
    print("[CONFIG] domain_rand.push_robots=False")
    print("[CONFIG] evaluator physics-substep force trace disabled")

    ppo_runner, _ = task_registry.make_alg_runner(
        env=env,
        name=NFORCE_TASK,
        args=legged_args,
        train_cfg=train_cfg,
        log_root=None,
    )
    checkpoint = load_actor_only_for_inference(
        ppo_runner,
        legged_args.resume_path,
        device=env.device,
    )
    policy = ppo_runner.get_inference_policy(device=env.device)

    samples, summary = run_trial(
        env,
        policy,
        checkpoint=checkpoint,
        seed=seed,
        command=command,
        eval_args=eval_args,
        seed_fn=set_seed,
    )
    if eval_args.save_csv:
        output_dir = eval_args.output_dir or _default_output_dir(checkpoint)
        logger = NForceVelocityCsvLogger(output_dir)
        trace_path = logger.write_trace("T0001", samples)
        logger.append_summary(summary)
        print(f"[OUTPUT] summary={logger.summary_path}")
        print(f"[OUTPUT] trace={trace_path}")
    return summary


if __name__ == "__main__":
    evaluator_args = parse_evaluator_args()
    args = get_args()
    play(evaluator_args, args)
