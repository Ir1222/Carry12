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
from evaluate import (  # noqa: E402
    assert_startup_compatibility,
    load_actor_only_for_inference,
)
from evaluation.commands import set_fixed_evaluation_command  # noqa: E402
from evaluation.nforce_velocity import (  # noqa: E402
    NForceVelocityCsvLogger,
    sample_velocity_tracking,
    summarize_velocity_tracking,
)
from legged_gym.envs.g1.carrybox_config import G1Cfg, G1CfgPPO  # noqa: E402
from legged_gym.utils import get_args, task_registry  # noqa: E402
from legged_gym.utils.helpers import set_seed  # noqa: E402


NFORCE_TASK = "carrybox_nforce_velocity_eval"
DEFAULT_STEADY_CARRY_WARMUP_S = 0.20
DEFAULT_STEADY_DURATION_S = 5.0
DEFAULT_EPISODE_LENGTH_S = 30.0

WAIT_CARRY = "WAIT_CARRY"
CONFIRMED_CARRY_WARMUP = "CONFIRMED_CARRY_WARMUP"
STEADY_CARRY = "STEADY_CARRY"
END = "END"


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


def _duration_steps(seconds, policy_dt, allow_zero=False):
    steps = int(round(float(seconds) / float(policy_dt)))
    return max(0 if allow_zero else 1, steps)


def _yaw_scalar(env):
    return float(env.yaw[0].reshape(-1)[0].item())


def _reset_for_trial(env, seed, command):
    set_seed(int(seed))
    env.reset_evaluation_trial_state(clear_actor_history=True)
    env.nforce_terminal_base_yaw_rad[:] = float("nan")
    env.reset()
    obs = set_fixed_evaluation_command(env, command)
    env.clear_summary_snapshot(env_id=0)
    return obs


def _summary_scalar(env, name):
    value = env.summary_scalar(name, env_id=0)
    return float(value.item())


def _summary_bool(env, name):
    return bool(_summary_scalar(env, name))


def _final_confirmed_carry(env):
    if bool(env.clean_eval_has_terminal_snapshot[0].item()):
        return bool(env.clean_eval_terminal_confirmed_carry_buf[0].item())
    return bool(env.confirmed_carry_buf[0].item())


def _final_base_yaw(env):
    if bool(env.clean_eval_has_terminal_snapshot[0].item()):
        terminal_yaw = float(env.nforce_terminal_base_yaw_rad[0].item())
        if math.isfinite(terminal_yaw):
            return terminal_yaw
    return _yaw_scalar(env)


def _assert_no_force_state(env):
    cfg = env.cfg
    if bool(cfg.clean_perturbation.enabled):
        raise AssertionError("No-force evaluator requires clean perturbation disabled")
    if bool(cfg.domain_rand.disturbance):
        raise AssertionError("No-force evaluator requires domain_rand.disturbance=False")
    if bool(cfg.domain_rand.push_robots):
        raise AssertionError("No-force evaluator requires domain_rand.push_robots=False")
    if int(_summary_scalar(env, "clean_eval_event_count_buf")) != 0:
        raise AssertionError("No-force evaluator unexpectedly scheduled a force event")
    if int(torch.sum(env.clean_eval_remaining_physics_steps).item()) != 0:
        raise AssertionError("No-force evaluator has pending external-force steps")
    if float(torch.linalg.vector_norm(env.clean_eval_force_tensor).item()) != 0.0:
        raise AssertionError("No-force evaluator has a nonzero box-force tensor")
    if float(torch.linalg.vector_norm(env.disturbance).item()) != 0.0:
        raise AssertionError("No-force evaluator has a nonzero robot disturbance tensor")


def _termination_reason_from_done(env, termination_ids):
    reason = env.clean_eval_last_termination_reason[0]
    if not reason and termination_ids.numel() > 0:
        reason = "termination"
    return reason or "done"


def _latched_failure_reason(env):
    if bool(env.clean_eval_humanoid_failure_buf[0].item()):
        return env.clean_eval_humanoid_failure_reason[0] or "humanoid_failure"
    if bool(env.clean_eval_box_failure_buf[0].item()):
        return env.clean_eval_box_failure_reason[0] or "box_failure"
    return ""


def _print_trial_header(checkpoint, seed, command):
    print("=" * 60)
    print("[NFORCE] Trial T0001")
    print(f"checkpoint={checkpoint}")
    print(f"seed={seed}")
    print(
        "raw_command="
        f"({command[0]:.3f}, {command[1]:.3f}, {command[2]:.3f})"
    )
    print("=" * 60)


def run_trial(env, policy, checkpoint, seed, command, eval_args):
    _print_trial_header(checkpoint, seed, command)
    obs = _reset_for_trial(env, seed, command)
    assert_startup_compatibility(env, obs, expected_command=command)
    _assert_no_force_state(env)

    print("[NFORCE ASSERT]")
    print("tracking target = carry_policy_commands")
    print("actual velocity = base_lin_vel / base_ang_vel")
    print("force_scheduled = 0")
    print(f"[STATE] {WAIT_CARRY}")

    policy_dt = float(env.dt)
    warmup_steps = _duration_steps(
        eval_args.steady_carry_warmup, policy_dt, allow_zero=True
    )
    steady_steps = _duration_steps(eval_args.steady_duration, policy_dt)
    phase = WAIT_CARRY
    warmup_count = 0
    steady_start_time_s = float("nan")
    samples = []
    termination_reason = ""

    rollout_steps = int(env.max_episode_length) + 2
    for step_id in range(rollout_steps):
        with torch.no_grad():
            actions = policy(obs.detach())
        step_result = env.step(actions.detach())
        obs, _, _, dones, _, termination_ids, _, _ = step_result
        policy_step = step_id + 1
        time_s = policy_step * policy_dt

        if bool(dones[0].item()):
            termination_reason = _termination_reason_from_done(env, termination_ids)
            phase = END
            break

        confirmed_carry = bool(env.confirmed_carry_buf[0].item())
        if phase == WAIT_CARRY and confirmed_carry:
            if samples:
                raise AssertionError("Velocity samples exist before STEADY_CARRY")
            print(f"[STATE] {WAIT_CARRY} -> {CONFIRMED_CARRY_WARMUP}")
            phase = CONFIRMED_CARRY_WARMUP
            warmup_count = 0
            if warmup_steps == 0:
                phase = STEADY_CARRY
                steady_start_time_s = time_s
                print(
                    f"[STATE] {CONFIRMED_CARRY_WARMUP} -> {STEADY_CARRY}"
                )
                print(
                    f"[STEADY] collecting {eval_args.steady_duration:.2f} s "
                    "velocity tracking"
                )

        elif phase == CONFIRMED_CARRY_WARMUP:
            if not confirmed_carry:
                if samples:
                    raise AssertionError("Velocity samples exist during carry warmup")
                print(f"[STATE] {CONFIRMED_CARRY_WARMUP} -> {WAIT_CARRY}")
                phase = WAIT_CARRY
                warmup_count = 0
            else:
                warmup_count += 1
                if warmup_count >= warmup_steps:
                    if samples:
                        raise AssertionError("Velocity samples exist before steady carry")
                    phase = STEADY_CARRY
                    steady_start_time_s = time_s
                    print(
                        f"[STATE] {CONFIRMED_CARRY_WARMUP} -> {STEADY_CARRY}"
                    )
                    print(
                        f"[STEADY] collecting {eval_args.steady_duration:.2f} s "
                        "velocity tracking"
                    )

        if phase == STEADY_CARRY:
            samples.append(
                sample_velocity_tracking(
                    env,
                    time_s=time_s,
                    policy_step=policy_step,
                )
            )
        elif samples:
            raise AssertionError("Velocity samples may only be recorded in STEADY_CARRY")

        failure_reason = _latched_failure_reason(env)
        if failure_reason:
            termination_reason = failure_reason
            phase = END
            break
        if phase == STEADY_CARRY and len(samples) >= steady_steps:
            termination_reason = "steady_carry_complete"
            print(f"[STATE] {STEADY_CARRY} -> {END}")
            phase = END
            break

    if not termination_reason:
        termination_reason = "rollout_limit"
    if not samples:
        print(f"[STATE] {END} without {STEADY_CARRY}")

    _assert_no_force_state(env)
    summary = summarize_velocity_tracking(
        trial_id="T0001",
        checkpoint=checkpoint,
        seed=seed,
        raw_command=command,
        samples=samples,
        policy_dt=policy_dt,
        steady_carry_start_time_s=steady_start_time_s,
        final_confirmed_carry=_final_confirmed_carry(env),
        carry_achieved=_summary_bool(env, "clean_eval_carry_achieved_buf"),
        humanoid_failure=_summary_bool(env, "clean_eval_humanoid_failure_buf"),
        humanoid_failure_reason=env.summary_reason(
            "clean_eval_humanoid_failure_reason", env_id=0
        ),
        box_failure=_summary_bool(env, "clean_eval_box_failure_buf"),
        box_failure_reason=env.summary_reason(
            "clean_eval_box_failure_reason", env_id=0
        ),
        timeout=_summary_bool(env, "clean_eval_timeout_buf"),
        termination_reason=termination_reason,
        force_scheduled=int(_summary_scalar(env, "clean_eval_event_count_buf") > 0),
        final_base_yaw_rad=_final_base_yaw(env),
    )
    if summary["force_scheduled"] != 0:
        raise AssertionError("No-force rollout ended with force_scheduled != 0")

    print("[RESULT]")
    for key in (
        "steady_carry_achieved",
        "steady_carry_duration_s",
        "steady_carry_steps",
        "policy_vx_mean",
        "actual_vx_mean",
        "vx_mae",
        "vx_rmse",
        "policy_vy_mean",
        "actual_vy_mean",
        "vy_mae",
        "yaw_rate_mae",
        "lin_vel_error_norm_mean",
        "confirmed_carry_fraction_steady",
        "force_scheduled",
        "termination_reason",
    ):
        print(f"{key}={summary[key]}")
    return samples, summary


def play(eval_args, legged_args):
    if legged_args.resume_path is None:
        raise ValueError("evaluator_NForce.py requires --resume_path.")

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
