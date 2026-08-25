import argparse
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

from configs.nominal_clean_config import (  # noqa: E402
    DEFAULT_NOMINAL_OBSERVATION_S,
    NOMINAL_CLEAN_COMMAND,
    apply_nominal_clean_config,
)
from envs.carrybox_nominal_clean_env import LeggedRobot as NominalCleanCarryBoxEnv  # noqa: E402
from evaluate import (  # noqa: E402
    CLEAN_SOURCE_TASK,
    assert_startup_compatibility,
    load_actor_only_for_inference,
    quat_rotate_for_eval,
)
from evaluation.force_profiles import (  # noqa: E402
    DEFAULT_BETAS,
    DEFAULT_DIRECTIONS,
    DEFAULT_HOLD_DURATION_S,
    DEFAULT_HOLD_DURATIONS,
    DEFAULT_POST_FORCE_OBSERVATION_S,
    DEFAULT_PRE_FORCE_DELAY_S,
    DEFAULT_PULSE_DURATION_S,
    DEFAULT_RAMP_DOWN_S,
    DEFAULT_RAMP_UP_S,
    DEFAULT_SEEDS,
    VALID_DIRECTIONS,
    VALID_PROFILES,
)
from evaluation.commands import set_fixed_evaluation_command  # noqa: E402
from evaluation.logger import EvaluationCsvLogger  # noqa: E402
from evaluation.metrics import sample_policy_metrics, summarize_trial  # noqa: E402
from evaluation.trial import generate_sweep, make_single_trial  # noqa: E402
from legged_gym import LEGGED_GYM_ROOT_DIR  # noqa: E402
from legged_gym.envs.g1.carrybox import LeggedRobot as CarryBoxBase  # noqa: E402
from legged_gym.envs.g1.carrybox_config import G1Cfg, G1CfgPPO  # noqa: E402
from legged_gym.utils import get_args, task_registry  # noqa: E402
from legged_gym.utils.helpers import set_seed  # noqa: E402


NOMINAL_CLEAN_TASK = "carrybox_nominal_clean_eval"


def _parse_float_list(text):
    return tuple(float(item) for item in str(text).split(",") if item != "")


def _parse_int_list(text):
    return tuple(int(item) for item in str(text).split(",") if item != "")


def _parse_str_list(text):
    return tuple(item.strip() for item in str(text).split(",") if item.strip())


def _parse_command(text):
    values = _parse_float_list(text)
    if len(values) != 3:
        raise argparse.ArgumentTypeError("--command must contain vx,vy,yaw_rate")
    return values


def parse_evaluator_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--command", type=_parse_command, default=NOMINAL_CLEAN_COMMAND)
    parser.add_argument(
        "--nominal_observation",
        type=float,
        default=DEFAULT_NOMINAL_OBSERVATION_S,
        help="Seconds to observe after confirmed carry when --with_force is not set.",
    )
    parser.add_argument("--episode_length", type=float, default=40.0)
    parser.add_argument("--with_force", action="store_true", default=False)
    parser.add_argument("--no_force", action="store_true", default=False)
    parser.add_argument("--profile", choices=VALID_PROFILES, default="half_sine")
    parser.add_argument("--direction", choices=VALID_DIRECTIONS, default="+box_x")
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--pulse_duration", type=float, default=DEFAULT_PULSE_DURATION_S)
    parser.add_argument("--hold_duration", type=float, default=DEFAULT_HOLD_DURATION_S)
    parser.add_argument("--ramp_up", type=float, default=DEFAULT_RAMP_UP_S)
    parser.add_argument("--ramp_down", type=float, default=DEFAULT_RAMP_DOWN_S)
    parser.add_argument("--pre_force_delay", type=float, default=DEFAULT_PRE_FORCE_DELAY_S)
    parser.add_argument(
        "--post_force_observation",
        type=float,
        default=DEFAULT_POST_FORCE_OBSERVATION_S,
    )
    parser.add_argument("--sweep", action="store_true", default=False)
    parser.add_argument("--save_csv", action="store_true", default=False)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--directions", type=_parse_str_list, default=DEFAULT_DIRECTIONS)
    parser.add_argument("--betas", type=_parse_float_list, default=DEFAULT_BETAS)
    parser.add_argument("--seeds", type=_parse_int_list, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--hold_durations",
        type=_parse_float_list,
        default=DEFAULT_HOLD_DURATIONS,
    )
    eval_args, remaining = parser.parse_known_args()
    if eval_args.no_force:
        eval_args.with_force = False
    sys.argv = [sys.argv[0], *remaining]
    return eval_args


def _register_clean_source_task():
    task_registry.register(CLEAN_SOURCE_TASK, CarryBoxBase, G1Cfg(), G1CfgPPO())


def _register_nominal_task(env_cfg, train_cfg):
    task_registry.register(NOMINAL_CLEAN_TASK, NominalCleanCarryBoxEnv, env_cfg, train_cfg)


def _checkpoint_label(path):
    if not path:
        return "checkpoint"
    stem = os.path.splitext(os.path.basename(path))[0]
    parent = os.path.basename(os.path.dirname(path))
    return f"{parent}_{stem}" if parent else stem


def _default_output_dir(resume_path):
    return os.path.join(
        EXPERIMENT_DIR,
        "results",
        "nominal_clean",
        _checkpoint_label(resume_path),
        "metrics_v2",
    )


def _make_trials(eval_args, legged_args):
    seed = 1 if legged_args.seed is None else int(legged_args.seed)
    if eval_args.sweep:
        return generate_sweep(
            profile=eval_args.profile,
            directions=eval_args.directions,
            betas=eval_args.betas,
            seeds=eval_args.seeds,
            hold_durations=eval_args.hold_durations,
            pulse_duration_s=eval_args.pulse_duration,
            ramp_up_s=eval_args.ramp_up,
            ramp_down_s=eval_args.ramp_down,
        )
    return [
        make_single_trial(
            profile=eval_args.profile,
            direction=eval_args.direction,
            beta=eval_args.beta,
            seed=seed,
            pulse_duration_s=eval_args.pulse_duration,
            hold_duration_s=eval_args.hold_duration,
            ramp_up_s=eval_args.ramp_up,
            ramp_down_s=eval_args.ramp_down,
        )
    ]


def _policy_action(policy, obs):
    with torch.no_grad():
        return policy(obs.detach())


def _policy_step(env, policy, obs):
    actions = _policy_action(policy, obs)
    return actions, env.step(actions.detach())


def _phase_steps(seconds, policy_dt):
    return max(1, int(round(float(seconds) / float(policy_dt))))


def _reset_for_trial(env, seed):
    set_seed(int(seed))
    if env.clean_eval_trace_enabled:
        env.end_trace()
    env.reset_evaluation_trial_state(clear_actor_history=True)
    env.reset()
    obs = set_fixed_evaluation_command(env, env.cfg.nominal_clean.command)
    env.clear_summary_snapshot(env_id=0)
    return obs


def _direction_world_from_box(env, direction_name, env_id=0):
    local = torch.zeros((1, 3), dtype=torch.float, device=env.device)
    component = 0 if direction_name[-1] == "x" else 1
    local[0, component] = 1.0 if direction_name[0] == "+" else -1.0
    world = quat_rotate_for_eval(env, local, env_id)
    return world / torch.clamp(torch.linalg.vector_norm(world), min=1.0e-6)


def _current_yaw(env, env_id=0):
    return float(env.yaw[env_id].reshape(-1)[0].item())


def _print_trial_header(condition, command, with_force):
    print("=" * 60)
    print(f"Trial {condition.trial_id}")
    print(f"seed={condition.seed}")
    print(f"command={command}")
    print(f"with_force={bool(with_force)}")
    if with_force:
        print(f"profile={condition.profile}")
        print(f"direction={condition.direction}")
        print(f"beta={condition.beta:.3f}")
        if condition.profile == "smooth_hold":
            print(f"hold={condition.hold_duration_s:.3f}s")
        else:
            print(f"pulse={condition.pulse_duration_s:.3f}s")
    print("=" * 60)


def run_nominal_trial(env, policy, condition, checkpoint, eval_args):
    _print_trial_header(condition, eval_args.command, eval_args.with_force)
    obs = _reset_for_trial(env, condition.seed)
    assert_startup_compatibility(env, obs, expected_command=eval_args.command)

    signature = env.evaluation_initial_state_signature(env_id=0)
    if eval_args.save_csv:
        print(f"[SIGNATURE] sha1={signature['sha1']}")
        env.begin_trace(
            metadata={
                "trial_id": condition.trial_id,
                "checkpoint": checkpoint,
                "profile": condition.profile,
                "direction": condition.direction,
                "seed": condition.seed,
                "hold_duration_s": condition.hold_duration_s,
                "pulse_duration_s": condition.pulse_duration_s,
                "ramp_up_s": condition.ramp_up_s,
                "ramp_down_s": condition.ramp_down_s,
                "nominal_clean": 1,
            }
        )

    env.cfg.clean_perturbation.enabled = bool(eval_args.with_force)
    phase = "WAIT_CARRY"
    pre_force_steps = _phase_steps(eval_args.pre_force_delay, env.dt)
    nominal_steps = _phase_steps(eval_args.nominal_observation, env.dt)
    post_force_steps = _phase_steps(eval_args.post_force_observation, env.dt)
    pre_count = 0
    nominal_count = 0
    post_count = 0
    samples = []
    force_start = {
        "box_pos": env.box_states[0, 0:3].detach().clone(),
        "robot_pos": env.root_states[0, 0:3].detach().clone(),
        "robot_yaw": _current_yaw(env),
    }
    direction_world_t = _direction_world_from_box(env, condition.direction)
    termination = {"termination_reason": ""}
    printed_confirmed = False

    rollout_steps = int(env.max_episode_length) + post_force_steps + 10
    for _ in range(rollout_steps):
        _, step_result = _policy_step(env, policy, obs)
        obs, _, _, dones, _, termination_ids, _, _ = step_result
        if bool(dones[0].item()):
            reason = env.clean_eval_last_termination_reason[0]
            if not reason and termination_ids.numel() > 0:
                reason = "termination"
            termination["termination_reason"] = reason or "done"
            break

        samples.append(sample_policy_metrics(env, direction_world_t, force_start, phase.lower()))

        if phase == "WAIT_CARRY" and bool(env.confirmed_carry_buf[0].item()):
            printed_confirmed = True
            force_start = {
                "box_pos": env.box_states[0, 0:3].detach().clone(),
                "robot_pos": env.root_states[0, 0:3].detach().clone(),
                "robot_yaw": _current_yaw(env),
            }
            env.set_force_start_reference(env_id=0)
            direction_world_t = _direction_world_from_box(env, condition.direction)
            print("[STATE] WAIT_CARRY -> CONFIRMED_CARRY")
            if eval_args.with_force:
                print("[STATE] CONFIRMED_CARRY -> PRE_FORCE")
                phase = "PRE_FORCE"
                pre_count = 0
                env.set_trace_phase("pre_force")
            else:
                print("[STATE] CONFIRMED_CARRY -> NOMINAL_LOCOMOTION")
                phase = "NOMINAL_LOCOMOTION"
                nominal_count = 0
                env.set_trace_phase("nominal_locomotion")

        elif phase == "NOMINAL_LOCOMOTION":
            if not bool(env.confirmed_carry_buf[0].item()):
                print("[STATE] NOMINAL_LOCOMOTION -> WAIT_CARRY")
                phase = "WAIT_CARRY"
                nominal_count = 0
                env.set_trace_phase("wait_carry")
            else:
                nominal_count += 1
                if nominal_count >= nominal_steps:
                    print("[STATE] NOMINAL_LOCOMOTION -> END")
                    break

        elif phase == "PRE_FORCE":
            if not bool(env.confirmed_carry_buf[0].item()):
                print("[STATE] PRE_FORCE -> WAIT_CARRY")
                phase = "WAIT_CARRY"
                pre_count = 0
                env.set_trace_phase("wait_carry")
            else:
                pre_count += 1
                if pre_count >= pre_force_steps:
                    force_start = {
                        "box_pos": env.box_states[0, 0:3].detach().clone(),
                        "robot_pos": env.root_states[0, 0:3].detach().clone(),
                        "robot_yaw": _current_yaw(env),
                    }
                    env.set_force_start_reference(env_id=0)
                    force_info = env.schedule_evaluation_force(
                        condition.direction,
                        condition.beta,
                        condition.profile,
                        pulse_duration_s=condition.pulse_duration_s,
                        hold_duration_s=condition.hold_duration_s,
                        ramp_up_s=condition.ramp_up_s,
                        ramp_down_s=condition.ramp_down_s,
                    )
                    direction_world_t = torch.tensor(
                        force_info["direction_world"],
                        dtype=torch.float,
                        device=env.device,
                    )
                    print("[STATE] PRE_FORCE -> FORCE")
                    print("[FORCE ON]")
                    print(
                        "world_direction="
                        f"{tuple(round(x, 6) for x in force_info['direction_world'])}"
                    )
                    print(f"target_force={force_info['target_force_N']:.6f}N")
                    print("[ASSERT] GLOBAL_SPACE is used")
                    phase = "FORCE"

        elif phase == "FORCE":
            if int(env.clean_eval_remaining_physics_steps[0].item()) == 0:
                env.set_trace_phase("post_force")
                print("[STATE] FORCE -> POST_FORCE")
                print("[FORCE OFF]")
                print(f"impulse={float(env.clean_eval_impulse_Ns[0].item()):.6f}Ns")
                phase = "POST_FORCE"
                post_count = 0

        elif phase == "POST_FORCE":
            post_count += 1
            if post_count >= post_force_steps:
                print("[STATE] POST_FORCE -> END")
                break

    if not printed_confirmed:
        print("[STATE] END without CONFIRMED_CARRY")

    trace_rows = env.end_trace() if eval_args.save_csv else []
    summary = summarize_trial(
        condition=condition,
        checkpoint=checkpoint,
        signature=signature,
        samples=samples,
        env=env,
        termination=termination,
    )
    print("[RESULT]")
    print(f"humanoid_failure={summary['humanoid_failure']}")
    print(f"box_failure={summary['box_failure']}")
    print(f"timeout={summary['timeout']}")
    print(f"termination_reason={summary['termination_reason']}")
    print(f"final_confirmed_carry={summary['final_confirmed_carry']}")
    print(f"command={eval_args.command}")
    print(f"final_yaw_delta={summary['final_base_yaw_delta_from_start_rad']}")
    print(f"mean_body_yaw_rate={summary['base_yaw_rate_body_nominal_mean']}")
    print(f"mean_abs_body_yaw_rate={summary['base_abs_yaw_rate_body_nominal_mean']}")
    print(f"integrated_body_yaw_rate={summary['base_yaw_rate_integral_nominal_rad']}")
    print(
        "robot_lateral_displacement_from_confirmed="
        f"{summary['robot_lateral_displacement_from_start']}"
    )
    return obs, trace_rows, summary


def play(eval_args, legged_args):
    command = tuple(float(value) for value in eval_args.command)
    eval_args.command = command

    _register_clean_source_task()
    env_cfg, train_cfg = task_registry.get_cfgs(name=CLEAN_SOURCE_TASK)
    env_cfg = apply_nominal_clean_config(
        env_cfg,
        verbose=eval_args.verbose,
        trace_enabled=eval_args.save_csv,
        command=command,
        episode_length_s=eval_args.episode_length,
    )
    env_cfg.nominal_clean.command = command
    train_cfg.runner.resume = False
    legged_args.resume = False
    legged_args.num_envs = env_cfg.env.num_envs
    legged_args.task = NOMINAL_CLEAN_TASK

    _register_nominal_task(env_cfg, train_cfg)
    env, _ = task_registry.make_env(
        name=NOMINAL_CLEAN_TASK,
        args=legged_args,
        env_cfg=env_cfg,
    )
    print("[CONFIG] nominal-clean evaluator inherits experiment perturb env")
    print("[CONFIG] training CarryBox env/config/task registration unchanged")
    print("[CONFIG] full task path: default stand -> approach -> pickup -> carry")
    print("[CONFIG] training uses dynamic carry-command resampling")
    print("[CONFIG] evaluator disables carry-command resampling")
    print(f"[CONFIG] fixed body-frame command={command}")
    print("[CONFIG] domain randomization, reset randomization, and obs noise disabled")
    if not eval_args.with_force:
        print("[CONFIG] external force disabled; use --with_force to enable it")

    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env,
        name=NOMINAL_CLEAN_TASK,
        args=legged_args,
        train_cfg=train_cfg,
        log_root=None,
    )
    checkpoint_path = load_actor_only_for_inference(
        ppo_runner,
        legged_args.resume_path,
        device=env.device,
    )
    policy = ppo_runner.get_inference_policy(device=env.device)

    trials = _make_trials(eval_args, legged_args)
    logger = None
    if eval_args.save_csv:
        output_dir = eval_args.output_dir or _default_output_dir(legged_args.resume_path)
        logger = EvaluationCsvLogger(output_dir)
        print(f"[OUTPUT] {logger.output_dir}")

    obs = None
    for condition in trials:
        obs, trace_rows, summary = run_nominal_trial(
            env,
            policy,
            condition,
            legged_args.resume_path,
            eval_args,
        )
        if logger is not None:
            logger.write_trace(condition.trial_id, trace_rows)
            logger.append_summary(summary)
    return obs


if __name__ == "__main__":
    evaluator_args = parse_evaluator_args()
    args = get_args()
    if args.resume_path is None:
        raise ValueError("evaluate_nominal_clean.py requires --resume_path.")
    args.resume_path = os.path.expanduser(
        args.resume_path.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
    )
    play(evaluator_args, args)
