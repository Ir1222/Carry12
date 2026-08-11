import argparse
import os
import sys


EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(EXPERIMENT_DIR, "..", ".."))
for path in (
    REPO_ROOT,
    os.path.join(REPO_ROOT, "legged_gym"),
    os.path.join(REPO_ROOT, "rsl_rl"),
):
    if path not in sys.path:
        sys.path.insert(0, path)

import isaacgym  # noqa: F401,E402
import torch  # noqa: E402

from legged_gym.envs import *  # noqa: F401,F403,E402
from legged_gym.scripts.play_ActorOnly import (  # noqa: E402
    load_actor_only_for_inference,
)
from legged_gym.utils import get_args, task_registry  # noqa: E402
from legged_gym.utils.helpers import set_seed  # noqa: E402

from configs.evaluation_config import apply_evaluation_config  # noqa: E402
from envs.carrybox_perturb_eval import LeggedRobot as CarryBoxPerturbEval  # noqa: E402
from evaluation.force_profiles import (  # noqa: E402
    DEFAULT_BETAS,
    DEFAULT_DIRECTIONS,
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
from evaluation.logger import EvaluationCsvLogger  # noqa: E402
from evaluation.metrics import sample_policy_metrics, summarize_trial  # noqa: E402
from evaluation.trial import generate_sweep, make_single_trial  # noqa: E402


BASE_TASK = "carrybox_perturb"
EVAL_TASK = "carrybox_perturb_eval"


def _parse_float_list(text):
    return tuple(float(item) for item in str(text).split(",") if item != "")


def _parse_int_list(text):
    return tuple(int(item) for item in str(text).split(",") if item != "")


def _parse_str_list(text):
    return tuple(item.strip() for item in str(text).split(",") if item.strip())


def parse_evaluator_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--profile", choices=VALID_PROFILES, default="half_sine")
    parser.add_argument("--direction", choices=VALID_DIRECTIONS, default="+box_x")
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--pulse_duration", type=float, default=DEFAULT_PULSE_DURATION_S)
    parser.add_argument("--hold_duration", type=float, default=1.0)
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
    parser.add_argument("--no_force", action="store_true", default=False)
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
    sys.argv = [sys.argv[0], *remaining]
    return eval_args


def _base_task_from_args(task_name):
    if task_name in (BASE_TASK, EVAL_TASK, "carrybox_perturb_debug"):
        return BASE_TASK
    raise ValueError("evaluate.py only supports --task carrybox_perturb.")


def _register_eval_task(env_cfg, train_cfg):
    task_registry.register(EVAL_TASK, CarryBoxPerturbEval, env_cfg, train_cfg)


def _checkpoint_label(path):
    if not path:
        return "checkpoint"
    stem = os.path.splitext(os.path.basename(path))[0]
    parent = os.path.basename(os.path.dirname(path))
    return f"{parent}_{stem}" if parent else stem


def _default_output_dir(resume_path):
    return os.path.join(EXPERIMENT_DIR, "results", _checkpoint_label(resume_path))


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


def _print_trial_header(condition):
    print("=" * 60)
    print(f"Trial {condition.trial_id}")
    print(f"profile={condition.profile}")
    print(f"direction={condition.direction}")
    print(f"beta={condition.beta:.3f}")
    if condition.profile == "smooth_hold":
        print(f"hold={condition.hold_duration_s:.3f}s")
    else:
        print(f"pulse={condition.pulse_duration_s:.3f}s")
    print(f"seed={condition.seed}")
    print("=" * 60)


def _policy_step(env, policy, obs):
    env.commands[:, 0] = float(getattr(env.cfg.debug_command, "lin_vel_x", 0.4))
    env.commands[:, 1] = float(getattr(env.cfg.debug_command, "lin_vel_y", 0.0))
    env.commands[:, 2] = float(getattr(env.cfg.debug_command, "yaw_rate", 0.0))
    env.gym.fetch_results(env.sim, True)
    with torch.no_grad():
        actions = policy(obs.detach())
    return env.step(actions.detach())


def _short_values(tensor, count=10):
    values = tensor.detach().cpu().reshape(-1).tolist()
    return [round(float(value), 6) for value in values[:count]]


def _print_pre_rollout_snapshot(env, obs):
    cfg = env.cfg
    reset_mode = getattr(getattr(cfg.asset, "box", None), "reset_mode", "unknown")
    randomize_initial_joint_pos = getattr(
        cfg.domain_rand, "randomize_initial_joint_pos", "unknown"
    )
    obs_norm = float(torch.linalg.vector_norm(obs[0]).item())
    obs_digest = torch.sum(obs[0] * torch.arange(
        1, obs.shape[1] + 1, device=obs.device, dtype=obs.dtype
    ))
    print("[PRE_FORCE_SNAPSHOT]")
    print(f"box.reset_mode={reset_mode}")
    print(f"env.test={bool(cfg.env.test)}")
    print(f"domain_rand.disturbance={bool(cfg.domain_rand.disturbance)}")
    print(f"domain_rand.delay={bool(cfg.domain_rand.delay)}")
    print(f"domain_rand.push_robots={bool(cfg.domain_rand.push_robots)}")
    print(f"noise.add_noise={bool(cfg.noise.add_noise)}")
    print(f"randomize_initial_joint_pos={bool(randomize_initial_joint_pos)}")
    print(f"root_pos={_short_values(env.root_states[0, 0:3])}")
    print(f"root_quat={_short_values(env.root_states[0, 3:7])}")
    print(f"root_lin_vel={_short_values(env.root_states[0, 7:10])}")
    print(f"root_ang_vel={_short_values(env.root_states[0, 10:13])}")
    print(f"dof_pos_norm={float(torch.linalg.vector_norm(env.dof_pos[0]).item()):.6f}")
    print(f"dof_pos_first10={_short_values(env.dof_pos[0], count=10)}")
    print(f"dof_vel_norm={float(torch.linalg.vector_norm(env.dof_vel[0]).item()):.6f}")
    print(f"box_pose={_short_values(torch.cat((env.box_states[0, 0:3], env.box_states[0, 3:7])))}")
    print(f"goal={_short_values(env.goal_pos[0])}")
    print(f"obs_norm={obs_norm:.6f}")
    print(f"obs_digest={float(obs_digest.item()):.6f}")
    print(f"obs_first10={_short_values(obs[0], count=10)}")
    print(f"actor_history_shape={tuple(obs.shape)}")


def _reset_for_trial(env, seed):
    set_seed(int(seed))
    if env.box_perturb_trace_enabled:
        env.end_box_perturb_trace()
    env.reset()
    return env.get_observations()


def _phase_steps(seconds, policy_dt):
    return max(1, int(round(float(seconds) / float(policy_dt))))


def run_trial(env, policy, obs, condition, checkpoint, eval_args):
    _print_trial_header(condition)
    obs = _reset_for_trial(env, condition.seed)
    if eval_args.verbose:
        _print_pre_rollout_snapshot(env, obs)
    signature = env.evaluation_initial_state_signature(env_id=0)
    if eval_args.save_csv:
        print(f"[SIGNATURE] sha1={signature['sha1']}")

    trace_metadata = {
        "trial_id": condition.trial_id,
        "checkpoint": checkpoint,
        "profile": condition.profile,
        "direction": condition.direction,
        "seed": condition.seed,
        "hold_duration_s": condition.hold_duration_s,
        "ramp_up_s": condition.ramp_up_s,
        "ramp_down_s": condition.ramp_down_s,
    }
    if eval_args.save_csv:
        env.begin_box_perturb_trace(metadata=trace_metadata, verbose=eval_args.verbose)
        env.set_box_perturb_trace_phase("wait_carry")

    threshold = int(env.cfg.box_perturbation.stable_confirmed_carry_policy_steps)
    pre_force_steps = _phase_steps(eval_args.pre_force_delay, env.dt)
    post_force_steps = _phase_steps(eval_args.post_force_observation, env.dt)
    phase = "WAIT_CARRY"
    pre_count = 0
    post_count = 0
    force_info = None
    force_start = None
    direction_world_t = None
    samples = []
    failure = {"physical_failure": False, "termination_reason": ""}
    if eval_args.no_force:
        print("[NO_FORCE]")
        print("External force scheduling disabled; running nominal CarryBox rollout.")

    for _ in range(int(env.max_episode_length) + post_force_steps + 10):
        obs, _, _, dones, _, termination_ids, _, _ = _policy_step(env, policy, obs)
        done = bool(dones[0].item())
        if done:
            failure["physical_failure"] = True
            reason = env.box_perturb_last_termination_reason[0]
            if not reason and termination_ids.numel() > 0:
                reason = "termination"
            failure["termination_reason"] = reason or "done"
            break

        if eval_args.no_force:
            continue

        if phase == "WAIT_CARRY":
            if int(env.confirmed_carry_streak[0].item()) >= threshold:
                print("[STATE]")
                print("WAIT_CARRY -> CONFIRMED_CARRY")
                print("[STATE]")
                print("CONFIRMED_CARRY -> PRE_FORCE")
                phase = "PRE_FORCE"
                pre_count = 0
                if eval_args.save_csv:
                    env.set_box_perturb_trace_phase("pre")

        elif phase == "PRE_FORCE":
            if not bool(env.confirmed_carry_buf[0].item()):
                print("[STATE]")
                print("PRE_FORCE -> WAIT_CARRY")
                phase = "WAIT_CARRY"
                pre_count = 0
                if eval_args.save_csv:
                    env.set_box_perturb_trace_phase("wait_carry")
            else:
                pre_count += 1
                if pre_count >= pre_force_steps:
                    force_start = {
                        "box_pos": env.box_states[0, 0:3].detach().clone(),
                        "robot_pos": env.root_states[0, 0:3].detach().clone(),
                    }
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
                    print("[FORCE ON]")
                    print(f"direction={condition.direction}")
                    print(f"world_direction={tuple(round(x, 6) for x in force_info['direction_world'])}")
                    print(f"beta={condition.beta:.3f}")
                    print(f"box_mass={force_info['box_mass_kg']:.6f}kg")
                    print(f"target_force={force_info['target_force_N']:.6f}N")
                    if condition.profile == "smooth_hold":
                        print(f"ramp_up={condition.ramp_up_s:.3f}s")
                        print(f"hold={condition.hold_duration_s:.3f}s")
                        print(f"ramp_down={condition.ramp_down_s:.3f}s")
                    else:
                        print(f"pulse_duration={condition.pulse_duration_s:.3f}s")
                    phase = "FORCE"

        elif phase == "FORCE":
            if force_start is not None and direction_world_t is not None:
                samples.append(
                    sample_policy_metrics(env, direction_world_t, force_start)
                )
            if int(env.box_perturb_remaining_physics_steps[0].item()) == 0:
                if eval_args.save_csv:
                    env.set_box_perturb_trace_phase("post_force")
                print("[FORCE OFF]")
                print(f"duration={float(env.box_perturb_pulse_duration_s[0].item()):.6f}s")
                print(f"impulse={float(env.box_perturb_eval_impulse_Ns[0].item()):.6f}Ns")
                phase = "POST_FORCE"
                post_count = 0

        elif phase == "POST_FORCE":
            if force_start is not None and direction_world_t is not None:
                samples.append(
                    sample_policy_metrics(env, direction_world_t, force_start)
                )
            post_count += 1
            if post_count >= post_force_steps:
                break

    trace_rows = env.end_box_perturb_trace() if eval_args.save_csv else []
    summary = summarize_trial(
        condition=condition,
        checkpoint=checkpoint,
        signature=signature,
        trace_rows=trace_rows,
        samples=samples,
        env=env,
        failure=failure,
    )
    print("[RESULT]")
    print(f"physical_failure={'yes' if failure['physical_failure'] else 'no'}")
    print(f"contact_loss={summary['contact_loss']}")
    print(f"box_displacement_along_force={summary['box_displacement_along_force']}")
    print(f"robot_displacement_along_force={summary['robot_displacement_along_force']}")
    print(f"max_hand_box_relative_speed={summary['max_hand_box_relative_speed']}")
    print(f"final_confirmed_carry={summary['final_confirmed_carry']}")
    return obs, trace_rows, summary


def play(eval_args, legged_args):
    requested_task = legged_args.task
    if requested_task == "g1":
        print("[ARGS] --task was not provided; defaulting to carrybox_perturb.")
        requested_task = BASE_TASK
        legged_args.task = BASE_TASK
    base_task = _base_task_from_args(requested_task)

    env_cfg, train_cfg = task_registry.get_cfgs(name=base_task)
    env_cfg = apply_evaluation_config(
        env_cfg,
        verbose=eval_args.verbose,
        trace_enabled=eval_args.save_csv,
        draw_force=True,
    )

    train_cfg.runner.resume = False
    legged_args.resume = False
    legged_args.num_envs = 1

    _register_eval_task(env_cfg, train_cfg)
    env, _ = task_registry.make_env(name=EVAL_TASK, args=legged_args, env_cfg=env_cfg)
    print(f"[CONFIG] box.reset_mode={env.box_cfg.reset_mode}")
    obs = env.get_observations()

    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env,
        name=EVAL_TASK,
        args=legged_args,
        train_cfg=train_cfg,
        log_root=None,
    )
    load_actor_only_for_inference(ppo_runner, legged_args.resume_path, device=env.device)
    policy = ppo_runner.get_inference_policy(device=env.device)

    trials = _make_trials(eval_args, legged_args)
    logger = None
    if eval_args.save_csv:
        output_dir = eval_args.output_dir or _default_output_dir(legged_args.resume_path)
        logger = EvaluationCsvLogger(output_dir)
        print(f"[OUTPUT] {logger.output_dir}")

    for condition in trials:
        obs, trace_rows, summary = run_trial(
            env, policy, obs, condition, legged_args.resume_path, eval_args
        )
        if logger is not None:
            logger.write_trace(condition.trial_id, trace_rows)
            logger.append_summary(summary)


if __name__ == "__main__":
    evaluator_args = parse_evaluator_args()
    args = get_args()
    play(evaluator_args, args)
