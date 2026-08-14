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

from configs.evaluation_config import FIXED_COMMAND, apply_evaluation_config  # noqa: E402
from envs.carrybox_perturb_env import LeggedRobot as CarryBoxCleanPerturbEnv  # noqa: E402
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
    VALID_DIRECTIONS,
    VALID_PROFILES,
)
from evaluation.logger import EvaluationCsvLogger  # noqa: E402
from evaluation.metrics import sample_policy_metrics, summarize_trial  # noqa: E402
from evaluation.trial import generate_sweep, make_single_trial  # noqa: E402
from legged_gym import LEGGED_GYM_ROOT_DIR  # noqa: E402
from legged_gym.envs.g1.carrybox import LeggedRobot as CarryBoxBase  # noqa: E402
from legged_gym.envs.g1.carrybox_config import G1Cfg, G1CfgPPO  # noqa: E402
from legged_gym.utils import get_args, task_registry  # noqa: E402
from legged_gym.utils.helpers import set_seed  # noqa: E402


CLEAN_SOURCE_TASK = "carrybox_clean_source"
EVAL_TASK = "carrybox_clean_perturb_eval"
EXPECTED_ACTOR_INPUT_DIM = 738
EXPECTED_TASK_OBS_DIM = 15


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


def _resolve_checkpoint_path(checkpoint_path):
    if checkpoint_path is None:
        raise ValueError("evaluate.py requires --resume_path.")
    resolved_path = os.path.expanduser(
        checkpoint_path.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
    )
    if not os.path.isfile(resolved_path):
        raise FileNotFoundError(f"Checkpoint not found: {resolved_path}")
    return resolved_path


def load_actor_only_for_inference(ppo_runner, checkpoint_path, device):
    checkpoint_path = _resolve_checkpoint_path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint does not contain 'model_state_dict': {checkpoint_path}")

    checkpoint_state = checkpoint["model_state_dict"]
    actor_input_dim = checkpoint_state["actor.0.weight"].shape[1]
    if int(actor_input_dim) != EXPECTED_ACTOR_INPUT_DIM:
        raise AssertionError(
            f"Expected checkpoint actor input {EXPECTED_ACTOR_INPUT_DIM}, got {actor_input_dim}"
        )

    actor_state = {
        key: value
        for key, value in checkpoint_state.items()
        if key.startswith("actor.") or key == "std"
    }
    if not any(key.startswith("actor.") for key in actor_state):
        raise RuntimeError(f"Checkpoint has no actor parameters: {checkpoint_path}")

    actor_critic = ppo_runner.alg.actor_critic
    current_state = actor_critic.state_dict()
    current_actor_input_dim = current_state["actor.0.weight"].shape[1]
    if int(current_actor_input_dim) != EXPECTED_ACTOR_INPUT_DIM:
        raise AssertionError(
            "Current policy actor input mismatch: "
            f"expected {EXPECTED_ACTOR_INPUT_DIM}, got {current_actor_input_dim}"
        )

    shape_mismatches = {}
    for key, value in actor_state.items():
        if key not in current_state:
            shape_mismatches[key] = (tuple(value.shape), None)
        elif value.shape != current_state[key].shape:
            shape_mismatches[key] = (tuple(value.shape), tuple(current_state[key].shape))
    if shape_mismatches:
        raise RuntimeError(
            "Checkpoint actor is incompatible with the current policy network: "
            f"{shape_mismatches}"
        )

    incompatible = actor_critic.load_state_dict(actor_state, strict=False)
    missing_required = [
        key
        for key in incompatible.missing_keys
        if not key.startswith("critic.") and key != "std"
    ]
    if missing_required or incompatible.unexpected_keys:
        raise RuntimeError(
            "Actor-only checkpoint load was incomplete: "
            f"missing={missing_required}, unexpected={incompatible.unexpected_keys}"
        )

    print(
        "[ASSERT] checkpoint Actor loads successfully "
        f"(actor_input_dim={actor_input_dim}, current_actor_input_dim={current_actor_input_dim})"
    )


def _register_clean_source_task():
    task_registry.register(CLEAN_SOURCE_TASK, CarryBoxBase, G1Cfg(), G1CfgPPO())


def _register_eval_task(env_cfg, train_cfg):
    task_registry.register(EVAL_TASK, CarryBoxCleanPerturbEnv, env_cfg, train_cfg)


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


def _print_trial_header(condition, no_force):
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
    print(f"no_force={bool(no_force)}")
    print("=" * 60)


def _policy_step(env, policy, obs):
    with torch.no_grad():
        actions = policy(obs.detach())
    return env.step(actions.detach())


def _reset_for_trial(env, seed):
    set_seed(int(seed))
    if env.clean_eval_trace_enabled:
        env.end_trace()
    obs, _ = env.reset()
    env.clear_summary_snapshot(env_id=0)
    return obs


def _phase_steps(seconds, policy_dt):
    return max(1, int(round(float(seconds) / float(policy_dt))))


def _current_task_obs(env, obs):
    return obs[0, -int(env.num_task_obs):]


def assert_startup_compatibility(env, obs):
    if int(env.num_task_obs) != EXPECTED_TASK_OBS_DIM:
        raise AssertionError(f"Expected task obs dim 15, got {env.num_task_obs}")
    if int(env.actor_obs_length) != EXPECTED_ACTOR_INPUT_DIM:
        raise AssertionError(
            f"Expected actor obs dim 738, got {env.actor_obs_length}"
        )
    if tuple(obs.shape) != (env.num_envs, EXPECTED_ACTOR_INPUT_DIM):
        raise AssertionError(f"Unexpected obs shape: {tuple(obs.shape)}")
    task_obs = _current_task_obs(env, obs)
    if task_obs.numel() != EXPECTED_TASK_OBS_DIM:
        raise AssertionError(f"Expected current task obs length 15, got {task_obs.numel()}")
    command = task_obs[-3:]
    expected = torch.tensor(FIXED_COMMAND, dtype=command.dtype, device=command.device)
    if not torch.allclose(command, expected, atol=1.0e-6, rtol=0.0):
        raise AssertionError(
            "Current task observation command mismatch: "
            f"expected={FIXED_COMMAND}, got={command.detach().cpu().tolist()}"
        )
    forbidden_goal_attrs = (
        "goal_pos",
        "goal_rot",
        "robot2goal_dir",
        "robot2goal_dist",
        "object2goal_pos",
        "object2goal_dist_xy",
        "object2goal_dist_xyz",
    )
    present = [name for name in forbidden_goal_attrs if hasattr(env, name)]
    if present:
        raise AssertionError(f"Goal/relocation evaluator attributes present: {present}")
    print(f"[ASSERT] Actor input dimension = {env.actor_obs_length}")
    print(f"[ASSERT] task obs dimension = {env.num_task_obs}")
    print(
        "[ASSERT] current task obs env0 = "
        f"{[round(float(v), 6) for v in task_obs.detach().cpu().tolist()]}"
    )
    print(
        "[ASSERT] last 3 task-obs values = "
        f"{[round(float(v), 6) for v in command.detach().cpu().tolist()]}"
    )
    print("[ASSERT] no goal-conditioned task observation")


def _direction_world_from_box(env, direction_name, env_id=0):
    local = torch.zeros((1, 3), dtype=torch.float, device=env.device)
    component = 0 if direction_name[-1] == "x" else 1
    local[0, component] = 1.0 if direction_name[0] == "+" else -1.0
    world = quat_rotate_for_eval(env, local, env_id)
    return world / torch.clamp(torch.linalg.vector_norm(world), min=1.0e-6)


def quat_rotate_for_eval(env, local, env_id):
    from isaacgym.torch_utils import quat_rotate

    return quat_rotate(env.box_states[env_id, 3:7].unsqueeze(0), local)[0]


def run_trial(env, policy, condition, checkpoint, eval_args):
    _print_trial_header(condition, eval_args.no_force)
    obs = _reset_for_trial(env, condition.seed)
    assert_startup_compatibility(env, obs)

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
            }
        )

    pre_force_steps = _phase_steps(eval_args.pre_force_delay, env.dt)
    post_force_steps = _phase_steps(eval_args.post_force_observation, env.dt)
    phase = "WAIT_CARRY"
    pre_count = 0
    post_count = 0
    force_info = None
    force_start = {
        "box_pos": env.box_states[0, 0:3].detach().clone(),
        "robot_pos": env.root_states[0, 0:3].detach().clone(),
    }
    direction_world_t = _direction_world_from_box(env, condition.direction)
    samples = []
    failure = {"physical_failure": False, "termination_reason": ""}
    printed_confirmed = False

    if eval_args.no_force:
        env.cfg.clean_perturbation.enabled = False
        print("[NO_FORCE] External force scheduling disabled.")
    else:
        env.cfg.clean_perturbation.enabled = True

    for _ in range(int(env.max_episode_length) + post_force_steps + 10):
        obs, _, _, dones, _, termination_ids, _, _ = _policy_step(env, policy, obs)
        done = bool(dones[0].item())
        if done:
            failure["physical_failure"] = True
            reason = env.clean_eval_last_termination_reason[0]
            if not reason and termination_ids.numel() > 0:
                reason = "termination"
            failure["termination_reason"] = reason or "done"
            break

        samples.append(sample_policy_metrics(env, direction_world_t, force_start, phase))

        if phase == "WAIT_CARRY" and bool(env.confirmed_carry_buf[0].item()):
            print("[STATE] WAIT_CARRY -> CONFIRMED_CARRY")
            printed_confirmed = True
            if eval_args.no_force:
                phase = "CONFIRMED_CARRY"
            else:
                print("[STATE] CONFIRMED_CARRY -> PRE_FORCE")
                phase = "PRE_FORCE"
                pre_count = 0
                env.set_trace_phase("pre_force")

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
                    print(f"direction={condition.direction}")
                    print(
                        "world_direction="
                        f"{tuple(round(x, 6) for x in force_info['direction_world'])}"
                    )
                    print(f"beta={condition.beta:.3f}")
                    print(f"box_mass={force_info['box_mass_kg']:.6f}kg")
                    print(f"target_force={force_info['target_force_N']:.6f}N")
                    print(
                        "force_duration="
                        f"{force_info['measured_duration_s']:.6f}s "
                        f"({force_info['physics_steps']} physics steps)"
                    )
                    print("[ASSERT] force direction_world frozen at event commitment")
                    print("[ASSERT] force acts at box COM via rigid-body force tensor")
                    print("[ASSERT] GLOBAL_SPACE is used")
                    print("[ASSERT] force is applied before every gym.simulate substep")
                    phase = "FORCE"

        elif phase == "FORCE":
            if int(env.clean_eval_remaining_physics_steps[0].item()) == 0:
                env.set_trace_phase("post_force")
                print("[STATE] FORCE -> POST_FORCE")
                print("[FORCE OFF]")
                print(f"duration={float(env.clean_eval_force_duration_s[0].item()):.6f}s")
                print(f"impulse={float(env.clean_eval_impulse_Ns[0].item()):.6f}Ns")
                phase = "POST_FORCE"
                post_count = 0

        elif phase == "POST_FORCE":
            post_count += 1
            if post_count >= post_force_steps:
                print("[STATE] POST_FORCE -> END")
                break

    if eval_args.no_force and not printed_confirmed:
        print("[STATE] END without CONFIRMED_CARRY")

    trace_rows = env.end_trace() if eval_args.save_csv else []
    summary = summarize_trial(
        condition=condition,
        checkpoint=checkpoint,
        signature=signature,
        samples=samples,
        env=env,
        failure=failure,
    )
    print("[RESULT]")
    print(f"physical_failure={'yes' if failure['physical_failure'] else 'no'}")
    print(f"termination_reason={summary['termination_reason']}")
    print(f"contact_loss={summary['contact_loss']}")
    print(f"box_displacement_along_force={summary['box_displacement_along_force']}")
    print(f"robot_displacement_along_force={summary['robot_displacement_along_force']}")
    print(f"max_hand_box_relative_speed={summary['max_hand_box_relative_speed']}")
    print(f"final_confirmed_carry={summary['final_confirmed_carry']}")
    print(f"recovery_success={summary['recovery_success']}")
    print(f"recovery_time={summary['recovery_time']}")
    return obs, trace_rows, summary


def play(eval_args, legged_args):
    _register_clean_source_task()
    env_cfg, train_cfg = task_registry.get_cfgs(name=CLEAN_SOURCE_TASK)
    env_cfg = apply_evaluation_config(
        env_cfg,
        verbose=eval_args.verbose,
        trace_enabled=eval_args.save_csv,
    )
    if eval_args.no_force:
        env_cfg.clean_perturbation.enabled = False
        env_cfg.clean_perturbation.debug_draw_force = False

    train_cfg.runner.resume = False
    legged_args.resume = False
    legged_args.num_envs = 1
    legged_args.task = EVAL_TASK

    _register_eval_task(env_cfg, train_cfg)
    env, _ = task_registry.make_env(name=EVAL_TASK, args=legged_args, env_cfg=env_cfg)
    print("[CONFIG] clean CarryBox source registered through task_registry")
    print("[CONFIG] evaluator env inherits directly from clean carrybox.LeggedRobot")
    print(f"[CONFIG] fixed command={FIXED_COMMAND}")
    print(
        "[CONFIG] Stage-1 training command range was "
        "vx in [0.4,0.8], vy=0, yaw=0; evaluator uses vx=0.6."
    )

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

    obs = None
    for condition in trials:
        obs, trace_rows, summary = run_trial(
            env, policy, condition, legged_args.resume_path, eval_args
        )
        if logger is not None:
            logger.write_trace(condition.trial_id, trace_rows)
            logger.append_summary(summary)
    return obs


if __name__ == "__main__":
    evaluator_args = parse_evaluator_args()
    args = get_args()
    play(evaluator_args, args)
