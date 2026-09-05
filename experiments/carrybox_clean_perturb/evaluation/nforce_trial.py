"""No-force rollout state machine, independent of Isaac Gym imports."""

import math

import torch

from evaluation.inference import assert_startup_compatibility
from evaluation.nforce_velocity import sample_velocity_tracking, summarize_velocity_tracking


WAIT_CARRY = "WAIT_CARRY"
CONFIRMED_CARRY_WARMUP = "CONFIRMED_CARRY_WARMUP"
STEADY_CARRY = "STEADY_CARRY"
END = "END"


def _duration_steps(seconds, policy_dt, allow_zero=False):
    steps = int(round(float(seconds) / float(policy_dt)))
    return max(0 if allow_zero else 1, steps)


def _yaw_scalar(env):
    return float(env.yaw[0].reshape(-1)[0].item())


def _reset_for_trial(env, seed, command, seed_fn):
    seed_fn(int(seed))
    env.reset_evaluation_trial_state(clear_actor_history=True)
    env.nforce_terminal_base_yaw_rad[:] = float("nan")
    # NominalCleanCarryBoxEnv restores the fixed raw command in _reset_task.
    # reset() advances one zero-action step and builds the observation using
    # this branch's vy suppression and heading feedback. Do not overwrite it
    # with raw commands or append an extra history frame here.
    obs, _ = env.reset()
    expected = env.commands.new_tensor(command).expand(env.num_envs, -1)
    if not torch.allclose(env.commands[:, :3], expected, atol=1.0e-6, rtol=0.0):
        raise AssertionError("Nominal reset did not preserve the fixed raw command")
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
    if bool(cfg.commands.resample_carry_commands) or cfg.commands.resampling_time != 0.0:
        raise AssertionError("No-force evaluator requires command resampling disabled")
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


def run_trial(env, policy, checkpoint, seed, command, eval_args, *, seed_fn):
    _print_trial_header(checkpoint, seed, command)
    obs = _reset_for_trial(env, seed, command, seed_fn)
    assert_startup_compatibility(
        env, obs, expected_command=env.carry_policy_commands[0, :3].tolist()
    )
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


