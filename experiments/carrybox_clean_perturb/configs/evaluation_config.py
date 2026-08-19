"""Experiment-only config overrides for clean CarryBox perturbation evaluation."""

from types import SimpleNamespace

from evaluation.force_profiles import (
    DEFAULT_POST_FORCE_OBSERVATION_S,
    DEFAULT_PRE_FORCE_DELAY_S,
    DEFAULT_RECOVERY_CONFIRMED_CARRY_STEPS,
    DEFAULT_STABLE_CONFIRMED_CARRY_STEPS,
)


FIXED_COMMAND = (0.6, 0.0, 0.0)
PLAY_NOMINAL_COMMAND = (0.8, 0.0, 0.0)


def apply_evaluation_config(
    env_cfg,
    verbose=False,
    trace_enabled=False,
    play_nominal_parity=False,
):
    """Apply clean evaluator overrides before task_registry.make_env()."""
    if play_nominal_parity:
        env_cfg.env.num_envs = min(env_cfg.env.num_envs, 10)
        env_cfg.env.episode_length_s = 10
    else:
        env_cfg.env.num_envs = 1
        env_cfg.env.episode_length_s = 30
    env_cfg.env.test = True

    env_cfg.commands.curriculum = False
    env_cfg.commands.resampling_time = 0.0
    env_cfg.commands.resample_carry_commands = False
    env_cfg.commands.heading_command = False
    env_cfg.commands.heading_to_ang_vel = False
    env_cfg.commands.lin_vel_clip = 0.0
    env_cfg.commands.ang_vel_clip = 0.0
    if not play_nominal_parity:
        env_cfg.commands.ranges.lin_vel_x = [FIXED_COMMAND[0], FIXED_COMMAND[0]]
        env_cfg.commands.ranges.lin_vel_y = [FIXED_COMMAND[1], FIXED_COMMAND[1]]
        env_cfg.commands.ranges.ang_vel_yaw = [FIXED_COMMAND[2], FIXED_COMMAND[2]]
        env_cfg.commands.ranges.heading = [0.0, 0.0]

    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.delay = False
    env_cfg.domain_rand.push_robots = False

    if hasattr(env_cfg.asset, "box"):
        env_cfg.asset.box.random_props = False
        env_cfg.asset.box.reset_mode = "default"
        if not play_nominal_parity:
            env_cfg.asset.box.random_size = False
            env_cfg.asset.box.random_density = False

    env_cfg.clean_perturbation = SimpleNamespace(
        enabled=True,
        evaluation_trace_enabled=bool(trace_enabled),
        evaluation_verbose=bool(verbose),
        contact_force_threshold=1.0,
        max_box_rel_lin_vel=1.0,
        max_box_ang_vel=3.0,
        stable_confirmed_carry_policy_steps=DEFAULT_STABLE_CONFIRMED_CARRY_STEPS,
        recovery_confirmed_carry_steps=DEFAULT_RECOVERY_CONFIRMED_CARRY_STEPS,
        pre_force_delay_s=DEFAULT_PRE_FORCE_DELAY_S,
        post_force_observation_s=DEFAULT_POST_FORCE_OBSERVATION_S,
        max_events_per_episode=1,
        debug_draw_force=True,
        debug_force_draw_scale_m_per_N=1.0,
        debug_force_bundle_line_count=20,
        debug_force_bundle_jitter_m=0.01,
        debug_force_draw_max_envs=10,
    )
    return env_cfg
