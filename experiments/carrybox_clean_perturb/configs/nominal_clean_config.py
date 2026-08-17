"""Experiment-only config for deterministic nominal CarryBox evaluation."""

from types import SimpleNamespace

from configs.evaluation_config import FIXED_COMMAND, apply_evaluation_config


NOMINAL_CLEAN_COMMAND = FIXED_COMMAND
DEFAULT_NOMINAL_OBSERVATION_S = 30.0


def apply_nominal_clean_config(
    env_cfg,
    verbose=False,
    trace_enabled=False,
    command=NOMINAL_CLEAN_COMMAND,
    episode_length_s=40.0,
):
    """Apply deterministic evaluator overrides before task_registry.make_env()."""
    env_cfg = apply_evaluation_config(
        env_cfg,
        verbose=verbose,
        trace_enabled=trace_enabled,
        play_nominal_parity=False,
    )
    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = float(episode_length_s)
    env_cfg.env.test = True

    env_cfg.commands.ranges.lin_vel_x = [float(command[0]), float(command[0])]
    env_cfg.commands.ranges.lin_vel_y = [float(command[1]), float(command[1])]
    env_cfg.commands.ranges.ang_vel_yaw = [float(command[2]), float(command[2])]
    env_cfg.commands.ranges.heading = [0.0, 0.0]

    for name in (
        "randomize_actuation_offset",
        "randomize_motor_strength",
        "randomize_payload_mass",
        "randomize_com_displacement",
        "randomize_link_mass",
        "randomize_friction",
        "randomize_restitution",
        "randomize_kp",
        "randomize_kd",
        "randomize_initial_joint_pos",
        "disturbance",
        "delay",
        "push_robots",
    ):
        if hasattr(env_cfg.domain_rand, name):
            setattr(env_cfg.domain_rand, name, False)

    if hasattr(env_cfg, "noise"):
        env_cfg.noise.add_noise = False

    if hasattr(env_cfg.asset, "box"):
        env_cfg.asset.box.random_props = False
        env_cfg.asset.box.random_size = False
        env_cfg.asset.box.random_density = False
        env_cfg.asset.box.reset_mode = "default"

    env_cfg.clean_perturbation.debug_draw_force = False
    env_cfg.nominal_clean = SimpleNamespace(
        command=tuple(float(value) for value in command),
        fixed_box_x=float(env_cfg.asset.box.base_size[0]) / 2.0 + 0.4,
        fixed_box_y=0.0,
        box_vertical_clearance=0.01,
        fixed_camera_hfov_rad=0.5 * sum(env_cfg.asset.camera.hfov_rad),
        fixed_camera_vfov_rad=0.5 * sum(env_cfg.asset.camera.vfov_rad),
        fixed_camera_facing_angle=0.5 * sum(env_cfg.asset.camera.facing_angle),
        fixed_thresh_tag=max(env_cfg.asset.box.thresh_tag),
    )
    return env_cfg
