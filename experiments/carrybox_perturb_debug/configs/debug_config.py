"""Experiment-only config overrides for carrybox perturbation debugging."""

from types import SimpleNamespace


DEBUG_NUM_ENVS = 1
DEBUG_EPISODE_LENGTH_S = 15
# Trigger carry phase gate
DEBUG_STABLE_CARRY_STEPS = 5
# Egibility: True for constraint perturbation add
DEBUG_FORCE_EVENT = True
# Draw force vector in the scene
DEBUG_DRAW_FORCE = True
DEBUG_CARRY_GATE_LOG_INTERVAL_POLICY_STEPS = 5

# Experiment-only deterministic reset scene. Positions are env-local; the env
# origin is added by the debug subclass when writing simulator state tensors.
FIXED_SCENE_ENABLED = True
FIXED_ROBOT_POSITION = [2.3, 0.0, 0.8]
FIXED_ROBOT_ORIENTATION = [0.0, 0.0, 1.0, 0.0]
# Robot-local xyz offset. With the default 180-degree yaw this puts the box at
# env-local [0.55, 0.0, 0.135] without changing the parent-sampled box rotation.
FIXED_BOX_OFFSET_ROBOT_LOCAL = [1.75, 0.0, -0.665]
FIXED_GOAL_DISTANCE_M = 8.0
FIXED_GOAL_BEARING_DEG = 0.0


def apply_debug_config(env_cfg):
    """Apply debug/evaluation overrides before task_registry.make_env()."""
    env_cfg.env.num_envs = DEBUG_NUM_ENVS
    env_cfg.env.episode_length_s = DEBUG_EPISODE_LENGTH_S
    env_cfg.env.test = True

    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.delay = False
    env_cfg.domain_rand.push_robots = False

    if hasattr(env_cfg.asset, "box"):
        env_cfg.asset.box.random_props = False

    env_cfg.box_perturbation.enabled = True
    env_cfg.box_perturbation.debug_force_event = DEBUG_FORCE_EVENT
    env_cfg.box_perturbation.debug_draw_force = DEBUG_DRAW_FORCE
    env_cfg.box_perturbation.stable_confirmed_carry_policy_steps = (
        DEBUG_STABLE_CARRY_STEPS
    )
    env_cfg.box_perturbation.debug_carry_gate_log_interval_policy_steps = (
        DEBUG_CARRY_GATE_LOG_INTERVAL_POLICY_STEPS
    )
    env_cfg.fixed_scene = SimpleNamespace(
        enabled=FIXED_SCENE_ENABLED,
        robot_position=FIXED_ROBOT_POSITION,
        robot_orientation=FIXED_ROBOT_ORIENTATION,
        box_offset_robot_local=FIXED_BOX_OFFSET_ROBOT_LOCAL,
        goal_distance_m=FIXED_GOAL_DISTANCE_M,
        goal_bearing_deg=FIXED_GOAL_BEARING_DEG,
    )
    return env_cfg
