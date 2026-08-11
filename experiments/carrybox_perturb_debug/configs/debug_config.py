"""Experiment-only config overrides for carrybox perturbation debugging."""

from types import SimpleNamespace


DEBUG_NUM_ENVS = 1
DEBUG_EPISODE_LENGTH_S = 30
# Trigger carry phase gate
DEBUG_STABLE_CARRY_STEPS = 5
# Egibility: True for constraint perturbation add
DEBUG_FORCE_EVENT = True
# Draw force vector in the scene
DEBUG_DRAW_FORCE = True
DEBUG_CARRY_GATE_LOG_INTERVAL_POLICY_STEPS = 5

# Debug-only trigger mode:
# - confirmed_carry: keep baseline perturb gate semantics.
# - time_after_reset: schedule once after DEBUG_TRIGGER_POLICY_STEP, regardless
#   of carry_phase. Use this only to validate trigger -> commit -> apply.
# - relaxed_carry: schedule using the relaxed masks below and log both gates.
DEBUG_TRIGGER_MODE = "confirmed_carry"
DEBUG_TRIGGER_POLICY_STEP = 100
DEBUG_RELAXED_REQUIRE_HEIGHT_GATE = True
DEBUG_RELAXED_REQUIRE_STATIC_GATE = False
DEBUG_RELAXED_CONTACT_MODE = "either"  # "both", "either", or "none"
DEBUG_RELAXED_STABLE_STEPS = 1

DEBUG_COMMAND_X = 0.4
DEBUG_COMMAND_Y = 0.0
DEBUG_COMMAND_YAW = 0.0

# Experiment-only deterministic reset scene. Positions are env-local; the env
# origin is added by the debug subclass when writing simulator state tensors.
FIXED_SCENE_ENABLED = True
FIXED_ROBOT_POSITION = [2.3, 0.0, 0.8]
FIXED_ROBOT_ORIENTATION = [0.0, 0.0, 1.0, 0.0]
# Robot-local xyz offset. With the default 180-degree yaw this puts the box at
# env-local [0.55, 0.0, 0.135] without changing the parent-sampled box rotation.
FIXED_BOX_OFFSET_ROBOT_LOCAL = [1.75, 0.0, -0.665]
FIXED_GOAL_DISTANCE_M = 4.0
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
    env_cfg.box_perturbation.debug_trigger_mode = DEBUG_TRIGGER_MODE
    env_cfg.box_perturbation.debug_trigger_policy_step = DEBUG_TRIGGER_POLICY_STEP
    env_cfg.box_perturbation.debug_relaxed_require_height_gate = (
        DEBUG_RELAXED_REQUIRE_HEIGHT_GATE
    )
    env_cfg.box_perturbation.debug_relaxed_require_static_gate = (
        DEBUG_RELAXED_REQUIRE_STATIC_GATE
    )
    env_cfg.box_perturbation.debug_relaxed_contact_mode = DEBUG_RELAXED_CONTACT_MODE
    env_cfg.box_perturbation.debug_relaxed_stable_policy_steps = (
        DEBUG_RELAXED_STABLE_STEPS
    )
    env_cfg.debug_command = SimpleNamespace(
        lin_vel_x=DEBUG_COMMAND_X,
        lin_vel_y=DEBUG_COMMAND_Y,
        yaw_rate=DEBUG_COMMAND_YAW,
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
