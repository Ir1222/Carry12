"""Experiment-only config overrides for carrybox perturbation debugging."""

DEBUG_NUM_ENVS = 1
DEBUG_EPISODE_LENGTH_S = 15
DEBUG_STABLE_CARRY_STEPS = 5
DEBUG_FORCE_EVENT = True
DEBUG_DRAW_FORCE = True
DEBUG_CARRY_GATE_LOG_INTERVAL_POLICY_STEPS = 5


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
        env_cfg.asset.box.reset_mode = "default"

    env_cfg.box_perturbation.enabled = True
    env_cfg.box_perturbation.debug_force_event = DEBUG_FORCE_EVENT
    env_cfg.box_perturbation.debug_draw_force = DEBUG_DRAW_FORCE
    env_cfg.box_perturbation.stable_confirmed_carry_policy_steps = (
        DEBUG_STABLE_CARRY_STEPS
    )
    env_cfg.box_perturbation.debug_carry_gate_log_interval_policy_steps = (
        DEBUG_CARRY_GATE_LOG_INTERVAL_POLICY_STEPS
    )
    return env_cfg
