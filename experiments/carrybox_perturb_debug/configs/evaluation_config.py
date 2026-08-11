"""Experiment-only config overrides for controlled perturbation evaluation."""

from configs.debug_config import apply_debug_config
from evaluation.force_profiles import (
    DEFAULT_POST_FORCE_OBSERVATION_S,
    DEFAULT_STABLE_CONFIRMED_CARRY_STEPS,
)


def apply_evaluation_config(env_cfg, verbose=False, trace_enabled=False, draw_force=True):
    """Apply controlled evaluator overrides before task_registry.make_env()."""
    env_cfg = apply_debug_config(env_cfg)

    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = 30
    env_cfg.env.test = True

    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.delay = False
    env_cfg.domain_rand.push_robots = False

    if hasattr(env_cfg.asset, "box"):
        env_cfg.asset.box.random_props = False

    perturb_cfg = env_cfg.box_perturbation
    perturb_cfg.enabled = True
    perturb_cfg.evaluation_mode = True
    perturb_cfg.evaluation_manual_schedule = True
    perturb_cfg.evaluation_trace_enabled = bool(trace_enabled)
    perturb_cfg.evaluation_verbose_substeps = bool(verbose)
    perturb_cfg.evaluation_post_window_s = DEFAULT_POST_FORCE_OBSERVATION_S
    perturb_cfg.debug_sweep_enabled = False
    perturb_cfg.debug_force_event = False
    perturb_cfg.debug_draw_force = bool(draw_force)
    perturb_cfg.debug_carry_gate_log_interval_policy_steps = 1 if verbose else 0
    perturb_cfg.stable_confirmed_carry_policy_steps = (
        DEFAULT_STABLE_CONFIRMED_CARRY_STEPS
    )
    perturb_cfg.max_events_per_episode = 1
    perturb_cfg.evaluation_quiet = not bool(verbose)
    perturb_cfg.evaluation_verbose = bool(verbose)

    return env_cfg
