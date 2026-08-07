from .carrybox_resume_config import G1Cfg as CarryBoxResumeCfg
from .carrybox_resume_config import G1CfgPPO as CarryBoxResumeCfgPPO


class G1Cfg(CarryBoxResumeCfg):
    """Resume-compatible box-COM perturbation curriculum configuration."""

    class env(CarryBoxResumeCfg.env):
        # Explicit compatibility contract for existing carry-box checkpoints.
        num_actor_obs = 738
        num_privileged_obs = 143
        num_interaction_priv_obs = 17

    class asset(CarryBoxResumeCfg.asset):
        class box(CarryBoxResumeCfg.asset.box):
            random_size = False
            random_density = False
            density_default = 50.0

    class domain_rand(CarryBoxResumeCfg.domain_rand):
        # Keep all inherited robot DR; disable only the legacy torso force.
        disturbance = False

    class box_perturbation:
        # Master experiment gate. False preserves nominal carry dynamics.
        enabled = True
        stable_confirmed_carry_policy_steps = 20
        max_events_per_episode = 1
        pulse_duration_s = 0.10
        force_sign_verification_samples = 12
        force_closure_residual_max = 0.20

        schedule_mode = "staged_policy_steps"
        manual_stage_override = None
        stage_start_policy_steps = {
            "C1": 0,
            "C2": 250_000,
            "C3": 750_000,
            "C4": 1_500_000,
        }
        stages = {
            "C1": {
                "directions": ("+box_x", "-box_x"),
                "beta": {"x": (0.05, 0.10)},
                "probability": 0.25,
            },
            "C2": {
                "directions": ("+box_x", "-box_x"),
                "beta": {"x": (0.10, 0.20)},
                "probability": 0.40,
            },
            "C3": {
                "directions": ("+box_x", "-box_x", "-z_world"),
                "beta": {"x": (0.10, 0.25), "z": (0.05, 0.15)},
                "probability": 0.50,
            },
            "C4": {
                "directions": (
                    "+box_x",
                    "-box_x",
                    "+box_y",
                    "-box_y",
                    "-z_world",
                ),
                "beta": {
                    "x": (0.10, 0.30),
                    "y": (0.05, 0.15),
                    "z": (0.05, 0.20),
                },
                "probability": 0.60,
            },
        }

        recovery_window_s = 1.0
        recovery_confirmed_carry_steps = 5
        debug_force_event = False
        debug_draw_force = False
        # Viewer arrow length is |F| times this scale.
        # CarryBox forces are much smaller than FALCON hand forces, so retain
        # the FALCON bundle style but use a task-appropriate length scale.
        debug_force_draw_scale_m_per_N = 0.12
        debug_force_bundle_line_count = 20
        debug_force_bundle_jitter_m = 0.01
        # Viewer-only persistence after the physical pulse ends.  This does
        # not change applied force, trace, reward, or termination.
        debug_force_arrow_hold_s = 1.25
        debug_force_draw_max_envs = 10
        debug_force_log_interval_policy_steps = 1

        # Evaluation-only deterministic sweep. For every beta level, test all
        # six evaluation directions once before increasing the force level.
        debug_sweep_enabled = False
        debug_sweep_beta_values = (0.10, 0.25, 0.50, 0.75)
        debug_sweep_directions = (
            "+box_x",
            "-box_x",
            "+box_y",
            "-box_y",
            "+z_world",
            "-z_world",
        )
        debug_sweep_inter_event_policy_steps = 75

        # Evaluation-only instrumentation. Disabled for training.
        evaluation_mode = False
        evaluation_manual_schedule = False
        evaluation_trace_enabled = False
        evaluation_verbose_substeps = False
        evaluation_ignore_task_success_reset = False
        evaluation_precondition_timeout_s = 5.0
        evaluation_post_window_s = 2.0
        evaluation_goal_mode = "default"
        evaluation_goal_distance_range = (4.0, 8.0)
        evaluation_goal_bearing_offset_deg = (15.0, 75.0)


class G1CfgPPO(CarryBoxResumeCfgPPO):
    class runner(CarryBoxResumeCfgPPO.runner):
        run_name = "carrybox_boxperturb_resume"

    amp = G1Cfg.amp
