import numpy as np

from .carrybox_config import G1Cfg as CarryBoxCfg
from .carrybox_config import G1CfgPPO as CarryBoxCfgPPO


class G1Cfg(CarryBoxCfg):
    """Stage2A Phase 1 config; the Stage1 policy interface is inherited verbatim."""

    class domain_rand(CarryBoxCfg.domain_rand):
        # Isolate the box-force -> teacher -> response causal chain.
        disturbance = False

    class external_force:
        enable_external_force = False
        force_event_probability = 1.0
        force_directions = ("-box_x",)
        beta_range = (0.10, 0.10)

        force_ramp_up_duration_s = 0.4
        force_hold_duration_range_s = (2.0, 4.0)
        force_ramp_down_duration_s = 0.4
        stable_carry_policy_steps = 10
        max_force_events_per_episode = 1

        admittance_D_bar = 12.0
        max_teacher_heading_offset_rad = np.pi / 4.0
        teacher_vx_min = 0.0
        teacher_vx_max = 0.8

        debug_logging = False
        debug_env_id = 0
        debug_log_interval_policy_steps = 25
        debug_draw_force = False
        debug_force_draw_scale_m_per_N = 0.08


class G1CfgPPO(CarryBoxCfgPPO):
    class runner(CarryBoxCfgPPO.runner):
        run_name = "stage2a_force_phase1"
        experiment_name = "Ampstage2a_force_phase1"

    amp = G1Cfg.amp
