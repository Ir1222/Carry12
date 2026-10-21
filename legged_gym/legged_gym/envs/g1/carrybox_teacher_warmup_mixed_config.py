"""M1: trained B2 gait -> real teacher pickup -> held, slew-limited Full Mixed."""
from .carrybox_teacher_warmup_config import G1Cfg as WarmupCfg, G1CfgPPO as WarmupPPO


class G1Cfg(WarmupCfg):
    class env(WarmupCfg.env):
        num_envs = 4096

    class handoff(WarmupCfg.handoff):
        ablation = "A5"
        adapter = "rate_limited"
        warmup_vx = 0.8
        fixed_command = [0.8, 0.0, 0.0]
        command_suite = "mixed"
        mixed_command_hold_s = 2.0
        mixed_command_slew_rates = [0.6, 0.5, 0.6]
        mixed_command_manager = "hold_then_slew_v1"

    class rewards(WarmupCfg.rewards):
        class scales(WarmupCfg.rewards.scales):
            carry_upper_body_pose = 0.5
            gait_foot_slip = -0.08
            gait_swing_clearance = 0.15
            gait_foot_scuff = -0.08
            gait_air_time = 0.05
            feet_air_time = 0.0
            feet_slip = 0.0
            feet_clearance = 0.0


class G1CfgPPO(WarmupPPO):
    class runner(WarmupPPO.runner):
        experiment_name = "carrybox_teacher_warmup_mixed"
        run_name = "M1_mixed"
        max_iterations = 10000
        num_steps_per_env = 100
        save_interval = 500

    class algorithm(WarmupPPO.algorithm):
        min_valid_samples = 2048
