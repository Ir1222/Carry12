"""Isolated full-pickup -> locomotion fine-tuning configuration."""
from .carrybox_config import G1Cfg as FullCfg
from .carrybox_locomotion_config import G1Cfg as LocoCfg, G1CfgPPO as LocoPPO


class G1Cfg(LocoCfg):
    class env(FullCfg.env):
        num_envs = 64
        # 126 original privileged + phase one-hot, phase age, handoff age,
        # previous applied action (29). Actor remains exactly 738D.
        num_privileged_obs = 160
        episode_length_s = 100.0

    class asset(FullCfg.asset):
        class box(FullCfg.asset.box):
            reset_mode = "default"
            random_props = False

    class dataset(FullCfg.dataset):
        pass

    class commands(LocoCfg.commands):
        resample_carry_commands = False  # Full pre-carry semantics

    class domain_rand(LocoCfg.domain_rand):
        delay = False  # adapter state must be observed; unobserved delay forbidden

    class handoff:
        teacher_checkpoint = ""
        ablation = "A3"
        warmup_timeout_s = 45.0
        transition_duration_s = 1.0
        transition_timeout_s = 3.0
        student_timeout_s = 45.0
        stable_s = 0.30
        lift_min_m = 0.10
        tilt_max_deg = 20.0
        relative_velocity_max = 0.30
        box_omega_max = 1.0
        hysteresis_ratio = 1.25
        warmup_vx = 0.4
        command_suite = "fixed"  # fixed or mixed (stand/vx/vy/yaw/mixed)
        fixed_command = [0.4, 0.0, 0.0]
        adapter = "hard"
        target_rate_arms = 2.0  # rad/s; initial research settings, not calibrated
        target_rate_waist = 1.5
        target_rate_legs = 12.0
        joint_target_rates = None  # optional 29D override in simulator DOF order
        imitation_window_s = 2.0
        imitation_max_vy = 0.15
        imitation_max_yaw = 0.20
        imitation_max_vx = 0.8
        posture_reference = ""  # required if reward scale is nonzero
        posture_box_size_tolerance_m = 0.04
        posture_penalty_cap = 10.0

    class gait:
        contact_threshold_N = 1.0
        moving_threshold = 0.1
        slip_tolerance_m_s = 0.05
        horizontal_speed_scale_m_s = 0.30
        target_height_m = 0.04
        scuff_height_m = 0.02
        air_time_min_s = 0.06
        air_time_range_s = 0.20
        geometry_max_points = 128
        geometry_fit_tolerance_m = 0.0005
        geometry_validation_tolerance_m = 0.001

    class rewards(LocoCfg.rewards):
        class scales(LocoCfg.rewards.scales):
            carry_upper_body_pose = 0.0
            # Opt in through config_B2_gait_vx08.json; A5 is unchanged.
            gait_foot_slip = 0.0
            gait_swing_clearance = 0.0
            gait_foot_scuff = 0.0
            gait_air_time = 0.0


class G1CfgPPO(LocoPPO):
    runner_class_name = "TeacherWarmupOnPolicyRunner"

    class runner(LocoPPO.runner):
        runner_class_name = "TeacherWarmupOnPolicyRunner"
        algorithm_class_name = "TeacherWarmupPPO"
        experiment_name = "carrybox_teacher_warmup"
        run_name = "A3_fixed"
        logger = "tensorboard"
        num_steps_per_env = 100
        max_iterations = 5000
        save_interval = 100

    class algorithm(LocoPPO.algorithm):
        learning_rate = 5e-5
        min_valid_samples = 64
        handoff_imitation_coef = 0.5
        imitation_arms_weight = 2.0
        imitation_waist_weight = 2.0
        imitation_legs_weight = 0.25
        imitation_joint_weights = None
