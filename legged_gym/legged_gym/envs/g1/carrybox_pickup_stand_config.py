"""Scratch PhysHSI approach/pickup AMP + PPO with a stationary loaded hold."""
from .carrybox_config import G1Cfg as CarryCfg
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfgPPO


class G1Cfg(CarryCfg):
    observation_semantics = 'pickup_stand_phase_onehot_v2'
    class asset(CarryCfg.asset):
        class box(CarryCfg.asset.box):
            skill = ["loco", "pickUp"]
            skill_init_prob = [0.2, 0.8]  # task uses explicit reset_weights below
            reset_mode = 'hybrid'

    class commands(CarryCfg.commands):
        curriculum = False
        resample_carry_commands = False
        heading_command = False
        heading_to_ang_vel = False
        resampling_time = 0.
        carry_moving_vx_range = [0., 0.]
        carry_turn_probability = 0.
        carry_turn_yaw_range = [0., 0.]
        class ranges:
            lin_vel_x = [0., 0.]
            lin_vel_y = [0., 0.]
            ang_vel_yaw = [0., 0.]
            heading = [0., 0.]

    class dataset(CarryCfg.dataset):
        motion_file = "{LEGGED_GYM_ROOT_DIR}/resources/config/carrybox_pickup_stand.yaml"

    class pickup_stand:
        # Seconds, metres, radians, m/s, rad/s. See task README for rationale.
        approach_seconds = 0.10
        pickup_facing_tolerance = 0.50
        completion_seconds = 0.20
        min_base_height = 0.70
        max_base_tilt = 0.25
        min_box_height = 0.67
        max_box_height = 1.05
        max_box_tilt = 0.25
        min_clearance = 0.08
        max_relative_speed = 0.20
        max_vertical_speed = 0.15
        palm_surface_tolerance = 0.06
        foot_force = 10.0
        support_body_names = ['left_ankle_roll_link', 'right_ankle_roll_link']
        stable_base_speed = 0.12
        stable_base_angular_speed = 0.20
        stable_box_speed = 0.12
        stable_box_angular_speed = 0.20
        stable_foot_speed = 0.08
        stable_seconds = 0.50
        success_hold_seconds = 3.0
        terminate_on_success = True
        failure_grace_seconds = 0.50
        # [validated HOLD, early/mid/late pickup RSI, full approach]. No fallback.
        reset_weights = [0., 0.20, 0.20, 0.20, 0.40]
        approach_rsi_probability = 0.50  # remaining full approaches start at default
        hold_bank_path = None
        bank_size_tolerance = 0.015
        bank_mass_tolerance = 0.15

    class rewards(CarryCfg.rewards):
        target_speed_loco = 0.85
        # Explicit coefficients: no velocity-tracking objective inherited.
        class scales:
            dof_acc = -1e-7
            action_rate = -0.03
            torques = -1e-4
            dof_vel = -2e-4
            dof_pos_limits = -5.0
            dof_vel_limits = -1e-3
            torque_limits = -0.03
            walk_task = 1.0
            carryup_task = 1.0
            pickup_grasp = 0.5
            pickup_standup = 0.75
            loaded_balance = 1.0
            carry_velocity_task = 0.
            carry_lin_vel_tracking = 0.
            carry_yaw_vel_tracking = 0.
            carry_heading_hold = 0.
            carry_contact_task = 0.
            hold_base_still = 0.75
            hold_box_still = 0.75
            hold_contact = 1.0
            hold_box_pose = 0.75
            hold_posture = 0.75
            hold_feet = 0.5
            hold_action_rate = -0.02
            hold_footsteps = -0.25
            task_success = 5.0


class G1CfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        learning_rate = 5.e-4
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        experiment_name = 'carrybox_pickup_stand_scratch'
        run_name = 'scratch'
        num_steps_per_env = 100
        max_iterations = 20000
        save_interval = 500
        use_muon_optim = False
        logger = 'tensorboard'
        resume = False
        resume_path = None
        finetune_path = None
        scratch_only = True

    amp = G1Cfg.amp
