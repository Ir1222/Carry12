"""Configuration for the dedicated CarryBox locomotion specialist."""

from .carrybox_config import G1Cfg as CarryBoxCfg
from .carrybox_config import G1CfgPPO as CarryBoxCfgPPO


class G1Cfg(CarryBoxCfg):
    class terrain(CarryBoxCfg.terrain):
        curriculum = False

    class asset(CarryBoxCfg.asset):
        class box(CarryBoxCfg.asset.box):
            reset_mode = "random"
            skill = ["carryWith"]
            skill_init_prob = [1.0]
            carry_reset_phase_range = [0.10, 0.90]
            fixed_carry_reset = False
            fixed_carry_motion_id = 0
            fixed_carry_phase = 0.5
            use_mass_size_mixture = True
            random_props = False

    class commands(CarryBoxCfg.commands):
        curriculum = False
        resampling_time = 0.0
        resample_carry_commands = True
        carry_command_mode_probabilities = [0.10, 0.15, 0.10, 0.15, 0.50]
        carry_vx_range = [-0.5, 1.2]
        carry_vy_range = [-0.4, 0.4]
        carry_yaw_rate_range = [-0.6, 0.6]
        carry_mixed_ranges = [
            [-0.4, 0.96],
            [-0.32, 0.32],
            [-0.4, 0.4],
        ]
        carry_command_resample_interval_s = [4.0, 6.0]
        carry_moving_vx_range = [-0.5, 1.2]
        heading_command = False
        heading_to_ang_vel = False
        lin_vel_clip = 0.0
        ang_vel_clip = 0.0

        class ranges:
            lin_vel_x = [-0.5, 1.2]
            lin_vel_y = [-0.4, 0.4]
            ang_vel_yaw = [-0.6, 0.6]
            heading = [0.0, 0.0]

    class rewards(CarryBoxCfg.rewards):
        class scales:
            carry_lin_vel_tracking = 3.0
            carry_yaw_vel_tracking = 3.0

            carry_bilateral_contact = 0.5
            carry_hand_box_surface = 1.5
            carry_hand_slip = 0.5
            carry_relative_velocity = 0.5
            carry_relative_position = 0.75
            carry_relative_orientation = 0.0
            carry_arm_range = 0.2
            zero_command_stillness = 0.2

            lin_vel_z = -1.0
            ang_vel_xy = -0.05
            orientation = -1.0
            carry_box_tilt = -0.5
            carry_upper_body_pose = 0.0

            feet_air_time = 0.2
            feet_slip = -0.2
            feet_stumble = -1.0
            collision = -0.5
            feet_contact_forces = -0.001

            dof_acc = -1e-7
            action_rate = -0.03
            torques = -1e-4
            dof_vel = -2e-4
            dof_pos_limits = -5.0
            dof_vel_limits = -1e-3
            torque_limits = -0.03

            walk_task = 0.0
            carryup_task = 0.0
            carry_velocity_task = 0.0
            # Legacy coupled term contains uncompensated world velocity and a
            # saturated distance reward; the separate terms above replace it.
            carry_contact_task = 0.0
            feet_clearance = 0.0
            no_fly = 0.0
            base_height = 0.0
            joint_power = 0.0

        # Physical carry constraints; CarryWith is used only for RSI.
        carry_torso_link = "torso_link"
        carry_hand_links = ["left_palm_link", "right_palm_link"]

        # Left +Y / right -Y; use actual randomized box half-extents.
        carry_hand_side_tolerance = 0.03  # m
        carry_hand_face_margin = 0.01  # m outside x/z face edges
        carry_hand_surface_violation_scale = 0.04  # m, post-dead-zone softness
        carry_hand_slip_tolerance = 0.35  # m/s, norm of local x/z velocity
        carry_hand_slip_violation_scale = 0.35  # m/s

        # Arm-only guardrail, radians; no waist or leg supervision.
        carry_arm_joint_names = [
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
            "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
        ]
        carry_arm_range_lower = [
            -0.90, -0.25, -0.85, -0.25, -1.10, -0.60, -0.60,
            -1.10, -1.00, -0.55, -0.25, -0.55, -0.60, -0.60,
        ]
        carry_arm_range_upper = [
            0.55, 0.80, 0.70, 1.15, 0.55, 0.60, 0.60,
            0.25, 0.25, 1.00, 1.40, 1.10, 0.60, 0.60,
        ]
        carry_arm_violation_scale = 0.35  # rad beyond any one joint's range

        # Box workspace in torso_link coordinates (m), with no preferred center.
        carry_box_relative_position_lower = [0.18, -0.20, -0.15]
        carry_box_relative_position_upper = [0.53, 0.20, 0.25]
        carry_box_position_violation_scale = [0.10, 0.10, 0.10]  # m
        # Dead zone for the derivative of torso-relative box position.
        carry_box_relative_velocity_tolerance = [0.35, 0.45, 0.45]  # m/s
        carry_box_velocity_violation_scale = [0.35, 0.45, 0.45]  # m/s

        box_drop_height = 0.15
        robot_box_max_distance = 1.0
        box_tilt_termination_deg = 70.0
        grasp_loss_grace_s = 0.30

    class domain_rand(CarryBoxCfg.domain_rand):
        randomize_actuation_offset = False

        randomize_motor_strength = True
        motor_strength_range = [0.95, 1.05]

        randomize_payload_mass = False
        randomize_com_displacement = False

        randomize_link_mass = True
        link_mass_range = [0.95, 1.05]

        randomize_friction = True
        friction_range = [0.6, 1.2]

        randomize_restitution = True
        restitution_range = [0.0, 0.05]

        randomize_kp = True
        kp_range = [0.95, 1.05]

        randomize_kd = True
        kd_range = [0.95, 1.05]

        randomize_initial_joint_pos = False
        disturbance = False
        push_robots = False

        delay = True
        max_delay_timesteps = 2

    class dataset(CarryBoxCfg.dataset):
        motion_file = (
            "{LEGGED_GYM_ROOT_DIR}/resources/config/"
            "carrybox_locomotion.yaml"
        )


class G1CfgPPO(CarryBoxCfgPPO):
    runner_class_name = "CarryLocomotionOnPolicyRunner"

    class algorithm(CarryBoxCfgPPO.algorithm):
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 2e-4
        schedule = "adaptive"
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01

    class runner(CarryBoxCfgPPO.runner):
        policy_class_name = "ActorCritic"
        algorithm_class_name = "CarryLocomotionPPO"
        runner_class_name = "CarryLocomotionOnPolicyRunner"
        use_muon_optim = False
        num_steps_per_env = 100
        max_iterations = 20000
        save_interval = 500
        experiment_name = "carrybox_locomotion"
        run_name = "loaded_velocity_tracking"

    amp = G1Cfg.amp
