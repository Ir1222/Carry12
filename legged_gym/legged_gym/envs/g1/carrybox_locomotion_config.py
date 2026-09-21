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
        # Controlled V1 ablation: restore the pre-7115 command distribution so
        # the first experiment isolates lower-body reward design. This is not
        # intended to be the final target command distribution.
        carry_command_mode_probabilities = [0.10, 0.25, 0.15, 0.15, 0.35]
        carry_vx_range = [-0.5, 1.2]
        carry_vy_range = [-0.4, 0.4]
        carry_yaw_rate_range = [-0.5, 0.5]
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
            ang_vel_yaw = [-0.5, 0.5]
            heading = [0.0, 0.0]

    class rewards(CarryBoxCfg.rewards):
        class scales:
            carry_lin_vel_tracking = 3.0
            carry_yaw_vel_tracking = 2.5

            carry_bilateral_contact = 0.5
            carry_hand_box_surface = 1.5
            carry_hand_slip = 0.5
            carry_relative_velocity = 0.5
            carry_relative_position = 0.75
            carry_relative_orientation = 0.0
            carry_arm_range = 0.2

            # Lower-body feasible-family constraints. The broad 12-joint box
            # and legacy one-sided stance cap are disabled for the clean V1
            # ablation; their constants remain available as a safety envelope.
            carry_hip_posture = 0.8
            carry_foot_heading = 0.35
            carry_feet_width = 0.30
            carry_knee_width = 0.20
            carry_leg_range = 0.0
            carry_stance_width = 0.0

            # Waist/torso quantities are diagnostics only in this experiment.
            carry_waist_posture = 0.0
            carry_torso_pelvis_alignment = 0.0
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

        # Physical carry constraints; reference statistics below are offline only.
        carry_torso_link = "torso_link"
        carry_hand_links = ["left_palm_link", "right_palm_link"]

        # Left +Y / right -Y; use actual randomized box half-extents.
        carry_hand_side_tolerance = 0.03  # m
        carry_hand_face_margin = 0.01  # m outside x/z face edges
        carry_hand_surface_violation_scale = 0.04  # m, post-dead-zone softness
        carry_hand_slip_tolerance = 0.35  # m/s, norm of local x/z velocity
        carry_hand_slip_violation_scale = 0.35  # m/s

        # Arm-only guardrail, radians.
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

        # All 773 frames of carrywith1/2/3.pt: P1/P99 plus margins (rad).
        # Hip roll/yaw: max(0.04, 0.15*span); others: max(0.06, 0.20*span).
        # Clipped to URDF limits with 0.02 rad clearance, except saturated knees
        # and ankle rolls: 0.0001 rad. Raw references slightly exceed those limits.
        # Rounded outward where not URDF-clipped; no preferred pose in the range.
        carry_leg_joint_names = [
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        ]
        carry_leg_range_lower = [
            -0.6077, -0.3364, -1.2562, -0.087167, -0.4483, -0.2617,
            -0.5830, -0.1907, -0.4138, -0.087167, -0.6344, -0.1565,
        ]
        carry_leg_range_upper = [
            0.8094, 0.1562, 0.4030, 1.2273, 0.2624, 0.2617,
            0.8535, 0.4900, 0.8239, 1.3980, 0.2731, 0.2617,
        ]
        carry_leg_violation_scale = [
            0.25, 0.15, 0.20, 0.25, 0.20, 0.15,
            0.25, 0.15, 0.20, 0.25, 0.20, 0.15,
        ]  # rad beyond each joint's feasible range

        # Focus the posture prior on the four joints that create crab geometry.
        # Targets are read from default_dof_pos at runtime, not demonstrations.
        carry_hip_joint_names = [
            "left_hip_roll_joint", "left_hip_yaw_joint",
            "right_hip_roll_joint", "right_hip_yaw_joint",
        ]
        carry_hip_posture_deadzone = [0.08, 0.10, 0.08, 0.10]  # rad
        carry_hip_posture_softness = [0.15, 0.15, 0.15, 0.15]  # rad

        carry_foot_links = ["left_ankle_pitch_link", "right_ankle_pitch_link"]
        carry_knee_links = ["left_knee_link", "right_knee_link"]
        carry_foot_heading_deadzone = 0.12  # rad relative to pelvis heading
        carry_foot_heading_softness = 0.20  # rad beyond the dead zone

        # Offline FK over all 773 CarryWith frames, pelvis-heading frame (m):
        #                 min     P01     P05     P50     P95     P99     max
        # feet width:   .0754   .0841   .1087   .1562   .2503   .2727   .2772
        # knee width:   .1471   .1486   .1579   .1928   .2306   .2487   .2524
        # Feasible intervals are P05/P95 plus about 2 cm engineering margin.
        carry_feet_width_range = [0.09, 0.27]
        carry_feet_width_softness = 0.08  # m beyond the interval
        carry_knee_width_range = [0.14, 0.25]
        carry_knee_width_softness = 0.06  # m beyond the interval

        # Explicitly enumerate all three waist joints for diagnostics. The
        # inherited asset.waist_joints intentionally contains yaw only.
        carry_waist_joint_names = [
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        ]

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


class G1CfgAblationA(G1Cfg):
    """Current upper body + pre-7115 commands + lower constraints OFF."""

    class rewards(G1Cfg.rewards):
        class scales(G1Cfg.rewards.scales):
            carry_hip_posture = 0.0
            carry_foot_heading = 0.0
            carry_feet_width = 0.0
            carry_knee_width = 0.0


class G1CfgAblationB(G1CfgAblationA):
    """Current upper body + current harder commands + lower constraints OFF."""

    class commands(G1Cfg.commands):
        carry_command_mode_probabilities = [0.10, 0.15, 0.10, 0.15, 0.50]
        carry_yaw_rate_range = [-0.6, 0.6]

        class ranges(G1Cfg.commands.ranges):
            ang_vel_yaw = [-0.6, 0.6]

    class rewards(G1CfgAblationA.rewards):
        class scales(G1CfgAblationA.rewards.scales):
            carry_yaw_vel_tracking = 3.0
