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
            carry_relative_velocity = 0.5
            carry_relative_position = 0.75
            carry_relative_orientation = 0.25
            carry_arm_pose = 0.2
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

        # Final CarryWith calibration, copied without rounding from
        # resources/config/carry_preservation.json at commit 7cb664d.
        # That JSON and analysis/carry_preservation/statistics.json stay offline;
        # training reads only these constants. Regression tests check equality.
        carry_reference_policy_dt = 0.02
        carry_torso_link = "torso_link"
        carry_hand_links = ["left_palm_link", "right_palm_link"]

        # Unit box-to-palm rays in the torso frame, from the CarryWith clips.
        # Left must meet the box's +Y face; right must meet its -Y face.
        carry_hand_direction_left = [-0.18807008519022791, 0.981842058522644, -0.024815623557696863]
        carry_hand_direction_right = [0.2292152654769457, -0.9729077921719884, 0.030179297595546023]
        carry_hand_normal_sigma = [0.02, 0.02]  # m; also used for face-boundary overflow
        carry_hand_direction_sigma = [0.1302497874351225, 0.11122129436107886]  # rad

        # Arm-only prior: the order below matches target/sigma, with no waist.
        carry_arm_joint_names = [
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
            "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
        ]
        carry_arm_target = [  # rad
            -0.12775056064128876, 0.07655864953994751, -0.0941256582736969,
            0.1573413610458374, -0.24472947418689728, 0.0, 0.0,
            -0.43349742889404297, -0.28036007285118103, 0.24257469177246094,
            0.46407607197761536, 0.2483402043581009, 0.0, 0.0,
        ]
        carry_arm_sigma = [0.15] * 14  # rad

        # torso_link origin -> box center, expressed in torso axes (m).
        carry_box_relative_position_target = [
            0.34750078866225004, -0.003911388585662426, 0.049450910653684545,
        ]
        carry_box_relative_position_sigma = [0.03, 0.038154325120839785, 0.03466218700457088]
        # Torso-relative XYZW quaternion; SO(3) widths in rad, with weak yaw.
        carry_box_relative_orientation_target = [
            -0.0163041585521531, -0.055856736281097885, 0.022884953268880424, 0.9980433248811456,
        ]
        carry_box_relative_orientation_sigma = [
            0.09012323316733016, 0.11665271531784734, 0.47389821765272633,
        ]
        # Finite difference of torso-frame position at policy dt, target zero (m/s).
        carry_box_relative_velocity_sigma = [0.1751883601826239, 0.314361221419571, 0.3053264617085051]

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
