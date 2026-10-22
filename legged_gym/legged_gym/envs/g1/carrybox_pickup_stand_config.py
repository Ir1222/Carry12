"""Independent approach -> pickup -> stationary loaded hold configuration."""
from .carrybox_config import G1Cfg as CarryCfg, G1CfgPPO as CarryPPO


class G1Cfg(CarryCfg):
    class asset(CarryCfg.asset):
        class box(CarryCfg.asset.box):
            skill = ["loco", "pickUp"]
            skill_init_prob = [0.2, 0.8]
            reset_mode = 'random'

    class commands(CarryCfg.commands):
        resample_carry_commands = False
        approach_vx_range = [0.2, 0.8]

    class dataset(CarryCfg.dataset):
        motion_file = "{LEGGED_GYM_ROOT_DIR}/resources/config/carrybox_pickup_stand.yaml"

    class pickup_stand:
        # Seconds, metres, radians, m/s, rad/s. See task README for rationale.
        contact_seconds = 0.10
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
        stable_base_speed = 0.12
        stable_base_angular_speed = 0.20
        stable_box_speed = 0.12
        stable_box_angular_speed = 0.20
        stable_foot_speed = 0.08
        stable_seconds = 0.50
        success_hold_seconds = 3.0
        failure_grace_seconds = 0.50
        # Explicit stages keep advancement contingent on evaluation results.
        curriculum_stage = 'hold'  # hold, pickup, full
        hold_bank_path = None
        # Columns: validated hold, early/mid/late pickup, full approach.
        hold_reset_weights = [0.65, 0.05, 0.05, 0.20, 0.05]
        pickup_reset_weights = [0.20, 0.20, 0.25, 0.25, 0.10]
        full_reset_weights = [0.10, 0.10, 0.10, 0.15, 0.55]
        require_hold_bank = False
        bank_size_tolerance = 0.015
        bank_mass_tolerance = 0.15

    class rewards(CarryCfg.rewards):
        class scales(CarryCfg.rewards.scales):
            hold_base_still = 0.75
            hold_box_still = 0.75
            hold_contact = 1.0
            hold_box_pose = 0.75
            hold_posture = 0.75
            hold_feet = 0.5
            hold_action_rate = -0.02


class G1CfgPPO(CarryPPO):
    class algorithm(CarryPPO.algorithm):
        learning_rate = 1.e-4
        schedule = 'fixed'
        entropy_coef = 0.003

    class runner(CarryPPO.runner):
        experiment_name = 'carrybox_pickup_stand'
        run_name = 'hold'
        finetune_path = None  # pass --finetune_path explicitly
