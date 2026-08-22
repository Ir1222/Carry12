"""PI-only configuration layered on the current carry-box task."""

from .carrybox_config import G1Cfg as CarryBoxCfg
from .carrybox_config import G1CfgPPO as CarryBoxCfgPPO


class G1Cfg(CarryBoxCfg):
    class env(CarryBoxCfg.env):
        num_base_lin_vel_priv = 3
        num_current_frame_critic_obs = CarryBoxCfg.env.num_privileged_obs
        num_interaction_kinematic_obs = 6
        num_interaction_contact_flag_obs = 2
        num_interaction_force_vector_obs = 9
        num_interaction_priv_obs = (
            num_interaction_kinematic_obs + num_interaction_contact_flag_obs
        )
        num_privileged_obs = (
            num_current_frame_critic_obs + num_interaction_priv_obs
        )

    class interaction_priv:
        enabled = True
        include_force_vectors = False
        hand_contact_force_threshold = 1.0
        box_lin_vel_scale = 1.0
        box_ang_vel_scale = 1.0
        net_contact_force_scale = 1.0
        clip_value = 10.0

    def __init__(self):
        super().__init__()
        force_obs_dim = (
            self.env.num_interaction_force_vector_obs
            if self.interaction_priv.include_force_vectors
            else 0
        )
        self.env.num_interaction_priv_obs = (
            self.env.num_interaction_kinematic_obs
            + self.env.num_interaction_contact_flag_obs
            + force_obs_dim
        )
        self.env.num_privileged_obs = (
            self.env.num_current_frame_critic_obs
            + self.env.num_interaction_priv_obs
        )


class G1CfgPPO(CarryBoxCfgPPO):
    class runner(CarryBoxCfgPPO.runner):
        run_name = "stage1_UpAndWalk_PIOnly"
        experiment_name = "Ampstage1_UpAndWalk_PIOnly"
        resume = False
        resume_path = None
