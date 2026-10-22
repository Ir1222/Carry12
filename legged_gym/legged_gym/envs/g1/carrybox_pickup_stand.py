"""G1 approaches, picks up and immediately holds. No loaded locomotion phase."""
import warnings
import torch
from isaacgym.torch_utils import quat_rotate_inverse

from .carrybox import LeggedRobot as CarryBox
from .pickup_stand_logic import (APPROACH, PICKUP, HOLD, advance_phase,
                                  completion_conditions, palm_box_contact, ResetKinematics)
from ..motionlib.pickup_stand_sampling import PickupStandMotionSampler


class LeggedRobot(CarryBox):
    def _init_buffers(self):
        super()._init_buffers()
        self.motionlib = PickupStandMotionSampler(self.motionlib)
        from legged_gym import LEGGED_GYM_ROOT_DIR
        self.reset_kinematics = ResetKinematics(self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR), self.dof_names)
        self.phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.reset_category = torch.zeros_like(self.phase)
        self.phase_age = torch.zeros_like(self.phase)
        self.pending_observation_reset = torch.zeros_like(self.phase, dtype=torch.bool)
        for name in ('contact_seconds', 'completion_seconds', 'hold_elapsed',
                     'stable_elapsed', 'longest_hold', 'hold_failure_seconds',
                     'completion_at', 'stable_delay', 'approach_elapsed', 'pickup_elapsed'):
            setattr(self, name, torch.zeros(self.num_envs, device=self.device))
        self.pickup_completed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.hold_success = torch.zeros_like(self.pickup_completed)
        self.hold_box_height = torch.full((self.num_envs,), self.cfg.rewards.target_box_height, device=self.device)
        self.hold_box_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.hold_box_quat[:, 3] = 1
        self.reset_ee = torch.zeros(self.num_envs, 15, device=self.device)
        self.reset_torso_inverse = torch.eye(3, device=self.device).repeat(self.num_envs, 1, 1)
        self.stable_delay.fill_(-1.)
        self.hold_bank = None
        path = self.cfg.pickup_stand.hold_bank_path
        if path:
            self.hold_bank = torch.load(path, map_location=self.device, weights_only=False)
            self._validate_bank()
        elif self.cfg.pickup_stand.require_hold_bank:
            raise ValueError('Validated HOLD bank required; collect one with pickup_stand.py --mode collect')
        else:
            warnings.warn('No validated HOLD bank: HOLD reset weight goes to late pickup; no reference frame is labeled static HOLD.')
        expected = list(self.cfg.init_state.default_joint_angles)
        if self.dof_names != expected or self.num_obs != 738 or self.num_actions != 29:
            raise ValueError('Full AMP interface/joint order mismatch')

    def reset(self):
        # BaseTask.reset advances physics with a zero action, which can wrench
        # a loaded RSI's arms toward the unloaded default pose before inference.
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.compute_observations()
        return self.obs_buf, self.privileged_obs_buf

    def post_physics_step(self):
        result = super().post_physics_step()
        ids = result[0]
        # The parent updates these AFTER autoreset; remove terminal-row leakage.
        self.actions[ids] = 0.
        self.last_actions[ids] = 0.
        self.last_last_actions[ids] = 0.
        self.last_torques[ids] = 0.
        self.last_dof_vel[ids] = self.dof_vel[ids]
        self.last_root_vel[ids] = self.root_states[ids, 7:13]
        return result

    def _validate_bank(self):
        bank, cfg = self.hold_bank, self.cfg.pickup_stand
        if bank.get('dof_names') != self.dof_names or bank.get('version') != 1:
            raise ValueError('HOLD bank joint ordering/version mismatch')
        count = len(bank['root'])
        for key, width in [('root', 13), ('box', 13), ('dof_pos', 29), ('dof_vel', 29), ('size', 3), ('ee', 15)]:
            if bank[key].shape != (count, width) or not torch.isfinite(bank[key]).all():
                raise ValueError(f'Invalid HOLD bank {key}')
        if not count or bank['mass'].shape != (count,) or not torch.isfinite(bank['mass']).all():
            raise ValueError('Empty bank or invalid mass metadata')
        if bank['validated_seconds'].shape != (count,) or not torch.isfinite(bank['validated_seconds']).all() or (bank['validated_seconds'] < cfg.stable_seconds).any():
            raise ValueError('HOLD bank contains unvalidated states')
        for key in ('bilateral_contact', 'both_feet_contact'):
            if bank[key].shape != (count,) or not bank[key].all():
                raise ValueError('HOLD bank lacks contact validation')
        if bank['foot_slip'].shape != (count,) or not torch.isfinite(bank['foot_slip']).all() or (bank['foot_slip'] > cfg.stable_foot_speed).any():
            raise ValueError('HOLD bank has slipping feet')
        for body, linear, angular in [('root', cfg.stable_base_speed, cfg.stable_base_angular_speed),
                                      ('box', cfg.stable_box_speed, cfg.stable_box_angular_speed)]:
            q = bank[body][:, 3:7]
            tilt = torch.acos((1 - 2 * q[:, :2].square().sum(-1)).clamp(-1, 1))
            tilt_limit = cfg.max_base_tilt if body == 'root' else cfg.max_box_tilt
            if not torch.allclose(q.norm(dim=-1), torch.ones(count, device=self.device), atol=1.e-3) or (tilt > tilt_limit).any():
                raise ValueError('HOLD bank has invalid orientation')
            if (bank[body][:, 7:10].norm(dim=-1) > linear).any() or (bank[body][:, 10:13].norm(dim=-1) > angular).any():
                raise ValueError('HOLD bank contains moving states')
        if ((bank['root'][:, 2] < cfg.min_base_height).any()
                or (bank['box'][:, 2] < cfg.min_box_height).any()
                or (bank['box'][:, 2] > cfg.max_box_height).any()):
            raise ValueError('HOLD bank contains invalid standing heights')

    def _bilateral_contact(self):
        return palm_box_contact(self.rigid_body_states[:, self.hand_pos_indices, :3],
            self.box_states[:, :3], self.box_states[:, 3:7], self._box_size,
            self.contact_forces[:, self.hand_colli_indices],
            self.cfg.rewards.hand_contact_threshold,
            self.cfg.pickup_stand.palm_surface_tolerance)[0]

    def _physical_status(self):
        cfg = self.cfg.pickup_stand
        feet_down = (self.contact_forces[:, self.feet_indices, 2] > cfg.foot_force).all(-1)
        bilateral = self._bilateral_contact()
        base_height = self.root_states[:, 2] - self.env_origins[:, 2]
        box_height = self.box_states[:, 2] - self.env_origins[:, 2]
        base_tilt = torch.acos((-self.projected_gravity[:, 2]).clamp(-1, 1))
        box_tilt = torch.acos((-self.projected_gravity_box[:, 2]).clamp(-1, 1))
        # Both ground clearance and separation from the source platform top.
        bottom = self.box_states[:, 2] - self._box_size[:, 2] / 2
        support_top = torch.maximum(self.env_origins[:, 2], self.platform_pos[:, 2] + self._platform_height / 2)
        relative = (self.box_states[:, 7:10] - self.root_states[:, 7:10]).norm(dim=-1)
        complete = completion_conditions(base_height, base_tilt, box_height, box_tilt,
            bottom - support_top, bilateral, feet_down, relative,
            self.root_states[:, 9], self.box_states[:, 9], cfg)
        return complete, bilateral, feet_down

    def _compute_carry_velocity_active(self):
        # The parent calls this before our phase update; no dynamic carry gate.
        return self.phase == HOLD

    def _update_carry_heading_commands(self):
        old = self.phase.clone()
        complete, bilateral, _ = self._physical_status()
        cfg = self.cfg.pickup_stand
        self.phase, self.contact_seconds, self.completion_seconds, entered = advance_phase(
            self.phase, self.contact_seconds, self.completion_seconds,
            self.robot2object_dist < self.cfg.rewards.thresh_robot2object,
            bilateral, complete, self.dt, cfg.contact_seconds, cfg.completion_seconds)
        self.phase_age = torch.where(old == self.phase, self.phase_age + 1, 0)
        self.pickup_completed |= entered
        self.completion_at[entered] = self.episode_length_buf[entered] * self.dt
        self.hold_box_height[entered] = self.box_states[entered, 2] - self.env_origins[entered, 2]
        self.hold_box_quat[entered] = self.box_states[entered, 3:7]
        # Remove movement commands upon grasping, BEFORE lift completion. The
        # inherited approach reward uses the original speed only while APPROACH.
        self.commands[self.phase != APPROACH] = 0.
        self.carry_policy_commands[:] = self.commands
        self.carry_heading_error.zero_()
        self.carry_velocity_active = self.phase == HOLD
        self.carry_tracking_started = self.phase == HOLD
        self.approach_elapsed += (self.phase == APPROACH) * self.dt
        self.pickup_elapsed += (self.phase == PICKUP) * self.dt
        self.hold_elapsed += (self.phase == HOLD) * self.dt
        stable = (self.phase == HOLD) & complete
        stable &= self.root_states[:, 7:10].norm(dim=-1) <= cfg.stable_base_speed
        stable &= self.root_states[:, 10:13].norm(dim=-1) <= cfg.stable_base_angular_speed
        stable &= self.box_states[:, 7:10].norm(dim=-1) <= cfg.stable_box_speed
        stable &= self.box_states[:, 10:13].norm(dim=-1) <= cfg.stable_box_angular_speed
        stable &= self.feet_vel.norm(dim=-1).amax(-1) <= cfg.stable_foot_speed
        self.stable_elapsed = torch.where(stable, self.stable_elapsed + self.dt, 0.)
        self.longest_hold = torch.maximum(self.longest_hold, self.stable_elapsed)
        first_stable = (self.stable_elapsed >= cfg.stable_seconds) & (self.stable_delay < 0)
        self.stable_delay[first_stable] = self.hold_elapsed[first_stable] - cfg.stable_seconds
        self.hold_success |= self.stable_elapsed >= cfg.success_hold_seconds
        self.hold_failure_seconds = torch.where((self.phase == HOLD) & ~complete,
                                                 self.hold_failure_seconds + self.dt, 0.)

    def check_termination(self):
        super().check_termination()
        self.reset_buf |= self.hold_failure_seconds > self.cfg.pickup_stand.failure_grace_seconds

    def compute_amp_observations(self):
        result = super().compute_amp_observations()
        # Snapshot BEFORE parent autoreset. Exclude mixed-stage/initial histories.
        valid = (self.phase != HOLD) & (self.phase_age >= self.cfg.amp.window_length)
        self.extras['amp_stage'] = torch.where(valid, self.phase, -1).clone()
        self.extras['pickup_stand'] = {k: v.clone() for k, v in self.diagnostics().items()}
        return result

    def _reward_walk_task(self):
        return super()._reward_walk_task() * (self.phase == APPROACH)

    def _reward_hold_base_still(self):
        return (self.phase == HOLD) * torch.exp(-self.root_states[:, 7:9].square().sum(-1) / 0.02
                                                - self.root_states[:, 10:13].square().sum(-1) / 0.08)

    def _reward_hold_box_still(self):
        return (self.phase == HOLD) * torch.exp(-self.box_states[:, 7:10].square().sum(-1) / 0.02
                                                - self.box_states[:, 10:13].square().sum(-1) / 0.08)

    def _reward_hold_contact(self):
        return (self.phase == HOLD) * self._bilateral_contact().float()

    def _reward_hold_box_pose(self):
        height_error = self.box_states[:, 2] - self.env_origins[:, 2] - self.hold_box_height
        angle = 2 * torch.acos((self.box_states[:, 3:7] * self.hold_box_quat).sum(-1).abs().clamp(0, 1))
        return (self.phase == HOLD) * torch.exp(-height_error.square() / 0.01 - angle.square() / 0.09)

    def _reward_hold_posture(self):
        # Wide height dead zone, upright torso, soft leg posture. Arms remain free
        # to support randomized box masses/sizes; no zero-joint-velocity target.
        height = self.root_states[:, 2] - self.env_origins[:, 2]
        excess = (0.70 - height).clamp_min(0) + (height - 0.88).clamp_min(0)
        leg_error = (self.dof_pos[:, :12] - self.default_dof_pos[:, :12]).square().mean(-1)
        return (self.phase == HOLD) * torch.exp(-excess.square() / 0.01
            - self.projected_gravity[:, :2].square().sum(-1) / 0.09 - 0.3 * leg_error)

    def _reward_hold_feet(self):
        down = (self.contact_forces[:, self.feet_indices, 2] > self.cfg.pickup_stand.foot_force).all(-1)
        return (self.phase == HOLD) * down * torch.exp(-self.feet_vel.square().sum((1, 2)) / 0.02)

    def _reward_hold_action_rate(self):
        return (self.phase == HOLD) * super()._reward_action_rate()

    def _reset_actors(self, env_ids):
        cfg = self.cfg.pickup_stand
        weights = torch.tensor(getattr(cfg, cfg.curriculum_stage + '_reset_weights'), dtype=torch.float, device=self.device)
        if self.hold_bank is None:
            weights[3] += weights[0]
            weights[0] = 0
        categories = torch.multinomial(weights, len(env_ids), replacement=True)
        self.reset_category[env_ids] = categories
        self.phase[env_ids] = APPROACH
        # Full approach always starts unloaded, using the inherited default reset.
        approach_ids = env_ids[categories == 4]
        if len(approach_ids):
            self._reset_default(approach_ids)
            self.reset_ee[approach_ids] = 0.
        for category in (1, 2, 3):
            ids = env_ids[categories == category]
            if not len(ids):
                continue
            motions = self.motionlib.sample_motions('pickUp', len(ids))
            contact = self.motionlib.contact_index[motions].float()
            end = self.motionlib.motion_len['pickUp'][motions].float() - 1
            # Early includes pre-contact. Mid/late are stratified AFTER grasp.
            lower = [None, torch.zeros_like(contact), contact, contact + .5 * (end - contact)][category]
            upper = [None, contact, contact + .5 * (end - contact), end][category]
            times = lower + torch.rand(len(ids), device=self.device) * (upper - lower)
            root, quat, linear, angular, joints, velocity, ee = self.motionlib.get_motion_state('pickUp', motions, times)
            # Include the reference height correction in root Z velocity.
            t0, t1 = (times - .5).clamp_min(0), torch.minimum(times + .5, end)
            p0 = self.motionlib.get_motion_state('pickUp', motions, t0)[0]
            p1 = self.motionlib.get_motion_state('pickUp', motions, t1)[0]
            linear[:, 2] = (p1[:, 2] - p0[:, 2]) * self.motionlib.fps / (t1 - t0).clamp_min(1.e-6)
            self.root_states[ids, :3] = root + self.env_origins[ids]
            self.root_states[ids, 3:7] = quat
            self.root_states[ids, 7:10] = linear
            self.root_states[ids, 10:13] = angular
            self.dof_pos[ids], self.dof_vel[ids], self.reset_ee[ids] = joints, velocity, ee
            # Parent reset-box helper consumes these exact same motion/time IDs.
            self._reset_ref_env_ids.setdefault('pickUp', []).append(ids)
            self._reset_ref_motion_ids.setdefault('pickUp', []).append(motions)
            self._reset_ref_motion_times.setdefault('pickUp', []).append(times)
            self.phase[ids] = torch.where(times >= contact, PICKUP, APPROACH)
        for mapping in (self._reset_ref_env_ids, self._reset_ref_motion_ids, self._reset_ref_motion_times):
            if 'pickUp' in mapping:
                mapping['pickUp'] = torch.cat(mapping['pickUp'])
        self._bank_reset_ids = env_ids[categories == 0]
        if len(self._bank_reset_ids):
            bank = self.hold_bank
            matches = (self._box_size[self._bank_reset_ids, None] - bank['size'][None]).abs().amax(-1) <= cfg.bank_size_tolerance
            matches &= (self.box_masses[self._bank_reset_ids, None] - bank['mass'][None]).abs() <= cfg.bank_mass_tolerance
            if not matches.any(-1).all():
                raise ValueError('HOLD bank lacks states for sampled box size/mass; collect matching coverage or set HOLD reset weight to zero')
            self._bank_indices = torch.multinomial(matches.float(), 1).squeeze(-1)
            ids, rows = self._bank_reset_ids, self._bank_indices
            self.root_states[ids] = bank['root'][rows]
            self.root_states[ids, :3] += self.env_origins[ids]
            self.dof_pos[ids], self.dof_vel[ids] = bank['dof_pos'][rows], bank['dof_vel'][rows]
            self.reset_ee[ids] = bank['ee'][rows]
            self.phase[ids] = HOLD

    def _reset_boxes(self, env_ids):
        super()._reset_boxes(env_ids)
        ids = self._reset_ref_env_ids.get('pickUp')
        if ids is not None:
            linear, angular = self.motionlib.object_velocity('pickUp', self._reset_ref_motion_ids['pickUp'], self._reset_ref_motion_times['pickUp'])
            supported = self._reset_ref_motion_times['pickUp'] < self.motionlib.contact_index[self._reset_ref_motion_ids['pickUp']]
            linear[supported] = 0.
            angular[supported] = 0.
            self.box_states[ids, 7:10], self.box_states[ids, 10:13] = linear, angular
        if len(self._bank_reset_ids):
            ids = self._bank_reset_ids
            self.box_states[ids] = self.hold_bank['box'][self._bank_indices]
            self.box_states[ids, :3] += self.env_origins[ids]
            self.platform_pos[ids] = self.platform_default_pos[ids]
            self.has_seen_tag[ids] = False

    def _reset_task(self, env_ids):
        super()._reset_task(env_ids)
        self.commands[env_ids] = 0.
        ids = env_ids[self.phase[env_ids] == APPROACH]
        lo, hi = self.cfg.commands.approach_vx_range
        self.commands[ids, 0] = lo + torch.rand(len(ids), device=self.device) * (hi - lo)
        self.carry_policy_commands[env_ids] = self.commands[env_ids]

    def reset_idx(self, env_ids):
        if not len(env_ids):
            return
        # Parent clears episode_sums and resets phase-independent physics buffers.
        stats = self.episode_diagnostics(env_ids)
        super().reset_idx(env_ids)
        self.reset_ee[env_ids], self.reset_torso_inverse[env_ids] = self.reset_kinematics.end_effectors(self.dof_pos[env_ids], return_torso=True)
        self.extras['episode'].update(stats)
        for name in ('contact_seconds', 'completion_seconds', 'hold_elapsed', 'stable_elapsed',
                     'longest_hold', 'hold_failure_seconds', 'completion_at', 'approach_elapsed', 'pickup_elapsed'):
            getattr(self, name)[env_ids] = 0.
        self.stable_delay[env_ids] = -1.
        self.pickup_completed[env_ids] = False
        self.hold_success[env_ids] = False
        self.phase_age[env_ids] = 0
        self.pending_observation_reset[env_ids] = True
        self.amp_obs_buf[env_ids] = 0.
        self.obs_buf[env_ids] = 0.
        self.last_contacts[env_ids] = False
        self.contact_filt[env_ids] = False
        hold = env_ids[self.phase[env_ids] == HOLD]
        self.hold_box_height[hold] = self.box_states[hold, 2] - self.env_origins[hold, 2]
        self.hold_box_quat[hold] = self.box_states[hold, 3:7]

    def compute_observations(self):
        super().compute_observations()
        ids = self.pending_observation_reset.nonzero().flatten()
        if not len(ids):
            return
        # Gym's rigid-body tensor still describes the terminal state immediately
        # after indexed resets. Seed histories from the reset state, not that tensor.
        q = self.root_states[ids, 3:7]
        torso_inverse = self.reset_torso_inverse[ids]
        def in_torso(value):
            return (torso_inverse @ quat_rotate_inverse(q, value).unsqueeze(-1)).squeeze(-1)
        angular = in_torso(self.root_states[ids, 10:13])
        gravity = in_torso(self.gravity_vec[ids])
        box_local = in_torso(self.box_states[ids, :3] - self.root_states[ids, :3])
        from isaacgym.torch_utils import quat_mul, quat_conjugate
        from legged_gym.utils.torch_utils import quat_to_tan_norm
        box_rot = quat_to_tan_norm(quat_mul(quat_conjugate(q), self.box_states[ids, 3:7]))
        box_rot = (torso_inverse[:, None] @ box_rot.reshape(-1, 2, 3, 1)).reshape(-1, 6)
        proprio = torch.cat((angular * self.obs_scales.ang_vel, gravity,
            (self.dof_pos[ids] - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel[ids] * self.obs_scales.dof_vel, self.reset_ee[ids],
            torch.zeros(len(ids), 29, device=self.device)), -1)
        task = torch.cat((box_local, box_rot, self._box_size[ids], self.carry_policy_commands[ids, :3]), -1)
        self.obs_buf[ids] = torch.cat((proprio, task), -1).repeat(1, self.actor_history_length)
        self.privileged_obs_buf[ids] = torch.cat((proprio,
            in_torso(self.root_states[ids, 7:10]) * self.obs_scales.lin_vel, task), -1)
        self.pending_observation_reset[ids] = False

    def diagnostics(self):
        _, bilateral, feet = self._physical_status()
        angle = 2 * torch.acos((self.box_states[:, 3:7] * self.hold_box_quat).sum(-1).abs().clamp(0, 1))
        return dict(phase=self.phase, reset_category=self.reset_category,
            approach_completed=self.phase != APPROACH,
            pickup_completed=self.pickup_completed, hold_success=self.hold_success,
            hold_elapsed=self.hold_elapsed, stable_elapsed=self.stable_elapsed,
            longest_hold=self.longest_hold, stable_delay=self.stable_delay,
            approach_seconds=self.approach_elapsed, pickup_seconds=self.pickup_elapsed,
            base_linear_speed=self.root_states[:, 7:10].norm(dim=-1),
            base_angular_speed=self.root_states[:, 10:13].norm(dim=-1),
            box_linear_speed=self.box_states[:, 7:10].norm(dim=-1),
            box_angular_speed=self.box_states[:, 10:13].norm(dim=-1),
            bilateral_contact=bilateral, both_feet_contact=feet,
            foot_slip=self.feet_vel[:, :, :2].norm(dim=-1).amax(-1),
            box_height_deviation=(self.box_states[:, 2] - self.env_origins[:, 2] - self.hold_box_height).abs(),
            box_orientation_deviation=angle, command_speed=self.carry_policy_commands[:, :3].norm(dim=-1))

    def episode_diagnostics(self, ids):
        valid = (self.stable_delay[ids] >= 0) & self.pickup_completed[ids]
        # RSI HOLD resets are excluded from pickup success denominators.
        pickup = self.reset_category[ids] != 0
        return {'pickup_success_rate': self.pickup_completed[ids][pickup].float().mean() if pickup.any() else 0.,
                'successful_hold_seconds': self.longest_hold[ids].mean(),
                'hold_success_rate': self.hold_success[ids].float().mean(),
                'pickup_to_stable_hold_seconds': self.stable_delay[ids][valid].mean() if valid.any() else -1.}
