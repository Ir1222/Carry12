"""G1 approaches, picks up and immediately holds. No loaded locomotion phase."""
import math
import torch
from isaacgym import gymapi
from isaacgym.torch_utils import quat_rotate_inverse

from .carrybox import LeggedRobot as CarryBox
from .pickup_stand_logic import (APPROACH, PICKUP, HOLD, advance_phase,
                                  completion_predicates, palm_box_contact, ResetKinematics,
                                  validate_reset_weights, per_foot_support)
from ..motionlib.pickup_stand_sampling import PickupStandMotionSampler
from legged_gym.utils.helpers import class_to_dict


class LeggedRobot(CarryBox):
    def _parse_cfg(self, cfg):
        # CarryBox's parser requires a positive moving-carry command range.
        # This task has no moving-carry objective: parse the common simulator
        # fields directly, without changing that range or the parent's guard.
        self.dt = cfg.control.decimation * self.sim_params.dt
        self.obs_scales = cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(cfg.rewards.scales)
        self.command_ranges = class_to_dict(cfg.commands.ranges)
        if cfg.terrain.mesh_type not in ('heightfield', 'trimesh'):
            cfg.terrain.curriculum = False
        self.max_episode_length_s = cfg.env.episode_length_s
        self.max_episode_length = math.ceil(self.max_episode_length_s / self.dt)
        cfg.domain_rand.push_interval = math.ceil(cfg.domain_rand.push_interval_s / self.dt)

    def _init_buffers(self):
        super()._init_buffers()
        # reset() runs before the first physics step for loaded RSI. The parent
        # otherwise creates this buffer only inside post_physics_step().
        self.contact_filt = torch.zeros_like(self.last_contacts)
        self._init_support_buffers()
        self.motionlib = PickupStandMotionSampler(self.motionlib)
        from legged_gym import LEGGED_GYM_ROOT_DIR
        self.reset_kinematics = ResetKinematics(self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR), self.dof_names)
        self.phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.reset_category = torch.zeros_like(self.phase)
        self.phase_age = torch.zeros_like(self.phase)
        self.pending_observation_reset = torch.zeros_like(self.phase, dtype=torch.bool)
        for name in ('approach_ready_seconds', 'completion_seconds', 'hold_elapsed',
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
        validate_reset_weights(self.cfg.pickup_stand.reset_weights, path is not None)
        if path:
            self.hold_bank = torch.load(path, map_location=self.device, weights_only=False)
            self._validate_bank()
        expected = list(self.cfg.init_state.default_joint_angles)
        if self.dof_names != expected or self.num_obs != 738 or self.num_actions != 29:
            raise ValueError('Full AMP interface/joint order mismatch')
        self.policy_metadata = dict(task='carrybox_pickup_stand',
            observation_semantics=self.cfg.observation_semantics, actor_obs=738,
            critic_obs=126, actions=29, dof_names=self.dof_names,
            initialization='scratch', approach_speed=self.cfg.rewards.target_speed_loco)

    def _init_support_buffers(self):
        """Tensor rows are DOMAIN_ENV: platform, target platform, box, robot."""
        names = self.gym.get_actor_rigid_body_names(self.envs[0], self.actor_handles[0])
        offset = self.rigid_body_states.shape[1] - len(names)
        if offset != 3:
            raise ValueError('Unexpected actor/body tensor layout')
        def resolve(name):
            index = self.gym.find_actor_rigid_body_index(self.envs[0], self.actor_handles[0], name, gymapi.DOMAIN_ENV)
            if name not in names or index != offset + names.index(name):
                raise ValueError(f'Invalid DOMAIN_ENV body index for {name}: {index}')
            return index
        support = self.cfg.pickup_stand.support_body_names
        self.support_indices = torch.tensor([resolve(name) for name in support], device=self.device)
        pitch = ['left_ankle_pitch_link', 'right_ankle_pitch_link']
        self.pitch_contact_indices = torch.tensor([resolve(name) for name in pitch], device=self.device)
        self.support_manifest = dict(domain='DOMAIN_ENV', robot_offset=offset,
            support_names=support, support_indices=self.support_indices.cpu().tolist(),
            pitch_indices=self.pitch_contact_indices.cpu().tolist(),
            amp_foot_indices=self.feet_indices.cpu().tolist())
        self.previous_support = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
        self.footstep_events = torch.zeros(self.num_envs, device=self.device)

    def _foot_support(self):
        return per_foot_support(self.contact_forces, self.support_indices, self.cfg.pickup_stand.foot_force)

    def _support_velocities(self):
        return self.rigid_body_states[:, self.support_indices, 7:10]

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
        if bank.get('dof_names') != self.dof_names or bank.get('version') != 2:
            raise ValueError('HOLD bank joint ordering/version mismatch')
        if bank.get('support_manifest', {}).get('support_names') != self.cfg.pickup_stand.support_body_names:
            raise ValueError('HOLD bank must validate the actual shoe collision bodies')
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
        feet_down = self._foot_support()[0].all(-1)
        bilateral = self._bilateral_contact()
        base_height = self.root_states[:, 2] - self.env_origins[:, 2]
        box_height = self.box_states[:, 2] - self.env_origins[:, 2]
        base_tilt = torch.acos((-self.projected_gravity[:, 2]).clamp(-1, 1))
        box_tilt = torch.acos((-self.projected_gravity_box[:, 2]).clamp(-1, 1))
        # Both ground clearance and separation from the source platform top.
        bottom = self.box_states[:, 2] - self._box_size[:, 2] / 2
        support_top = torch.maximum(self.env_origins[:, 2], self.platform_pos[:, 2] + self._platform_height / 2)
        relative = (self.box_states[:, 7:10] - self.root_states[:, 7:10]).norm(dim=-1)
        self.completion_values = dict(base_height=base_height, base_tilt=base_tilt,
            box_height=box_height, box_tilt=box_tilt, clearance=bottom - support_top,
            relative_speed=relative, base_vertical_speed=self.root_states[:, 9],
            box_vertical_speed=self.box_states[:, 9])
        self.completion_checks = completion_predicates(base_height, base_tilt, box_height, box_tilt,
            bottom - support_top, bilateral, feet_down, relative,
            self.root_states[:, 9], self.box_states[:, 9], cfg)
        complete = torch.stack(list(self.completion_checks.values()), -1).all(-1)
        return complete, bilateral, feet_down

    def _compute_carry_velocity_active(self):
        # The parent calls this before our phase update; no dynamic carry gate.
        return torch.zeros_like(self.phase, dtype=torch.bool)

    def _clear_velocity_commands(self, env_ids=None):
        if env_ids is None:
            self.commands.zero_()
            self.carry_policy_commands.zero_()
        else:
            self.commands[env_ids] = 0.
            self.carry_policy_commands[env_ids] = 0.

    def _resample_commands(self, env_ids):
        self._clear_velocity_commands(env_ids)

    _sample_episode_vx_commands = _resample_commands
    _resample_carry_yaw_commands = _resample_commands

    def _sample_carry_yaw_resample_time(self, env_ids):
        self.carry_yaw_resample_time[env_ids] = 0.

    def update_command_curriculum(self, env_ids):
        self._clear_velocity_commands(env_ids)

    def _facing_error(self):
        target = torch.atan2(self.robot2object_dir[:, 1], self.robot2object_dir[:, 0])
        error = target - self.yaw
        return torch.atan2(torch.sin(error), torch.cos(error)).abs()

    def _update_carry_heading_commands(self):
        old = self.phase.clone()
        complete, bilateral, _ = self._physical_status()
        cfg = self.cfg.pickup_stand
        ready = (self.robot2object_dist < self.cfg.rewards.thresh_robot2object) & (self._facing_error() <= cfg.pickup_facing_tolerance)
        self.phase, self.approach_ready_seconds, self.completion_seconds, entered = advance_phase(
            self.phase, self.approach_ready_seconds, self.completion_seconds,
            ready, complete, self.dt, cfg.approach_seconds, cfg.completion_seconds)
        self.phase_age = torch.where(old == self.phase, self.phase_age + 1, 0)
        self.pickup_completed |= entered
        self.completion_at[entered] = self.episode_length_buf[entered] * self.dt
        self.hold_box_height[entered] = self.box_states[entered, 2] - self.env_origins[entered, 2]
        self.hold_box_quat[entered] = self.box_states[entered, 3:7]
        self._clear_velocity_commands()
        self.carry_heading_error.zero_()
        self.carry_velocity_active.zero_()
        self.carry_tracking_started.zero_()
        support = self._foot_support()[0]
        self.footstep_events += ((self.phase == HOLD)[:, None] & ~support & self.previous_support).sum(-1)
        self.previous_support[:] = support
        self.approach_elapsed += (self.phase == APPROACH) * self.dt
        self.pickup_elapsed += (self.phase == PICKUP) * self.dt
        self.hold_elapsed += (self.phase == HOLD) * self.dt
        stable = (self.phase == HOLD) & complete
        stable &= self.root_states[:, 7:10].norm(dim=-1) <= cfg.stable_base_speed
        stable &= self.root_states[:, 10:13].norm(dim=-1) <= cfg.stable_base_angular_speed
        stable &= self.box_states[:, 7:10].norm(dim=-1) <= cfg.stable_box_speed
        stable &= self.box_states[:, 10:13].norm(dim=-1) <= cfg.stable_box_angular_speed
        stable &= self._support_velocities().norm(dim=-1).amax(-1) <= cfg.stable_foot_speed
        self.stable_elapsed = torch.where(stable, self.stable_elapsed + self.dt, 0.)
        self.longest_hold = torch.maximum(self.longest_hold, self.stable_elapsed)
        first_stable = (self.stable_elapsed >= cfg.stable_seconds) & (self.stable_delay < 0)
        self.stable_delay[first_stable] = self.hold_elapsed[first_stable] - cfg.stable_seconds
        self.hold_success |= self.stable_elapsed >= max(cfg.success_hold_seconds, cfg.stable_seconds)
        self.hold_failure_seconds = torch.where((self.phase == HOLD) & ~complete,
                                                 self.hold_failure_seconds + self.dt, 0.)

    def check_termination(self):
        super().check_termination()
        self.reset_buf |= self.hold_failure_seconds > self.cfg.pickup_stand.failure_grace_seconds
        if self.cfg.pickup_stand.terminate_on_success:
            self.reset_buf |= self.hold_success

    def compute_amp_observations(self):
        result = super().compute_amp_observations()
        # Snapshot BEFORE parent autoreset. Exclude mixed-stage/initial histories.
        valid = (self.phase != HOLD) & (self.phase_age >= self.cfg.amp.window_length)
        self.extras['amp_stage'] = torch.where(valid, self.phase, -1).clone()
        self.extras['pickup_stand'] = {k: v.clone() for k, v in self.diagnostics().items()}
        return result

    def _reward_walk_task(self):
        # Original PhysHSI world-frame box-direction speed + facing objective.
        direction = torch.nn.functional.normalize(self.robot2object_dir, dim=-1)
        speed = (direction * self.rigid_body_states[:, self.upper_body_index, 7:9]).sum(-1)
        velocity = torch.exp(-5 * (self.cfg.rewards.target_speed_loco - speed).square())
        near = self.robot2object_dist < self.cfg.rewards.thresh_robot2object
        velocity = torch.where(near, 1., velocity)
        position = torch.where(near, 1., torch.exp(-.5 * self.robot2object_dist))
        facing = torch.exp(-.75 * self._facing_error())
        cfg = self.cfg.rewards
        return (self.phase == APPROACH) * (cfg.robot2object_pos * position + cfg.robot2object_vel * velocity + cfg.start_heading * facing)

    def _reward_carryup_task(self):
        return (self.phase != APPROACH) * super()._reward_carryup_task()

    def _reward_pickup_grasp(self):
        return (self.phase == PICKUP) * self._bilateral_contact().float()

    def _lift_progress(self):
        bottom = self.box_states[:, 2] - self._box_size[:, 2] / 2
        top = torch.maximum(self.env_origins[:, 2], self.platform_pos[:, 2] + self._platform_height / 2)
        return ((bottom - top) / self.cfg.pickup_stand.min_clearance).clamp(0., 1.)

    def _standing_posture(self):
        height = self.root_states[:, 2] - self.env_origins[:, 2]
        excess = (self.cfg.pickup_stand.min_base_height - height).clamp_min(0) + (height - .88).clamp_min(0)
        leg_error = (self.dof_pos[:, :12] - self.default_dof_pos[:, :12]).square().mean(-1)
        return torch.exp(-excess.square() / .01 - self.projected_gravity[:, :2].square().sum(-1) / .09 - .3 * leg_error)

    def _reward_pickup_standup(self):
        return (self.phase == PICKUP) * self._lift_progress() * self._standing_posture()

    def _reward_loaded_balance(self):
        # Continuous feedback before the hard completion gate (including a foot
        # or posture predicate still failing); never demands moving-carry motion.
        base = torch.exp(-self.root_states[:, 7:10].square().sum(-1) / .02 - self.root_states[:, 10:13].square().sum(-1) / .08)
        box = torch.exp(-self.box_states[:, 7:10].square().sum(-1) / .02 - self.box_states[:, 10:13].square().sum(-1) / .08)
        support = self._foot_support()[1].clamp_min(0) / self.cfg.pickup_stand.foot_force
        feet = support.clamp_max(1).mean(-1) * torch.exp(-self._support_velocities().square().sum((1, 2)) / .02)
        box_level = torch.exp(-self.projected_gravity_box[:, :2].square().sum(-1) / .09)
        gate = (self.phase == PICKUP) * self._lift_progress() * self._bilateral_contact().float()
        return gate * (.2 * base + .2 * box + .25 * self._standing_posture() + .2 * feet + .15 * box_level)

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
        return (self.phase == HOLD) * self._standing_posture()

    def _reward_hold_feet(self):
        down = self._foot_support()[0].all(-1)
        return (self.phase == HOLD) * down * torch.exp(-self._support_velocities().square().sum((1, 2)) / .02)

    def _reward_hold_action_rate(self):
        return (self.phase == HOLD) * super()._reward_action_rate()

    def _reward_hold_footsteps(self):
        support = self._foot_support()[0]
        # Both lost support and unnecessary foot motion cost reward; no action
        # freezing or rejection of small physical recovery adjustments.
        return (self.phase == HOLD) * ((~support).float().sum(-1) + self._support_velocities().square().sum((1, 2)))

    def _reward_task_success(self):
        # _prepare_reward_function multiplies scales by dt. Undo that here so
        # the configured success scale is a terminal bonus, not a rate.
        return self.hold_success.float() / self.dt

    def _reset_actors(self, env_ids):
        cfg = self.cfg.pickup_stand
        weights = validate_reset_weights(cfg.reset_weights, self.hold_bank is not None).to(self.device)
        categories = torch.multinomial(weights, len(env_ids), replacement=True)
        self.reset_category[env_ids] = categories
        self.phase[env_ids] = APPROACH
        # PhysHSI-style hybrid coverage: unloaded default + loco RSI approaches,
        # with pickup RSI stratified separately. Every category is explicit.
        approach_ids = env_ids[categories == 4]
        if len(approach_ids):
            use_rsi = torch.rand(len(approach_ids), device=self.device) < cfg.approach_rsi_probability
            default = approach_ids[~use_rsi]
            if len(default):
                self._reset_default(default)
            reference = approach_ids[use_rsi]
            if len(reference):
                motions = self.motionlib.sample_motions('loco', len(reference))
                times = self.motionlib.sample_time_rsi('loco', motions)
                self._set_reference_reset(reference, 'loco', motions, times)
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
            self._set_reference_reset(ids, 'pickUp', motions, times)
            self.phase[ids] = PICKUP  # reaching/grasping is part of pickup AMP
        for mapping in (self._reset_ref_env_ids, self._reset_ref_motion_ids, self._reset_ref_motion_times):
            for skill in mapping:
                mapping[skill] = torch.cat(mapping[skill])
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

    def _set_reference_reset(self, ids, skill, motions, times):
        root, quat, linear, angular, joints, velocity, ee = self.motionlib.get_motion_state(skill, motions, times)
        end = self.motionlib.motion_len[skill][motions].float() - 1
        t0, t1 = (times - .5).clamp_min(0), torch.minimum(times + .5, end)
        p0 = self.motionlib.get_motion_state(skill, motions, t0)[0]
        p1 = self.motionlib.get_motion_state(skill, motions, t1)[0]
        linear[:, 2] = (p1[:, 2] - p0[:, 2]) * self.motionlib.fps / (t1 - t0).clamp_min(1.e-6)
        self.root_states[ids, :3] = root + self.env_origins[ids]
        self.root_states[ids, 3:7] = quat
        self.root_states[ids, 7:10] = linear
        self.root_states[ids, 10:13] = angular
        self.dof_pos[ids], self.dof_vel[ids], self.reset_ee[ids] = joints, velocity, ee
        for mapping, value in [(self._reset_ref_env_ids, ids), (self._reset_ref_motion_ids, motions), (self._reset_ref_motion_times, times)]:
            mapping.setdefault(skill, []).append(value)

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
        self.tar_platform_states[env_ids] = self.tar_platform_default_states[env_ids]
        self._clear_velocity_commands(env_ids)

    def reset_idx(self, env_ids):
        if not len(env_ids):
            return
        # Parent clears episode_sums and resets phase-independent physics buffers.
        stats = self.episode_diagnostics(env_ids)
        super().reset_idx(env_ids)
        self.reset_ee[env_ids], self.reset_torso_inverse[env_ids] = self.reset_kinematics.end_effectors(self.dof_pos[env_ids], return_torso=True)
        self.extras['episode'].update(stats)
        for name in ('approach_ready_seconds', 'completion_seconds', 'hold_elapsed', 'stable_elapsed',
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
        self.previous_support[env_ids] = False
        self.footstep_events[env_ids] = 0.
        hold = env_ids[self.phase[env_ids] == HOLD]
        self.hold_box_height[hold] = self.box_states[hold, 2] - self.env_origins[hold, 2]
        self.hold_box_quat[hold] = self.box_states[hold, 3:7]

    def compute_task_observations(self):
        actor, critic = super().compute_task_observations()
        phase = torch.nn.functional.one_hot(self.phase, 3).to(actor.dtype)
        actor[:, -3:] = phase
        critic[:, -3:] = phase
        return actor, critic

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
        phase = torch.nn.functional.one_hot(self.phase[ids], 3).to(proprio.dtype)
        task = torch.cat((box_local, box_rot, self._box_size[ids], phase), -1)
        self.obs_buf[ids] = torch.cat((proprio, task), -1).repeat(1, self.actor_history_length)
        self.privileged_obs_buf[ids] = torch.cat((proprio,
            in_torso(self.root_states[ids, 7:10]) * self.obs_scales.lin_vel, task), -1)
        self.pending_observation_reset[ids] = False

    def diagnostics(self):
        _, bilateral, feet = self._physical_status()
        angle = 2 * torch.acos((self.box_states[:, 3:7] * self.hold_box_quat).sum(-1).abs().clamp(0, 1))
        support, forces = self._foot_support()
        result = dict(phase=self.phase, reset_category=self.reset_category,
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
            left_foot_support=support[:, 0], right_foot_support=support[:, 1],
            left_foot_force_z=forces[:, 0], right_foot_force_z=forces[:, 1],
            left_pitch_force_z=self.contact_forces[:, self.pitch_contact_indices[0], 2],
            right_pitch_force_z=self.contact_forces[:, self.pitch_contact_indices[1], 2],
            foot_slip=self._support_velocities()[:, :, :2].norm(dim=-1).amax(-1),
            footstep_events=self.footstep_events, completion_hysteresis_seconds=self.completion_seconds,
            box_height_deviation=(self.box_states[:, 2] - self.env_origins[:, 2] - self.hold_box_height).abs(),
            box_orientation_deviation=angle, command_speed=self.carry_policy_commands[:, :3].norm(dim=-1))
        result.update({'completion/' + name: value for name, value in self.completion_checks.items()})
        result.update({'completion_value/' + name: value for name, value in self.completion_values.items()})
        result['completion/all'] = torch.stack(list(self.completion_checks.values()), -1).all(-1)
        return result

    def episode_diagnostics(self, ids):
        valid = (self.stable_delay[ids] >= 0) & self.pickup_completed[ids]
        # RSI HOLD resets are excluded from pickup success denominators.
        pickup = self.reset_category[ids] != 0
        return {'pickup_success_rate': self.pickup_completed[ids][pickup].float().mean() if pickup.any() else 0.,
                'successful_hold_seconds': self.longest_hold[ids].mean(),
                'hold_success_rate': self.hold_success[ids].float().mean(),
                'pickup_to_stable_hold_seconds': self.stable_delay[ids][valid].mean() if valid.any() else -1.}
