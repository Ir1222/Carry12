"""CPU tests execute production tensor code; Gym integration is a separate smoke."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace as NS
import os
import sys
import torch
import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'experiments/pickup_stand'))
from audit import inspect_references, inspect_checkpoint, load_source


def source_without_imports(path, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding='utf-8'))
    tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    exec(compile(tree, str(path), 'exec'), namespace)
    return namespace


logic = load_source('pickup_logic', 'legged_gym/legged_gym/envs/g1/pickup_stand_logic.py')
sampling = load_source('pickup_sampling', 'legged_gym/legged_gym/envs/motionlib/pickup_stand_sampling.py')
amp_module = load_source('pickup_amp', 'rsl_rl/rsl_rl/modules/amp.py')
actor_module = load_source('pickup_actor', 'rsl_rl/rsl_rl/modules/actor_critic.py')


def rotate_inverse(q, v):
    u, w = q[..., :3], q[..., 3:4]
    return v - 2 * w * torch.cross(u, v, dim=-1) + 2 * torch.cross(u, torch.cross(u, v, dim=-1), dim=-1)


def multiply(a, b):
    return torch.cat((a[..., 3:4] * b[..., :3] + b[..., 3:4] * a[..., :3] + torch.cross(a[..., :3], b[..., :3], dim=-1),
                      a[..., 3:4] * b[..., 3:4] - (a[..., :3] * b[..., :3]).sum(-1, keepdim=True)), -1)


def heading(q):
    yaw = torch.atan2(2 * (q[:, 3] * q[:, 2] + q[:, 0] * q[:, 1]), 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2))
    out = torch.zeros_like(q)
    out[:, 2], out[:, 3] = torch.sin(yaw / 2), torch.cos(yaw / 2)
    return out


def conjugate(q):
    return q * q.new_tensor([-1, -1, -1, 1])


def tan_norm(q):
    return torch.cat((rotate_inverse(conjugate(q), q.new_tensor([1., 0, 0]).expand(len(q), -1)),
                      rotate_inverse(conjugate(q), q.new_tensor([0., 0, 1]).expand(len(q), -1))), -1)


@pytest.fixture(scope='module')
def cfg():
    base = load_source('base_config', 'legged_gym/legged_gym/envs/base/base_config.py')
    ns = source_without_imports('legged_gym/legged_gym/envs/base/legged_robot_config.py', {'BaseConfig': base.BaseConfig})
    carry = source_without_imports('legged_gym/legged_gym/envs/g1/carrybox_config.py', dict(ns, np=np))
    selected = source_without_imports('legged_gym/legged_gym/envs/g1/carrybox_pickup_stand_config.py',
        dict(CarryCfg=carry['G1Cfg'], CarryPPO=carry['G1CfgPPO']))
    return selected['G1Cfg']()


@pytest.fixture(scope='module')
def library():
    ns = source_without_imports('legged_gym/legged_gym/envs/motionlib/motionlib_carrybox.py',
        dict(torch=torch, np=np, os=os, yaml=yaml, torch_utils=NS(
            calc_heading_quat_inv=lambda q: conjugate(heading(q)), quat_mul=multiply,
            quat_to_tan_norm=tan_norm, quat_rotate_inverse=rotate_inverse, calc_heading_quat=heading)))
    names = [s.split()[1] for s in (ROOT / 'legged_gym/resources/config/joint_id.txt').read_text().splitlines()]
    lib = ns['MotionLib'](str(ROOT / 'legged_gym/resources/config/carrybox_pickup_stand.yaml'),
        str(ROOT / 'legged_gym/resources/config/joint_id.txt'), names, 60, 'cpu', 10, [.95, 1.05], .7)
    return sampling.PickupStandMotionSampler(lib)


def test_motion_audit():
    result = inspect_references()
    assert result['available_expert_clips'] == 14
    assert all(r['kinematic_hold_candidates_in_tail'] == 0 for r in result['pickup'])
    assert all(r['root_angular_speed']['final'] > 1 for r in result['pickup'])


def test_checkpoint_strict_load():
    result = inspect_checkpoint(ROOT / 'shared_artifacts/2026-10-19/carry_track/model_33999.pt')
    assert result['strict_actor_and_amp_load'] and result['urdf_matches_motion_joint_order']


def test_actual_finetune_loader_does_not_restore_optimizer():
    ns = source_without_imports('rsl_rl/rsl_rl/runners/him_on_policy_runner.py',
                                dict(torch=torch, VecEnv=object))
    cls = ns['HIMOnPolicyRunner']
    runner = cls.__new__(cls)
    runner.device = 'cpu'
    model = actor_module.ActorCritic(738, 126, 6, 29, actor_hidden_dims=[512, 256, 256], critic_hidden_dims=[512, 256, 256])
    amp = amp_module.AMP(600, .25, device='cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=1.e-4)
    runner.alg = NS(actor_critic=model, amp=amp, optimizer=optimizer)
    runner.current_learning_iteration = 900
    runner.load_finetune(ROOT / 'shared_artifacts/2026-10-19/carry_track/model_33999.pt')
    assert runner.alg.optimizer is optimizer and not optimizer.state
    assert runner.current_learning_iteration == 0


def test_new_config_is_independent(cfg):
    assert cfg.env.num_actor_obs == 738 and cfg.env.num_actions == 29 and cfg.amp.num_obs == 600
    assert cfg.asset.box.skill == ['loco', 'pickUp']
    assert cfg.commands.resample_carry_commands is False
    base_cfg = type(cfg).__mro__[1]()
    assert base_cfg.asset.box.skill == ['loco', 'pickUp', 'carryWith']
    assert base_cfg.commands.resample_carry_commands is True
    assert base_cfg.dataset.motion_file.endswith('carrybox_no_relocation.yaml')


def test_phase_contact_hysteresis_and_irreversibility():
    phase = torch.tensor([0, 0, 2])
    contact = torch.zeros(3)
    complete = torch.zeros(3)
    for _ in range(4):
        phase, contact, complete, _ = logic.advance_phase(phase, contact, complete,
            torch.ones(3, dtype=torch.bool), torch.tensor([True, False, False]), torch.ones(3, dtype=torch.bool), .02, .1, .2)
    assert phase.tolist() == [0, 0, 2]
    phase, contact, complete, _ = logic.advance_phase(phase, contact, complete,
        torch.ones(3, dtype=torch.bool), torch.ones(3, dtype=torch.bool), torch.ones(3, dtype=torch.bool), .021, .1, .2)
    assert phase.tolist() == [1, 0, 2]
    # A failed physical condition clears, rather than pauses, completion evidence.
    _, _, complete, entered = logic.advance_phase(phase, contact, torch.ones(3) * .19,
        torch.ones(3, dtype=torch.bool), torch.ones(3, dtype=torch.bool), torch.tensor([False, True, False]), .02, .1, .2)
    assert complete[0] == 0 and not entered[0]


@pytest.mark.parametrize('field,value', [('base_height', .5), ('base_tilt', .4), ('box_height', .5),
    ('box_tilt', .4), ('clearance', .02), ('bilateral', False), ('feet_down', False),
    ('relative_speed', .4), ('base_vertical_speed', .4), ('box_vertical_speed', .4)])
def test_premature_lift_rejected(cfg, field, value):
    data = dict(base_height=.78, base_tilt=.1, box_height=.75, box_tilt=.1, clearance=.2,
                bilateral=True, feet_down=True, relative_speed=.05, base_vertical_speed=.02, box_vertical_speed=.02)
    good = {k: torch.tensor([v]) for k, v in data.items()}
    assert logic.completion_conditions(**good, cfg=cfg.pickup_stand).item()
    good[field] = torch.tensor([value])
    assert not logic.completion_conditions(**good, cfg=cfg.pickup_stand).item()


def test_contact_rejects_unrelated_force_same_side_and_rotates_box():
    palms = torch.tensor([[[0., .175, 0], [0, -.175, 0]]])
    forces = torch.ones(1, 2, 3) * 5
    pos, quat, size = torch.zeros(1, 3), torch.tensor([[0., 0, 0, 1]]), torch.ones(1, 3) * .35
    call = lambda p, q=quat: logic.palm_box_contact(p, pos, q, size, forces, 1., .06)[0].item()
    assert call(palms)
    assert not call(palms + 2)
    assert not call(palms.abs())
    rotated = palms.clone()
    rotated[..., 0], rotated[..., 1] = -palms[..., 1], palms[..., 0]
    assert call(rotated, torch.tensor([[0., 0, math.sqrt(.5), math.sqrt(.5)]]))


def test_stage_sampler_and_no_clip_leak(library):
    torch.manual_seed(1)
    out = library.get_expert_obs_for_stages(torch.tensor([0, 1] * 16))
    assert out.shape == (32, 600) and torch.isfinite(out).all()
    with pytest.raises(ValueError):
        library.get_expert_obs_for_stages(torch.tensor([2]))
    ids = torch.tensor([2])  # final pickup clip immediately precedes loco
    end = library.motion_len['pickUp'][ids].float() - 1
    history = library.get_amp_hist_obs('pickUp', ids, end).reshape(1, 10, 60)
    index = library.motion_end_ids['pickUp'][2] - 1
    assert torch.allclose(history[0, -1, :1], library.motion_base_height[index])
    early = library.get_amp_hist_obs('pickUp', ids, torch.zeros(1)).reshape(1, 10, 60)
    assert torch.allclose(early[:, 0], early[:, -1])


def test_reward_mixing():
    amp = amp_module.AMP(600, .25, hidden_dims=[8, 4], device='cpu')
    reward = amp.combine_reward_for_stages(torch.ones(4), torch.ones(4) * 4, torch.tensor([0, 1, 2, -1]))
    assert torch.allclose(reward, torch.tensor([3.25, 3.25, 4., 4.]))
    assert torch.allclose(amp.combine_reward(torch.ones(4), torch.ones(4) * 4), torch.full((4,), 3.25))


def runtime_class():
    # The harness supplies only the Gym boundary; all task methods are production.
    class Parent:
        def _reward_walk_task(self):
            return torch.ones(self.num_envs)
        def _reward_action_rate(self):
            return torch.ones(self.num_envs)
        def _reset_default(self, ids):
            self._reset_default_env_ids = ids
            self.root_states[ids, :3] = self.env_origins[ids] + torch.tensor([2.3, 0., .8])
            self.dof_pos[ids] = 0
            self.dof_vel[ids] = 0
        def _reset_boxes(self, ids):
            for skill, selected in self._reset_ref_env_ids.items():
                p, q, supported, platform = self.motionlib.get_obj_motion_state(skill,
                    self._reset_ref_motion_ids[skill], self._reset_ref_motion_times[skill])
                self.box_states[selected, :3] = p + self.env_origins[selected]
                self.box_states[selected, 3:7] = q
        def _reset_task(self, ids):
            pass
        def reset_idx(self, ids):
            self._reset_default_env_ids = []
            self._reset_ref_env_ids, self._reset_ref_motion_ids, self._reset_ref_motion_times = {}, {}, {}
            self._reset_actors(ids)
            self._reset_boxes(ids)
            self._reset_task(ids)
            self.extras['episode'] = {}
        def compute_amp_observations(self):
            return torch.zeros(self.num_envs, 600)
        def compute_observations(self):
            self.obs_buf[:] = 99
            self.privileged_obs_buf[:] = 99
        def post_physics_step(self):
            return torch.tensor([1, 3]), torch.zeros(2, 126), torch.zeros(4, 600)
    ns = {k: getattr(logic, k) for k in ('APPROACH', 'PICKUP', 'HOLD', 'advance_phase', 'completion_conditions', 'palm_box_contact', 'ResetKinematics')}
    ns.update(CarryBox=Parent, torch=torch, warnings=__import__('warnings'), PickupStandMotionSampler=sampling.PickupStandMotionSampler,
              quat_rotate_inverse=rotate_inverse)
    return source_without_imports('legged_gym/legged_gym/envs/g1/carrybox_pickup_stand.py', ns)['LeggedRobot']


@pytest.fixture
def env(cfg, library):
    import copy
    cls = runtime_class()
    obj = cls.__new__(cls)
    obj.cfg, obj.motionlib, obj.num_envs, obj.device, obj.dt = copy.deepcopy(cfg), library, 4, 'cpu', .02
    obj.phase = torch.zeros(4, dtype=torch.long)
    obj.phase_age = torch.zeros(4, dtype=torch.long)
    obj.reset_category = torch.zeros(4, dtype=torch.long)
    obj.root_states = torch.zeros(4, 13)
    obj.root_states[:, 2], obj.root_states[:, 6] = .78, 1
    obj.box_states = obj.root_states.clone()
    obj.box_states[:, 0], obj.box_states[:, 2] = .4, .75
    obj.env_origins, obj.platform_pos = torch.zeros(4, 3), torch.zeros(4, 3)
    obj._platform_height = .1
    obj.platform_default_pos = torch.ones(4, 3) * -10
    obj._box_size = torch.tensor([[.35, .35, .3]]).repeat(4, 1)
    obj.contact_forces = torch.zeros(4, 4, 3)
    obj.contact_forces[:, :, 2] = 20
    obj.hand_pos_indices = obj.hand_colli_indices = [0, 1]
    obj.feet_indices = [2, 3]
    obj.rigid_body_states = torch.zeros(4, 4, 13)
    obj.rigid_body_states[:, :2, :3] = obj.box_states[:, None, :3]
    obj.rigid_body_states[:, 0, 1], obj.rigid_body_states[:, 1, 1] = .175, -.175
    obj.feet_vel = torch.zeros(4, 2, 3)
    obj.projected_gravity = obj.projected_gravity_box = torch.tensor([[0., 0., -1.]]).repeat(4, 1)
    obj.robot2object_dist = torch.ones(4) * .4
    obj.commands = torch.ones(4, 4) * .5
    obj.carry_policy_commands = obj.commands.clone()
    obj.carry_heading_error = torch.zeros(4)
    obj.episode_length_buf = torch.ones(4) * 10
    obj.dof_pos, obj.dof_vel, obj.default_dof_pos = torch.zeros(4, 29), torch.zeros(4, 29), torch.zeros(1, 29)
    for name in ('contact_seconds', 'completion_seconds', 'hold_elapsed', 'stable_elapsed', 'longest_hold',
                 'hold_failure_seconds', 'completion_at', 'stable_delay', 'approach_elapsed', 'pickup_elapsed'):
        setattr(obj, name, torch.zeros(4))
    obj.stable_delay[:] = -1
    obj.pickup_completed, obj.hold_success = torch.zeros(4, dtype=torch.bool), torch.zeros(4, dtype=torch.bool)
    obj.hold_box_height = torch.ones(4) * .75
    obj.hold_box_quat = obj.box_states[:, 3:7].clone()
    obj.hold_bank = None
    obj.reset_ee = torch.zeros(4, 15)
    obj.reset_torso_inverse = torch.eye(3).repeat(4, 1, 1)
    obj.dof_names = list(library.mapping)
    obj.reset_kinematics = logic.ResetKinematics(ROOT / 'legged_gym/resources/robots/g1/urdf/g1_29dof.urdf', obj.dof_names)
    obj.pending_observation_reset = torch.zeros(4, dtype=torch.bool)
    obj.amp_obs_buf, obj.obs_buf = torch.ones(4, 600), torch.ones(4, 738)
    obj.privileged_obs_buf = torch.ones(4, 126)
    obj.gravity_vec = torch.tensor([[0., 0., -1.]]).repeat(4, 1)
    obj.obs_scales = NS(ang_vel=.25, dof_pos=1., dof_vel=.05, lin_vel=2.)
    obj.actor_history_length = 6
    obj.last_contacts = torch.ones(4, 2, dtype=torch.bool)
    obj.contact_filt = obj.last_contacts.clone()
    obj.extras = {}
    obj.has_seen_tag = torch.zeros(4, dtype=torch.bool)
    return obj


def test_runtime_direct_hold_zero_commands_and_no_walk(env):
    for _ in range(18):
        env._update_carry_heading_commands()
    assert (env.phase == 2).all() and env.pickup_completed.all()
    assert (env.commands == 0).all() and (env.carry_policy_commands[:, :3] == 0).all()
    assert (env._reward_walk_task() == 0).all() and env.carry_tracking_started.all()
    env.commands[:] = .8  # any external resampling is overridden on next step
    env._update_carry_heading_commands()
    assert (env.commands == 0).all()
    env.compute_amp_observations()
    assert (env.extras['amp_stage'] == -1).all()


def test_runtime_loaded_moving_completes_without_walk_to_stop(env):
    env.root_states[:, 7] = env.box_states[:, 7] = .5
    for _ in range(18):
        env._update_carry_heading_commands()
    assert (env.phase == 2).all()  # no speed-dependent moving-carry phase
    assert (env.stable_elapsed == 0).all() and (env.commands == 0).all()


def test_reset_pickup_preserves_reference_velocity_and_clears_buffers(env):
    env.cfg.pickup_stand.hold_reset_weights = [0, 0, 1, 0, 0]
    env.phase[:] = 2
    env.hold_elapsed[:] = 8
    env.reset_idx(torch.tensor([1, 3]))
    ids = env._reset_ref_env_ids['pickUp']
    assert (env.phase[ids] == 1).all() and (env.commands[ids] == 0).all()
    assert torch.any(env.dof_vel[ids] != 0) and torch.any(env.box_states[ids, 7:10] != 0)
    assert (env.obs_buf[ids] == 0).all() and (env.amp_obs_buf[ids] == 0).all()
    assert (env.hold_elapsed[ids] == 0).all() and (env.stable_delay[ids] == -1).all()
    assert (env.hold_elapsed[[0, 2]] == 8).all() and (env.phase[[0, 2]] == 2).all()


def test_reset_full_approach_keeps_movement(env):
    env.cfg.pickup_stand.hold_reset_weights = [0, 0, 0, 0, 1]
    env.reset_idx(torch.arange(4))
    assert (env.phase == 0).all() and (env.commands[:, 0] > 0).all()


def test_missing_hold_bank_falls_back_to_dynamic_late_pickup(env):
    env.cfg.pickup_stand.hold_reset_weights = [1, 0, 0, 0, 0]
    env.reset_idx(torch.arange(4))
    assert (env.reset_category == 3).all() and (env.phase == 1).all()


def test_reset_kinematics_matches_symmetry(env):
    ee = env.reset_kinematics.end_effectors(torch.zeros(4, 29)).reshape(4, 5, 3)
    assert torch.isfinite(ee).all()
    assert torch.allclose(ee[:, 2, 2], ee[:, 3, 2], atol=1.e-5)
    assert (ee[:, 2:4, 2] < -.6).all() and (ee[:, 4, 2] > .3).all()


def test_hold_bank_reset_preserves_state_and_requires_matching_physics(env):
    env.cfg.pickup_stand.hold_reset_weights = [1, 0, 0, 0, 0]
    env.box_masses = torch.ones(4) * 2.5
    env.hold_bank = dict(version=1, dof_names=env.dof_names,
        root=env.root_states[:1].clone(), box=env.box_states[:1].clone(),
        dof_pos=env.dof_pos[:1].clone(), dof_vel=env.dof_vel[:1].clone(), size=env._box_size[:1].clone(),
        mass=torch.tensor([2.5]), ee=env.reset_ee[:1].clone(), validated_seconds=torch.tensor([1.]),
        bilateral_contact=torch.tensor([True]), both_feet_contact=torch.tensor([True]), foot_slip=torch.tensor([0.]))
    env._validate_bank()
    env.reset_idx(torch.arange(4))
    assert (env.phase == 2).all() and (env.commands == 0).all()
    assert (env.box_states[:, 2] == .75).all() and not env.pickup_completed.any()
    env.box_masses[0] = 9
    with pytest.raises(ValueError, match='size/mass'):
        env.reset_idx(torch.tensor([0]))
    env.hold_bank['root'][0, 7] = 1.
    with pytest.raises(ValueError, match='moving'):
        env._validate_bank()


def test_initial_reset_seeds_complete_histories_without_zero_action_step(env, monkeypatch):
    from types import ModuleType
    gym_utils, observation_utils = ModuleType('isaacgym.torch_utils'), ModuleType('legged_gym.utils.torch_utils')
    gym_utils.quat_mul, gym_utils.quat_conjugate = multiply, conjugate
    observation_utils.quat_to_tan_norm = tan_norm
    monkeypatch.setitem(sys.modules, 'isaacgym.torch_utils', gym_utils)
    monkeypatch.setitem(sys.modules, 'legged_gym.utils.torch_utils', observation_utils)
    env.cfg.pickup_stand.hold_reset_weights = [0, 0, 1, 0, 0]
    obs, critic = env.reset()
    assert obs.shape == (4, 738) and critic.shape == (4, 126)
    history = obs.reshape(4, 6, 123)
    assert torch.allclose(history[:, 0], history[:, -1]) and not (obs == 99).any()
    assert torch.allclose(history[:, 0, 64:79], env.reset_ee)
    assert (history[:, 0, 79:108] == 0).all() and (history[:, 0, -3:] == 0).all()
    assert torch.isfinite(critic).all() and not env.pending_observation_reset.any()


def test_autoreset_removes_terminal_action_leak(env):
    for key in ('actions', 'last_actions', 'last_last_actions', 'last_dof_vel', 'last_torques'):
        setattr(env, key, torch.ones(4, 29) * 99)
    env.last_root_vel = torch.ones(4, 6) * 99
    env.post_physics_step()
    assert (env.last_actions[[1, 3]] == 0).all()
    assert (env.last_actions[[0, 2]] == 99).all()
    assert torch.equal(env.last_root_vel[[1, 3]], env.root_states[[1, 3], 7:13])


@pytest.mark.parametrize('stages', [None, [-1, 2, 2, -1], [0, 1, 2, -1]])
def test_actual_ppo_update_masks_discriminator(stages):
    storage_ns = source_without_imports('rsl_rl/rsl_rl/storage/him_rollout_storage.py', dict(torch=torch, np=np))
    ns = source_without_imports('rsl_rl/rsl_rl/algorithms/him_ppo.py', dict(torch=torch, nn=torch.nn,
        optim=torch.optim, ActorCritic=actor_module.ActorCritic, HIMRolloutStorage=storage_ns['HIMRolloutStorage']))
    model = actor_module.ActorCritic(738, 126, 6, 29, actor_hidden_dims=[16, 8], critic_hidden_dims=[16, 8])
    amp = amp_module.AMP(600, .25, hidden_dims=[16, 8], device='cpu')
    calls = []
    class Buffer:
        def get_expert_obs(self, batch_size):
            calls.append('legacy')
            return torch.randn(batch_size, 600)
        def get_expert_obs_for_stages(self, selected):
            calls.append(selected.clone())
            assert ((selected == 0) | (selected == 1)).all()
            return torch.randn(len(selected), 600)
    algorithm = ns['HIMPPO'](model, amp=amp, motion_buffer=Buffer(), device='cpu', num_mini_batches=1)
    algorithm.init_storage(4, 3, [738], [126], [29], [600])
    before = {k: v.clone() for k, v in amp.state_dict().items()}
    for _ in range(3):
        with torch.inference_mode():
            algorithm.act(torch.randn(4, 738), torch.randn(4, 126))
            algorithm.process_amp_state(torch.randn(4, 600), None if stages is None else torch.tensor(stages))
            algorithm.process_env_step(torch.ones(4), torch.zeros(4), {}, torch.randn(4, 126))
    with torch.inference_mode():
        algorithm.compute_returns(torch.randn(4, 126))
    losses = algorithm.update()
    assert np.isfinite(losses).all()
    if stages is not None and 0 not in stages:
        assert calls == [] and losses[2:] == (0., 0., 0.)
        assert all(torch.equal(before[k], v) for k, v in amp.state_dict().items())
    else:
        assert calls
