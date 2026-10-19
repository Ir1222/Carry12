"""CPU semantic and integration tests. No Isaac Gym/contact simulation is mocked as real physics."""
import ast
import copy
import csv
import importlib.util
import inspect
import json
import math
from pathlib import Path
import sys
import tempfile
from typing import Tuple
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "rsl_rl"))
from rsl_rl.teacher_handoff import adapt_action, checkpoint_actor, joint_groups, pose_excess
from rsl_rl.teacher_handoff_reference import validate_reference
from rsl_rl.modules import ActorCritic
from rsl_rl.storage.teacher_warmup_rollout_storage import TeacherWarmupRolloutStorage
from rsl_rl.algorithms.teacher_warmup_ppo import TeacherWarmupPPO
from rsl_rl.runners.teacher_warmup_on_policy_runner import TeacherWarmupOnPolicyRunner

torch.set_num_threads(1)
NAMES = [line.split()[1] for line in (ROOT / "legged_gym/resources/config/joint_id.txt").read_text().splitlines()]
GROUPS = joint_groups(NAMES)


def load_script(name):
    path = ROOT / "legged_gym/legged_gym/scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def definitions(path, ns):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))]
    exec(compile(tree, str(path), "exec"), ns)


def load_env_class():
    # Execute actual environment method bodies without importing/initializing Gym.
    ns = dict(torch=torch, np=np, math=math, F=F, BaseTask=object,
              LeggedRobotCfg=object, wrap_to_pi=lambda x: (x + math.pi) % (2 * math.pi) - math.pi,
              adapt_action=adapt_action, pose_excess=pose_excess,
              TEACHER_WARMUP=0, HANDOFF_TRANSITION=1, STUDENT_CARRY=2)
    definitions("legged_gym/legged_gym/envs/g1/carrybox.py", ns)
    full = ns["LeggedRobot"]
    ns["CarryBoxBase"] = full
    definitions("legged_gym/legged_gym/envs/g1/carrybox_locomotion.py", ns)
    ns.update(FullCarryBox=full, CarryLocomotion=ns["LeggedRobot"])
    definitions("legged_gym/legged_gym/envs/g1/carrybox_teacher_warmup.py", ns)
    return ns["LeggedRobot"]


def load_configs():
    ns = dict(np=np, inspect=inspect)
    definitions("legged_gym/legged_gym/envs/base/base_config.py", ns)
    definitions("legged_gym/legged_gym/envs/base/legged_robot_config.py", ns)
    definitions("legged_gym/legged_gym/envs/g1/carrybox_config.py", ns)
    full = ns["G1Cfg"]
    ns.update(FullCfg=full, CarryBoxCfg=full, CarryBoxCfgPPO=ns["G1CfgPPO"])
    definitions("legged_gym/legged_gym/envs/g1/carrybox_locomotion_config.py", ns)
    ns.update(LocoCfg=ns["G1Cfg"], LocoPPO=ns["G1CfgPPO"])
    definitions("legged_gym/legged_gym/envs/g1/carrybox_teacher_warmup_config.py", ns)
    return ns["G1Cfg"](), ns["G1CfgPPO"](), full


def supervisor_fixture():
    cls = load_env_class()
    env = cls.__new__(cls)
    h = NS(stable_s=.06, hysteresis_ratio=1.25, lift_min_m=.10, tilt_max_deg=20,
           relative_velocity_max=.3, box_omega_max=1., warmup_timeout_s=.3,
           transition_duration_s=.04, transition_timeout_s=.10, student_timeout_s=.10,
           command_suite="fixed", fixed_command=[.4, .2, -.3])
    env.cfg = NS(handoff=h, rewards=NS(hand_contact_threshold=1., box_drop_height=.15,
                    box_tilt_termination_deg=70., robot_box_max_distance=1., grasp_loss_grace_s=.04,
                    thresh_robot2object=1.),
                 asset=NS(), commands=NS(resample_carry_commands=False,
                    carry_command_resample_interval_s=[4., 6.]),
                 domain_rand=NS())
    env.box_cfg = NS(box_termination=True)
    env.num_envs, env.device, env.dt = 3, "cpu", .02
    env.phase = torch.zeros(3, dtype=torch.long)
    env.phase_age = torch.zeros(3)
    env.handoff_age = torch.zeros(3)
    env.warmup_elapsed = torch.zeros(3)
    env.stable_time = torch.zeros(3)
    env.pickup_seen = torch.zeros(3, dtype=torch.bool)
    env.handoff_seen = env.pickup_seen.clone()
    env.failure_reason = env.phase.clone()
    env.grasp_loss_time = torch.zeros(3)
    env.contact_forces = torch.zeros(3, 4, 3)
    env.contact_forces[:, 1:3, 2] = 2
    env.hand_colli_indices = torch.tensor([1, 2])
    env.last_hand_contacts = torch.zeros(3, 2, dtype=torch.bool)
    env.hand_contact_filt = env.last_hand_contacts.clone()
    env.termination_contact_indices = torch.tensor([], dtype=torch.long)
    env.hip_yaw_indices = torch.tensor([], dtype=torch.long)
    env.episode_length_buf = torch.ones(3, dtype=torch.long)
    env.max_episode_length = 10000
    env.rigid_body_states = torch.zeros(3, 4, 13)
    env.rigid_body_states[:, :, 2] = 1
    env.head_index, env.upper_body_index = 3, 0
    env.root_states = torch.zeros(3, 13)
    env.root_states[:, 2] = .8
    env.roll = env.pitch = env.yaw = torch.zeros(3)
    env.projected_gravity = torch.tensor([[0., 0., -1.]]).repeat(3, 1)
    env.projected_gravity_box = env.projected_gravity.clone()
    env.robot2object_dist = torch.full((3,), .4)
    env.box_states = torch.zeros(3, 13)
    env.box_states[:, 2] = 1.1
    env._box_size = torch.full((3, 3), .4)
    env.platform_pos = torch.zeros(3, 3)
    env.platform_pos[:, 2] = -.1
    env._platform_height = .2
    env.env_origins = torch.zeros(3, 3)
    env.commands = torch.zeros(3, 4)
    env.commands[:, 0] = .4
    env.carry_policy_commands = env.commands.clone()
    env.carry_command_resample_time = torch.ones(3)
    env.carry_heading_error = torch.zeros(3)
    env.carry_heading_ref = torch.zeros(3)
    env.carry_heading_initialized = torch.zeros(3, dtype=torch.bool)
    env.is_stage_carry = torch.ones(3, dtype=torch.bool)
    env.extras = {}
    # Only the sampler RNG is replaced, not supervision or physics-state logic.
    env._sample_carry_command_resample_time = lambda ids: None
    return env


def make_alg(**kwargs):
    ac = ActorCritic(4, 6, 1, 29, actor_hidden_dims=[8], critic_hidden_dims=[8])
    alg = TeacherWarmupPPO(ac, device="cpu", groups=GROUPS, num_learning_epochs=2,
                           num_mini_batches=2, min_valid_samples=2,
                           schedule="fixed", **kwargs)
    alg.init_storage(2, 4, [4], [6], [29])
    return alg


def collect(alg, valid=True, weights=1.0):
    with torch.no_grad():
        for t in range(4):
            obs = torch.randn(2, 4)
            critic = torch.randn(2, 6)
            alg.act(obs, critic)
            mask = torch.tensor([valid and t >= 1, valid and t >= 2])
            info = dict(student_valid=mask, time_outs=torch.zeros(2, dtype=torch.bool),
                        teacher_action=torch.zeros(2, 29), imitation_weight=torch.full((2,), weights))
            alg.process_env_step(torch.ones(2), torch.zeros(2, dtype=torch.bool), info, critic + .1, obs + .1)
        alg.compute_returns(critic)


class AdapterTests(unittest.TestCase):
    def test_exact_rad_per_second_bound_and_group_rates(self):
        raw = torch.full((2, 29), 4.)
        previous = torch.zeros_like(raw)
        rates = torch.full((29,), 12.)
        rates[GROUPS["arms"]] = 2
        rates[GROUPS["waist"]] = 1.5
        applied = adapt_action(raw, previous, rates, .02, .25, 100, torch.tensor([True, False]), "rate_limited")
        self.assertTrue(torch.allclose(applied[0] * .25, rates * .02))
        self.assertTrue(torch.equal(applied[1], raw[1]))

    def test_hard_clips_only_and_limiter_tracks_executed_action(self):
        raw = torch.full((1, 29), 5.)
        previous = torch.ones_like(raw)
        rates = torch.ones(29)
        mask = torch.ones(1, dtype=torch.bool)
        self.assertTrue(torch.equal(adapt_action(raw, previous, rates, .02, .25, 3, mask, "hard"), raw.clamp(-3, 3)))
        first = adapt_action(raw, previous, rates, .02, .25, 100, mask, "rate_limited")
        second = adapt_action(raw, first, rates, .02, .25, 100, mask, "rate_limited")
        self.assertTrue(torch.allclose(second - first, torch.full_like(raw, .08)))

    def test_pose_deadzone(self):
        q = torch.tensor([[0., .5, 2.]])
        self.assertTrue(torch.equal(pose_excess(q, torch.zeros(3), torch.ones(3), torch.ones(3)), torch.tensor([[0., 0., 1.]])))


class SupervisorTests(unittest.TestCase):
    def test_reward_episode_sums_exclude_teacher_slots(self):
        env = supervisor_fixture()
        env.control_mask = torch.tensor([False, True, False])
        env.reward_scales = {"dof_vel": -1.}
        env._reward_dof_vel = lambda: torch.ones(3)
        env.rew_buf = torch.zeros(3)
        env.cfg.rewards.only_positive_rewards = False
        env._prepare_reward_function()
        full = env.compute_reward.__globals__["FullCarryBox"]
        full.compute_reward(env)
        self.assertTrue(torch.allclose(env.rew_buf, torch.tensor([0., -.02, 0.])))
        self.assertTrue(torch.equal(env.rew_buf, env.episode_sums["dof_vel"]))

    def step_fixture(self, initializing=False):
        env = supervisor_fixture()
        env.phase[:] = torch.tensor([0, 1, 2])
        env._initializing = initializing
        env.cfg.handoff.imitation_window_s = 2.
        env.cfg.handoff.imitation_max_vy = .15
        env.cfg.handoff.imitation_max_yaw = .2
        env.cfg.handoff.imitation_max_vx = .8
        env.cfg.handoff.adapter = "rate_limited"
        env.cfg.control = NS(action_scale=.25)
        env.cfg.normalization = NS(clip_actions=100.)
        env.obs_buf = torch.zeros(3, 738)
        env.actions = torch.zeros(3, 29)
        env.last_actions = env.actions.clone()
        env.last_last_actions = env.actions.clone()
        env.raw_student_action = env.actions.clone()
        env.applied_action = env.actions.clone()
        env.applied_pd_target = env.actions.clone()
        env.default_dof_pos = torch.zeros(1, 29)
        env.rates = torch.ones(29)
        env.teacher = lambda obs: torch.full((3, 29), .1)

        def physical_step_stub(self, applied):
            # Only the physics stepping seam is stubbed. Execute the real
            # supervisor/adapter wrapper and inspect its ownership metadata.
            self.actions = applied.clone()
            self.phase[:] = 1
            self.last_step_snapshot = {"applied_action": applied.clone(), "pd_target": applied * .25}
            return (self.obs_buf, torch.zeros(3, 160), torch.ones(3), torch.zeros(3, dtype=torch.bool),
                    {}, torch.tensor([], dtype=torch.long), torch.zeros(0, 160), None)

        full = env.step.__globals__["FullCarryBox"]
        raw = torch.full((3, 29), 5.)
        with patch.object(full, "step", physical_step_stub):
            result = env.step(raw)
        return env, raw, result

    def test_control_ownership_is_pre_step_and_raw_action_is_not_mutated(self):
        env, raw, result = self.step_fixture()
        self.assertEqual(result[4]["student_valid"].tolist(), [False, True, True])
        self.assertEqual(result[4]["control_phase"].tolist(), [0, 1, 2])
        self.assertTrue(torch.all(raw == 5))
        self.assertTrue(torch.all(env.raw_student_action == 5))
        self.assertTrue(torch.allclose(env.actions[0], torch.full((29,), .1)))
        self.assertTrue(torch.allclose(env.actions[1:], torch.full((2, 29), .08)))
        self.assertFalse(result[4]["teacher_action"].requires_grad)

    def test_initial_reset_uses_full_zero_action_step(self):
        env, _, _ = self.step_fixture(initializing=True)
        self.assertTrue(torch.all(env.actions == 0))

    def test_reset_only_selected_phase_and_adapter_state(self):
        env = supervisor_fixture()
        env.phase[:] = 2
        env.phase_age[:] = 3
        env.actions = torch.ones(3, 29)
        env.obs_buf = torch.ones(3, 738)
        env.carry_history_valid = torch.ones(3, dtype=torch.bool)
        for name in ("carry_log_sums", "carry_log_steps", "carry_lower_log_sums", "carry_lower_log_steps",
                     "raw_student_action", "applied_action", "applied_pd_target"):
            setattr(env, name, torch.ones(3, 29))
        full = env.reset_idx.__globals__["FullCarryBox"]
        with patch.object(full, "reset_idx", lambda self, ids: self.extras.update(episode={})):
            env.reset_idx(torch.tensor([1]))
        self.assertEqual(env.phase.tolist(), [2, 0, 2])
        self.assertEqual(env.phase_age.tolist(), [3, 0, 3])
        self.assertTrue(torch.all(env.actions[1] == 0))
        self.assertTrue(torch.all(env.actions[[0, 2]] == 1))
        self.assertTrue(torch.all(env.obs_buf[1] == 0))
        self.assertTrue(torch.all(env.obs_buf[[0, 2]] == 1))

    def test_pose_penalty_persists_beyond_imitation_window(self):
        env = supervisor_fixture()
        env.pose_reference = dict(box_size_min=[.3] * 3, box_size_max=[.5] * 3)
        env.cfg.handoff.posture_box_size_tolerance_m = .04
        env.cfg.handoff.posture_penalty_cap = 10.
        env.control_mask = torch.tensor([True, True, False])
        env.pose_ids = GROUPS["waist"] + GROUPS["arms"]
        env.pose_lower, env.pose_upper, env.pose_softness = torch.zeros(17), torch.ones(17), torch.ones(17)
        env.dof_pos = torch.full((3, 29), 2.)
        early = env._reward_carry_upper_body_pose()
        env.handoff_age[:] = 30
        late = env._reward_carry_upper_body_pose()
        self.assertTrue(torch.equal(early, late))
        self.assertEqual(late.tolist(), [-1., -1., 0.])

    def test_asynchronous_handoff_and_history_unchanged(self):
        env = supervisor_fixture()
        env.contact_forces[1, 1] = 0
        env.box_states[2, 10] = 2
        physical = (env.root_states.clone(), env.box_states.clone())
        env.obs_buf = torch.randn(3, 738)
        history = env.obs_buf.clone()
        for _ in range(3):
            env.check_termination()
        self.assertEqual(env.phase.tolist(), [1, 0, 0])
        self.assertTrue(torch.equal(history, env.obs_buf))
        self.assertTrue(torch.equal(physical[0], env.root_states))
        self.assertTrue(torch.equal(physical[1], env.box_states))
        self.assertTrue(torch.allclose(env.carry_policy_commands[0, :3], torch.tensor([.4, .2, -.3])))

    def test_hysteresis_does_not_accumulate_loose_only_time(self):
        env = supervisor_fixture()
        env.check_termination()
        env.box_states[:, 10] = 1.1  # outside strict, inside hysteresis
        for _ in range(5):
            env.check_termination()
        self.assertTrue(torch.all(env.phase == 0))
        self.assertTrue(torch.allclose(env.stable_time, torch.full((3,), .02)))
        env.box_states[:, 10] = 2
        env.check_termination()
        self.assertTrue(torch.all(env.stable_time == 0))

    def test_raw_three_dimensional_command_preserved(self):
        env = supervisor_fixture()
        env.phase[:] = torch.tensor([0, 1, 2])
        env.commands[:] = torch.tensor([.4, .2, -.3, 0.])
        env._update_carry_heading_commands()
        self.assertEqual(env.carry_policy_commands[0, 1].item(), 0.)
        self.assertTrue(torch.allclose(env.carry_policy_commands[1:, :3], env.commands[1:, :3]))

    def test_independent_timeouts_and_physical_priority(self):
        env = supervisor_fixture()
        env.phase[:] = torch.tensor([0, 1, 2])
        env.phase_age[:] = torch.tensor([.3, .1, .1])
        env.check_termination()
        self.assertEqual(env.failure_reason.tolist(), [2, 3, 4])
        self.assertEqual(env.time_out_buf.tolist(), [False, False, True])
        env.root_states[2, 2] = .1
        env.check_termination()
        self.assertEqual(env.failure_reason[2].item(), 1)
        self.assertFalse(env.time_out_buf[2])

    def test_transition_is_student_owned_and_becomes_carry(self):
        env = supervisor_fixture()
        env.phase[:] = 1
        env.phase_age[:] = .03
        env.check_termination()
        self.assertTrue(torch.all(env.phase == 2))
        self.assertTrue(torch.all(env.phase_age == 0))
        self.assertTrue(torch.all(env.handoff_age > 0))


class StorageAndPPOTests(unittest.TestCase):
    def test_gae_terminal_timeout_and_teacher_boundaries(self):
        storage = TeacherWarmupRolloutStorage(3, 3, [4], [6], [29])
        storage.step = 3
        storage.valid[:] = True
        storage.valid[0, 0] = False
        storage.valid[2, 0] = False
        storage.values[:] = 1
        storage.next_values[:] = 2
        storage.rewards[:] = 3
        storage.rewards[~storage.valid] = 1e10
        storage.dones[1, 1] = 1
        storage.dones[1, 2] = 1
        storage.timeout[1, 2] = True
        storage.compute_returns(None, .9, 1.)
        self.assertAlmostEqual(storage.returns[1, 0, 0].item(), 4.8, places=5)
        self.assertAlmostEqual(storage.returns[1, 1, 0].item(), 3., places=5)
        self.assertAlmostEqual(storage.returns[1, 2, 0].item(), 4.8, places=5)
        self.assertTrue(torch.all(storage.returns[~storage.valid] == 0))
        self.assertTrue(torch.isfinite(storage.advantages).all())

    def test_minibatches_exclude_teacher_and_include_final_slot(self):
        alg = make_alg()
        collect(alg)
        alg.storage.observations[~alg.storage.valid.squeeze(-1)] = 999
        seen = 0
        for batch in alg.storage.mini_batch_generator(2, 1):
            self.assertFalse(torch.any(batch[0] == 999))
            seen += len(batch[0])
        self.assertEqual(seen, 5)

    def test_no_valid_data_performs_no_update(self):
        alg = make_alg()
        collect(alg, valid=False)
        before = copy.deepcopy(alg.actor_critic.state_dict())
        self.assertEqual(alg.update(), (0.,) * 4)
        self.assertFalse(alg.optimizer.state)
        for k, v in before.items():
            self.assertTrue(torch.equal(v, alg.actor_critic.state_dict()[k]))

    def test_imitation_has_actor_gradient_no_teacher_gradient(self):
        alg = make_alg()
        label = torch.randn(3, 29, requires_grad=True)
        alg.storage.batch_teacher_actions = label
        alg.storage.batch_imitation_weights = torch.ones(3, 1)
        alg._diag_sums = dict.fromkeys(("handoff_imitation_loss", "imitation_active_ratio", "gap_arms", "gap_waist", "gap_legs"), 0.)
        loss = alg._auxiliary_loss(None, alg.actor_critic.actor(torch.randn(3, 4)))
        loss.backward()
        self.assertIsNone(label.grad)
        self.assertGreater(sum(p.grad.abs().sum().item() for p in alg.actor_critic.actor.parameters()), 0)

    def test_zero_temporal_weights_have_zero_auxiliary_gradient(self):
        alg = make_alg()
        alg.storage.batch_teacher_actions = torch.ones(3, 29)
        alg.storage.batch_imitation_weights = torch.zeros(3, 1)
        alg._diag_sums = dict.fromkeys(("handoff_imitation_loss", "imitation_active_ratio", "gap_arms", "gap_waist", "gap_legs"), 0.)
        loss = alg._auxiliary_loss(None, alg.actor_critic.actor(torch.randn(3, 4)))
        loss.backward()
        self.assertEqual(loss.item(), 0)
        self.assertTrue(all(torch.all(p.grad == 0) for p in alg.actor_critic.actor.parameters()))

    def test_actual_ppo_update_is_finite_and_raw_sample_kept(self):
        alg = make_alg()
        collect(alg)
        with torch.no_grad():
            obs = alg.storage.observations[1]
            alg.actor_critic.act(obs)
            expected = alg.actor_critic.get_actions_log_prob(alg.storage.actions[1])
        self.assertTrue(torch.allclose(expected, alg.storage.actions_log_prob[1, :, 0]))
        losses = alg.update()
        self.assertTrue(all(math.isfinite(x) for x in losses))
        self.assertEqual(alg.diagnostics["valid_samples"], 5)
        self.assertTrue(alg.optimizer.state)


class SyntheticVecEnv:
    """Deterministic vector transition fixture; explicitly not a physics simulator."""
    def __init__(self, teacher):
        self.num_envs, self.num_obs, self.num_privileged_obs = 2, 738, 160
        self.num_actions, self.actor_history_length, self.device = 29, 6, "cpu"
        self.groups, self.dof_names, self.dt = GROUPS, NAMES, .02
        self.cfg = NS(handoff=NS(teacher_checkpoint=str(teacher), posture_reference=""),
                      normalization=NS(clip_observations=100.), control=NS(action_scale=.25))
        self.phase = torch.zeros(2, dtype=torch.long)
        self.carry_policy_commands = torch.zeros(2, 4)
        self.base_lin_vel_yaw = torch.zeros(2, 3)
        self.base_yaw_rate_world = torch.zeros(2)
        self.step_diagnostics = {"bilateral": torch.ones(2)}
        self.t = 0

    def reset(self):
        self.t = 0
        self.obs = torch.zeros(2, 738)
        self.critic = torch.zeros(2, 160)
        return self.obs, self.critic

    def get_observations(self):
        return self.obs

    def get_privileged_observations(self):
        return self.critic

    def step(self, action):
        self.t += 1
        valid = torch.tensor([self.t % 8 >= 2, self.t % 8 >= 4])
        self.phase = valid.long() * 2
        self.obs = torch.full_like(self.obs, self.t / 100)
        self.critic = torch.full_like(self.critic, self.t / 100)
        dones = torch.full((2,), self.t % 8 == 0)
        ids = dones.nonzero().flatten()
        info = dict(student_valid=valid, time_outs=dones.clone(), teacher_action=torch.zeros_like(action),
                    imitation_weight=valid.float())
        return self.obs, self.critic, torch.ones(2), dones, info, ids, self.critic[ids].clone(), None


class RunnerIntegrationTests(unittest.TestCase):
    def test_small_env_three_iteration_smoke_and_resume(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "experiments/teacher_warmup") as directory:
            path = Path(directory)
            source = ActorCritic(738, 126, 6, 29, actor_hidden_dims=[8, 8], critic_hidden_dims=[8])
            ckpt = path / "source.pt"
            torch.save({"model_state_dict": source.state_dict()}, ckpt)
            cfg = dict(runner=dict(policy_class_name="ActorCritic", num_steps_per_env=8, save_interval=2),
                       policy=dict(actor_hidden_dims=[3], critic_hidden_dims=[8]),
                       algorithm=dict(num_learning_epochs=1, num_mini_batches=2, learning_rate=1e-4,
                           min_valid_samples=2, handoff_imitation_coef=.5, imitation_arms_weight=2.,
                           imitation_waist_weight=2., imitation_legs_weight=.25, imitation_joint_weights=None),
                       handoff_contract={"adapter": "hard"})
            env = SyntheticVecEnv(ckpt)
            run = TeacherWarmupOnPolicyRunner(env, cfg, str(ckpt), str(path / "run"))
            critic_before = copy.deepcopy(run.alg.actor_critic.critic.state_dict())
            run.load_init_policy(str(ckpt))
            for k, v in source.actor.state_dict().items():
                self.assertTrue(torch.equal(v, run.alg.actor_critic.actor.state_dict()[k]))
            for k, v in critic_before.items():
                self.assertTrue(torch.equal(v, run.alg.actor_critic.critic.state_dict()[k]))
            self.assertFalse(run.alg.optimizer.state)
            run.learn(3)
            saved = path / "run/model_3.pt"
            self.assertTrue(saved.exists())
            resumed = TeacherWarmupOnPolicyRunner(SyntheticVecEnv(ckpt), cfg, str(saved), str(path / "resume"))
            resumed.load(str(saved))
            self.assertEqual(resumed.current_learning_iteration, 3)
            self.assertTrue(resumed.alg.optimizer.state)
            resumed.learn(1)
            self.assertEqual(resumed.current_learning_iteration, 4)
            resumed.contract["handoff"] = {"adapter": "rate_limited"}
            with self.assertRaises(ValueError):
                resumed.load(str(saved))
            run.writer.close()
            resumed.writer.close()

    def test_actual_checkpoint_architectures_and_frozen_actor(self):
        for relative in ("carry_track/model_33999.pt", "track_vel_fina/model_19999 (2).pt"):
            path = ROOT / "shared_artifacts/2026-10-19" / relative
            if not path.exists():
                self.skipTest("Historical local checkpoint unavailable")
            actor, spec = checkpoint_actor(path)
            self.assertEqual(spec["dims"], [738, 512, 256, 256, 29])
            self.assertEqual(spec["critic_input"], 126)
            self.assertTrue(all(not p.requires_grad for p in actor.parameters()))
            self.assertEqual(actor(torch.zeros(2, 738)).shape, (2, 29))


class AnalysisAndReferenceTests(unittest.TestCase):
    def test_generic_amp_runner_rejects_teacher_warmup_task(self):
        ns = dict(VecEnv=object, LeggedRobotCfg=object, LeggedRobotCfgPPO=object,
                  HIMOnPolicyRunner=object, Tuple=Tuple)
        definitions("legged_gym/legged_gym/utils/task_registry.py", ns)
        registry = ns["TaskRegistry"]()
        cfg = NS(runner=NS(runner_class_name="TeacherWarmupOnPolicyRunner"))
        with self.assertRaisesRegex(ValueError, "student-masked PPO"):
            registry.make_alg_runner(None, args=NS(), train_cfg=cfg)

    def test_configuration_ablation_isolation(self):
        env, train, full = load_configs()
        own = NS(teacher_checkpoint="teacher.pt", ablation="A5", command_suite="mixed",
                 fixed_command=[.4, 0, 0], posture_reference="reference.json", handoff_config=None)
        load_script("train_teacher_warmup_loco").apply_configuration(env, train, own)
        self.assertEqual(env.env.num_actor_obs, 738)
        self.assertEqual(env.env.num_privileged_obs, 160)
        self.assertEqual(env.asset.box.reset_mode, "default")
        self.assertEqual(full.asset.box.reset_mode, "hybrid")
        self.assertEqual(env.handoff.adapter, "rate_limited")
        self.assertEqual(env.rewards.scales.carry_upper_body_pose, .5)
        self.assertFalse(env.domain_rand.delay)
        own.ablation = "A2"
        load_script("train_teacher_warmup_loco").apply_configuration(env, train, own)
        self.assertEqual(train.algorithm.handoff_imitation_coef, 0)
        self.assertEqual(env.rewards.scales.carry_upper_body_pose, 0)

    def test_build_reference_from_successful_teacher_only(self):
        build = load_script("build_teacher_posture_reference").build
        with tempfile.TemporaryDirectory(dir=ROOT / "experiments/teacher_warmup") as directory:
            p = Path(directory)
            manifest = dict(kind="full_teacher_physical_pickup", teacher_sha256="test_fixture_only",
                            dof_names=NAMES)
            (p / "manifest.json").write_text(json.dumps(manifest))
            rows = []
            for episode in range(4):
                for t in range(40):
                    row = dict(episode=episode, done=int(t == 39), reason=(4 if episode < 3 else 1) if t == 39 else 0,
                               ready=1, student_valid=1, handoff_relative_s=2)
                    row.update({"q:" + name: .01 * t if episode < 3 else 100 for name in NAMES})
                    row.update({"box_" + axis: .4 for axis in "xyz"})
                    rows.append(row)
            with (p / "steps.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            reference = build(p, p / "reference.json")
            self.assertEqual(reference["sample_count"], 120)
            self.assertEqual(reference["successful_episodes"], 3)
            self.assertLess(max(reference["upper"]), 1)
            manifest["kind"] = "dual_policy"
            (p / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                build(p, p / "invalid.json")

    def test_failure_denominators_and_survival(self):
        analyze = load_script("analyze_teacher_warmup").analyze
        with tempfile.TemporaryDirectory(dir=ROOT / "experiments/teacher_warmup") as directory:
            p = Path(directory)
            (p / "manifest.json").write_text(json.dumps(dict(episodes=2, dof_names=[])))
            rows = [dict(episode=0, done=1, reason=2, eligible=0, pickup=0, warmup_s=45,
                         student_valid=0, handoff_relative_s=-1),
                    dict(episode=1, done=1, reason=4, eligible=1, pickup=1, warmup_s=10,
                         student_valid=1, handoff_relative_s=45)]
            for row in rows:
                row.update({k: 0 for k in ("command_vx", "actual_vx", "command_vy", "actual_vy",
                            "command_yaw", "actual_yaw", "bilateral", "tilt", "omega", "relative",
                            "posture_error", "gap_arms", "gap_waist", "gap_legs")})
            with (p / "steps.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            report = analyze(p)
            self.assertEqual(report["end_to_end_completion"], .5)
            self.assertEqual(report["handoff_conditional_completion"], 1)
            self.assertEqual(report["survival_3s_end_to_end"], .5)
            self.assertEqual(report["handoff_eligibility_rate"], .5)

    def test_reference_provenance_validation(self):
        upper_names = [NAMES[i] for i in GROUPS["waist"] + GROUPS["arms"]]
        ref = dict(schema=1, source_kind="full_teacher_physical_pickup", teacher_sha256="abc",
                   sample_count=100, successful_episodes=3, joint_names=upper_names,
                   lower=[-.1] * 17, upper=[.1] * 17, softness=[.2] * 17,
                   box_size_min=[.3] * 3, box_size_max=[.4] * 3)
        validate_reference(ref, NAMES)
        for key, value in (("sample_count", 0), ("source_kind", "student"), ("softness", [0] * 17)):
            invalid = dict(ref, **{key: value})
            with self.assertRaises(ValueError):
                validate_reference(invalid, NAMES)


if __name__ == "__main__":
    unittest.main()
