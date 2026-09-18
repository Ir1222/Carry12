"""CPU semantic tests of the physical carry constraints, without Isaac Gym.

Execute the real specialist methods and parent reward dispatch. Only simulator
setup/reset is stubbed; quaternion operations use the project's TorchScript
utilities (the same flat-batch API as Isaac Gym), not the old oracle's math.
Run: python -m unittest discover -s experiments/carry_preservation/tests -v
"""

import ast
import importlib.util
import math
from pathlib import Path
import types
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "legged_gym"))


ROOT = Path(__file__).resolve().parents[3]
ENV_PATH = "legged_gym/legged_gym/envs/g1/carrybox_locomotion.py"
CFG_PATH = "legged_gym/legged_gym/envs/g1/carrybox_locomotion_config.py"


def definitions(source, filename, namespace):
    nodes = [n for n in ast.parse(source).body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, "exec"), namespace)


def load_config(source=None):
    ns = dict(np=np, BaseConfig=object)
    for path in ("legged_gym/legged_gym/envs/base/legged_robot_config.py",
                 "legged_gym/legged_gym/envs/g1/carrybox_config.py"):
        definitions((ROOT / path).read_text(), path, ns)
    ns.update(CarryBoxCfg=ns["G1Cfg"], CarryBoxCfgPPO=ns["G1CfgPPO"])
    definitions(source or (ROOT / CFG_PATH).read_text(), CFG_PATH, ns)
    return ns["G1Cfg"], ns["G1CfgPPO"]


class SimulatorStub:
    def _init_buffers(self):
        pass

    def _post_physics_step_callback(self):
        pass

    def reset_idx(self, ids):
        self.extras["episode"] = {"rew_existing": torch.tensor(1.0)}
        self.rigid_body_states[ids, :, :3] = 100
        self.episode_length_buf[ids] = 0
        for sums in self.episode_sums.values():
            sums[ids] = 0


REWARDS = ("carry_hand_box_surface", "carry_hand_slip", "carry_arm_range",
           "carry_relative_position", "carry_relative_velocity", "carry_bilateral_contact")


class RuntimeRewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.ppo = load_config()
        spec = importlib.util.spec_from_file_location(
            "project_torch_utils", ROOT / "legged_gym/legged_gym/utils/torch_utils.py")
        cls.math_utils = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.math_utils)
        ns = dict(torch=torch, np=np, math=math, SimulatorStub=SimulatorStub,
                  quat_rotate_inverse=cls.math_utils.quat_rotate_inverse)
        parent_path = ROOT / "legged_gym/legged_gym/envs/g1/carrybox.py"
        parent = next(n for n in ast.parse(parent_path.read_text()).body if isinstance(n, ast.ClassDef))
        parent.bases = [ast.Name(id="SimulatorStub", ctx=ast.Load())]
        parent.body = [n for n in parent.body if isinstance(n, ast.FunctionDef)
                       and n.name in ("compute_reward", "_prepare_reward_function")]
        tree = ast.fix_missing_locations(ast.Module(body=[parent], type_ignores=[]))
        exec(compile(tree, str(parent_path), "exec"), ns)
        ns["CarryBoxBase"] = ns["LeggedRobot"]
        definitions((ROOT / ENV_PATH).read_text(), ENV_PATH, ns)
        cls.env_type = ns["LeggedRobot"]
        definitions((ROOT / "experiments/carrybox_locomotion_eval/evaluation/trial.py").read_text(),
                    "trial.py", ns)
        cls.eval_sample = staticmethod(ns["_carry_sample"])

    def make_env(self, n=1, dtype=torch.float32):
        env = self.env_type()
        env.num_envs, env.device, env.dt = n, "cpu", 0.02
        env.cfg = types.SimpleNamespace(rewards=self.cfg.rewards)
        env.dof_names = ["unused_%d" % i for i in range(15)] + list(self.cfg.rewards.carry_arm_joint_names)
        env.dof_pos = torch.zeros(n, 29, dtype=dtype)
        env.gym = types.SimpleNamespace(find_actor_rigid_body_handle=lambda e, a, name:
            {"torso_link": 1, "left_palm_link": 2, "right_palm_link": 3}[name])
        env.envs, env.actor_handles = [0], [0]
        env.upper_body_index = 0
        env._init_buffers()
        env.rigid_body_states = env.dof_pos.new_zeros(n, 4, 13)
        env.box_states = env.dof_pos.new_zeros(n, 13)
        env.rigid_body_states[..., 6] = 1
        env.box_states[:, 6] = 1
        env._box_size = env.dof_pos.new_tensor([.35, .35, .30]).repeat(n, 1)
        env.dof_pos[:, env.carry_arm_indices] = (env.carry_arm_range_lower + env.carry_arm_range_upper) / 2
        env.box_states[:, :3] = env.dof_pos.new_tensor([.35, 0, .05])
        env.rigid_body_states[:, 2:4, :3] = env.box_states[:, None, :3]
        env.rigid_body_states[:, 2, 1] += .175
        env.rigid_body_states[:, 3, 1] -= .175
        env.hand_contact_filt = torch.ones(n, 2, dtype=torch.bool)
        env.extras = {"episode": {"stale": 1}}
        env.carry_policy_commands = env.commands = env.dof_pos.new_zeros(n, 3)
        env.obs_buf = env.dof_pos.new_zeros(n, 6)
        for name in ("is_stage_carry", "carry_velocity_active", "carry_tracking_started", "has_seen_tag", "can_see_tag"):
            setattr(env, name, torch.zeros(n, dtype=torch.bool))
        env.carry_command_resample_time[:] = 1
        env.episode_length_buf = torch.arange(n) + 50
        env.reward_scales = {name: getattr(self.cfg.rewards.scales, name) for name in REWARDS}
        env.rew_buf = env.dof_pos.new_zeros(n)
        env._prepare_reward_function()
        env.last_rewards = {}
        def capture(name, function):
            def reward():
                value = function()
                env.last_rewards[name] = value.clone()
                return value
            return reward
        env.reward_functions = [capture(name, fn) for name, fn in zip(env.reward_names, env.reward_functions)]
        return env

    def step(self, env):
        env._post_physics_step_callback()
        env.compute_reward()
        rewards = env.last_rewards.copy()
        for value in rewards.values():
            self.assertEqual(value.shape, (env.num_envs,))
            self.assertTrue(bool(torch.isfinite(value).all()))
            self.assertTrue(bool(((value >= 0) & (value <= 1)).all()))
        torch.testing.assert_close(env.rew_buf,
            sum(rewards[name] * scale for name, scale in env.reward_scales.items()))
        return rewards

    def test_valid_region_and_first_sample(self):
        env = self.make_env(4)
        rewards = self.step(env)
        for name in REWARDS:
            expected = 0 if name in ("carry_hand_slip", "carry_relative_velocity") else 1
            torch.testing.assert_close(rewards[name], torch.full_like(rewards[name], expected))
        env.dof_pos[:, env.carry_arm_indices] += .1
        env.dof_pos[:, :15] = 100  # Arms do not supervise waist or legs.
        for reward in self.step(env).values():
            torch.testing.assert_close(reward, torch.ones_like(reward))

    def test_wrong_hand_side_and_face_overflow(self):
        env = self.make_env(3)
        env.rigid_body_states[0, 2:4, :3] = env.rigid_body_states[0, [3, 2], :3]
        env.rigid_body_states[1, 3, :3] = env.rigid_body_states[1, 2, :3]
        env.rigid_body_states[2, 2, 0] += .5
        self.assertTrue((self.step(env)["carry_hand_box_surface"] < .1).all())

    def test_outside_arm_and_box_ranges(self):
        env = self.make_env(3)
        env.dof_pos[:, env.carry_arm_indices[0]] = env.carry_arm_range_upper[0] + torch.tensor([0, .2, .5])
        env.box_states[:, 0] = env.carry_box_relative_position_upper[0] + torch.tensor([0, .1, .3])
        rewards = self.step(env)
        for name in ("carry_arm_range", "carry_relative_position"):
            self.assertAlmostEqual(rewards[name][0].item(), 1.)
            self.assertTrue((rewards[name][:-1] > rewards[name][1:]).all())

    def test_hand_slip_and_box_local_velocity(self):
        env = self.make_env(3)
        self.step(env)
        env.rigid_body_states[:, 2, 0] += torch.tensor([0, .02, .04])
        env.box_states[:, 2] += torch.tensor([0, .02, .04])
        rewards = self.step(env)
        for name in ("carry_hand_slip", "carry_relative_velocity"):
            self.assertEqual(rewards[name][0].item(), 1.)
            self.assertTrue((rewards[name][:-1] > rewards[name][1:]).all())

    def test_partial_reset_history(self):
        env = self.make_env(2)
        self.step(env)
        env.rigid_body_states[:, 2, 0] += .02
        self.step(env)
        previous = env.previous_hand_box[1].clone()
        env.reset_idx(torch.tensor([0]))
        torch.testing.assert_close(env.previous_hand_box[1], previous)
        self.assertAlmostEqual(env.extras["episode"]["carry/hand_slip"].item(), .5, places=5)
        # The stub reset moves bodies by 100 m; do not differentiate across it.
        rewards = self.step(env)
        self.assertNotIn("episode", env.extras)
        self.assertEqual(env.carry_log_steps[:, 1].tolist(), [0, 2])
        for name in ("carry_hand_slip", "carry_relative_velocity"):
            self.assertEqual(rewards[name][0].item(), 0.)
        env.reset_idx(torch.tensor([0]))
        self.assertTrue(torch.isnan(env.extras["episode"]["carry/hand_slip"]))

    def test_rigid_yaw_translation(self):
        env = self.make_env(4)
        box, hands = env.box_states[:, :3].clone(), env.rigid_body_states[:, 2:4, :3].clone()
        self.step(env)
        for t in range(1, 6):
            q = torch.tensor(Rotation.from_euler("z", [.1*t]*4).as_quat(), dtype=torch.float32)
            shift = torch.tensor([.03*t, -.02*t, 0.])
            env.rigid_body_states[:, 1, :3] = shift
            env.rigid_body_states[:, 1, 3:7] = q
            env.box_states[:, :3] = self.math_utils.quat_rotate(q, box) + shift
            env.box_states[:, 3:7] = q
            env.rigid_body_states[:, 2:4, :3] = self.math_utils.quat_rotate(
                q[:, None].expand(-1, 2, -1).reshape(-1, 4), hands.reshape(-1, 3)
            ).reshape(-1, 2, 3) + shift
            for reward in self.step(env).values():
                torch.testing.assert_close(reward, torch.ones_like(reward))

    def test_batched_finite_and_evaluator_history(self):
        env = self.make_env(4096)
        torch.manual_seed(7)
        env.box_states[:, :3] += torch.randn(4096, 3) * .6
        env.rigid_body_states[:, 2:4, :3] += torch.randn(4096, 2, 3) * .4
        env.dof_pos += torch.randn(4096, 29)
        self.step(env)
        self.step(env)
        sample, hands, box = self.eval_sample(env)
        self.assertTrue(math.isnan(sample["box_relative_motion_error_mps"]))
        env.rigid_body_states[0, 2, 0] += .02
        sample, _, _ = self.eval_sample(env, hands, box)
        self.assertAlmostEqual(sample["left_hand_tangential_slip_mps"], 1., places=4)
        self.assertEqual(sample["box_relative_motion_error_mps"], 0.)


if __name__ == "__main__":
    unittest.main()
