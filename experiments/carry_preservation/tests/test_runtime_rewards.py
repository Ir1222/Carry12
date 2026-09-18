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
import subprocess
import types
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "legged_gym"))
from legged_gym.carry_constraint_metrics import METRIC_NAMES, TEMPORAL_METRICS
from experiments.carry_preservation import analyze


ROOT = Path(__file__).resolve().parents[3]
BASELINE = "a070bb2"
ENV_PATH = "legged_gym/legged_gym/envs/g1/carrybox_locomotion.py"
CFG_PATH = "legged_gym/legged_gym/envs/g1/carrybox_locomotion_config.py"
EVAL_PATH = "experiments/carrybox_locomotion_eval/envs/carrybox_locomotion_eval_env.py"


def old_source(path):
    return subprocess.check_output(
        ["git", "show", BASELINE + ":" + path], cwd=ROOT, text=True, encoding="utf-8",
    )


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
                  quat_rotate_inverse=cls.math_utils.quat_rotate_inverse,
                  METRIC_NAMES=METRIC_NAMES, TEMPORAL_METRICS=TEMPORAL_METRICS)
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
        ns["carrybox_locomotion"] = types.SimpleNamespace(LeggedRobot=cls.env_type)
        definitions((ROOT / EVAL_PATH).read_text(), EVAL_PATH, ns)
        cls.eval_type = ns["CarryBoxLocomotionEvalEnv"]
        cls.kernel = staticmethod(ns["constraint_reward"])
        cls.violation = staticmethod(ns["range_violation"])

    def make_env(self, n=1, evaluation=False, dtype=torch.float32):
        env = (self.eval_type if evaluation else self.env_type)()
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
        return env

    def step(self, env):
        env._post_physics_step_callback()
        env.compute_reward()
        rewards = {name: getattr(env, "_reward_" + name)().clone() for name in REWARDS}
        for value in rewards.values():
            self.assertEqual(value.shape, (env.num_envs,))
            self.assertTrue(bool(torch.isfinite(value).all()))
            self.assertTrue(bool(((value >= 0) & (value <= 1)).all()))
        torch.testing.assert_close(env.rew_buf,
            sum(rewards[name] * scale for name, scale in env.reward_scales.items()))
        return rewards

    def test_valid_region_is_flat_and_temporal_first_sample_excluded(self):
        env = self.make_env(8)
        rewards = self.step(env)
        for name in REWARDS:
            expected = 0 if name in ("carry_hand_slip", "carry_relative_velocity") else 1
            torch.testing.assert_close(rewards[name], torch.full_like(rewards[name], expected))
        for reward in self.step(env).values():
            torch.testing.assert_close(reward, torch.ones_like(reward))
        # Static state at another point within the region has exactly the same reward.
        env.box_states[:, :3] += torch.tensor([.08, -.10, .08])
        env.rigid_body_states[:, 2:4, :3] += torch.tensor([.08, -.10, .08])
        env.rigid_body_states[:, 2:4, 0] += .10
        env.dof_pos[:, env.carry_arm_indices] += .1
        self.step(env)
        for reward in self.step(env).values():
            torch.testing.assert_close(reward, torch.ones_like(reward))
        # Zero gradients inside, continuous first derivative at a boundary.
        x = torch.tensor([[.2], [1.0], [1.001]], requires_grad=True)
        self.kernel(self.violation(x, -1., 1.)).sum().backward()
        torch.testing.assert_close(x.grad[:2], torch.zeros_like(x.grad[:2]))
        self.assertLess(abs(x.grad[2].item()), .003)

    def test_bad_states_increase_violation_and_reduce_reward(self):
        cases = (
            ("side", "carry_hand_box_surface", "left_hand_side_violation_m"),
            ("edge", "carry_hand_box_surface", "left_hand_face_violation_m"),
            ("slip", "carry_hand_slip", "left_hand_slip_violation_mps"),
            ("arm", "carry_arm_range", "arm_range_violation_rad"),
            ("position", "carry_relative_position", "box_relative_region_violation_m"),
            ("velocity", "carry_relative_velocity", "box_relative_velocity_violation_mps"),
        )
        for mode, reward_name, metric in cases:
            rewards, violations = [], []
            for excess in (0., .05, .15, .35):
                env = self.make_env()
                self.step(env)
                if mode == "side":
                    env.rigid_body_states[:, 2, 1] -= .03 + excess
                elif mode == "edge":
                    env.rigid_body_states[:, 2, 0] += .185 + excess
                elif mode == "slip":
                    env.rigid_body_states[:, 2, 0] += (.35 + excess * 10) * env.dt
                elif mode == "arm":
                    env.dof_pos[:, env.carry_arm_indices[0]] = env.carry_arm_range_upper[0] + excess
                elif mode == "position":
                    env.box_states[:, 0] = env.carry_box_relative_position_upper[0] + excess
                else:
                    env.box_states[:, 0] += (.35 + excess * 10) * env.dt
                rewards.append(self.step(env)[reward_name].item())
                violations.append(env.carry_metrics[metric].item())
            self.assertTrue(all(a > b for a, b in zip(rewards, rewards[1:])), (mode, rewards))
            self.assertTrue(all(a < b for a, b in zip(violations, violations[1:])), (mode, violations))
        env = self.make_env(3)
        env.rigid_body_states[0, 2:4, :3] = env.rigid_body_states[0, [3, 2], :3]
        env.rigid_body_states[1, 3, :3] = env.rigid_body_states[1, 2, :3]
        env.rigid_body_states[2, 2, :3] = env.box_states[2, :3]
        self.assertTrue(bool((self.step(env)["carry_hand_box_surface"] < .1).all()))

    def test_all_box_size_corners_and_arm_scope(self):
        # Includes asymmetric dimensions beyond the configured mixture.
        sizes = torch.cartesian_prod(torch.tensor([.20, .60]), torch.tensor([.20, .60]), torch.tensor([.20, .50]))
        env = self.make_env(len(sizes))
        env._box_size = sizes
        env.rigid_body_states[:, 2, 1] = sizes[:, 1] / 2
        env.rigid_body_states[:, 3, 1] = -sizes[:, 1] / 2
        env.rigid_body_states[:, 2:4, 0] += sizes[:, None, 0] / 2 - .01
        env.dof_pos[:, :15] = 100  # No waist/leg coupling in arm guardrail.
        rewards = self.step(env)
        torch.testing.assert_close(rewards["carry_hand_box_surface"], torch.ones(len(sizes)))
        torch.testing.assert_close(rewards["carry_arm_range"], torch.ones(len(sizes)))

    def test_rigid_translation_rotation_and_quaternion_signs(self):
        env = self.make_env(5)
        local_box = env.box_states[:, :3].clone()
        local_palms = env.rigid_body_states[:, 2:4, :3].clone()
        self.step(env)
        for t in range(1, 16):
            # Stand/vx/vy/yaw/mixed, including full 3D rigid-body rotations.
            rotvec = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, .1], [.04, -.03, .08]]) * t
            q = torch.tensor(Rotation.from_rotvec(rotvec).as_quat(), dtype=torch.float32)
            translation = torch.tensor([[0, 0, 0], [.02, 0, 0], [0, .02, 0], [0, 0, 0], [.02, -.01, .005]]) * t
            env.rigid_body_states[:, 1, :3] = translation
            env.rigid_body_states[:, 1, 3:7] = q
            env.box_states[:, :3] = self.math_utils.quat_rotate(q, local_box) + translation
            env.box_states[:, 3:7] = q * (-1 if t % 2 else 1)
            env.rigid_body_states[:, 2:4, :3] = self.math_utils.quat_rotate(
                q[:, None].expand(-1, 2, -1).reshape(-1, 4), local_palms.reshape(-1, 3)
            ).reshape(-1, 2, 3) + translation[:, None]
            rewards = self.step(env)
            for name in ("carry_hand_slip", "carry_relative_velocity"):
                torch.testing.assert_close(rewards[name], torch.ones_like(rewards[name]))
            for name in ("left_hand_tangential_slip_mps", "right_hand_tangential_slip_mps", "box_relative_motion_error_mps"):
                self.assertLess(env.carry_metrics[name].max().item(), 2e-5)

    def test_partial_reset_terminal_metrics_and_evaluation(self):
        env = self.make_env(3, evaluation=True)
        self.step(env)
        env.rigid_body_states[:, 2, 0] += .02
        self.step(env)
        expected = env.carry_metrics["left_hand_tangential_slip_mps"][0].clone()
        previous = env.carry_previous_hand_box[1:].clone()
        env.reset_idx(torch.tensor([0]))
        torch.testing.assert_close(env.extras["episode"]["carry/left_hand_tangential_slip_mps"], expected)
        torch.testing.assert_close(env.carry_previous_hand_box[1:], previous)
        self.assertEqual(env.carry_motion_samples.tolist(), [0, 1, 1])
        self.assertEqual(env.carry_history_valid.tolist(), [False, True, True])
        # Stub reset moves bodies by 100 m. That must never become a velocity sample.
        rewards = self.step(env)
        self.assertNotIn("episode", env.extras)
        self.assertEqual(env.carry_motion_metric_valid.tolist(), [False, True, True])
        for name in ("carry_hand_slip", "carry_relative_velocity"):
            self.assertEqual(rewards[name][0].item(), 0)
        for name in TEMPORAL_METRICS:
            self.assertEqual(env.carry_metrics[name][0].item(), 0)
        env.reset_idx(torch.tensor([0]))
        self.assertTrue(torch.isnan(env.extras["episode"]["carry/left_hand_tangential_slip_mps"]))
        env.reset_idx(torch.tensor([], dtype=torch.long))

    def test_disabled_rewards_still_update_history_and_diagnostics(self):
        env = self.make_env()
        env.reward_scales = {"carry_hand_box_surface": .03}
        env._prepare_reward_function()
        self.step(env)
        env.rigid_body_states[:, 2, 0] += .02
        self.step(env)
        self.assertAlmostEqual(env.carry_metrics["left_hand_tangential_slip_mps"].item(), 1., places=5)
        self.assertEqual(env.carry_motion_samples.item(), 1)
        # Reward access is pure and never consumes a history sample.
        for _ in range(3):
            env._reward_carry_hand_slip()
            env._reward_carry_relative_velocity()
        self.assertEqual(env.carry_motion_samples.item(), 1)

    def test_float32_batch_finite_independent(self):
        torch.manual_seed(7)
        env = self.make_env(4096)
        env.box_states[:, :3] += torch.randn(4096, 3) * .6
        env.rigid_body_states[:, 2:4, :3] += torch.randn(4096, 2, 3) * .4
        env.dof_pos += torch.randn(4096, 29)
        env._box_size *= .5 + torch.rand(4096, 3)
        self.step(env)
        env.box_states[:, :3] += torch.randn(4096, 3) * .05
        first = self.step(env)
        for value in env.carry_metrics.values():
            self.assertTrue(bool(torch.isfinite(value).all()))
        self.assertTrue(torch.isfinite(env.rew_buf).all())
        # A changed environment cannot affect any other environment's spatial reward.
        env.rigid_body_states[0, 2, 1] += 5
        second = self.step(env)
        torch.testing.assert_close(first["carry_hand_box_surface"][1:], second["carry_hand_box_surface"][1:])

    def test_reference_data_fits_guardrails_without_online_reference(self):
        data, _, _ = analyze.load_dataset()
        for clip in data:
            sampled = clip["policy_sampled"]
            env = self.make_env(len(sampled["bp"]))
            pos, rotation = sampled["poses"]["torso_link"]
            env.rigid_body_states[:, 1, :3] = torch.tensor(pos)
            env.rigid_body_states[:, 1, 3:7] = torch.tensor(Rotation.from_matrix(rotation).as_quat())
            env.box_states[:, :3] = torch.tensor(sampled["bp"])
            env.box_states[:, 3:7] = torch.tensor(sampled["bq"])
            for i, side in enumerate(("left", "right")):
                env.rigid_body_states[:, i + 2, :3] = torch.tensor(sampled["poses"][side + "_palm_link"][0])
            env.dof_pos[:] = torch.tensor(sampled["dofs"])
            self.step(env)
            env.carry_previous_relative_pos[1:] = env.carry_previous_relative_pos[:-1].clone()
            env.carry_previous_hand_box[1:] = env.carry_previous_hand_box[:-1].clone()
            env.carry_history_valid[0] = False
            rewards = self.step(env)
            for name in ("carry_arm_range", "carry_relative_position"):
                torch.testing.assert_close(rewards[name], torch.ones_like(rewards[name]))
            for name in ("carry_hand_slip", "carry_relative_velocity"):
                torch.testing.assert_close(rewards[name][1:], torch.ones_like(rewards[name][1:]))
            # Raw palms do NOT establish a valid simulated contact: no fake retargeting.
            self.assertTrue(bool((env.carry_metrics["left_hand_side_violation_m"] > 0).all()))

    def test_protected_configuration_and_online_dependency_boundary(self):
        r = self.cfg.rewards
        self.assertEqual(r.scales.carry_lin_vel_tracking, 3.)
        self.assertEqual(r.scales.carry_yaw_vel_tracking, 2.5)
        self.assertEqual(r.scales.carry_relative_orientation, 0.)
        def non_reward_config(source):
            parsed = ast.parse(source)
            for node in parsed.body:
                if isinstance(node, ast.ClassDef) and node.name == "G1Cfg":
                    node.body = [n for n in node.body if not isinstance(n, ast.ClassDef) or n.name != "rewards"]
            return ast.dump(parsed)
        self.assertEqual(non_reward_config((ROOT / CFG_PATH).read_text()), non_reward_config(old_source(CFG_PATH)))
        def methods(source):
            cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef))
            return {n.name: ast.dump(n) for n in cls.body if isinstance(n, ast.FunctionDef)}
        current, baseline = methods((ROOT / ENV_PATH).read_text()), methods(old_source(ENV_PATH))
        changed = {"_init_buffers", "reset_idx", "_post_physics_step_callback",
                   "_reward_carry_hand_box_surface", "_reward_carry_relative_velocity",
                   "_reward_carry_relative_position", "_reward_carry_relative_orientation", "_reward_carry_arm_pose"}
        for name in baseline.keys() - changed:
            self.assertEqual(current[name], baseline[name], name)
        for path in (ENV_PATH, CFG_PATH, EVAL_PATH):
            source = (ROOT / path).read_text()
            for forbidden in ("CarryCalibration", "compute_preservation", "carry_hand_direction",
                              "carry_arm_target", "carry_arm_sigma", "position_target",
                              "position_sigma", "orientation_target", "orientation_sigma", "velocity_sigma"):
                self.assertNotIn(forbidden, source)
        for name, body in current.items():
            if name not in ("_reset_actors", "_reset_boxes"):
                self.assertNotIn("motionlib", body)
        from experiments.carrybox_locomotion_eval.evaluation.metrics import PRESERVATION_METRICS
        self.assertEqual(METRIC_NAMES, PRESERVATION_METRICS)


if __name__ == "__main__":
    unittest.main()
