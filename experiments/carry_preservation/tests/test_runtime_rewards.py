"""CPU regression against commit 7cb664d, without importing Isaac Gym.

Execute the real specialist methods and parent reward dispatch. Only simulator
setup/reset is stubbed; quaternion operations use the project's TorchScript
utilities (the same flat-batch API as Isaac Gym), not the old oracle's math.
Run: python -m unittest discover -s experiments/carry_preservation/tests -v
"""

import ast
import copy
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import types
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from experiments.carry_preservation.tests.test_preservation import analyze, scene


ROOT = Path(__file__).resolve().parents[3]
BASELINE = "7cb664d35c797de2fca84bb4e9d6cbf7db3825f0"
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


class RuntimeRewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old = types.ModuleType("preservation_at_7cb664d")
        exec(compile(old_source("legged_gym/legged_gym/carry_preservation.py"),
                     "preservation_at_7cb664d.py", "exec"), cls.old.__dict__)
        cls.calibration_data = json.loads(old_source("legged_gym/resources/config/carry_preservation.json"))
        cls.cfg, cls.ppo = load_config()
        spec = importlib.util.spec_from_file_location(
            "project_torch_utils", ROOT / "legged_gym/legged_gym/utils/torch_utils.py")
        cls.math_utils = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.math_utils)
        ns = dict(torch=torch, np=np, math=math, SimulatorStub=SimulatorStub,
                  quat_mul=cls.math_utils.quat_mul, quat_conjugate=cls.math_utils.quat_conjugate,
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
        cls.runtime_log = staticmethod(ns["quaternion_log"])
        cls.parent_type = ns["CarryBoxBase"]
        ns.update(carrybox_locomotion=types.SimpleNamespace(LeggedRobot=cls.env_type),
                  LEGGED_GYM_ROOT_DIR=str(ROOT / "legged_gym"),
                  CarryCalibration=cls.old.CarryCalibration,
                  compute_preservation=cls.old.compute_preservation)
        definitions((ROOT / EVAL_PATH).read_text(), EVAL_PATH, ns)
        cls.eval_type = ns["CarryBoxLocomotionEvalEnv"]
        cls.max_errors = {name: 0.0 for name in cls.old.REWARD_NAMES}

    @classmethod
    def tearDownClass(cls):
        print("\nMaximum float32 absolute reward differences vs " + BASELINE + ":")
        for name, value in cls.max_errors.items():
            print(f"  {name}: {value:.9g}")

    def make_env(self, state, evaluation=False):
        env = (self.eval_type if evaluation else self.env_type)()
        n = state["box_pos"].shape[0]
        env.num_envs, env.device, env.dt = n, "cpu", state["dt"]
        env.cfg = types.SimpleNamespace(rewards=self.cfg.rewards)
        env.dof_names = ["unused_%d" % i for i in range(15)] + list(self.calibration_data["arm_joint_names"])
        env.dof_pos = state["arm_pos"].new_zeros(n, 29)
        env.gym = types.SimpleNamespace(find_actor_rigid_body_handle=lambda e, a, name:
            {"torso_link": 1, "left_palm_link": 2, "right_palm_link": 3}[name])
        env.envs, env.actor_handles = [0], [0]
        env.upper_body_index = 0
        env._init_buffers()
        env.rigid_body_states = env.dof_pos.new_zeros(n, 4, 13)
        env.box_states = env.dof_pos.new_zeros(n, 13)
        self.set_state(env, state)
        env.carry_previous_relative_pos.copy_(state["previous_relative_pos"])
        env.carry_history_valid.copy_(state["history_valid"])
        env.extras = {"episode": {"stale": 1}}
        env.carry_policy_commands = env.commands = env.dof_pos.new_zeros(n, 3)
        env.obs_buf = env.dof_pos.new_zeros(n, 6)
        for name in ("is_stage_carry", "carry_velocity_active", "carry_tracking_started", "has_seen_tag", "can_see_tag"):
            setattr(env, name, torch.zeros(n, dtype=torch.bool))
        env.carry_command_resample_time[:] = 1
        env.episode_length_buf = torch.arange(n) + 50
        env.reward_scales = {name: getattr(self.cfg.rewards.scales, name) for name in self.old.REWARD_NAMES}
        env.rew_buf = env.dof_pos.new_zeros(n)
        env._prepare_reward_function()
        env.last_rewards = {}

        def record(name, function):
            def reward():
                result = function()
                env.last_rewards[name] = result.clone()
                return result
            return reward

        env.reward_functions = [record(name, fn) for name, fn in zip(env.reward_names, env.reward_functions)]
        return env

    @staticmethod
    def set_state(env, s):
        env.rigid_body_states[:, 1, :3] = s["torso_pos"]
        env.rigid_body_states[:, 1, 3:7] = s["torso_quat"]
        env.rigid_body_states[:, 2:4, :3] = s["hand_pos"]
        env.box_states[:, :3], env.box_states[:, 3:7] = s["box_pos"], s["box_quat"]
        env.dof_pos[:, env.carry_arm_indices] = s["arm_pos"]
        env._box_size = s["box_size"]

    def assert_equivalent(self, state, env=None):
        c = self.old.CarryCalibration(self.calibration_data, dtype=state["arm_pos"].dtype)
        expected, metrics, relative = self.old.compute_preservation(calibration=c, **state)
        env = env or self.make_env(state)
        env._post_physics_step_callback()
        env.compute_reward()
        for name, reward in expected.items():
            actual = env.last_rewards[name]
            torch.testing.assert_close(actual, reward, atol=2e-5, rtol=2e-5)
            self.assertTrue(bool(torch.isfinite(actual).all()))
            if actual.dtype == torch.float32:
                self.max_errors[name] = max(self.max_errors[name], (actual - reward).abs().max().item())
        torch.testing.assert_close(env.rew_buf, sum(expected[n] * env.reward_scales[n] for n in expected),
                                   atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(env.carry_previous_relative_pos, relative, atol=1e-6, rtol=2e-5)
        self.assertTrue(bool(env.carry_history_valid.all()))
        return env, metrics

    def test_exact_constants_scales_and_unrelated_configuration(self):
        r, data = self.cfg.rewards, self.calibration_data
        mapping = {
            "carry_reference_policy_dt": "policy_dt", "carry_torso_link": "torso_link",
            "carry_hand_links": "hand_links", "carry_arm_joint_names": "arm_joint_names",
            "carry_hand_normal_sigma": "hand_normal_sigma", "carry_hand_direction_sigma": "hand_direction_sigma",
            "carry_arm_target": "arm_target", "carry_arm_sigma": "arm_sigma",
            "carry_box_relative_position_target": "position_target", "carry_box_relative_position_sigma": "position_sigma",
            "carry_box_relative_orientation_target": "orientation_target_xyzw",
            "carry_box_relative_orientation_sigma": "orientation_sigma", "carry_box_relative_velocity_sigma": "motion_sigma",
        }
        for field, key in mapping.items():
            self.assertEqual(getattr(r, field), data[key], field)
        self.assertEqual([r.carry_hand_direction_left, r.carry_hand_direction_right], data["hand_directions"])
        self.assertEqual(json.loads(analyze.DEFAULT_CALIBRATION.read_text()), data)
        old_cfg, _ = load_config(old_source(CFG_PATH))
        self.assertEqual({k: v for k, v in vars(r.scales).items() if not k.startswith("_")},
                         {k: v for k, v in vars(old_cfg.rewards.scales).items() if not k.startswith("_")})
        # Changes to commands, observations, PPO, etc. are outside this refactor.
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
        changed = {"_init_buffers", "reset_idx", "compute_reward"}
        changed.update("_reward_" + name for name in self.old.REWARD_NAMES)
        for name in baseline.keys() - changed:
            self.assertEqual(current[name], baseline[name], name)

    def test_runtime_dispatch_and_dependency_boundary(self):
        self.assertIs(self.env_type.compute_reward, self.parent_type.compute_reward)
        self.assertNotIn("compute_reward", self.env_type.__dict__)
        source = (ROOT / ENV_PATH).read_text()
        for forbidden in ("CarryCalibration", "CarryMetricAccumulator", "compute_preservation",
                          "carry_preservation_rewards", "statistics.json", "carry_preservation_metrics"):
            self.assertNotIn(forbidden, source)
        q = torch.tensor(Rotation.from_rotvec([[0, 0, 0], [1e-10, 0, 0], [.2, -.3, .5], [0, 0, 3.14159]]).as_quat())
        torch.testing.assert_close(self.runtime_log(q), self.old.quaternion_log(q))
        torch.testing.assert_close(self.runtime_log(-q), self.old.quaternion_log(q))

    def test_randomized_float32_batch_and_edge_grasps(self):
        rng = np.random.default_rng(739)
        c = self.old.CarryCalibration(self.calibration_data)
        s = scene(c, 4096)
        n = len(s["box_pos"])
        def tensor(x):
            return torch.tensor(x, dtype=torch.float32)
        s["torso_pos"] = tensor(rng.uniform(-10, 10, (n, 3)))
        s["torso_quat"] = tensor(Rotation.random(n, random_state=rng).as_quat())
        relative = c.position_target + tensor(rng.normal(0, .04, (n, 3)))
        s["box_pos"] = s["torso_pos"] + self.old.quat_rotate(s["torso_quat"], relative)
        dq = tensor(Rotation.from_rotvec(rng.normal(0, [.09, .12, .47], (n, 3))).as_quat())
        s["box_quat"] = self.old.quat_multiply(s["torso_quat"], self.old.quat_multiply(c.orientation_target_xyzw, dq))
        s["box_size"] *= tensor(rng.uniform([.85, .85, .85], [1.15, 1.15, 1.20], (n, 3)))
        offsets = c.hand_directions[None] * s["box_size"][:, None, 1:2] * .5
        offsets += tensor(rng.normal(0, .025, (n, 2, 3)))
        s["hand_pos"] = s["box_pos"][:, None] + self.old.quat_rotate(s["torso_quat"][:, None], offsets)
        s["hand_pos"][0, 0] = s["box_pos"][0]  # degenerate ray
        s["hand_pos"][1] = s["hand_pos"][1].flip(0)  # swapped sides
        s["hand_pos"][2, 1] = s["hand_pos"][2, 0]  # both on one side
        s["hand_pos"][3, 0, 0] += 1  # outside face boundary
        s["previous_relative_pos"] = relative + tensor(rng.normal(0, .004, (n, 3)))
        s["history_valid"][::7] = False
        s["previous_relative_pos"][::7] = 1e6
        s["arm_pos"] += tensor(rng.normal(0, .15, (n, 14)))
        env, metrics = self.assert_equivalent(s)
        for name, sums in env.carry_error_sums.items():
            torch.testing.assert_close(sums, metrics[:, self.old.METRIC_NAMES.index(name)], atol=1e-4, rtol=2e-5)
        signed = copy.deepcopy(s)
        signed["torso_quat"] *= -1
        signed["box_quat"] *= -1
        self.assert_equivalent(signed)
        env.dof_pos[:, :15] = 123  # waist/legs have no influence on arm reward
        torch.testing.assert_close(env._reward_carry_arm_pose(), env.last_rewards["carry_arm_pose"])

    def test_rigid_yaw_and_translation_sequence(self):
        c = self.old.CarryCalibration(self.calibration_data)
        nominal = scene(c, 5)
        commands = torch.tensor([[0, 0, 0], [1.2, 0, 0], [0, -.4, 0], [0, 0, .5], [.96, -.32, -.4]])
        nominal["history_valid"][:] = False
        env = self.make_env(nominal)
        previous = nominal["previous_relative_pos"]
        for step in range(60):
            s = copy.deepcopy(nominal)
            t = step * .02
            q = torch.tensor(Rotation.from_euler("z", (commands[:, 2] * t).numpy()).as_quat(), dtype=torch.float32)
            translation = torch.cat((commands[:, :2] * t, torch.zeros(5, 1)), dim=-1)
            s["torso_pos"], s["torso_quat"] = translation, q
            s["box_pos"] = translation + self.old.quat_rotate(q, s["box_pos"])
            s["box_quat"] = self.old.quat_multiply(q, s["box_quat"])
            s["hand_pos"] = translation[:, None] + self.old.quat_rotate(q[:, None], s["hand_pos"])
            s["previous_relative_pos"] = previous
            s["history_valid"][:] = step > 0
            self.set_state(env, s)
            self.assert_equivalent(s, env)
            previous = self.old.quat_rotate_inverse(q, s["box_pos"] - translation)
            if step:
                torch.testing.assert_close(env.last_rewards["carry_relative_velocity"], torch.ones(5))

    def test_partial_reset_terminal_metrics_and_evaluation_trace(self):
        c = self.old.CarryCalibration(self.calibration_data)
        s = scene(c, 2)
        s["history_valid"][:] = False
        env = self.make_env(s, evaluation=True)
        self.assertEqual((env.upper_body_index, env.carry_torso_index), (0, 1))
        self.assert_equivalent(s, env)
        self.assertNotIn("episode", env.extras)
        self.assertFalse(bool(env.carry_motion_metric_valid.any()))
        s["box_pos"][:, 0] += .02
        s["history_valid"][:] = True
        self.set_state(env, s)
        _, terminal_metrics = self.assert_equivalent(s, env)
        torch.testing.assert_close(env.carry_preservation_metrics, terminal_metrics)
        env.reset_idx(torch.tensor([0]))
        logged = env.extras["episode"]
        self.assertEqual(logged["rew_existing"].item(), 1)
        self.assertAlmostEqual(logged["carry/box_relative_motion_error_mps"].item(), 1, places=4)
        self.assertAlmostEqual(logged["carry/box_relative_position_error_m"].item(), .01, places=5)
        self.assertEqual(env.carry_history_valid.tolist(), [False, True])
        self.assertEqual(env.carry_error_steps.tolist(), [0, 2])
        self.assertEqual(env.carry_motion_samples.tolist(), [0, 1])
        self.assertEqual(env.carry_previous_relative_pos[0].abs().sum().item(), 0)
        torch.testing.assert_close(env.carry_preservation_metrics, terminal_metrics)
        s["previous_relative_pos"] = env.carry_previous_relative_pos.clone()
        s["history_valid"] = env.carry_history_valid.clone()
        self.set_state(env, s)
        self.assert_equivalent(s, env)
        self.assertNotIn("episode", env.extras)
        self.assertEqual(env.last_rewards["carry_relative_velocity"][0].item(), 0)
        self.assertEqual(env.carry_preservation_metrics[0, self.old.MOTION_METRIC_INDEX].item(), 0)
        env.reset_idx(torch.tensor([0]))
        self.assertTrue(math.isnan(env.extras["episode"]["carry/box_relative_motion_error_mps"].item()))

    def test_recorded_carrywith_policy_states(self):
        data, _, _ = analyze.load_dataset()
        for motion in data:
            d = motion["policy_sampled"]
            pos, rot = d["poses"]["torso_link"]
            n = len(pos)
            def tensor(x):
                return torch.tensor(np.asarray(x), dtype=torch.float32)
            relative = analyze.local(rot, d["bp"] - pos)
            state = dict(torso_pos=tensor(pos), torso_quat=tensor(Rotation.from_matrix(rot).as_quat()),
                         box_pos=tensor(d["bp"]), box_quat=tensor(d["bq"]),
                         hand_pos=tensor(np.stack([d["poses"][side + "_palm_link"][0] for side in ("left", "right")], axis=1)),
                         box_size=tensor(np.tile([.35, .35, .30], (n, 1))), arm_pos=tensor(d["dofs"][:, 15:]),
                         previous_relative_pos=tensor(np.concatenate((relative[:1], relative[:-1]))),
                         history_valid=torch.arange(n) > 0, dt=.02)
            self.assert_equivalent(state)


if __name__ == "__main__":
    unittest.main()
