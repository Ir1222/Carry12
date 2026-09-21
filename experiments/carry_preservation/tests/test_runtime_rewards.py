"""CPU semantic tests of the physical carry constraints, without Isaac Gym.

Execute the real specialist methods and parent reward dispatch. Only simulator
setup/reset is stubbed; quaternion operations use the project's TorchScript
utilities (the same flat-batch API as Isaac Gym), not the old oracle's math.
Run: python -m unittest discover -s experiments/carry_preservation/tests -v
"""

import ast
import hashlib
import importlib.util
import math
from pathlib import Path
import types
import unittest
import xml.etree.ElementTree as ET

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


UPPER_BODY_REWARDS = (
    "carry_hand_box_surface", "carry_hand_slip", "carry_arm_range",
    "carry_relative_position", "carry_relative_velocity",
    "carry_bilateral_contact",
)
LOWER_BODY_REWARDS = (
    "carry_hip_posture", "carry_foot_heading",
    "carry_feet_width", "carry_knee_width",
    "carry_waist_reference", "carry_torso_pelvis_alignment",
)
REWARDS = UPPER_BODY_REWARDS + LOWER_BODY_REWARDS

UPPER_BODY_AST_HASHES = {
    "_reward_carry_bilateral_contact": "d6e25c5a4057416482f1ef48abf139b85023656b00d8252b7907a3eb8e50426d",
    "_reward_carry_hand_box_surface": "7704857c00a0d7250fb0a26297b2c00a507bdaccf40e7c8bbf38d6f62c85c402",
    "_reward_carry_hand_slip": "1325ac1cf23a3917068b46c84ce97879ce1454ca0a2d45ef9cd0159434921c6d",
    "_reward_carry_relative_velocity": "e152aa929c556cf09abf8b0c9ea2576eeb422d22307076694eee6b700844b50c",
    "_reward_carry_relative_position": "baedf8f48c2e23a492f74a129c9bdd3e348014212fd3f946fa86e021e0f945b9",
    "_reward_carry_arm_range": "56f60576c659998fbbc6549d051600aff5f6d45c171dbc2e0d9a1472f2b9d34e",
}


class RuntimeRewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.ppo = load_config()
        spec = importlib.util.spec_from_file_location(
            "project_torch_utils", ROOT / "legged_gym/legged_gym/utils/torch_utils.py")
        cls.math_utils = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.math_utils)
        ns = dict(
            torch=torch, np=np, math=math, SimulatorStub=SimulatorStub,
            quat_conjugate=cls.math_utils.quat_conjugate,
            quat_mul=cls.math_utils.quat_mul,
            quat_rotate_inverse=cls.math_utils.quat_rotate_inverse,
            calc_heading_quat=cls.math_utils.calc_heading_quat,
            wrap_to_pi=lambda angles: torch.atan2(torch.sin(angles), torch.cos(angles)),
        )
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
        cls.quaternion_rotation_vector = staticmethod(
            ns["quaternion_rotation_vector"])
        definitions((ROOT / "experiments/carrybox_locomotion_eval/evaluation/trial.py").read_text(),
                    "trial.py", ns)
        cls.eval_sample = staticmethod(ns["_carry_sample"])

    def make_env(self, n=1, dtype=torch.float32):
        env = self.env_type()
        env.num_envs, env.device, env.dt = n, "cpu", 0.02
        env.cfg = types.SimpleNamespace(rewards=self.cfg.rewards)
        env.dof_names = list(self.cfg.init_state.default_joint_angles)
        env.dof_pos = torch.zeros(n, 29, dtype=dtype)
        env.default_dof_pos = env.dof_pos.new_tensor([
            self.cfg.init_state.default_joint_angles[name]
            for name in env.dof_names
        ])
        body_indices = {
            "torso_link": 1,
            "left_palm_link": 2, "right_palm_link": 3,
            "left_ankle_pitch_link": 4, "right_ankle_pitch_link": 5,
            "left_knee_link": 6, "right_knee_link": 7,
        }
        env.gym = types.SimpleNamespace(
            find_actor_rigid_body_handle=lambda e, a, name: body_indices[name])
        env.envs, env.actor_handles = [0], [0]
        env.upper_body_index = 0
        env._init_buffers()
        env.rigid_body_states = env.dof_pos.new_zeros(n, 8, 13)
        env.root_states = env.dof_pos.new_zeros(n, 13)
        env.box_states = env.dof_pos.new_zeros(n, 13)
        env.rigid_body_states[..., 6] = 1
        env.root_states[:, 6] = 1
        env.box_states[:, 6] = 1
        env.dof_pos[:, env.carry_waist_indices] = (
            env.carry_waist_reference_target)
        env.rigid_body_states[:, env.carry_torso_index, 3:7] = (
            env.carry_torso_pelvis_reference_quat)
        env._box_size = env.dof_pos.new_tensor([.35, .35, .30]).repeat(n, 1)
        env.dof_pos[:, env.carry_arm_indices] = (env.carry_arm_range_lower + env.carry_arm_range_upper) / 2
        env.box_states[:, :3] = env.dof_pos.new_tensor([.35, 0, .05])
        env.rigid_body_states[:, 2:4, :3] = env.box_states[:, None, :3]
        env.rigid_body_states[:, 2, 1] += .175
        env.rigid_body_states[:, 3, 1] -= .175
        env.rigid_body_states[:, 4, 1] = .10
        env.rigid_body_states[:, 5, 1] = -.10
        env.rigid_body_states[:, 6, 1] = .09
        env.rigid_body_states[:, 7, 1] = -.09
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
        env.dof_pos[:, :15] = 100
        rewards = self.step(env)
        # Upper-body carry preservation remains independent of waist and legs.
        for name in UPPER_BODY_REWARDS:
            torch.testing.assert_close(rewards[name], torch.ones_like(rewards[name]))

    def test_actor_observation_shape_is_unchanged(self):
        self.assertEqual(self.cfg.env.num_actor_history, 6)
        self.assertEqual(self.cfg.env.num_actor_obs, 738)
        self.assertEqual(
            self.cfg.env.num_actor_obs // self.cfg.env.num_actor_history, 123)
        self.assertEqual(self.cfg.env.num_actions, 29)
        self.assertEqual(self.cfg.env.num_dofs, 29)

    def test_v2_reward_scales_and_moderate_commands(self):
        expected_scales = {
            "carry_lin_vel_tracking": 3.0,
            "carry_yaw_vel_tracking": 2.5,
            "carry_bilateral_contact": 0.5,
            "carry_hand_box_surface": 1.5,
            "carry_hand_slip": 0.5,
            "carry_relative_velocity": 0.5,
            "carry_relative_position": 0.75,
            "carry_relative_orientation": 0.0,
            "carry_arm_range": 0.2,
            "carry_waist_reference": 0.4,
            "carry_torso_pelvis_alignment": 0.2,
            "carry_hip_posture": 0.8,
            "carry_foot_heading": 0.35,
            "carry_feet_width": 0.30,
            "carry_knee_width": 0.20,
            "carry_leg_range": 0.0,
            "carry_stance_width": 0.0,
            "carry_upper_body_pose": 0.0,
        }
        for name, expected in expected_scales.items():
            self.assertEqual(getattr(self.cfg.rewards.scales, name), expected)
        self.assertEqual(
            self.cfg.commands.carry_command_mode_probabilities,
            [0.10, 0.25, 0.15, 0.15, 0.35],
        )
        self.assertEqual(self.cfg.commands.carry_yaw_rate_range, [-0.5, 0.5])

    def test_upper_body_reward_implementations_are_unchanged(self):
        tree = ast.parse((ROOT / ENV_PATH).read_text())
        actual = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in UPPER_BODY_AST_HASHES:
                dump = ast.dump(node, include_attributes=False)
                actual[node.name] = hashlib.sha256(dump.encode()).hexdigest()
        self.assertEqual(actual, UPPER_BODY_AST_HASHES)

    def test_verified_carrywith_waist_calibration_and_joint_order(self):
        mapping_path = ROOT / "legged_gym/resources/config/joint_id.txt"
        mapping = {
            name: int(index)
            for index, name in (
                line.split() for line in mapping_path.read_text().splitlines()
            )
        }
        waist_names = (
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        )
        self.assertEqual(tuple(self.cfg.rewards.carry_waist_joint_names), waist_names)
        self.assertEqual(tuple(mapping[name] for name in waist_names), (12, 13, 14))

        urdf = ET.parse(
            ROOT / "legged_gym/resources/robots/g1/urdf/g1_29dof.urdf"
        ).getroot()
        joints = {joint.get("name"): joint for joint in urdf.findall("joint")}
        chain = (
            ("waist_yaw_joint", "pelvis", "waist_yaw_link"),
            ("waist_roll_joint", "waist_yaw_link", "waist_roll_link"),
            ("waist_pitch_joint", "waist_roll_link", "torso_link"),
        )
        for name, parent, child in chain:
            self.assertEqual(joints[name].find("parent").get("link"), parent)
            self.assertEqual(joints[name].find("child").get("link"), child)

        frames = []
        motion_dir = ROOT / "legged_gym/resources/dataset/dataset_carrybox/carryWith"
        for path in sorted(motion_dir.glob("carrywith*.pt")):
            try:
                motion = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                motion = torch.load(path, map_location="cpu")
            frames.append(motion["joint_position"][:, [12, 13, 14]])
        waist = torch.cat(frames).numpy()
        self.assertEqual(len(waist), 773)
        np.testing.assert_allclose(
            np.median(waist, axis=0),
            self.cfg.rewards.carry_waist_reference_target,
            atol=1.0e-8,
        )
        np.testing.assert_allclose(
            np.percentile(waist, [5, 95], axis=0),
            [[-0.310877460, -0.022957506, 0.155981871],
             [0.212081027, 0.093573584, 0.361138302]],
            atol=1.0e-8,
        )

        yaw, roll, pitch = waist.T
        relative = (
            Rotation.from_euler("z", yaw).as_matrix()
            @ Rotation.from_euler("x", roll).as_matrix()
            @ Rotation.from_euler("y", pitch).as_matrix()
        )
        reference = Rotation.from_matrix(relative).mean().as_quat()
        if reference[3] < 0.0:
            reference = -reference
        np.testing.assert_allclose(
            reference,
            self.cfg.rewards.carry_torso_pelvis_reference_quat,
            atol=1.0e-8,
        )

    def test_waist_reference_deadzone_symmetry_and_monotonicity(self):
        env = self.make_env(4)
        expected_names = (
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        )
        selected_names = tuple(env.dof_names[index] for index in env.carry_waist_indices)
        self.assertEqual(selected_names, expected_names)
        self.assertTrue(set(env.carry_waist_indices.tolist()).isdisjoint(
            env.carry_arm_indices.tolist()))
        self.assertTrue(set(env.carry_waist_indices.tolist()).isdisjoint(
            env.carry_leg_indices.tolist()))

        for axis in range(3):
            env = self.make_env(4)
            deadzone = env.carry_waist_reference_deadzone[axis]
            errors = torch.tensor(
                [0.5 * deadzone, deadzone + 0.05,
                 deadzone + 0.20, -(deadzone + 0.05)],
                dtype=env.dof_pos.dtype,
            )
            env.dof_pos[:, env.carry_waist_indices[axis]] += errors
            reward = self.step(env)["carry_waist_reference"]
            self.assertEqual(reward[0].item(), 1.0)
            self.assertGreater(reward[1].item(), reward[2].item())
            torch.testing.assert_close(reward[1], reward[3])

    def test_torso_pelvis_alignment_so3_properties(self):
        for axis in range(3):
            env = self.make_env(4)
            deadzone = env.carry_torso_pelvis_alignment_deadzone[axis].item()
            deviations = np.zeros((4, 3))
            deviations[:, axis] = [0.5 * deadzone, deadzone + 0.05,
                                   deadzone + 0.20, -(deadzone + 0.05)]
            delta = torch.tensor(
                Rotation.from_rotvec(deviations).as_quat(),
                dtype=env.dof_pos.dtype,
            )
            reference = env.carry_torso_pelvis_reference_quat.unsqueeze(
                0).expand_as(delta)
            env.rigid_body_states[:, env.carry_torso_index, 3:7] = (
                self.math_utils.quat_mul(reference, delta))
            reward = self.step(env)["carry_torso_pelvis_alignment"]
            self.assertEqual(reward[0].item(), 1.0)
            self.assertGreater(reward[1].item(), reward[2].item())
            torch.testing.assert_close(reward[1], reward[3])

        sign = self.make_env(2)
        sign.rigid_body_states[1, sign.carry_torso_index, 3:7] *= -1.0
        reward = self.step(sign)["carry_torso_pelvis_alignment"]
        torch.testing.assert_close(reward, torch.ones_like(reward))

        near_wrap = torch.tensor(
            Rotation.from_euler("z", [math.pi - 0.01, -math.pi + 0.01]).as_quat(),
            dtype=torch.float32,
        )
        error_quat = self.math_utils.quat_mul(
            self.math_utils.quat_conjugate(near_wrap[:1]), near_wrap[1:])
        rotvec = self.quaternion_rotation_vector(error_quat)
        self.assertAlmostEqual(abs(rotvec[0, 2].item()), 0.02, places=5)

    def test_hip_deadzone_monotonicity_and_left_right_symmetry(self):
        env = self.make_env(3)
        target = env.carry_hip_target[1]
        env.dof_pos[:, env.carry_hip_indices[1]] = target + torch.tensor([.05, .15, .35])
        reward = self.step(env)["carry_hip_posture"]
        self.assertEqual(reward[0].item(), 1.0)
        self.assertTrue(bool((reward[:-1] > reward[1:]).all()))

        symmetric = self.make_env(2)
        symmetric.dof_pos[0, symmetric.carry_hip_indices[1]] = .25
        symmetric.dof_pos[1, symmetric.carry_hip_indices[3]] = .25
        reward = self.step(symmetric)["carry_hip_posture"]
        torch.testing.assert_close(reward[0], reward[1])

    def test_foot_heading_deadzone_sideways_and_wrap(self):
        env = self.make_env(4)
        pelvis_yaw = torch.tensor([0.0, 0.0, math.pi - .02, 0.0])
        foot_yaw = torch.tensor([0.0, .08, -math.pi + .02, math.pi / 2])
        env.root_states[:, 3:7] = torch.tensor(
            Rotation.from_euler("z", pelvis_yaw.numpy()).as_quat(),
            dtype=env.dof_pos.dtype,
        )
        foot_quat = torch.tensor(
            Rotation.from_euler("z", foot_yaw.numpy()).as_quat(),
            dtype=env.dof_pos.dtype,
        )
        env.rigid_body_states[:, 4:6, 3:7] = foot_quat[:, None, :]
        reward = self.step(env)["carry_foot_heading"]
        torch.testing.assert_close(reward[:3], torch.ones_like(reward[:3]))
        self.assertLess(reward[3].item(), reward[2].item())
        self.assertAlmostEqual(
            abs(env.carry_foot_heading_error[2, 0].item()), .04, places=4)

    def test_two_sided_feet_and_knee_width_intervals(self):
        env = self.make_env(5)
        feet_width = torch.tensor([.18, .09, .27, .05, .31])
        knee_width = torch.tensor([.18, .14, .25, .10, .29])
        env.rigid_body_states[:, 4, 1] = feet_width / 2
        env.rigid_body_states[:, 5, 1] = -feet_width / 2
        env.rigid_body_states[:, 6, 1] = knee_width / 2
        env.rigid_body_states[:, 7, 1] = -knee_width / 2
        rewards = self.step(env)
        for name in ("carry_feet_width", "carry_knee_width"):
            torch.testing.assert_close(
                rewards[name][:3], torch.ones_like(rewards[name][:3]))
            self.assertTrue(bool((rewards[name][3:] < 1.0).all()))
            torch.testing.assert_close(rewards[name][3], rewards[name][4])

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
        for name in (
            "carry/waist_reference_reward",
            "carry/waist_yaw_error_abs",
            "carry/waist_roll_error_abs",
            "carry/waist_pitch_error_abs",
            "carry/waist_yaw_excess",
            "carry/waist_roll_excess",
            "carry/waist_pitch_excess",
            "carry/torso_pelvis_alignment_reward",
            "carry/torso_pelvis_rotvec_x_abs",
            "carry/torso_pelvis_rotvec_y_abs",
            "carry/torso_pelvis_rotvec_z_abs",
            "carry/torso_pelvis_alignment_excess",
        ):
            self.assertIn(name, env.extras["episode"])
            self.assertTrue(torch.isfinite(env.extras["episode"][name]))
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
        box = env.box_states[:, :3].clone()
        body_positions = env.rigid_body_states[:, :, :3].clone()
        body_orientations = env.rigid_body_states[:, :, 3:7].clone()
        self.step(env)
        for t in range(1, 6):
            q = torch.tensor(Rotation.from_euler("z", [.1*t]*4).as_quat(), dtype=torch.float32)
            shift = torch.tensor([.03*t, -.02*t, 0.])
            body_q = q[:, None, :].expand(-1, 8, -1).reshape(-1, 4)
            env.rigid_body_states[:, :, :3] = self.math_utils.quat_rotate(
                body_q, body_positions.reshape(-1, 3)).reshape(-1, 8, 3) + shift
            env.rigid_body_states[:, :, 3:7] = self.math_utils.quat_mul(
                body_q, body_orientations.reshape(-1, 4)).reshape(-1, 8, 4)
            env.root_states[:, :3] = shift
            env.root_states[:, 3:7] = q
            env.box_states[:, :3] = self.math_utils.quat_rotate(q, box) + shift
            env.box_states[:, 3:7] = q
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
