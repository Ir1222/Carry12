import ast
import importlib.util
import math
import pathlib
import unittest

import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
UTILS_PATH = (
    REPO_ROOT
    / "legged_gym"
    / "legged_gym"
    / "envs"
    / "g1"
    / "carrybox_force_utils.py"
)
SPEC = importlib.util.spec_from_file_location("carrybox_force_utils", UTILS_PATH)
UTILS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(UTILS)
compute_admittance_teacher = UTILS.compute_admittance_teacher
smooth_force_profile = UTILS.smooth_force_profile


class TestForceProfile(unittest.TestCase):
    def test_cosine_ramp_hold_and_down(self):
        elapsed = torch.tensor([0.0, 0.2, 0.4, 2.4, 2.6, 2.8, 2.81])
        ramp = torch.full_like(elapsed, 0.4)
        hold = torch.full_like(elapsed, 2.0)
        actual = smooth_force_profile(elapsed, ramp, hold, ramp)
        expected = torch.tensor([0.0, 0.5, 1.0, 1.0, 0.5, 0.0, 0.0])
        torch.testing.assert_close(actual, expected, atol=1.0e-6, rtol=0.0)


class TestAdmittanceTeacher(unittest.TestCase):
    def _teacher(self, force_xy):
        force = torch.tensor([[force_xy[0], force_xy[1], 0.0]], dtype=torch.float64)
        mass = torch.tensor([2.5], dtype=torch.float64)
        nominal_vx = torch.tensor([0.5], dtype=torch.float64)
        nominal_heading = torch.tensor([0.0], dtype=torch.float64)
        robot_yaw = torch.tensor([0.0], dtype=torch.float64)
        nominal_heading_error = torch.tensor([0.0], dtype=torch.float64)
        nominal_raw_yaw = torch.tensor([0.0], dtype=torch.float64)
        nominal_yaw_target = torch.tensor([0.0], dtype=torch.float64)
        return compute_admittance_teacher(
            force,
            mass,
            nominal_vx,
            nominal_heading,
            robot_yaw,
            nominal_heading_error,
            nominal_raw_yaw,
            nominal_yaw_target,
            heading_kp=1.0,
            max_yaw_rate=0.4,
            admittance_d_bar=12.0,
            max_heading_offset=math.pi / 4.0,
            teacher_vx_min=0.0,
            teacher_vx_max=0.8,
        )

    def test_backward_force_reduces_speed_by_beta_g_over_dbar(self):
        mass = 2.5
        peak = 0.10 * mass * 9.81
        teacher = self._teacher((-peak, 0.0))
        self.assertAlmostEqual(float(teacher[0]), 0.5 - 0.10 * 9.81 / 12.0, places=12)
        self.assertAlmostEqual(float(teacher[1]), 0.0, places=12)
        self.assertAlmostEqual(float(teacher[3]), 0.0, places=12)

    def test_strong_pure_backward_force_stops_without_turning(self):
        teacher = self._teacher((-18.0, 0.0))
        self.assertAlmostEqual(float(teacher[0]), 0.0, places=12)
        self.assertAlmostEqual(float(teacher[1]), 0.0, places=12)
        self.assertAlmostEqual(float(teacher[3]), 0.0, places=12)

    def test_backward_force_before_zero_crossing_remains_longitudinal(self):
        teacher = self._teacher((-7.5, 0.0))
        self.assertAlmostEqual(float(teacher[0]), 0.25, places=12)
        self.assertAlmostEqual(float(teacher[1]), 0.0, places=12)

    def test_direction_signs(self):
        mass = 2.5
        peak = 0.10 * mass * 9.81
        forward = self._teacher((peak, 0.0))
        left = self._teacher((0.0, peak))
        right = self._teacher((0.0, -peak))
        self.assertGreater(float(forward[0]), 0.5)
        self.assertGreater(float(left[1]), 0.0)
        self.assertGreater(float(left[3]), 0.0)
        self.assertLess(float(right[1]), 0.0)
        self.assertLess(float(right[3]), 0.0)

    def test_zero_force_returns_exact_stage1_targets(self):
        nominal_vx = torch.tensor([0.37], dtype=torch.float64)
        nominal_heading = torch.tensor([0.21], dtype=torch.float64)
        robot_yaw = torch.tensor([0.11], dtype=torch.float64)
        nominal_heading_error = torch.tensor([0.1], dtype=torch.float64)
        nominal_yaw_target = torch.tensor([0.13], dtype=torch.float64)
        teacher = compute_admittance_teacher(
            torch.zeros((1, 3), dtype=torch.float64),
            torch.tensor([2.5], dtype=torch.float64),
            nominal_vx,
            nominal_heading,
            robot_yaw,
            nominal_heading_error,
            torch.tensor([0.03], dtype=torch.float64),
            nominal_yaw_target,
            heading_kp=1.0,
            max_yaw_rate=0.4,
            admittance_d_bar=12.0,
            max_heading_offset=math.pi / 4.0,
            teacher_vx_min=0.0,
            teacher_vx_max=0.8,
        )
        self.assertTrue(torch.equal(teacher[0], nominal_vx))
        self.assertTrue(torch.equal(teacher[1], nominal_heading))
        self.assertTrue(torch.equal(teacher[2], nominal_heading_error))
        self.assertTrue(torch.equal(teacher[3], nominal_yaw_target))


class TestActorObservationIsolation(unittest.TestCase):
    def test_force_and_teacher_names_are_absent_from_observation_builders(self):
        source_path = (
            REPO_ROOT / "legged_gym" / "legged_gym" / "envs" / "g1" / "carrybox_force.py"
        )
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        forbidden = {
            "external_force_world",
            "external_force_beta",
            "external_force_active",
            "teacher_vx",
            "teacher_heading",
            "teacher_yaw_rate",
            "force_parallel",
            "force_perp",
        }
        observation_methods = {
            "compute_observations",
            "compute_task_observations",
            "compute_termination_observations",
        }
        force_class = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LeggedRobot"
        )
        defined_methods = {
            node.name for node in force_class.body if isinstance(node, ast.FunctionDef)
        }
        self.assertTrue(
            observation_methods.isdisjoint(defined_methods),
            "carrybox_force must inherit Stage1 observation construction unchanged",
        )
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in observation_methods:
                names = {child.attr for child in ast.walk(node) if isinstance(child, ast.Attribute)}
                self.assertTrue(forbidden.isdisjoint(names), f"leaked names: {forbidden & names}")

    def test_nominal_command_buffers_have_no_force_or_teacher_rhs(self):
        source_path = (
            REPO_ROOT / "legged_gym" / "legged_gym" / "envs" / "g1" / "carrybox_force.py"
        )
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        protected = {"commands", "carry_policy_commands", "carry_heading_ref"}
        forbidden = {
            "external_force_world",
            "external_force_beta",
            "external_force_active",
            "teacher_vx",
            "teacher_heading",
            "teacher_yaw_rate",
            "force_parallel",
            "force_perp",
        }
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AugAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            target_attrs = {
                child.attr for target in targets for child in ast.walk(target)
                if isinstance(child, ast.Attribute)
            }
            if protected.isdisjoint(target_attrs):
                continue
            value_attrs = {
                child.attr for child in ast.walk(node.value) if isinstance(child, ast.Attribute)
            }
            self.assertTrue(forbidden.isdisjoint(value_attrs), f"command leakage: {forbidden & value_attrs}")


class TestArchitectureParity(unittest.TestCase):
    def test_force_config_inherits_actor_critic_interface(self):
        config_path = (
            REPO_ROOT / "legged_gym" / "legged_gym" / "envs" / "g1" / "carrybox_force_config.py"
        )
        tree = ast.parse(config_path.read_text(encoding="utf-8"))
        env_cfg = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "G1Cfg")
        ppo_cfg = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "G1CfgPPO")
        self.assertEqual(ast.unparse(env_cfg.bases[0]), "CarryBoxCfg")
        self.assertEqual(ast.unparse(ppo_cfg.bases[0]), "CarryBoxCfgPPO")
        self.assertNotIn("env", {node.name for node in env_cfg.body if isinstance(node, ast.ClassDef)})
        self.assertNotIn("policy", {node.name for node in ppo_cfg.body if isinstance(node, ast.ClassDef)})
        self.assertNotIn("algorithm", {node.name for node in ppo_cfg.body if isinstance(node, ast.ClassDef)})

    def test_stage1_checkpoint_dimensions(self):
        checkpoint_path = REPO_ROOT / "legged_gym" / "resources" / "ckpt" / "carrybox.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint["model_state_dict"]
        self.assertEqual(tuple(state["actor.0.weight"].shape), (512, 738))
        self.assertEqual(tuple(state["actor.6.weight"].shape), (29, 256))
        self.assertEqual(tuple(state["critic.0.weight"].shape), (512, 126))


if __name__ == "__main__":
    unittest.main()
