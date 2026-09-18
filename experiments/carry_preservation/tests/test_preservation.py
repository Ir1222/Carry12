"""Historical calibration/oracle tests; these kernels are offline only."""

import copy
import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "legged_gym"))
from legged_gym.carry_preservation import (
    CarryCalibration, CarryMetricAccumulator, METRIC_NAMES, MOTION_METRIC_INDEX,
    REWARD_NAMES, compute_preservation, quat_multiply, quat_rotate,
    quat_rotate_inverse, quaternion_log,
)
from experiments.carry_preservation import analyze


def calibration(device="cpu", dtype=torch.float64):
    return CarryCalibration.load(analyze.DEFAULT_CALIBRATION, device=device, dtype=dtype)


def scene(c, n=1):
    def repeat(x):
        return x.unsqueeze(0).repeat(n, *([1] * x.ndim))
    torso_pos = torch.zeros(n, 3, dtype=c.arm_target.dtype, device=c.arm_target.device)
    torso_quat = torch.zeros(n, 4, dtype=c.arm_target.dtype, device=c.arm_target.device)
    torso_quat[:, 3] = 1
    box_pos = repeat(c.position_target)
    box_quat = repeat(c.orientation_target_xyzw)
    size = repeat(c.position_target.new_tensor([0.35, 0.35, 0.30]))
    direction_box = quat_rotate_inverse(box_quat[:, None], c.hand_directions[None])
    offset = c.hand_directions[None] * (size[:, None, 1:2] / (2 * direction_box[..., 1:2].abs()))
    return dict(torso_pos=torso_pos, torso_quat=torso_quat, box_pos=box_pos,
                box_quat=box_quat, hand_pos=box_pos[:, None] + offset,
                box_size=size, arm_pos=repeat(c.arm_target),
                previous_relative_pos=repeat(c.position_target),
                history_valid=torch.ones(n, dtype=torch.bool, device=c.arm_target.device), dt=0.02)


def evaluate(s, c):
    return compute_preservation(calibration=c, **s)


class KernelTests(unittest.TestCase):
    def setUp(self):
        self.c = calibration()

    def test_nominal_and_finite_degenerate_hand(self):
        s = scene(self.c, 4)
        rewards, metrics, _ = evaluate(s, self.c)
        for value in rewards.values():
            torch.testing.assert_close(value, torch.ones_like(value))
        self.assertEqual(metrics.shape, (4, 9))
        s["hand_pos"][:, 0] = s["box_pos"]
        rewards, metrics, _ = evaluate(s, self.c)
        self.assertTrue(bool(torch.isfinite(metrics).all()))
        self.assertTrue(bool((rewards["carry_hand_box_surface"] < 1e-10).all()))

    def test_swapped_same_side_and_single_hand_loss(self):
        for kind in ("swap", "same_side", "away", "overflow"):
            s = scene(self.c)
            if kind == "swap":
                s["hand_pos"] = s["hand_pos"].flip(1)
            elif kind == "same_side":
                s["hand_pos"][:, 1] = s["hand_pos"][:, 0]
            else:
                offset = self.c.arm_target.new_tensor([[0, 0.15, 0] if kind == "away" else [0.5, 0, 0]])
                s["hand_pos"][:, 0] += quat_rotate(s["box_quat"], offset)
            rewards, _, _ = evaluate(s, self.c)
            self.assertLess(rewards["carry_hand_box_surface"].item(), 1e-4, kind)

    def test_monotonic_translation_rotation_arm_and_sliding(self):
        for axis in range(3):
            for mode, reward_key in (("position", "carry_relative_position"),
                                     ("orientation", "carry_relative_orientation"),
                                     ("motion", "carry_relative_velocity")):
                scores = []
                for magnitude in (0.0, 0.1, 0.2, 0.4):
                    s = scene(self.c)
                    if mode == "position":
                        s["box_pos"][:, axis] += magnitude
                    elif mode == "motion":
                        s["previous_relative_pos"][:, axis] -= magnitude * s["dt"]
                    else:
                        rotvec = np.eye(3)[axis] * magnitude
                        dq = self.c.arm_target.new_tensor(Rotation.from_rotvec(rotvec).as_quat())
                        s["box_quat"] = quat_multiply(s["box_quat"], dq)
                    scores.append(evaluate(s, self.c)[0][reward_key].item())
                self.assertTrue(all(a > b for a, b in zip(scores, scores[1:])), (mode, axis, scores))
        scores = []
        for amount in (0.0, 0.1, 0.3, 0.6):
            s = scene(self.c)
            s["arm_pos"] += amount
            scores.append(evaluate(s, self.c)[0]["carry_arm_pose"].item())
        self.assertTrue(all(a > b for a, b in zip(scores, scores[1:])))

    def test_orientation_matches_scipy_and_quaternion_sign(self):
        q = torch.tensor(Rotation.from_rotvec([[.2, -.3, .5], [0, 0, 3.13], [1e-10, 0, 0]]).as_quat())
        expected = torch.tensor(Rotation.from_quat(q.numpy()).as_rotvec())
        torch.testing.assert_close(quaternion_log(q), expected)
        torch.testing.assert_close(quaternion_log(-q), expected)
        original = scene(self.c, 3)
        reference = evaluate(original, self.c)
        for key in ("torso_quat", "box_quat"):
            s = copy.deepcopy(original)
            s[key] *= -1
            actual = evaluate(s, self.c)
            for name in REWARD_NAMES:
                torch.testing.assert_close(actual[0][name], reference[0][name])
            torch.testing.assert_close(actual[1], reference[1])

    def test_rigid_turn_and_translation_all_command_families(self):
        for vx, vy, yaw in ((0, 0, 0), (1.2, 0, 0), (0, -.4, 0),
                            (0, 0, .5), (.96, -.32, -.4)):
            s = scene(self.c, 60)
            t = torch.arange(60, dtype=torch.float64) * .02
            global_q = torch.tensor(Rotation.from_euler("z", (t * yaw).numpy()).as_quat())
            translation = torch.stack((vx * t, vy * t, torch.zeros_like(t)), dim=-1)
            s["torso_pos"] = translation
            s["torso_quat"] = global_q
            s["box_pos"] = translation + quat_rotate(global_q, s["box_pos"])
            s["hand_pos"] = translation[:, None] + quat_rotate(global_q[:, None], s["hand_pos"])
            s["box_quat"] = quat_multiply(global_q, s["box_quat"])
            rewards, metrics, _ = evaluate(s, self.c)
            self.assertLess(metrics[:, MOTION_METRIC_INDEX].max().item(), 1e-11)
            for name, value in rewards.items():
                torch.testing.assert_close(value, torch.ones_like(value), atol=1e-10, rtol=0)
            if yaw:
                # A valid rotating offset has unequal world linear velocities.
                naive = np.linalg.norm(np.cross([0, 0, yaw], self.c.position_target.numpy()))
                self.assertGreater(naive, .1)

    def test_batch_independence_and_reset_mask(self):
        s = scene(self.c, 3)
        s["box_pos"][1, 0] += .08
        s["arm_pos"][2] += .2
        s["previous_relative_pos"][0] = 1e6
        s["history_valid"][0] = False
        batched = evaluate(s, self.c)
        self.assertEqual(batched[0]["carry_relative_velocity"][0].item(), 0)
        self.assertEqual(batched[1][0, MOTION_METRIC_INDEX].item(), 0)
        for index in range(3):
            one = {key: value[index:index + 1] if torch.is_tensor(value) else value for key, value in s.items()}
            actual = evaluate(one, self.c)
            for name in REWARD_NAMES:
                torch.testing.assert_close(actual[0][name][0], batched[0][name][index])
            torch.testing.assert_close(actual[1][0], batched[1][index])

    def test_float32_training_batch(self):
        c = calibration(dtype=torch.float32)
        s = scene(c, 4096)
        s["box_pos"][:, 0] += torch.linspace(-.1, .1, 4096)
        rewards, metrics, relative = evaluate(s, c)
        self.assertEqual(metrics.shape, (4096, 9))
        self.assertEqual(relative.shape, (4096, 3))
        self.assertTrue(bool(torch.isfinite(metrics).all()))
        for value in rewards.values():
            self.assertEqual(value.dtype, torch.float32)
            self.assertEqual(value.device.type, "cpu")
            self.assertTrue(bool(((value >= 0) & (value <= 1)).all()))

    def test_metric_counts_and_terminal_sample(self):
        a = CarryMetricAccumulator(2, "cpu", torch.float64)
        first = torch.full((2, 9), 2.0, dtype=torch.float64)
        first[0, MOTION_METRIC_INDEX] = 9999  # invalid, must never enter sums
        a.add(first, torch.tensor([False, True]))
        a.add(torch.full((2, 9), 4.0), torch.tensor([True, True]))
        saved = a.pop(torch.tensor([0]))
        self.assertEqual(saved["carry/left_hand_side_error_m"].item(), 3)
        self.assertEqual(saved["carry/box_relative_motion_error_mps"].item(), 4)
        self.assertEqual(a.counts[0].sum().item(), 0)
        self.assertEqual(a.counts[1, MOTION_METRIC_INDEX].item(), 2)
        a.add(torch.full((2, 9), 8.0), torch.tensor([False, True]))
        self.assertEqual(saved["carry/left_hand_side_error_m"].item(), 3)
        self.assertTrue(math.isnan(a.pop(torch.tensor([0]))["carry/box_relative_motion_error_mps"].item()))

    def test_schema_and_calibration_validation(self):
        self.assertEqual(len(self.c.arm_joint_names), 14)
        self.assertTrue(all("waist" not in name and "hip" not in name for name in self.c.arm_joint_names))
        data = json.loads(analyze.DEFAULT_CALIBRATION.read_text())
        data["motion_sigma"][0] = 0
        with self.assertRaises(ValueError):
            CarryCalibration(data)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA runtime unavailable")
    def test_cuda_matches_cpu(self):
        cpu_c, gpu_c = calibration(dtype=torch.float32), calibration("cuda", torch.float32)
        cpu = scene(cpu_c, 128)
        cpu["box_pos"][:, 0] += torch.linspace(0, .2, 128)
        gpu = {key: value.cuda() if torch.is_tensor(value) else value for key, value in cpu.items()}
        cr, cm, _ = evaluate(cpu, cpu_c)
        gr, gm, _ = evaluate(gpu, gpu_c)
        for key in cr:
            torch.testing.assert_close(cr[key], gr[key].cpu(), atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(cm, gm.cpu(), atol=2e-5, rtol=2e-5)


class ReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.kinematics, cls.provenance = analyze.load_dataset()
        cls.generated = analyze.calibrate(cls.data, cls.kinematics, cls.provenance)
        cls.c = CarryCalibration(cls.generated, dtype=torch.float64)

    def test_every_motion_and_reproducible_calibration(self):
        self.assertEqual([d["n"] for d in self.data], [256, 194, 323])
        saved = json.loads(analyze.DEFAULT_CALIBRATION.read_text())
        self.assertEqual(saved["source_sha256"], self.provenance)
        for name in ("hand_directions", "hand_direction_sigma", "hand_normal_sigma", "arm_target",
                     "arm_sigma", "position_target", "position_sigma", "orientation_target_xyzw",
                     "orientation_sigma", "motion_sigma"):
            np.testing.assert_allclose(saved[name], self.generated[name], rtol=2e-5, atol=2e-6)

    def test_raw_reference_quality_and_explicit_synthetic_limitations(self):
        for d in self.data:
            self.assertLess(abs(d["quantities"]["stored_hand_midpoint_error"]).max(), 2e-7)
            self.assertTrue(np.all(d["hand_box"][:, 0, 1] > 0))
            self.assertTrue(np.all(d["hand_box"][:, 1, 1] < 0))
            for side in ("left", "right"):
                self.assertLess(np.linalg.norm(d["quantities"][side + "/fk_minus_stored_palm"], axis=1).max(), .005)
            rewards, _, _ = analyze.evaluate_sample(d, self.c)
            self.assertLess(rewards["carry_hand_box_surface"].mean().item(), .025)
            self.assertGreater(rewards["carry_arm_pose"].mean().item(), .97)
            self.assertGreater(rewards["carry_relative_position"].mean().item(), .81)
            self.assertGreater(rewards["carry_relative_orientation"].mean().item(), .81)
            rewards, _, valid = analyze.evaluate_sample(d["policy_sampled"], self.c, dt=.02)
            self.assertGreater(rewards["carry_relative_velocity"][valid].mean().item(), .86)

    def test_surface_retargeting_at_box_size_extremes(self):
        import itertools
        for x, y, z in itertools.product((.35 * .85, .35 * 1.15),
                                          (.35 * .85, .35 * 1.15), (.30 * .85, .30 * 1.20)):
            for d in self.data:
                rewards, metrics, _ = analyze.evaluate_sample(
                    d, self.c, surface_retarget=True, box_size=np.array([x, y, z]))
                self.assertGreater(rewards["carry_hand_box_surface"].mean().item(), .94)
                self.assertLess(metrics[:, :2].max().item(), 1e-6)

    def test_compensated_velocity_matches_local_derivative(self):
        for d in self.data:
            q = d["quantities"]
            delta = q["torso_link/compensated_velocity"][1:-1] - q["torso_link/local_position_derivative"][1:-1]
            self.assertLess(np.sqrt((delta * delta).mean()), .002)


if __name__ == "__main__":
    unittest.main()
