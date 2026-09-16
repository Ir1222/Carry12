import math
import unittest

from experiments.carrybox_locomotion_eval.evaluation.command_suite import CommandCondition
from experiments.carrybox_locomotion_eval.evaluation.metrics import aggregate_by_mode, summarize_trial


def sample(vx):
    return {
        "command_vx": 0.5, "command_vy": 0.0, "command_yaw_rate": 0.0,
        "actual_vx_training_frame": vx, "actual_vy_training_frame": 0.1,
        "actual_yaw_rate_training_frame": -0.2, "vx_error": vx - 0.5,
        "vy_error": 0.1, "yaw_rate_error": -0.2,
        "normalized_vector_error": math.sqrt(((vx - 0.5) / 1.2) ** 2 + 0.25 ** 2 + 0.4 ** 2),
        "xy_speed": math.hypot(vx, 0.1), "base_pos_x": vx, "base_pos_y": 0.0,
        "base_yaw": 0.1 * vx, "box_vx_error": vx - 0.4,
        "box_vy_error": 0.05, "box_yaw_rate_error": -0.1,
        "robot_box_relative_linear_velocity_norm": 0.03,
        "bilateral_contact": 1, "grasp_loss": 0, "robot_box_distance": 0.4,
        "box_tilt_deg": 5.0, "confirmed_carry": 1,
        "action_delta_rms": 0.2, "action_rate_rms": 10.0,
        "torque_rms": 12.0, "feet_slip": 0.01,
    }


class MetricTests(unittest.TestCase):
    def test_signed_axis_metrics(self):
        condition = CommandCondition("T0001", "vx", 0.5, 0.0, 0.0, 1, 0, 0.5)
        row = summarize_trial(
            condition, [sample(0.3), sample(0.7)], policy_dt=0.02,
            requested_steps=2, executed_steps=3, termination_reason="completed",
        )
        self.assertEqual(row["trial_completed"], 1)
        self.assertAlmostEqual(row["vx_bias"], 0.0)
        self.assertAlmostEqual(row["vx_mae"], 0.2)
        self.assertAlmostEqual(row["vx_rmse"], 0.2)
        self.assertEqual(row["final_confirmed_carry"], 1)

    def test_failed_trial_excluded_from_tracking_aggregate(self):
        condition = CommandCondition("T0001", "vx", 0.0, 0.0, 0.0, 1, 0, 0.5)
        row = summarize_trial(
            condition, [sample(0.0)], policy_dt=0.02,
            requested_steps=2, executed_steps=1, termination_reason="grasp_loss",
        )
        aggregates = aggregate_by_mode([row])
        self.assertEqual(aggregates[0]["completion_rate"], 0.0)
        self.assertTrue(math.isnan(aggregates[0]["vx_mae_mean"]))
        self.assertEqual(aggregates[-2]["mode"], "macro_average")
        self.assertEqual(
            aggregates[-1]["mode"], "training_distribution_weighted_secondary"
        )


if __name__ == "__main__":
    unittest.main()
