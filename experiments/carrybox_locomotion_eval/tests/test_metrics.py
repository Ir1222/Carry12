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

    def test_completed_only_and_all_observed_expose_survivor_bias(self):
        completed = summarize_trial(
            CommandCondition("T0001", "vx", 0.5, 0.0, 0.0, 1, 0, 0.5),
            [sample(0.6), sample(0.4)],
            policy_dt=0.02,
            requested_steps=2,
            executed_steps=2,
            termination_reason="completed",
        )
        failed_sample = sample(0.8)
        failed_sample["bilateral_contact"] = 0
        failed_sample["grasp_loss"] = 1
        failed = summarize_trial(
            CommandCondition("T0002", "vx", 0.5, 0.0, 0.0, 1, 0, 0.5),
            [failed_sample],
            policy_dt=0.02,
            requested_steps=2,
            executed_steps=1,
            termination_reason="grasp_loss",
        )
        aggregates = aggregate_by_mode([completed, failed])
        vx = aggregates[0]
        self.assertEqual(vx["completion_rate"], 0.5)
        self.assertAlmostEqual(vx["completed_only_vx_rmse_mean"], 0.1)
        self.assertAlmostEqual(vx["all_observed_vx_rmse_mean"], 0.2)
        self.assertNotEqual(
            vx["completed_only_vx_rmse_mean"],
            vx["all_observed_vx_rmse_mean"],
        )
        self.assertEqual(vx["grasp_loss_occurrence_rate"], 0.5)
        self.assertEqual(vx["bilateral_hand_contact_fraction_mean"], 0.5)
        self.assertEqual(vx["grasp_loss_fraction_mean"], 0.5)
        self.assertEqual(vx["number_of_trials_with_measure_samples"], 2)
        self.assertEqual(aggregates[-2]["mode"], "macro_average")
        self.assertEqual(
            aggregates[-1]["mode"], "training_distribution_weighted_secondary"
        )

    def test_mode_specific_tracking_schema(self):
        rows = []
        for index, mode in enumerate(("stand", "vx", "vy", "yaw", "mixed"), 1):
            rows.append(
                summarize_trial(
                    CommandCondition(
                        f"T{index:04d}", mode, 0.5, 0.0, 0.0, 1, 0, 0.5
                    ),
                    [sample(0.6), sample(0.4)],
                    policy_dt=0.02,
                    requested_steps=2,
                    executed_steps=2,
                    termination_reason="completed",
                )
            )
        by_mode = {row["mode"]: row for row in aggregate_by_mode(rows)}
        expected = {
            "stand": (
                "xy_speed_mean", "xy_speed_rms", "yaw_rate_abs_mean",
                "yaw_rate_rms", "xy_displacement_drift",
                "final_xy_displacement", "yaw_drift_abs", "max_yaw_drift",
            ),
            "vx": (
                "vx_mae", "vx_rmse", "vx_bias", "vx_p95",
                "vy_leakage_rms", "yaw_leakage_rms", "box_vx_mae",
                "box_vx_rmse",
            ),
            "vy": (
                "vy_mae", "vy_rmse", "vy_bias", "vy_p95",
                "vx_leakage_rms", "yaw_leakage_rms", "box_vy_mae",
                "box_vy_rmse",
            ),
            "yaw": (
                "yaw_rate_mae", "yaw_rate_rmse", "yaw_rate_bias",
                "yaw_rate_p95", "vx_leakage_rms", "vy_leakage_rms",
                "xy_translation_speed_rms", "box_yaw_rate_mae",
                "box_yaw_rate_rmse",
            ),
            "mixed": (
                "vx_mae", "vx_rmse", "vy_mae", "vy_rmse",
                "yaw_rate_mae", "yaw_rate_rmse",
                "normalized_vector_error_mean",
                "normalized_vector_error_rmse",
                "normalized_vector_error_p95", "box_vx_mae", "box_vy_mae",
                "box_yaw_rate_mae",
            ),
        }
        for mode, metrics in expected.items():
            for metric in metrics:
                self.assertTrue(math.isfinite(
                    by_mode[mode][f"completed_only_{metric}_mean"]
                ))
                self.assertTrue(math.isfinite(
                    by_mode[mode][f"all_observed_{metric}_mean"]
                ))

        for mode in expected:
            for metric in (
                "survival_duration_s", "bilateral_hand_contact_fraction",
                "grasp_loss_fraction", "robot_box_distance_p95",
                "box_tilt_p95_deg",
                "robot_box_relative_linear_velocity_norm_mean",
                "robot_box_relative_linear_velocity_norm_p95",
            ):
                self.assertTrue(math.isfinite(by_mode[mode][f"{metric}_mean"]))
        self.assertTrue(math.isnan(
            by_mode["stand"]["all_observed_vx_mae_mean"]
        ))

    def test_pre_measure_failure_does_not_fabricate_tracking(self):
        failed = summarize_trial(
            CommandCondition("T0001", "vx", 0.5, 0.0, 0.0, 1, 0, 0.5),
            [],
            policy_dt=0.02,
            requested_steps=2,
            executed_steps=1,
            termination_reason="grasp_loss",
        )
        vx = aggregate_by_mode([failed])[0]
        self.assertEqual(vx["completion_rate"], 0.0)
        self.assertEqual(vx["number_of_trials_with_measure_samples"], 0)
        self.assertTrue(math.isnan(vx["completed_only_vx_rmse_mean"]))
        self.assertTrue(math.isnan(vx["all_observed_vx_rmse_mean"]))
        self.assertEqual(vx["grasp_loss_occurrence_rate"], 1.0)
        self.assertAlmostEqual(vx["survival_duration_s_mean"], 0.02)


if __name__ == "__main__":
    unittest.main()
