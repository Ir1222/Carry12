"""Pure-Python metric reduction and CSV output."""

import csv
import math
import os
from statistics import median

from .command_suite import FULL_RANGES, MODES, TRAINING_MODE_WEIGHTS


TRACKING_SCALES = {
    axis: max(abs(low), abs(high))
    for axis, (low, high) in FULL_RANGES.items()
}


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def rms(values):
    values = list(values)
    return math.sqrt(mean(value * value for value in values)) if values else float("nan")


def percentile(values, q):
    values = sorted(values)
    if not values:
        return float("nan")
    index = (len(values) - 1) * float(q) / 100.0
    lo, hi = math.floor(index), math.ceil(index)
    if lo == hi:
        return values[lo]
    weight = index - lo
    return values[lo] * (1.0 - weight) + values[hi] * weight


def wrapped_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def _values(samples, key):
    return [float(sample[key]) for sample in samples]


def _axis_metrics(samples, axis):
    target = _values(samples, f"command_{axis}")
    actual = _values(samples, f"actual_{axis}_training_frame")
    error = _values(samples, f"{axis}_error")
    absolute = [abs(value) for value in error]
    return {
        f"{axis}_target_mean": mean(target),
        f"{axis}_actual_mean": mean(actual),
        f"{axis}_actual_std": rms(value - mean(actual) for value in actual),
        f"{axis}_bias": mean(error),
        f"{axis}_error_mean_signed": mean(error),
        f"{axis}_mae": mean(absolute),
        f"{axis}_rmse": rms(error),
        f"{axis}_p95": percentile(absolute, 95.0),
        f"{axis}_abs_error_max": max(absolute, default=float("nan")),
    }


def _box_axis_metrics(samples, axis):
    error = _values(samples, f"box_{axis}_error")
    return {
        f"box_{axis}_mae": mean(abs(value) for value in error),
        f"box_{axis}_rmse": rms(error),
    }


def summarize_trial(condition, samples, *, policy_dt, requested_steps,
                    executed_steps, termination_reason):
    samples = list(samples)
    completed = len(samples) == requested_steps and termination_reason == "completed"
    row = condition.as_row()
    row.update(
        {
            "trial_completed": int(completed),
            "termination_reason": termination_reason,
            "survival_duration_s": executed_steps * policy_dt,
            "measurement_duration_s": len(samples) * policy_dt,
            "measurement_steps": len(samples),
        }
    )
    for axis in ("vx", "vy", "yaw_rate"):
        row.update(_axis_metrics(samples, axis))

    normalized = _values(samples, "normalized_vector_error")
    xy_speed = _values(samples, "xy_speed")
    yaw_abs = [abs(v) for v in _values(samples, "actual_yaw_rate_training_frame")]
    row.update(
        {
            "normalized_vector_error_mean": mean(normalized),
            "normalized_vector_error_rmse": rms(normalized),
            "normalized_vector_error_p95": percentile(normalized, 95.0),
            "xy_speed_mean": mean(xy_speed),
            "xy_speed_rms": rms(xy_speed),
            "yaw_rate_abs_mean": mean(yaw_abs),
            "yaw_rate_rms": rms(yaw_abs),
            "vx_leakage_rms": rms(
                _values(samples, "actual_vx_training_frame")
            ),
            "vy_leakage_rms": rms(
                _values(samples, "actual_vy_training_frame")
            ),
            "yaw_leakage_rms": rms(
                _values(samples, "actual_yaw_rate_training_frame")
            ),
            "xy_translation_speed_rms": rms(xy_speed),
        }
    )

    if samples:
        start_x = float(samples[0]["base_pos_x"])
        start_y = float(samples[0]["base_pos_y"])
        displacements = [
            math.hypot(float(s["base_pos_x"]) - start_x,
                       float(s["base_pos_y"]) - start_y)
            for s in samples
        ]
        start_yaw = float(samples[0]["base_yaw"])
        yaw_drifts = [
            abs(wrapped_angle(float(s["base_yaw"]) - start_yaw))
            for s in samples
        ]
        final_xy = displacements[-1]
        final_yaw = yaw_drifts[-1]
    else:
        displacements = []
        yaw_drifts = []
        final_xy = final_yaw = float("nan")
    row.update(
        {
            "xy_displacement_drift": rms(displacements),
            "final_xy_displacement": final_xy,
            "yaw_drift_abs": final_yaw,
            "max_yaw_drift": max(yaw_drifts, default=float("nan")),
        }
    )

    for axis in ("vx", "vy", "yaw_rate"):
        row.update(_box_axis_metrics(samples, axis))
    bilateral = _values(samples, "bilateral_contact")
    grasp_loss = _values(samples, "grasp_loss")
    distance = _values(samples, "robot_box_distance")
    tilt = _values(samples, "box_tilt_deg")
    relative_velocity = _values(
        samples, "robot_box_relative_linear_velocity_norm"
    )
    row.update(
        {
            "robot_box_relative_linear_velocity_norm_mean": mean(relative_velocity),
            "robot_box_relative_linear_velocity_norm_p95": percentile(
                relative_velocity, 95.0
            ),
            "bilateral_hand_contact_fraction": mean(bilateral),
            "grasp_loss_fraction": mean(grasp_loss),
            "grasp_loss_occurrence": int(
                any(grasp_loss) or termination_reason == "grasp_loss"
            ),
            "robot_box_distance_mean": mean(distance),
            "robot_box_distance_p95": percentile(distance, 95.0),
            "robot_box_distance_max": max(distance, default=float("nan")),
            "box_tilt_mean_deg": mean(tilt),
            "box_tilt_p95_deg": percentile(tilt, 95.0),
            "box_tilt_max_deg": max(tilt, default=float("nan")),
            "final_confirmed_carry": int(
                completed and bool(samples) and bool(samples[-1]["confirmed_carry"])
            ),
            "action_delta_rms": rms(_values(samples, "action_delta_rms")),
            "action_rate_rms": rms(_values(samples, "action_rate_rms")),
            "torque_rms": rms(_values(samples, "torque_rms")),
            "feet_slip_mean": mean(_values(samples, "feet_slip")),
        }
    )
    return row


AGGREGATE_METRICS = (
    "vx_mae", "vx_rmse", "vy_mae", "vy_rmse",
    "yaw_rate_mae", "yaw_rate_rmse",
    "normalized_vector_error_rmse",
    "box_vx_mae", "box_vy_mae", "box_yaw_rate_mae",
    "bilateral_hand_contact_fraction", "grasp_loss_fraction",
    "robot_box_distance_p95", "box_tilt_p95_deg",
    "survival_duration_s",
)
TRACKING_AGGREGATE_METRICS = {
    "vx_mae", "vx_rmse", "vy_mae", "vy_rmse",
    "yaw_rate_mae", "yaw_rate_rmse",
    "normalized_vector_error_rmse",
    "box_vx_mae", "box_vy_mae", "box_yaw_rate_mae",
}


def _finite(rows, metric):
    return [
        float(row[metric]) for row in rows
        if math.isfinite(float(row[metric]))
    ]


def _mode_row(mode, rows):
    completed = [row for row in rows if int(row["trial_completed"])]
    aggregate = {
        "mode": mode,
        "number_of_trials": len(rows),
        "number_of_completed_tracking_trials": len(completed),
        "completion_rate": mean(int(row["trial_completed"]) for row in rows),
        "final_confirmed_carry_rate": mean(
            int(row["final_confirmed_carry"]) for row in rows
        ),
        "grasp_loss_occurrence_rate": mean(
            int(row["grasp_loss_occurrence"]) for row in rows
        ),
    }
    # Incomplete trials never enter tracking aggregates. Integrity and survival
    # aggregates retain partial failures so that box loss cannot be hidden.
    for metric in AGGREGATE_METRICS:
        source = completed if metric in TRACKING_AGGREGATE_METRICS else rows
        values = _finite(source, metric)
        aggregate[f"{metric}_mean"] = mean(values)
        aggregate[f"{metric}_median"] = median(values) if values else float("nan")
        aggregate[f"{metric}_p95"] = percentile(values, 95.0)
    return aggregate


def aggregate_by_mode(summary_rows):
    mode_rows = []
    for mode in MODES:
        rows = [row for row in summary_rows if row["mode"] == mode]
        if rows:
            mode_rows.append(_mode_row(mode, rows))

    def combine(label, weights):
        result = {
            "mode": label,
            "number_of_trials": sum(row["number_of_trials"] for row in mode_rows),
            "number_of_completed_tracking_trials": sum(
                row["number_of_completed_tracking_trials"] for row in mode_rows
            ),
        }
        total_weight = sum(weights[row["mode"]] for row in mode_rows)
        for key in mode_rows[0]:
            if key in result or key == "mode":
                continue
            values = [
                (row[key], weights[row["mode"]]) for row in mode_rows
                if math.isfinite(float(row[key]))
            ]
            denom = sum(weight for _, weight in values)
            result[key] = (
                sum(value * weight for value, weight in values) / denom
                if denom and total_weight else float("nan")
            )
        return result

    if mode_rows:
        equal = {row["mode"]: 1.0 for row in mode_rows}
        macro = combine("macro_average", equal)
        weighted = combine(
            "training_distribution_weighted_secondary",
            TRAINING_MODE_WEIGHTS,
        )
        mode_rows.extend((macro, weighted))
    return mode_rows


def write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    if fieldnames is None:
        if not rows:
            raise ValueError(f"Cannot infer CSV schema for empty output: {path}")
        fieldnames = tuple(rows[0].keys())
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
