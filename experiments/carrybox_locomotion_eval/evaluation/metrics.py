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


COMMON_INTEGRITY_METRICS = (
    "survival_duration_s",
    "bilateral_hand_contact_fraction",
    "grasp_loss_fraction",
    "robot_box_distance_mean",
    "robot_box_distance_p95",
    "robot_box_distance_max",
    "box_tilt_mean_deg",
    "box_tilt_p95_deg",
    "box_tilt_max_deg",
    "robot_box_relative_linear_velocity_norm_mean",
    "robot_box_relative_linear_velocity_norm_p95",
)

# Keep the mode summary focused on the behavior each command family probes.
# The union is emitted as a stable CSV schema; non-applicable columns are NaN.
MODE_TRACKING_METRICS = {
    "stand": (
        "xy_speed_mean",
        "xy_speed_rms",
        "yaw_rate_abs_mean",
        "yaw_rate_rms",
        "xy_displacement_drift",
        "final_xy_displacement",
        "yaw_drift_abs",
        "max_yaw_drift",
    ),
    "vx": (
        "vx_mae",
        "vx_rmse",
        "vx_bias",
        "vx_p95",
        "vy_leakage_rms",
        "yaw_leakage_rms",
        "box_vx_mae",
        "box_vx_rmse",
    ),
    "vy": (
        "vy_mae",
        "vy_rmse",
        "vy_bias",
        "vy_p95",
        "vx_leakage_rms",
        "yaw_leakage_rms",
        "box_vy_mae",
        "box_vy_rmse",
    ),
    "yaw": (
        "yaw_rate_mae",
        "yaw_rate_rmse",
        "yaw_rate_bias",
        "yaw_rate_p95",
        "vx_leakage_rms",
        "vy_leakage_rms",
        "xy_translation_speed_rms",
        "box_yaw_rate_mae",
        "box_yaw_rate_rmse",
    ),
    "mixed": (
        "vx_mae",
        "vx_rmse",
        "vy_mae",
        "vy_rmse",
        "yaw_rate_mae",
        "yaw_rate_rmse",
        "normalized_vector_error_mean",
        "normalized_vector_error_rmse",
        "normalized_vector_error_p95",
        "box_vx_mae",
        "box_vy_mae",
        "box_yaw_rate_mae",
    ),
}
ALL_MODE_TRACKING_METRICS = tuple(
    dict.fromkeys(
        metric
        for mode in MODES
        for metric in MODE_TRACKING_METRICS[mode]
    )
)


def _finite(rows, metric):
    return [
        float(row[metric]) for row in rows
        if math.isfinite(float(row[metric]))
    ]


def _add_distribution(row, prefix, values):
    values = list(values)
    row[f"{prefix}_mean"] = mean(values)
    row[f"{prefix}_median"] = median(values) if values else float("nan")
    row[f"{prefix}_p95"] = percentile(values, 95.0)


def _mode_row(mode, rows):
    completed = [row for row in rows if int(row["trial_completed"])]
    observed = [row for row in rows if int(row["measurement_steps"]) > 0]
    aggregate = {
        "mode": mode,
        "number_of_trials": len(rows),
        "number_of_completed_tracking_trials": len(completed),
        "number_of_trials_with_measure_samples": len(observed),
        "completion_rate": mean(int(row["trial_completed"]) for row in rows),
        "final_confirmed_carry_rate": mean(
            int(row["final_confirmed_carry"]) for row in rows
        ),
        "grasp_loss_occurrence_rate": mean(
            int(row["grasp_loss_occurrence"]) for row in rows
        ),
    }

    # Integrity/survival always include failed and incomplete trials. A failure
    # before MEASURE naturally contributes NaN only to unavailable sample-based
    # fields; it still contributes to completion and failure rates.
    for metric in COMMON_INTEGRITY_METRICS:
        _add_distribution(aggregate, metric, _finite(rows, metric))

    # Emit a stable superset schema and make applicability explicit with NaN.
    for metric in ALL_MODE_TRACKING_METRICS:
        _add_distribution(aggregate, f"completed_only_{metric}", ())
        _add_distribution(aggregate, f"all_observed_{metric}", ())

    for metric in MODE_TRACKING_METRICS[mode]:
        _add_distribution(
            aggregate,
            f"completed_only_{metric}",
            _finite(completed, metric),
        )
        _add_distribution(
            aggregate,
            f"all_observed_{metric}",
            _finite(observed, metric),
        )
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
            "number_of_trials_with_measure_samples": sum(
                row["number_of_trials_with_measure_samples"] for row in mode_rows
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
