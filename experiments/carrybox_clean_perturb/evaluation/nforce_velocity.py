"""Policy-step velocity metrics for the no-force CarryBox evaluator."""

import csv
import math
import os


TRACE_FIELDS = (
    "time_s",
    "policy_step",
    "phase",
    "raw_command_vx",
    "raw_command_vy",
    "raw_command_yaw_rate",
    "policy_command_vx",
    "policy_command_vy",
    "policy_command_yaw_rate",
    "actual_vx_body",
    "actual_vy_body",
    "actual_yaw_rate_body",
    "vx_error_signed",
    "vy_error_signed",
    "yaw_rate_error_signed",
    "vx_error_abs",
    "vy_error_abs",
    "yaw_rate_error_abs",
    "lin_vel_error_norm",
    "confirmed_carry",
    "base_yaw_rad",
    "carry_heading_ref_rad",
    "carry_heading_error_rad",
)


SUMMARY_FIELDS = (
    "trial_id",
    "checkpoint",
    "seed",
    "steady_carry_achieved",
    "steady_carry_start_time_s",
    "steady_carry_duration_s",
    "steady_carry_steps",
    "raw_command_vx",
    "raw_command_vy",
    "raw_command_yaw_rate",
    "policy_vx_mean",
    "policy_vx_std",
    "actual_vx_mean",
    "actual_vx_std",
    "vx_error_mean_signed",
    "vx_mae",
    "vx_rmse",
    "vx_abs_error_p95",
    "vx_abs_error_max",
    "policy_vy_mean",
    "policy_vy_std",
    "actual_vy_mean",
    "actual_vy_std",
    "vy_error_mean_signed",
    "vy_mae",
    "vy_rmse",
    "vy_abs_error_p95",
    "vy_abs_error_max",
    "policy_yaw_rate_mean",
    "policy_yaw_rate_std",
    "actual_yaw_rate_mean",
    "actual_yaw_rate_std",
    "yaw_rate_error_mean_signed",
    "yaw_rate_mae",
    "yaw_rate_rmse",
    "yaw_rate_abs_error_p95",
    "yaw_rate_abs_error_max",
    "lin_vel_error_norm_mean",
    "lin_vel_error_norm_rmse",
    "lin_vel_error_norm_p95",
    "lin_vel_error_norm_max",
    "confirmed_carry_fraction_steady",
    "final_confirmed_carry",
    "carry_achieved",
    "humanoid_failure",
    "humanoid_failure_reason",
    "box_failure",
    "box_failure_reason",
    "timeout",
    "termination_reason",
    "force_scheduled",
    "final_base_yaw_rad",
    "max_abs_carry_heading_error_rad",
)


def _scalar(tensor, env_id=0):
    return float(tensor[env_id].reshape(-1)[0].item())


def sample_velocity_tracking(env, time_s, policy_step, env_id=0):
    """Sample one post-step STEADY_CARRY state in the policy/body frames."""
    raw_vx = float(env.commands[env_id, 0].item())
    raw_vy = float(env.commands[env_id, 1].item())
    raw_yaw_rate = float(env.commands[env_id, 2].item())
    policy_vx = float(env.carry_policy_commands[env_id, 0].item())
    policy_vy = float(env.carry_policy_commands[env_id, 1].item())
    policy_yaw_rate = float(env.carry_policy_commands[env_id, 2].item())
    actual_vx = float(env.base_lin_vel[env_id, 0].item())
    actual_vy = float(env.base_lin_vel[env_id, 1].item())
    actual_yaw_rate = float(env.base_ang_vel[env_id, 2].item())

    vx_error = actual_vx - policy_vx
    vy_error = actual_vy - policy_vy
    yaw_rate_error = actual_yaw_rate - policy_yaw_rate

    return {
        "time_s": float(time_s),
        "policy_step": int(policy_step),
        "phase": "STEADY_CARRY",
        "raw_command_vx": raw_vx,
        "raw_command_vy": raw_vy,
        "raw_command_yaw_rate": raw_yaw_rate,
        "policy_command_vx": policy_vx,
        "policy_command_vy": policy_vy,
        "policy_command_yaw_rate": policy_yaw_rate,
        "actual_vx_body": actual_vx,
        "actual_vy_body": actual_vy,
        "actual_yaw_rate_body": actual_yaw_rate,
        "vx_error_signed": vx_error,
        "vy_error_signed": vy_error,
        "yaw_rate_error_signed": yaw_rate_error,
        "vx_error_abs": abs(vx_error),
        "vy_error_abs": abs(vy_error),
        "yaw_rate_error_abs": abs(yaw_rate_error),
        "lin_vel_error_norm": math.hypot(vx_error, vy_error),
        "confirmed_carry": int(bool(env.confirmed_carry_buf[env_id].item())),
        "base_yaw_rad": _scalar(env.yaw, env_id),
        "carry_heading_ref_rad": _scalar(env.carry_heading_ref, env_id),
        "carry_heading_error_rad": _scalar(env.carry_heading_error, env_id),
    }


def _values(rows, key):
    return [float(row[key]) for row in rows]


def _mean(values):
    return sum(values) / len(values) if values else float("nan")


def _std(values):
    if not values:
        return float("nan")
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _rmse(values):
    return math.sqrt(_mean([value * value for value in values])) if values else float("nan")


def _percentile(values, percentile):
    """Return a linearly interpolated percentile matching NumPy's default rule."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = (len(ordered) - 1) * float(percentile) / 100.0
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _axis_metrics(rows, policy_key, actual_key, error_key, prefix):
    policy = _values(rows, policy_key)
    actual = _values(rows, actual_key)
    error = _values(rows, error_key)
    absolute_error = [abs(value) for value in error]
    return {
        f"policy_{prefix}_mean": _mean(policy),
        f"policy_{prefix}_std": _std(policy),
        f"actual_{prefix}_mean": _mean(actual),
        f"actual_{prefix}_std": _std(actual),
        f"{prefix}_error_mean_signed": _mean(error),
        f"{prefix}_mae": _mean(absolute_error),
        f"{prefix}_rmse": _rmse(error),
        f"{prefix}_abs_error_p95": _percentile(absolute_error, 95.0),
        f"{prefix}_abs_error_max": max(absolute_error, default=float("nan")),
    }


def summarize_velocity_tracking(
    *,
    trial_id,
    checkpoint,
    seed,
    raw_command,
    samples,
    policy_dt,
    steady_carry_start_time_s,
    final_confirmed_carry,
    carry_achieved,
    humanoid_failure,
    humanoid_failure_reason,
    box_failure,
    box_failure_reason,
    timeout,
    termination_reason,
    force_scheduled,
    final_base_yaw_rad,
):
    """Build one fixed-schema summary row; empty-sample metrics remain NaN."""
    samples = list(samples)
    linear_error = _values(samples, "lin_vel_error_norm")
    confirmed = _values(samples, "confirmed_carry")
    heading_error = [
        abs(value) for value in _values(samples, "carry_heading_error_rad")
    ]

    row = {
        "trial_id": str(trial_id),
        "checkpoint": str(checkpoint),
        "seed": int(seed),
        "steady_carry_achieved": int(bool(samples)),
        "steady_carry_start_time_s": (
            float(steady_carry_start_time_s) if samples else float("nan")
        ),
        "steady_carry_duration_s": len(samples) * float(policy_dt),
        "steady_carry_steps": len(samples),
        "raw_command_vx": float(raw_command[0]),
        "raw_command_vy": float(raw_command[1]),
        "raw_command_yaw_rate": float(raw_command[2]),
    }
    row.update(
        _axis_metrics(
            samples,
            "policy_command_vx",
            "actual_vx_body",
            "vx_error_signed",
            "vx",
        )
    )
    row.update(
        _axis_metrics(
            samples,
            "policy_command_vy",
            "actual_vy_body",
            "vy_error_signed",
            "vy",
        )
    )
    row.update(
        _axis_metrics(
            samples,
            "policy_command_yaw_rate",
            "actual_yaw_rate_body",
            "yaw_rate_error_signed",
            "yaw_rate",
        )
    )
    row.update(
        {
            "lin_vel_error_norm_mean": _mean(linear_error),
            "lin_vel_error_norm_rmse": _rmse(linear_error),
            "lin_vel_error_norm_p95": _percentile(linear_error, 95.0),
            "lin_vel_error_norm_max": max(linear_error, default=float("nan")),
            "confirmed_carry_fraction_steady": _mean(confirmed),
            "final_confirmed_carry": int(bool(final_confirmed_carry)),
            "carry_achieved": int(bool(carry_achieved)),
            "humanoid_failure": int(bool(humanoid_failure)),
            "humanoid_failure_reason": str(humanoid_failure_reason),
            "box_failure": int(bool(box_failure)),
            "box_failure_reason": str(box_failure_reason),
            "timeout": int(bool(timeout)),
            "termination_reason": str(termination_reason),
            "force_scheduled": int(force_scheduled),
            "final_base_yaw_rad": float(final_base_yaw_rad),
            "max_abs_carry_heading_error_rad": max(
                heading_error, default=float("nan")
            ),
        }
    )
    if tuple(row) != SUMMARY_FIELDS:
        raise AssertionError(
            f"No-force summary schema mismatch: got={tuple(row)}, expected={SUMMARY_FIELDS}"
        )
    return row


class NForceVelocityCsvLogger:
    """CSV writer isolated from the external-force evaluator's schema."""

    def __init__(self, output_dir):
        self.output_dir = os.path.abspath(output_dir)
        self.trace_dir = os.path.join(self.output_dir, "traces")
        self.summary_path = os.path.join(self.output_dir, "summary.csv")
        os.makedirs(self.trace_dir, exist_ok=True)

    def write_trace(self, trial_id, rows):
        path = os.path.join(self.trace_dir, f"{trial_id}.csv")
        with open(path, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=TRACE_FIELDS)
            writer.writeheader()
            for row in rows:
                if tuple(row) != TRACE_FIELDS:
                    raise ValueError(
                        f"No-force trace schema mismatch: got={tuple(row)}, "
                        f"expected={TRACE_FIELDS}"
                    )
                writer.writerow(row)
        return path

    def append_summary(self, row):
        if tuple(row) != SUMMARY_FIELDS:
            raise ValueError(
                f"No-force summary schema mismatch: got={tuple(row)}, "
                f"expected={SUMMARY_FIELDS}"
            )
        write_header = not os.path.exists(self.summary_path) or os.path.getsize(
            self.summary_path
        ) == 0
        if not write_header:
            with open(self.summary_path, newline="") as file:
                existing_fields = tuple(next(csv.reader(file), ()))
            if existing_fields != SUMMARY_FIELDS:
                raise ValueError(
                    "Existing no-force summary.csv schema does not match this evaluator. "
                    "Use a new --output_dir."
                )
        with open(self.summary_path, "a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
