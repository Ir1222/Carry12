import csv
import math
import sys
import tempfile
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))

from evaluation.nforce_velocity import (  # noqa: E402
    NForceVelocityCsvLogger,
    TRACE_FIELDS,
    sample_velocity_tracking,
    summarize_velocity_tracking,
)


class _TensorStub:
    def __init__(self, value):
        self.value = value

    def __getitem__(self, index):
        value = self.value
        indices = index if isinstance(index, tuple) else (index,)
        for item in indices:
            if isinstance(value, (list, tuple)):
                value = value[item]
            elif item != 0:
                raise IndexError(item)
        return _TensorStub(value)

    def reshape(self, *_shape):
        return self

    def item(self):
        return self.value


class _FakeEnv:
    commands = _TensorStub([[0.6, 0.2, 0.1, 0.0]])
    carry_policy_commands = _TensorStub([[0.6, 0.0, 0.15, 0.0]])
    base_lin_vel = _TensorStub([[0.5, -0.1, 0.0]])
    base_ang_vel = _TensorStub([[0.0, 0.0, 0.05]])
    confirmed_carry_buf = _TensorStub([True])
    yaw = _TensorStub([[0.3]])
    carry_heading_ref = _TensorStub([0.4])
    carry_heading_error = _TensorStub([0.1])


def _summary(samples):
    return summarize_velocity_tracking(
        trial_id="T0001",
        checkpoint="model.pt",
        seed=1,
        raw_command=(0.6, 0.2, 0.1),
        samples=samples,
        policy_dt=0.02,
        steady_carry_start_time_s=3.0,
        final_confirmed_carry=True,
        carry_achieved=True,
        humanoid_failure=False,
        humanoid_failure_reason="",
        box_failure=False,
        box_failure_reason="",
        timeout=False,
        termination_reason="steady_carry_complete",
        force_scheduled=0,
        final_base_yaw_rad=0.3,
    )


def test_trace_errors_use_policy_command_and_body_velocity():
    row = sample_velocity_tracking(_FakeEnv(), time_s=3.0, policy_step=150)
    assert tuple(row) == TRACE_FIELDS
    assert math.isclose(row["vx_error_signed"], 0.5 - 0.6, abs_tol=1.0e-6)
    assert math.isclose(row["vy_error_signed"], -0.1 - 0.0, abs_tol=1.0e-6)
    assert math.isclose(
        row["yaw_rate_error_signed"], 0.05 - 0.15, abs_tol=1.0e-6
    )
    assert math.isclose(row["vx_error_abs"], abs(row["vx_error_signed"]))
    assert math.isclose(
        row["lin_vel_error_norm"],
        math.hypot(row["vx_error_signed"], row["vy_error_signed"]),
    )
    assert math.isclose(row["raw_command_vy"], 0.2)
    assert math.isclose(row["policy_command_vy"], 0.0)


def test_summary_recomputes_trace_statistics():
    first = sample_velocity_tracking(_FakeEnv(), time_s=3.0, policy_step=150)
    second = dict(first)
    second["actual_vx_body"] = 0.7
    second["vx_error_signed"] = 0.1
    second["vx_error_abs"] = 0.1
    second["lin_vel_error_norm"] = math.hypot(0.1, second["vy_error_signed"])
    summary = _summary([first, second])
    expected_vx_errors = [first["vx_error_signed"], second["vx_error_signed"]]
    assert math.isclose(
        summary["actual_vx_mean"],
        (first["actual_vx_body"] + second["actual_vx_body"]) / 2.0,
    )
    assert math.isclose(
        summary["vx_mae"], sum(abs(value) for value in expected_vx_errors) / 2.0
    )
    assert math.isclose(
        summary["vx_rmse"],
        math.sqrt(sum(value * value for value in expected_vx_errors) / 2.0),
    )
    assert summary["vx_abs_error_max"] == max(
        abs(value) for value in expected_vx_errors
    )


def test_empty_trace_uses_nan_metrics_and_keeps_csv_schema():
    summary = _summary([])
    assert summary["steady_carry_achieved"] == 0
    assert summary["steady_carry_steps"] == 0
    assert math.isnan(summary["vx_mae"])
    assert math.isnan(summary["lin_vel_error_norm_mean"])

    with tempfile.TemporaryDirectory(dir=EXPERIMENT_DIR) as temp_dir:
        logger = NForceVelocityCsvLogger(temp_dir)
        trace_path = logger.write_trace("T0001", [])
        logger.append_summary(summary)
        with open(trace_path, newline="") as file:
            assert tuple(next(csv.reader(file))) == TRACE_FIELDS


if __name__ == "__main__":
    tests = (
        test_trace_errors_use_policy_command_and_body_velocity,
        test_summary_recomputes_trace_statistics,
        test_empty_trace_uses_nan_metrics_and_keeps_csv_schema,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS {len(tests)} no-force velocity tests")
