"""Command-line helpers for the no-force velocity evaluator."""

import argparse
import math

from configs.evaluation_config import FIXED_COMMAND


DEFAULT_VX_RANGE = (0.0, 1.2, 0.2)
DEFAULT_YAW_RANGE = (-0.4, 0.4, 0.2)


def _parse_command(text):
    parts = [part.strip() for part in str(text).split(",")]
    if len(parts) != 3 or any(not part for part in parts):
        raise argparse.ArgumentTypeError(
            "--command must contain exactly three values: VX,VY,YAW_RATE"
        )
    try:
        command = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--command must contain exactly three floating-point values"
        ) from exc
    if not all(math.isfinite(value) for value in command):
        raise argparse.ArgumentTypeError("--command values must be finite")
    return command


def _parse_range(text):
    parts = [part.strip() for part in str(text).split(",")]
    if len(parts) != 3 or any(not part for part in parts):
        raise argparse.ArgumentTypeError(
            "range must contain exactly three values: MIN,MAX,STEP"
        )
    try:
        start, stop, step = (float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "range must contain exactly three floating-point values"
        ) from exc
    if not all(math.isfinite(value) for value in (start, stop, step)):
        raise argparse.ArgumentTypeError("range values must be finite")
    if step <= 0.0:
        raise argparse.ArgumentTypeError("range STEP must be positive")
    if start > stop:
        raise argparse.ArgumentTypeError("range MIN must not exceed MAX")
    intervals = (stop - start) / step
    if not math.isclose(
        intervals, round(intervals), rel_tol=1.0e-9, abs_tol=1.0e-9
    ):
        raise argparse.ArgumentTypeError(
            "range STEP must divide MAX-MIN so both endpoints are included"
        )
    return start, stop, step


def _inclusive_values(range_spec):
    start, stop, step = range_spec
    interval_count = int(round((stop - start) / step))
    values = [start + index * step for index in range(interval_count + 1)]
    values[-1] = stop
    return tuple(0.0 if abs(value) < 1.0e-12 else value for value in values)


def generate_command_grid(vx_range=DEFAULT_VX_RANGE, yaw_range=DEFAULT_YAW_RANGE):
    """Return a deterministic vx-outer/yaw-inner Cartesian command grid."""
    vx_values = _inclusive_values(vx_range)
    yaw_values = _inclusive_values(yaw_range)
    return tuple((vx, 0.0, yaw_rate) for vx in vx_values for yaw_rate in yaw_values)


def parse_nforce_args(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    command_group = parser.add_mutually_exclusive_group()
    command_group.add_argument(
        "--command",
        type=_parse_command,
        default=None,
        metavar="VX,VY,YAW_RATE",
    )
    command_group.add_argument(
        "--command_sweep",
        action="store_true",
        default=False,
        help="Evaluate a Cartesian grid of static vx and yaw-rate commands.",
    )
    parser.add_argument(
        "--vx_range",
        type=_parse_range,
        default=None,
        metavar="MIN,MAX,STEP",
    )
    parser.add_argument(
        "--yaw_range",
        type=_parse_range,
        default=None,
        metavar="MIN,MAX,STEP",
    )
    parser.add_argument(
        "--steady_carry_warmup",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--steady_duration",
        type=float,
        default=5.0,
    )
    parser.add_argument("--save_csv", action="store_true", default=False)
    parser.add_argument("--output_dir", type=str, default=None)
    eval_args, remaining = parser.parse_known_args(argv)

    if not math.isfinite(eval_args.steady_carry_warmup):
        parser.error("--steady_carry_warmup must be finite")
    if eval_args.steady_carry_warmup < 0.0:
        parser.error("--steady_carry_warmup must be non-negative")
    if not math.isfinite(eval_args.steady_duration):
        parser.error("--steady_duration must be finite")
    if eval_args.steady_duration <= 0.0:
        parser.error("--steady_duration must be positive")
    if not eval_args.command_sweep and (
        eval_args.vx_range is not None or eval_args.yaw_range is not None
    ):
        parser.error("--vx_range and --yaw_range require --command_sweep")

    if eval_args.command_sweep:
        eval_args.vx_range = eval_args.vx_range or DEFAULT_VX_RANGE
        eval_args.yaw_range = eval_args.yaw_range or DEFAULT_YAW_RANGE
        eval_args.commands = generate_command_grid(
            eval_args.vx_range, eval_args.yaw_range
        )
    else:
        eval_args.command = (
            FIXED_COMMAND if eval_args.command is None else eval_args.command
        )
        eval_args.commands = (eval_args.command,)
    return eval_args, remaining
