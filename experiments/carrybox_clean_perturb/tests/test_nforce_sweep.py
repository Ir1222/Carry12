from types import SimpleNamespace
from unittest import TestCase

import torch

from nforce_test_support import RolloutEnv
from evaluation.nforce_cli import (
    DEFAULT_VX_RANGE,
    DEFAULT_YAW_RANGE,
    generate_command_grid,
    parse_nforce_args,
)
from evaluation.nforce_trial import run_trials


def test_default_command_grid_has_35_unique_inclusive_conditions():
    commands = generate_command_grid()
    assert len(commands) == len(set(commands)) == 35
    assert commands[0] == (0.0, 0.0, -0.4)
    assert commands[4] == (0.0, 0.0, 0.4)
    assert commands[5] == (0.2, 0.0, -0.4)
    assert commands[-1] == (1.2, 0.0, 0.4)
    assert DEFAULT_VX_RANGE == (0.0, 1.2, 0.2)
    assert DEFAULT_YAW_RANGE == (-0.4, 0.4, 0.2)


def test_custom_command_grid_includes_both_endpoints():
    commands = generate_command_grid((0.2, 0.6, 0.2), (-0.2, 0.2, 0.2))
    assert commands == (
        (0.2, 0.0, -0.2),
        (0.2, 0.0, 0.0),
        (0.2, 0.0, 0.2),
        (0.4, 0.0, -0.2),
        (0.4, 0.0, 0.0),
        (0.4, 0.0, 0.2),
        (0.6, 0.0, -0.2),
        (0.6, 0.0, 0.0),
        (0.6, 0.0, 0.2),
    )


def test_sweep_cli_builds_default_grid_and_preserves_remaining_args():
    args, remaining = parse_nforce_args(
        ["--command_sweep", "--seed", "1", "--headless"]
    )
    assert len(args.commands) == 35
    assert remaining == ["--seed", "1", "--headless"]


def test_single_command_cli_remains_compatible():
    args, remaining = parse_nforce_args(["--command", "0.8,0.0,0.2"])
    assert not args.command_sweep
    assert args.commands == ((0.8, 0.0, 0.2),)
    assert remaining == []


def test_sweep_cli_rejects_invalid_ranges_and_conflicts():
    invalid_argv = (
        ["--command_sweep", "--vx_range", "0,1,0"],
        ["--command_sweep", "--vx_range", "1,0,0.2"],
        ["--command_sweep", "--vx_range", "0,1,0.3"],
        ["--command_sweep", "--vx_range", "0,nan,0.2"],
        ["--command_sweep", "--vx_range", "0,1"],
        ["--vx_range", "0,1.2,0.2"],
        ["--command_sweep", "--command", "0.4,0,0"],
    )
    for argv in invalid_argv:
        with TestCase().assertRaises(SystemExit):
            parse_nforce_args(argv)


class _RecordingLogger:
    def __init__(self):
        self.traces = []
        self.summaries = []

    def write_trace(self, trial_id, rows):
        self.traces.append((trial_id, list(rows)))

    def append_summary(self, row):
        self.summaries.append(dict(row))


def test_run_trials_updates_command_seed_history_and_trial_ids():
    env = RolloutEnv()
    reset_commands = []
    original_reset_state = env.reset_evaluation_trial_state

    def record_reset(clear_actor_history=True):
        assert clear_actor_history
        reset_commands.append(tuple(env.cfg.nominal_clean.command))
        original_reset_state(clear_actor_history=clear_actor_history)

    env.reset_evaluation_trial_state = record_reset
    seeds = []
    logger = _RecordingLogger()
    commands = ((0.0, 0.0, -0.4), (1.2, 0.0, 0.4))
    summaries = run_trials(
        env,
        lambda obs: torch.zeros(1, 29),
        "model.pt",
        1,
        commands,
        SimpleNamespace(steady_carry_warmup=0.0, steady_duration=0.02),
        seed_fn=seeds.append,
        logger=logger,
    )

    assert reset_commands == list(commands)
    assert seeds == [1, 1]
    assert [row["trial_id"] for row in summaries] == ["T0001", "T0002"]
    assert [row["raw_command_vx"] for row in summaries] == [0.0, 1.2]
    assert [row["raw_command_yaw_rate"] for row in summaries] == [-0.4, 0.4]
    assert [trial_id for trial_id, _ in logger.traces] == ["T0001", "T0002"]
    assert [row["trial_id"] for row in logger.summaries] == ["T0001", "T0002"]
