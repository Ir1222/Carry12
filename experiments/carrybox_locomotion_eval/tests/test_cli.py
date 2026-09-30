import pytest

from experiments.carrybox_locomotion_eval.evaluator import parse_evaluator_args


def test_old_single_context_command_remains_default():
    args = parse_evaluator_args([
        "--resume_path", "checkpoint.pt", "--suite", "train_distribution",
        "--seed", "1", "--carry_motion_id", "0", "--carry_phase", "0.5",
        "--warmup", "0.20", "--duration", "3.0", "--save_csv", "--headless"])
    assert args.protocol == "constant"
    assert args.seeds == [1]
    assert args.carry_motion_ids == [0]
    assert args.carry_phases == [.5]
    assert args.remaining_isaac_args == ["--resume_path", "checkpoint.pt", "--headless"]


def test_full_grid_cli_and_invalid_combinations():
    args = parse_evaluator_args([
        "--protocol", "both", "--seeds", "1", "2", "3",
        "--carry_motion_ids", "0", "1", "2",
        "--carry_phases", ".25", ".5", ".75"])
    assert len(args.seeds) * len(args.carry_motion_ids) * len(args.carry_phases) == 27
    for invalid in (("--seed", "1", "--seeds", "2"),
                    ("--protocol", "step", "--mode", "stand"),
                    ("--output_dir", "x"),
                    ("--stable_hold", "2.1", "--settle_timeout", "2")):
        with pytest.raises(SystemExit):
            parse_evaluator_args(list(invalid))
