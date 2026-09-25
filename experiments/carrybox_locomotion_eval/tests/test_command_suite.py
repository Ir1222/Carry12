from collections import Counter

from experiments.carrybox_locomotion_eval.evaluation.command_suite import (
    FULL_RANGES,
    MIXED_RANGES,
    MODES,
    TRAINING_MODE_WEIGHTS,
    VX_VALUES,
    VY_VALUES,
    YAW_VALUES,
    build_command_suite,
)


def test_default_suite_counts_ranges_and_metadata():
    suite = build_command_suite(
        seed=1, carry_motion_id=0, carry_phase=0.5, modes=MODES
    )
    assert len(suite) == 52
    assert Counter(item.mode for item in suite) == {
        "stand": 1,
        "vx": 7,
        "vy": 6,
        "yaw": 6,
        "mixed": 32,
    }
    assert [item.trial_id for item in suite] == [
        f"T{index:04d}" for index in range(1, 53)
    ]
    assert all(item.seed == 1 for item in suite)
    assert all(item.carry_motion_id == 0 for item in suite)
    assert all(item.carry_phase == 0.5 for item in suite)
    assert suite[0].case_type == "stand"
    assert all(item.case_type == "axis" for item in suite if item.mode in ("vx", "vy", "yaw"))

    mixed = [item for item in suite if item.mode == "mixed"]
    for item in mixed:
        assert MIXED_RANGES["vx"][0] <= item.vx <= MIXED_RANGES["vx"][1]
        assert MIXED_RANGES["vy"][0] <= item.vy <= MIXED_RANGES["vy"][1]
        assert (
            MIXED_RANGES["yaw_rate"][0]
            <= item.yaw_rate
            <= MIXED_RANGES["yaw_rate"][1]
        )
    assert len({(item.vx, item.vy, item.yaw_rate) for item in mixed[:8]}) == 8
    assert all(item.case_type == "corner" for item in mixed[:8])
    assert all(item.case_type == "interior" for item in mixed[8:])
    assert VX_VALUES == (-0.60, -0.35, -0.15, 0.15, 0.40, 0.80, 1.20)
    assert VY_VALUES == (-0.50, -0.30, -0.15, 0.15, 0.30, 0.50)
    assert YAW_VALUES == (-0.70, -0.45, -0.20, 0.20, 0.45, 0.70)
    assert all(abs(value) >= 0.1 for value in VX_VALUES + VY_VALUES + YAW_VALUES)
    assert FULL_RANGES == MIXED_RANGES == {
        "vx": (-0.60, 1.20),
        "vy": (-0.50, 0.50),
        "yaw_rate": (-0.70, 0.70),
    }
    assert TRAINING_MODE_WEIGHTS == {
        "stand": 0.10, "vx": 0.10, "vy": 0.10, "yaw": 0.10, "mixed": 0.60,
    }


def test_mode_filter_restarts_trial_ids():
    suite = build_command_suite(
        seed=7, carry_motion_id=2, carry_phase=0.25, modes=("vy",)
    )
    assert len(suite) == 6
    assert suite[0].trial_id == "T0001"
    assert all(item.mode == "vy" for item in suite)
