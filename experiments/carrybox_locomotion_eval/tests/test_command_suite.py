from collections import Counter

from experiments.carrybox_locomotion_eval.evaluation.command_suite import (
    MIXED_RANGES,
    MODES,
    VX_VALUES,
    VY_VALUES,
    YAW_VALUES,
    build_command_suite,
)


def test_default_suite_counts_ranges_and_metadata():
    suite = build_command_suite(
        seed=1, carry_motion_id=0, carry_phase=0.5, modes=MODES
    )
    assert len(suite) == 43
    assert Counter(item.mode for item in suite) == {
        "stand": 1,
        "vx": 6,
        "vy": 6,
        "yaw": 6,
        "mixed": 24,
    }
    assert [item.trial_id for item in suite] == [
        f"T{index:04d}" for index in range(1, 44)
    ]
    assert all(item.seed == 1 for item in suite)
    assert all(item.carry_motion_id == 0 for item in suite)
    assert all(item.carry_phase == 0.5 for item in suite)

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
    assert VX_VALUES == (-0.50, -0.25, 0.10, 0.40, 0.90, 1.20)
    assert VY_VALUES == (-0.40, -0.20, -0.10, 0.10, 0.20, 0.40)
    assert YAW_VALUES == (-0.50, -0.25, -0.10, 0.10, 0.25, 0.50)
    assert all(abs(value) != 0.05 for value in VX_VALUES)
    assert all(abs(value) != 0.05 for value in VY_VALUES)
    assert all(abs(value) != 0.05 for value in YAW_VALUES)


def test_mode_filter_restarts_trial_ids():
    suite = build_command_suite(
        seed=7, carry_motion_id=2, carry_phase=0.25, modes=("vy",)
    )
    assert len(suite) == 6
    assert suite[0].trial_id == "T0001"
    assert all(item.mode == "vy" for item in suite)
