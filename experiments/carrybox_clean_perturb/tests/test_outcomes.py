import sys
from pathlib import Path

import torch


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))

from evaluation.outcomes import (  # noqa: E402
    box_instability_masks,
    humanoid_failure_masks,
    update_box_drop_state,
)


def test_box_drop_requires_confirmed_carry_and_five_steps():
    carry = torch.tensor([False])
    failure = torch.tensor([False])
    streak = torch.tensor([0], dtype=torch.long)
    confirmed = torch.tensor([False])
    box_bottom = torch.tensor([0.01])
    ground = torch.tensor([0.0])

    carry, failure, streak, _ = update_box_drop_state(
        carry, failure, streak, confirmed, box_bottom, ground, 0.03, 5
    )
    assert not carry.item()
    assert not failure.item()
    assert streak.item() == 0

    confirmed[:] = True
    box_bottom[:] = 0.5
    carry, failure, streak, _ = update_box_drop_state(
        carry, failure, streak, confirmed, box_bottom, ground, 0.03, 5
    )
    assert carry.item()
    confirmed[:] = False
    box_bottom[:] = 0.02

    for expected_streak in range(1, 5):
        carry, failure, streak, newly_failed = update_box_drop_state(
            carry, failure, streak, confirmed, box_bottom, ground, 0.03, 5
        )
        assert streak.item() == expected_streak
        assert not failure.item()
        assert not newly_failed.item()

    carry, failure, streak, newly_failed = update_box_drop_state(
        carry, failure, streak, confirmed, box_bottom, ground, 0.03, 5
    )
    assert failure.item()
    assert newly_failed.item()

    box_bottom[:] = 0.5
    carry, failure, streak, newly_failed = update_box_drop_state(
        carry, failure, streak, confirmed, box_bottom, ground, 0.03, 5
    )
    assert failure.item()
    assert not newly_failed.item()
    assert streak.item() == 0


def test_humanoid_failure_thresholds_are_strict():
    masks = humanoid_failure_masks(
        head_z=torch.tensor([0.59, 0.60, 1.0, 1.0]),
        base_z=torch.tensor([1.0, 0.19, 1.0, 1.0]),
        roll=torch.tensor([0.0, 0.0, 0.51, 0.0]),
        pitch=torch.tensor([0.0, 0.0, 0.0, 1.11]),
        hip_z=torch.tensor(
            [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.14, 0.5]]
        ),
    )
    assert masks["head_low"].tolist() == [True, False, False, False]
    assert masks["base_low"].tolist() == [False, True, False, False]
    assert masks["base_tilt"].tolist() == [False, False, True, True]
    assert masks["hip_low"].tolist() == [False, False, False, True]


def test_box_instability_classification():
    masks = box_instability_masks(
        torch.tensor([[3.0, 0.0], [3.01, 0.0]]),
        torch.tensor([-1.0, 0.0]),
        box_termination=True,
    )
    assert masks["box_unstable_speed"].tolist() == [False, True]
    assert masks["box_tilt"].tolist() == [False, True]
