"""Deterministic command manifests for carry-locomotion evaluation."""

from dataclasses import asdict, dataclass
from typing import Iterable, List, Sequence, Tuple


MODES = ("stand", "vx", "vy", "yaw", "mixed")
TRAINING_MODE_WEIGHTS = {
    "stand": 0.10,
    "vx": 0.25,
    "vy": 0.15,
    "yaw": 0.15,
    "mixed": 0.35,
}

VX_VALUES = (-0.50, -0.25, 0.05, 0.40, 0.90, 1.20)
VY_VALUES = (-0.40, -0.20, -0.05, 0.05, 0.20, 0.40)
YAW_VALUES = (-0.50, -0.25, -0.05, 0.05, 0.25, 0.50)

FULL_RANGES = {
    "vx": (-0.50, 1.20),
    "vy": (-0.40, 0.40),
    "yaw_rate": (-0.50, 0.50),
}
MIXED_RANGES = {
    "vx": (-0.40, 0.96),
    "vy": (-0.32, 0.32),
    "yaw_rate": (-0.40, 0.40),
}


@dataclass(frozen=True)
class CommandCondition:
    trial_id: str
    mode: str
    vx: float
    vy: float
    yaw_rate: float
    seed: int
    carry_motion_id: int
    carry_phase: float

    def as_row(self):
        return asdict(self)


def _radical_inverse(index: int, base: int) -> float:
    value = 0.0
    factor = 1.0 / base
    while index:
        index, digit = divmod(index, base)
        value += digit * factor
        factor /= base
    return value


def _scale(unit_value: float, value_range: Tuple[float, float]) -> float:
    low, high = value_range
    return low + unit_value * (high - low)


def mixed_commands() -> List[Tuple[float, float, float]]:
    """Return 24 fixed commands: all eight corners plus 16 Halton points."""
    vx = MIXED_RANGES["vx"]
    vy = MIXED_RANGES["vy"]
    yaw = MIXED_RANGES["yaw_rate"]
    corners = [
        (x, y, z)
        for x in vx
        for y in vy
        for z in yaw
    ]
    interior = [
        (
            _scale(_radical_inverse(i, 2), vx),
            _scale(_radical_inverse(i, 3), vy),
            _scale(_radical_inverse(i, 5), yaw),
        )
        for i in range(1, 17)
    ]
    return corners + interior


def _raw_commands() -> Iterable[Tuple[str, float, float, float]]:
    yield "stand", 0.0, 0.0, 0.0
    for value in VX_VALUES:
        yield "vx", value, 0.0, 0.0
    for value in VY_VALUES:
        yield "vy", 0.0, value, 0.0
    for value in YAW_VALUES:
        yield "yaw", 0.0, 0.0, value
    for vx, vy, yaw_rate in mixed_commands():
        yield "mixed", vx, vy, yaw_rate


def build_command_suite(
    *,
    seed: int,
    carry_motion_id: int,
    carry_phase: float,
    modes: Sequence[str] = MODES,
) -> List[CommandCondition]:
    selected = tuple(modes)
    unknown = set(selected).difference(MODES)
    if unknown:
        raise ValueError(f"Unknown command modes: {sorted(unknown)}")
    conditions = []
    for mode, vx, vy, yaw_rate in _raw_commands():
        if mode not in selected:
            continue
        conditions.append(
            CommandCondition(
                trial_id=f"T{len(conditions) + 1:04d}",
                mode=mode,
                vx=float(vx),
                vy=float(vy),
                yaw_rate=float(yaw_rate),
                seed=int(seed),
                carry_motion_id=int(carry_motion_id),
                carry_phase=float(carry_phase),
            )
        )
    return conditions


def modes_from_cli(mode: str) -> Tuple[str, ...]:
    if mode == "all":
        return MODES
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}")
    return (mode,)
