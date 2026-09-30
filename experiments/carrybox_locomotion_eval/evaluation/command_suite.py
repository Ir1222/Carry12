"""Deterministic command manifests for carry-locomotion evaluation."""

from dataclasses import asdict, dataclass, replace
import math
from typing import Iterable, List, Sequence, Tuple


MODES = ("stand", "vx", "vy", "yaw", "mixed")
STEP_COMMANDS = {
    "vx": ((.40, 0., 0.), (1.20, 0., 0.), (-.60, 0., 0.), (-.15, 0., 0.), (0., 0., 0.)),
    "vy": ((0., .15, 0.), (0., .50, 0.), (0., -.50, 0.), (0., -.15, 0.), (0., 0., 0.)),
    "yaw": ((0., 0., .20), (0., 0., .70), (0., 0., -.70), (0., 0., -.20), (0., 0., 0.)),
    "mixed": ((.40, .15, .20), (1.20, .50, .70), (-.60, -.50, -.70), (.40, -.15, .20), (0., 0., 0.)),
}
TRAINING_MODE_WEIGHTS = {
    "stand": 0.10,
    "vx": 0.10,
    "vy": 0.10,
    "yaw": 0.10,
    "mixed": 0.60,
}

VX_VALUES = (-0.60, -0.35, -0.15, 0.15, 0.40, 0.80, 1.20)
VY_VALUES = (-0.50, -0.30, -0.15, 0.15, 0.30, 0.50)
YAW_VALUES = (-0.70, -0.45, -0.20, 0.20, 0.45, 0.70)

FULL_RANGES = {
    "vx": (-0.60, 1.20),
    "vy": (-0.50, 0.50),
    "yaw_rate": (-0.70, 0.70),
}
MIXED_RANGES = FULL_RANGES.copy()


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
    case_type: str = "axis"
    protocol: str = "constant"
    segment_commands: Tuple[Tuple[float, float, float], ...] = ()

    @property
    def commands(self):
        return self.segment_commands or ((self.vx, self.vy, self.yaw_rate),)

    def as_row(self):
        row = asdict(self)
        row.pop("segment_commands")
        return row


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
    """Return 32 fixed commands: all eight corners plus 24 Halton points."""
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
        for i in range(1, 25)
    ]
    return corners + interior


def _raw_commands() -> Iterable[Tuple[str, float, float, float, str]]:
    yield "stand", 0.0, 0.0, 0.0, "stand"
    for value in VX_VALUES:
        yield "vx", value, 0.0, 0.0, "axis"
    for value in VY_VALUES:
        yield "vy", 0.0, value, 0.0, "axis"
    for value in YAW_VALUES:
        yield "yaw", 0.0, 0.0, value, "axis"
    for index, (vx, vy, yaw_rate) in enumerate(mixed_commands()):
        yield "mixed", vx, vy, yaw_rate, "corner" if index < 8 else "interior"


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
    for mode, vx, vy, yaw_rate, case_type in _raw_commands():
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
                case_type=case_type,
            )
        )
    return conditions


def modes_from_cli(mode: str) -> Tuple[str, ...]:
    if mode == "all":
        return MODES
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}")
    return (mode,)


def build_evaluation_suite(*, seeds, carry_motion_ids, carry_phases,
                           modes=MODES, protocol="constant"):
    """Expand a balanced context grid without changing the legacy 52 cases."""
    if protocol not in ("constant", "step", "both"):
        raise ValueError(f"Unknown protocol: {protocol}")
    modes = tuple(modes)
    if set(modes) - set(MODES):
        raise ValueError(f"Unknown modes: {set(modes) - set(MODES)}")
    if protocol == "step" and modes == ("stand",):
        raise ValueError("stand has no step protocol; use constant or both")
    seeds, carry_motion_ids, carry_phases = tuple(seeds), tuple(carry_motion_ids), tuple(carry_phases)
    if not seeds or not carry_motion_ids or not carry_phases:
        raise ValueError("Context lists must be nonempty")
    for values in (seeds, carry_motion_ids, carry_phases):
        if len(set(values)) != len(values):
            raise ValueError("Repeated context values would duplicate trials")
    if any(int(m) < 0 for m in carry_motion_ids):
        raise ValueError("Motion IDs must be nonnegative")
    if any(not math.isfinite(float(p)) or not 0. <= p <= 1. for p in carry_phases):
        raise ValueError("Carry phases must be finite and within [0, 1]")
    result = []
    for motion_id in carry_motion_ids:
        for phase in carry_phases:
            for seed in seeds:
                if protocol in ("constant", "both"):
                    for c in build_command_suite(seed=seed, carry_motion_id=motion_id,
                                                 carry_phase=phase, modes=modes):
                        result.append(replace(c, trial_id=f"T{len(result)+1:06d}"))
                if protocol in ("step", "both"):
                    for mode in modes:
                        if mode == "stand":
                            continue
                        commands = STEP_COMMANDS[mode]
                        result.append(CommandCondition(
                            trial_id=f"T{len(result)+1:06d}", mode=mode,
                            vx=commands[0][0], vy=commands[0][1], yaw_rate=commands[0][2],
                            seed=int(seed), carry_motion_id=int(motion_id),
                            carry_phase=float(phase), case_type="sequence", protocol="step",
                            segment_commands=commands,
                        ))
    return result
