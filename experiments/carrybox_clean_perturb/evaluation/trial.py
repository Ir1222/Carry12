"""Deterministic trial definitions for clean CarryBox perturbation evaluation."""

from dataclasses import dataclass

from .force_profiles import (
    DEFAULT_BETAS,
    DEFAULT_DIRECTIONS,
    DEFAULT_HOLD_DURATION_S,
    DEFAULT_HOLD_DURATIONS,
    DEFAULT_PULSE_DURATION_S,
    DEFAULT_RAMP_DOWN_S,
    DEFAULT_RAMP_UP_S,
    DEFAULT_SEEDS,
    VALID_DIRECTIONS,
    VALID_PROFILES,
)


DEFAULT_EVALUATION_COMMAND = (0.6, 0.0, 0.0)
DEFAULT_COMMAND_SWEEP = tuple(
    (vx, 0.0, yaw_rate)
    for vx in (0.2, 0.4, 0.6, 0.8)
    for yaw_rate in (-0.4, -0.2, 0.0, 0.2, 0.4)
) + ((0.0, 0.0, 0.0),)


@dataclass(frozen=True)
class TrialCondition:
    trial_id: str
    seed: int
    profile: str
    direction: str
    beta: float
    pulse_duration_s: float = DEFAULT_PULSE_DURATION_S
    hold_duration_s: float = DEFAULT_HOLD_DURATION_S
    ramp_up_s: float = DEFAULT_RAMP_UP_S
    ramp_down_s: float = DEFAULT_RAMP_DOWN_S
    command: tuple = DEFAULT_EVALUATION_COMMAND


def _normalize_directions(values):
    directions = tuple(values)
    invalid = [value for value in directions if value not in VALID_DIRECTIONS]
    if invalid:
        raise ValueError(f"Unsupported evaluator directions: {invalid}")
    return directions


def _normalize_commands(values):
    commands = []
    for value in values:
        command = tuple(float(component) for component in value)
        if len(command) != 3:
            raise ValueError(
                f"Expected a 3-D (vx, vy, yaw_rate) command, got {command}"
            )
        commands.append(command)
    return tuple(commands)


def make_single_trial(
    profile,
    direction,
    beta,
    seed,
    pulse_duration_s=DEFAULT_PULSE_DURATION_S,
    hold_duration_s=DEFAULT_HOLD_DURATION_S,
    ramp_up_s=DEFAULT_RAMP_UP_S,
    ramp_down_s=DEFAULT_RAMP_DOWN_S,
    command=DEFAULT_EVALUATION_COMMAND,
):
    if profile not in VALID_PROFILES:
        raise ValueError(f"Unsupported profile: {profile}")
    _normalize_directions((direction,))
    command = _normalize_commands((command,))[0]
    return TrialCondition(
        trial_id="T0001",
        seed=int(seed),
        profile=str(profile),
        direction=str(direction),
        beta=float(beta),
        pulse_duration_s=float(pulse_duration_s),
        hold_duration_s=float(hold_duration_s),
        ramp_up_s=float(ramp_up_s),
        ramp_down_s=float(ramp_down_s),
        command=command,
    )


def generate_sweep(
    profile,
    directions=DEFAULT_DIRECTIONS,
    betas=DEFAULT_BETAS,
    seeds=DEFAULT_SEEDS,
    hold_durations=DEFAULT_HOLD_DURATIONS,
    pulse_duration_s=DEFAULT_PULSE_DURATION_S,
    ramp_up_s=DEFAULT_RAMP_UP_S,
    ramp_down_s=DEFAULT_RAMP_DOWN_S,
    commands=(DEFAULT_EVALUATION_COMMAND,),
):
    if profile not in VALID_PROFILES:
        raise ValueError(f"Unsupported profile: {profile}")
    directions = _normalize_directions(directions)
    betas = tuple(float(value) for value in betas)
    seeds = tuple(int(value) for value in seeds)
    hold_durations = tuple(float(value) for value in hold_durations)
    commands = _normalize_commands(commands)

    conditions = []
    for command in commands:
        for seed in seeds:
            for direction in directions:
                for beta in betas:
                    durations = (
                        hold_durations
                        if profile == "smooth_hold"
                        else (DEFAULT_HOLD_DURATION_S,)
                    )
                    for hold_duration_s in durations:
                        trial_id = f"T{len(conditions) + 1:04d}"
                        conditions.append(
                            TrialCondition(
                                trial_id=trial_id,
                                seed=seed,
                                profile=str(profile),
                                direction=direction,
                                beta=beta,
                                pulse_duration_s=float(pulse_duration_s),
                                hold_duration_s=float(hold_duration_s),
                                ramp_up_s=float(ramp_up_s),
                                ramp_down_s=float(ramp_down_s),
                                command=command,
                            )
                        )
    return conditions
