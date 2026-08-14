"""Force profile definitions for clean CarryBox perturbation evaluation."""

import math


VALID_PROFILES = ("half_sine", "smooth_hold")
VALID_DIRECTIONS = ("+box_x", "-box_x", "+box_y", "-box_y")

DEFAULT_DIRECTIONS = VALID_DIRECTIONS
DEFAULT_BETAS = (0.1, 0.3, 0.5, 0.7, 0.9, 1.1)
DEFAULT_HOLD_DURATIONS = (0.5, 1.0, 2.0)
DEFAULT_SEEDS = (1, 2, 3)

DEFAULT_PULSE_DURATION_S = 0.10
DEFAULT_RAMP_UP_S = 0.15
DEFAULT_RAMP_DOWN_S = 0.15
DEFAULT_HOLD_DURATION_S = 1.0
DEFAULT_PRE_FORCE_DELAY_S = 0.20
DEFAULT_POST_FORCE_OBSERVATION_S = 2.0
DEFAULT_STABLE_CONFIRMED_CARRY_STEPS = 10
DEFAULT_RECOVERY_CONFIRMED_CARRY_STEPS = 10


def smooth_hold_total_duration(hold_duration_s, ramp_up_s, ramp_down_s):
    return float(ramp_up_s) + float(hold_duration_s) + float(ramp_down_s)


def profile_total_duration(profile, pulse_duration_s, hold_duration_s, ramp_up_s, ramp_down_s):
    if profile == "half_sine":
        return float(pulse_duration_s)
    if profile == "smooth_hold":
        return smooth_hold_total_duration(hold_duration_s, ramp_up_s, ramp_down_s)
    raise ValueError(f"Unknown force profile: {profile}")


def half_sine_scale(elapsed_s, duration_s):
    duration_s = max(float(duration_s), 1.0e-9)
    tau = min(max(float(elapsed_s) / duration_s, 0.0), 1.0)
    return math.sin(math.pi * tau)


def smooth_hold_scale(elapsed_s, ramp_up_s, hold_duration_s, ramp_down_s):
    elapsed_s = float(elapsed_s)
    ramp_up_s = max(float(ramp_up_s), 0.0)
    hold_duration_s = max(float(hold_duration_s), 0.0)
    ramp_down_s = max(float(ramp_down_s), 0.0)
    hold_start = ramp_up_s
    ramp_down_start = ramp_up_s + hold_duration_s
    total_s = ramp_down_start + ramp_down_s

    if elapsed_s < 0.0 or elapsed_s > total_s:
        return 0.0
    if ramp_up_s > 0.0 and elapsed_s < hold_start:
        tau = elapsed_s / ramp_up_s
        return 0.5 * (1.0 - math.cos(math.pi * tau))
    if elapsed_s < ramp_down_start:
        return 1.0
    if ramp_down_s > 0.0:
        tau = (elapsed_s - ramp_down_start) / ramp_down_s
        tau = min(max(tau, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * tau))
    return 0.0
