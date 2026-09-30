"""Evaluation-only continuous grasp qualification; force is a contact proxy."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ReadinessConfig:
    stable_hold_s: float
    settle_timeout_s: float
    contact_force_n: float
    side_tolerance_m: float
    face_margin_m: float
    slip_tolerance_mps: float
    relative_velocity_tolerance_mps: tuple
    box_drop_height_m: float
    robot_box_max_distance_m: float
    box_tilt_max_deg: float

    @classmethod
    def from_env(cls, env, stable_hold_s=.20, settle_timeout_s=2.):
        c = env.cfg.rewards
        if not (math.isfinite(stable_hold_s) and stable_hold_s > 0 and
                math.isfinite(settle_timeout_s) and settle_timeout_s >= stable_hold_s):
            raise ValueError("stable_hold must be positive and <= settle_timeout")
        if len(c.carry_box_relative_velocity_tolerance) != 3:
            raise ValueError("Carry box relative velocity tolerance must have xyz components")
        return cls(float(stable_hold_s), float(settle_timeout_s),
                   float(c.hand_contact_threshold), float(c.carry_hand_side_tolerance),
                   float(c.carry_hand_face_margin), float(c.carry_hand_slip_tolerance),
                   tuple(float(x) for x in c.carry_box_relative_velocity_tolerance),
                   float(c.box_drop_height), float(c.robot_box_max_distance),
                   float(c.box_tilt_termination_deg))


def readiness_checks(sample, config):
    """Return independent checks so a rejected start has an observable cause."""
    numeric = (
        "left_hand_force_n", "right_hand_force_n", "left_hand_side_error_m",
        "right_hand_side_error_m", "left_hand_face_excess_m", "right_hand_face_excess_m",
        "left_hand_tangential_slip_mps", "right_hand_tangential_slip_mps",
        "box_relative_vx_mps", "box_relative_vy_mps", "box_relative_vz_mps",
        "box_bottom_m", "robot_box_distance", "box_tilt_deg",
    )
    finite = bool(sample["physical_state_finite"]) and all(
        math.isfinite(float(sample[key])) for key in numeric)
    checks = {"finite": finite, "contact": False, "geometry": False,
              "slip": False, "relative_motion": False, "health": False}
    if finite:
        checks["contact"] = (sample["left_hand_force_n"] > config.contact_force_n
                             and sample["right_hand_force_n"] > config.contact_force_n)
        checks["geometry"] = all(
            sample[f"{side}_hand_side_error_m"] <= config.side_tolerance_m and
            sample[f"{side}_hand_face_excess_m"] <= 0.
            for side in ("left", "right"))
        checks["slip"] = all(
            sample[f"{side}_hand_tangential_slip_mps"] <= config.slip_tolerance_mps
            for side in ("left", "right"))
        checks["relative_motion"] = all(
            abs(sample[f"box_relative_v{axis}_mps"]) <= limit
            for axis, limit in zip("xyz", config.relative_velocity_tolerance_mps))
        checks["health"] = (
            sample["box_bottom_m"] >= config.box_drop_height_m and
            sample["robot_box_distance"] <= config.robot_box_max_distance_m and
            sample["box_tilt_deg"] <= config.box_tilt_max_deg)
    checks["all"] = all(checks.values())
    return checks


class ReadinessGate:
    def __init__(self, config, policy_dt):
        if policy_dt <= 0 or not math.isfinite(policy_dt):
            raise ValueError("policy_dt must be positive and finite")
        self.config = config
        self.dt = float(policy_dt)
        self.required_steps = max(1, math.ceil(config.stable_hold_s / self.dt - 1e-9))
        self.max_steps = max(1, math.ceil(config.settle_timeout_s / self.dt - 1e-9))
        self.steps = 0
        self.streak = 0
        self.longest_streak = 0
        self.last_checks = {}

    def update(self, sample):
        self.steps += 1
        self.last_checks = readiness_checks(sample, self.config)
        self.streak = self.streak + 1 if self.last_checks["all"] else 0
        self.longest_streak = max(self.longest_streak, self.streak)
        return self.streak >= self.required_steps

    @property
    def timed_out(self):
        return self.steps >= self.max_steps and self.streak < self.required_steps

    @property
    def elapsed_s(self):
        return self.steps * self.dt

    @property
    def longest_stable_s(self):
        return self.longest_streak * self.dt
