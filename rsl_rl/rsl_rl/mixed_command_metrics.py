"""Physical-step target-mode metrics shared by training and evaluation.

Missing samples are omitted, never converted into a zero tracking error.
"""
import math

import torch

MODES = ("stand", "vx", "vy", "yaw", "mixed")


class MixedCommandMetrics:
    def __init__(self):
        self.sums = {}
        self.counts = {}
        self.student_steps = 0
        self.hold_steps = 0
        self.ramp_steps = 0

    def _add(self, key, values):
        values = values.detach().float()
        finite = torch.isfinite(values)
        self.sums[key] = self.sums.get(key, 0.) + torch.where(finite, values, 0.).sum()
        self.counts[key] = self.counts.get(key, 0) + finite.sum()

    def update(self, snap, valid):
        if "target_mode" not in snap:
            return
        valid = valid.bool()
        self.student_steps += int(valid.sum())
        self.hold_steps += int((valid & snap["command_hold"]).sum())
        self.ramp_steps += int((valid & snap["command_ramp"]).sum())
        for i, mode in enumerate(MODES):
            selected = valid & (snap["target_mode"] == i)
            self._add(mode + "/target_samples", snap["target_sampled"][selected])
            for interval, mask in (("all", selected), ("ramping", selected & snap["command_ramp"]),
                                   ("settled", selected & ~snap["command_ramp"])):
                prefix = mode + "/" + interval + "/"
                self._add(prefix + "steps", torch.ones_like(snap["target_mode"][mask]))
                for axis, name in enumerate(("vx", "vy", "yaw")):
                    for reference, field in (("applied", "command"), ("target", "target_command")):
                        error = snap[field][mask, axis] - snap["actual"][mask, axis]
                        self._add(prefix + reference + "_rmse_" + name, error.square())
                for name in ("bilateral", "tilt", "relative", "omega", "gait_slip", "gait_clearance",
                             "gait_scuff", "gait_air_time", "gait_touchdown_air_time", "box_relative_speed", "hand_box_slip"):
                    if name in snap:
                        self._add(prefix + name, snap[name][mask])

    def scalars(self):
        result = {}
        if self.student_steps:
            result.update(hold_ratio=self.hold_steps / self.student_steps,
                          ramp_ratio=self.ramp_steps / self.student_steps)
        for key, count in self.counts.items():
            count = int(count)
            if not count:
                continue
            if key.endswith(("/steps", "/target_samples")):
                result[key] = float(self.sums[key])
            else:
                value = float(self.sums[key]) / count
                result[key] = math.sqrt(value) if "_rmse_" in key else value
        return result
