"""Streaming CPU evaluation analysis; every pickup episode stays in denominators."""
import argparse
import csv
import json
import math
from pathlib import Path


def add_stat(stats, key, value):
    if value == "":
        return
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Nonfinite evaluation metric: " + key)
    total, count, maximum = stats.get(key, (0., 0, -math.inf))
    stats[key] = total + value, count + 1, max(maximum, value)


def stat_pair(stats, key):
    total, count, maximum = stats.get(key, (0, 0, None))
    return total / count if count else None, maximum


def analyze(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    episodes, metrics = {}, {}
    horizons = (3, 15, 30)
    windows = [{"from_s": lo, "to_s": hi, "samples": 0, "stats": {}}
               for lo, hi in ((0, .25), (.25, 1), (1, 3), (3, 15), (15, 30), (30, 1e9))]
    scalar_keys = ("bilateral", "tilt", "omega", "relative", "posture_error", "gap_arms", "gap_waist", "gap_legs")
    window_keys = scalar_keys
    squared_errors = dict.fromkeys(("vx", "vy", "yaw"), 0.)
    jumps = {joint: None for joint in manifest["dof_names"]}
    total_steps, controlled_steps = 0, 0
    with (directory / "steps.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            episode = int(row["episode"])
            state = episodes.setdefault(episode, {"pickup": False, "eligible": False, "done": False,
                                                  "survived": dict.fromkeys(horizons, False),
                                                  "previous_target": None})
            if state["done"]:
                raise ValueError("Episode id reused after completion")
            total_steps += 1
            state["pickup"] |= bool(int(row["pickup"]))
            state["eligible"] |= bool(int(row["eligible"]))
            state["done"] = bool(int(row["done"]))
            state["reason"] = int(row["reason"])
            state["warmup"] = float(row["warmup_s"])
            age = float(row["handoff_relative_s"])
            for horizon in horizons:
                state["survived"][horizon] |= age + 1e-4 >= horizon and state["reason"] in (0, 4)
            controlled = bool(int(row["student_valid"]))
            target = [float(row["pd_target:" + joint]) for joint in jumps]
            target_step_max = None
            if controlled and state["previous_target"] is not None:
                target_step_max = 0.
                for joint, current, previous in zip(jumps, target, state["previous_target"]):
                    delta = abs(current - previous)
                    target_step_max = max(target_step_max, delta)
                    jumps[joint] = max(jumps[joint] or 0., delta)
            state["previous_target"] = target
            if not controlled:
                continue
            controlled_steps += 1
            for axis in squared_errors:
                squared_errors[axis] += (float(row["command_" + axis]) - float(row["actual_" + axis])) ** 2
            for key in scalar_keys:
                add_stat(metrics, key, row[key])
            for window in windows:
                if window["from_s"] <= age < window["to_s"]:
                    window["samples"] += 1
                    for key in window_keys:
                        add_stat(window["stats"], key, row[key])
                    if target_step_max is not None:
                        add_stat(window["stats"], "pd_target_step_max", target_step_max)
                    for axis in squared_errors:
                        add_stat(window["stats"], "tracking_squared_error_" + axis,
                                 (float(row["command_" + axis]) - float(row["actual_" + axis])) ** 2)
                    break
    if len(episodes) != manifest["episodes"] or any(not s["done"] for s in episodes.values()):
        raise ValueError("Incomplete evaluation: every requested pickup episode must finish")
    count = len(episodes)
    eligible = sum(s["eligible"] for s in episodes.values())
    success = sum(s["reason"] == 4 for s in episodes.values())
    report = {"episodes": count,
              "pickup_success": sum(s["pickup"] for s in episodes.values()) / count,
              "handoff_eligibility_rate": eligible / count,
              "end_to_end_completion": success / count,
              "handoff_conditional_completion": success / eligible if eligible else None,
              "warmup_mean_s": sum(s["warmup"] for s in episodes.values()) / count,
              "failure_counts": {str(code): sum(s["reason"] == code for s in episodes.values()) for code in range(5)},
              "student_valid_ratio": controlled_steps / max(1, total_steps)}
    for horizon in horizons:
        survived = sum(s["survived"][horizon] for s in episodes.values())
        report[f"survival_{horizon}s_conditional"] = survived / eligible if eligible else None
        report[f"survival_{horizon}s_end_to_end"] = survived / count
    if controlled_steps:
        for axis, squared_error in squared_errors.items():
            report["tracking_rmse_" + axis] = math.sqrt(squared_error / controlled_steps)
        for key in scalar_keys:
            report[key + "_mean"], report[key + "_max"] = stat_pair(metrics, key)
    for window in windows:
        stats = window.pop("stats")
        for key in window_keys:
            window[key + "_mean"], window[key + "_max"] = stat_pair(stats, key)
        window["pd_target_step_max_mean"], window["pd_target_step_max_max"] = stat_pair(stats, "pd_target_step_max")
        for axis in squared_errors:
            mean, _ = stat_pair(stats, "tracking_squared_error_" + axis)
            window["tracking_rmse_" + axis] = math.sqrt(mean) if mean is not None else None
    report["max_pd_target_step_rad"] = jumps
    report["handoff_time_windows"] = windows
    (directory / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("max_pd_target_step_rad", "handoff_time_windows")}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    analyze(parser.parse_args().directory)
