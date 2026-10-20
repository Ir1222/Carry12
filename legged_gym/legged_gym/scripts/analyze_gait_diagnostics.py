"""Streaming, CPU-only gait analysis; failed pickup episodes keep their denominator."""
import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path

import numpy as np

from gait_diagnostics import SIDES, SOURCES, WINDOWS, SETTINGS, window


def distribution(values):
    if not values:
        return dict(count=0, mean=None, p10=None, p50=None, p90=None)
    return dict(count=len(values), mean=float(np.mean(values)),
                **{f"p{p}": float(np.percentile(values, p)) for p in (10, 50, 90)})


class Bucket:
    def __init__(self, dt, names):
        self.dt, self.names = dt, names
        self.n = 0
        self.count = Counter()
        self.values = defaultdict(list)
        self.runs = {}
        self.last_touchdown = {}

    def add(self, row):
        self.n += 1
        ep = int(row["episode"])
        self.count[row["support_state"]] += 1
        for axis in ("vx", "vy", "yaw"):
            self.count["tracking_sse_" + axis] += (float(row["command_" + axis]) - float(row["actual_" + axis])) ** 2
        for key in ("bilateral", "physical_bilateral", "tilt", "relative", "omega", "posture_error", "gap_arms", "gap_waist", "gap_legs"):
            if row.get(key, "") != "":
                self.values[key].append(float(row[key]))
        self.count["limiter_any"] += int(row["leg_limiter_any"])
        for joint in self.names:
            self.count["limited:" + joint] += int(row["action_rate_limited:" + joint])
            self.count["clipped:" + joint] += int(row["action_clipped:" + joint])
        for side in SIDES:
            c = int(row[side + "_contact"])
            self.count[side + "_contact"] += c
            self.count[side + "_touchdowns"] += int(row[side + "_touchdown"])
            self.count[side + "_scuffs"] += int(row[side + "_scuff_onset"])
            self.count[side + "_near"] += int(row[side + "_near_ground_motion"])
            self.count[side + "_slip_m"] += float(row[side + "_slip_step_m"])
            for source in SOURCES:
                for mode in ("raw", "filtered"):
                    self.count[f"{side}_{source}_{mode}"] += int(row[f"{side}_{source}_contact_{mode}"])
            for mode in ("raw", "filtered"):
                p, r = (int(row[f"{side}_{source}_contact_{mode}"]) for source in SOURCES)
                self.count[f"{side}_{mode}_disagree"] += p != r
                self.count[f"{side}_{mode}_pitch_only"] += p and not r
                self.count[f"{side}_{mode}_roll_only"] += r and not p
            legacy = int(row[side + "_legacy_contact"])
            self.count[side + "_legacy_disagree"] += legacy != c
            for key in ("vx", "vy", "vz", "horizontal_speed", "sole_horizontal_speed"):
                self.values[side + "_" + key].append(float(row[side + "_" + key]))
            if c:
                self.values[side + "_contact_speed"].append(float(row[side + "_contact_speed"]))
                self.values[side + "_contact_clearance"].append(float(row[side + "_clearance_m"]))
            if int(row[side + "_valid_swing"]):
                self.values[side + "_swing_clearance"].append(float(row[side + "_clearance_m"]))
            # Complete bouts only: first and final bouts are censored at every
            # episode, owner/phase and reporting-window boundary.
            key = (ep, side)
            run = self.runs.get(key)
            time = float(row["time_s"])
            continuous = run is not None and abs(time - run["time"] - self.dt) < self.dt * .1
            if not continuous:
                run = dict(contact=c, n=1, left_censored=True, clearances=[], time=time)
            elif run["contact"] == c:
                run["n"] += 1
            else:
                duration = run["n"] * self.dt
                if not run["left_censored"]:
                    kind = "stance" if run["contact"] else "swing"
                    self.values[f"{side}_{kind}_duration"].append(duration)
                    if not run["contact"] and duration + 1e-8 >= SETTINGS["min_valid_swing_s"]:
                        self.values[side + "_swing_peak_clearance"].append(max(run["clearances"]))
                run = dict(contact=c, n=1, left_censored=False, clearances=[], time=time)
            run["clearances"].append(float(row[side + "_clearance_m"]))
            run["time"] = time
            self.runs[key] = run
            phase = "stance" if c else "swing"
            joint_ids = [j for j in self.names if j.startswith(side + "_")]
            self.count[side + "_" + phase + "_samples"] += 1
            for joint in joint_ids:
                self.count[f"{phase}_limited:{joint}"] += int(row["action_rate_limited:" + joint])
            if int(row[side + "_liftoff"]):
                self.count[side + "_liftoff_samples"] += 1
                for joint in joint_ids:
                    self.count["liftoff_limited:" + joint] += int(row["action_rate_limited:" + joint])
        touchdown_sides = [s for s in SIDES if int(row[s + "_touchdown"])]
        if len(touchdown_sides) == 2:
            self.count["simultaneous_touchdowns"] += 1
            self.last_touchdown.pop(ep, None)
        elif touchdown_sides:
            side = touchdown_sides[0]
            previous = self.last_touchdown.get(ep)
            if previous:
                self.count["touchdown_pairs"] += 1
                self.count["alternating_touchdown_pairs"] += previous != side
            self.last_touchdown[ep] = side

    def report(self):
        if not self.n:
            return None  # No invented gait values for nonexistent segments.
        report = dict(samples=self.n, exposure_s=self.n * self.dt,
                      double_support_ratio=self.count["double_support"] / self.n,
                      single_support_ratio=(self.count["left_only"] + self.count["right_only"]) / self.n,
                      left_only_ratio=self.count["left_only"] / self.n,
                      right_only_ratio=self.count["right_only"] / self.n,
                      no_contact_ratio=self.count["no_contact"] / self.n,
                      leg_limiter_any_ratio=self.count["limiter_any"] / self.n,
                      leg_limiter_joint_sample_ratio=sum(self.count["limited:" + j] for j in self.names) / (self.n * len(self.names)),
                      alternating_touchdown_pair_ratio=(self.count["alternating_touchdown_pairs"] / self.count["touchdown_pairs"]
                                                       if self.count["touchdown_pairs"] else None),
                      touchdown_pair_count=self.count["touchdown_pairs"],
                      simultaneous_touchdown_samples=self.count["simultaneous_touchdowns"])
        report["tracking_rmse"] = {axis: math.sqrt(self.count["tracking_sse_" + axis] / self.n) for axis in ("vx", "vy", "yaw")}
        report["preservation"] = {k: distribution(v) for k, v in self.values.items() if not k.startswith(SIDES)}
        report["feet"] = {}
        for side in SIDES:
            foot = dict(contact_duty_factor=self.count[side + "_contact"] / self.n,
                        touchdowns=self.count[side + "_touchdowns"],
                        touchdown_frequency_hz=self.count[side + "_touchdowns"] / (self.n * self.dt),
                        slip_distance_proxy_m=self.count[side + "_slip_m"],
                        near_ground_motion_ratio=self.count[side + "_near"] / self.n,
                        possible_scuff_onsets=self.count[side + "_scuffs"],
                        legacy_combined_contact_disagreement_ratio=self.count[side + "_legacy_disagree"] / self.n)
            foot["pitch_roll_comparison"] = {mode: dict(
                disagreement_ratio=self.count[f"{side}_{mode}_disagree"] / self.n,
                pitch_only_ratio=self.count[f"{side}_{mode}_pitch_only"] / self.n,
                roll_only_ratio=self.count[f"{side}_{mode}_roll_only"] / self.n,
                **{source + "_duty_factor": self.count[f"{side}_{source}_{mode}"] / self.n for source in SOURCES})
                for mode in ("raw", "filtered")}
            foot.update({key[len(side)+1:]: distribution(values) for key, values in self.values.items() if key.startswith(side + "_")})
            for key in ("vx", "vy", "vz", "horizontal_speed", "sole_horizontal_speed", "contact_speed", "contact_clearance",
                        "swing_clearance", "swing_duration", "stance_duration", "swing_peak_clearance"):
                foot.setdefault(key, distribution([]))
            report["feet"][side] = foot
        report["step_frequency_hz"] = sum(self.count[s + "_touchdowns"] for s in SIDES) / (self.n * self.dt)
        report["per_joint_limiter"] = {}
        for joint in self.names:
            side = joint.split("_")[0]
            data = dict(rate_limited_ratio=self.count["limited:" + joint] / self.n,
                        clipped_ratio=self.count["clipped:" + joint] / self.n)
            for phase in ("stance", "swing", "liftoff"):
                n = self.count[f"{side}_{phase}_samples"]
                data[phase + "_limited_ratio"] = self.count[f"{phase}_limited:{joint}"] / n if n else None
            report["per_joint_limiter"][joint] = data
        return report


def analyze_gait(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if "gait_diagnostics" not in manifest:
        raise ValueError("No gait instrumentation in this evaluation")
    dt = manifest["dt"]
    episodes = {}
    with (directory / "steps.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            # Validate every numeric column (including actions and terminal snapshots).
            for key, value in row.items():
                if value and key not in ("support_state", "gait_window") and not math.isfinite(float(value)):
                    raise ValueError("Nonfinite diagnostic: " + key)
            ep = int(row["episode"])
            state = episodes.setdefault(ep, dict(done=False, pickup=False, eligible=False,
                                                 survived={h: False for h in (3, 15, 30)}))
            if state["done"]:
                raise ValueError("Episode reused after terminal row")
            if bool(int(row["student_valid"])) != (int(row["control_phase"]) != 0):
                raise ValueError("Pre-step control ownership mismatch")
            if row["gait_window"] != window(row):
                raise ValueError("Phase/window mismatch")
            state["done"] = bool(int(row["done"]))
            state["reason"] = int(row["reason"])
            state["pickup"] |= bool(int(row["pickup"]))
            state["eligible"] |= bool(int(row["eligible"]))
            for horizon in state["survived"]:
                state["survived"][horizon] |= (float(row["handoff_relative_s"]) + 1e-4 >= horizon
                                               and int(row["reason"]) in (0, 4))
    if len(episodes) != manifest["episodes"] or any(not e["done"] for e in episodes.values()):
        raise ValueError("Incomplete gait evaluation; every requested physical episode must finish")
    names = [n for n in manifest["dof_names"] if n.startswith(SIDES) and any(k in n for k in ("hip_", "knee_", "ankle_"))]
    if len(names) != 12:
        raise ValueError("Expected exactly 12 leg joints")
    buckets = {cohort: {w: Bucket(dt, names) for w in WINDOWS} for cohort in ("all_episodes", "successful_carry")}
    terminal_causes = Counter()
    with (directory / "steps.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            buckets["all_episodes"][row["gait_window"]].add(row)
            if episodes[int(row["episode"])]["reason"] == 4 and int(row["student_valid"]) and not int(row["done"]):
                buckets["successful_carry"][row["gait_window"]].add(row)
            if int(row["done"]):
                terminal_causes.update({k: int(v) for k, v in row.items() if k.startswith("termination:")})
    n = len(episodes)
    eligible = sum(e["eligible"] for e in episodes.values())
    success = sum(e["reason"] == 4 for e in episodes.values())
    summary = dict(status="complete", episodes=n, pickup_success=sum(e["pickup"] for e in episodes.values()) / n,
                   handoff_eligibility_rate=eligible/n, end_to_end_completion=success/n,
                   handoff_conditional_completion=success/eligible if eligible else None,
                   termination_counts=dict(Counter(str(e["reason"]) for e in episodes.values())),
                   terminal_condition_counts=dict(terminal_causes),
                   settings=SETTINGS, limitations=manifest["gait_diagnostics"]["limitations"],
                   primary_cohort="successful_carry; terminal rows excluded; inspect all_episodes for survivor bias",
                   bout_censoring="Only complete bouts within an episode and reporting window; boundary bouts omitted",
                   windows={cohort: {w: b.report() for w, b in windows.items()} for cohort, windows in buckets.items()})
    summary["survival"] = {str(h): dict(end_to_end=sum(e["survived"][h] for e in episodes.values()) / n,
                                      conditional=sum(e["survived"][h] for e in episodes.values()) / eligible if eligible else None)
                           for h in (3, 15, 30)}
    (directory / "gait_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    make_plots(directory, summary)
    write_report(directory, summary)
    return summary


def make_plots(directory, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Plot first three full physical episodes, with a hard point cap. Aggregate
    # statistics above use every row. Plots never join separate episodes.
    traces = defaultdict(list)
    with (directory / "steps.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            ep = int(row["episode"])
            if ep < 3:
                traces[ep].append(row)
    for kind in ("contact_timeline", "swing_clearance", "pitch_roll_contacts"):
        fig, axes = plt.subplots(max(1, len(traces)), 1, figsize=(13, 3.5*max(1, len(traces))), squeeze=False)
        for ax, (ep, rows) in zip(axes[:, 0], sorted(traces.items())):
            stride = max(1, len(rows)//12000)
            rows = rows[::stride]
            t = [float(r["time_s"]) for r in rows]
            if kind == "contact_timeline":
                for i, side in enumerate(SIDES):
                    ax.step(t, [int(r[side+"_contact"])+1.3*i for r in rows], where="post", label=side)
            elif kind == "swing_clearance":
                for side in SIDES:
                    ax.plot(t, [float(r[side+"_clearance_m"]) for r in rows], label=side+" mesh clearance (m)")
                    ax.scatter([x for x, r in zip(t, rows) if int(r[side+"_valid_swing"])],
                               [float(r[side+"_clearance_m"]) for r in rows if int(r[side+"_valid_swing"])], s=3)
                ax.axhline(SETTINGS["near_ground_m"], color="gray", linestyle=":")
            else:
                for i, (side, source) in enumerate((s, c) for s in SIDES for c in SOURCES):
                    ax.step(t, [int(r[f"{side}_{source}_contact_raw"])+1.2*i for r in rows], where="post", label=side+" "+source)
            previous = None
            for row, x in zip(rows, t):
                phase = row["gait_window"]
                if phase != previous:
                    ax.axvline(x, color="gray", alpha=.3)
                    ax.text(x, ax.get_ylim()[1], phase, rotation=90, fontsize=6, va="top")
                previous = phase
            ax.set_title(f"Episode {ep}; full approach / pickup / handoff / carry")
            ax.set_xlabel("Physical episode time (s)")
            ax.legend(loc="lower right", fontsize=7)
        fig.tight_layout()
        fig.savefig(directory / (kind+".png"), dpi=140)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 5))
    windows = summary["windows"]["successful_carry"]
    present = [(w, b) for w, b in windows.items() if b is not None]
    if present:
        names = list(present[0][1]["per_joint_limiter"])
        matrix = np.array([[b["per_joint_limiter"][j]["rate_limited_ratio"] for j in names] for w, b in present])
        im = ax.imshow(matrix, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=75, ha="right", fontsize=7)
        ax.set_yticks(range(len(present)))
        ax.set_yticklabels([w for w, b in present])
        fig.colorbar(im, ax=ax, label="Limiter activation ratio")
    else:
        ax.text(.5, .5, "No successful student carry segments", ha="center")
    fig.tight_layout()
    fig.savefig(directory / "leg_limiter_activation.png", dpi=140)
    plt.close(fig)


def write_report(directory, summary):
    lines = ["# Experiment 0: A5 gait diagnostics", "", "Status: completed physical evaluation.", "",
             f"Episodes: {summary['episodes']}; pickup success: {summary['pickup_success']:.3f}; "
             f"end-to-end completion: {summary['end_to_end_completion']:.3f}.", "",
             "Primary statistics below use successful student-controlled episodes, excluding terminal rows. "
             "All failures remain in success denominators and the all_episodes cohort.", "",
             "| Window | Samples | Double support | Single support | No contact | Limiter joint fraction |",
             "|---|---:|---:|---:|---:|---:|"]
    for w, b in summary["windows"]["successful_carry"].items():
        if b is None:
            lines.append(f"| {w} | 0 | unavailable | unavailable | unavailable | unavailable |")
        else:
            lines.append(f"| {w} | {b['samples']} | {b['double_support_ratio']:.3f} | {b['single_support_ratio']:.3f} | "
                         f"{b['no_contact_ratio']:.3f} | {b['leg_limiter_joint_sample_ratio']:.3f} |")
    def fmt(value):
        return "unavailable" if value is None else f"{value:.4f}"
    lines += ["", "## Foot measurements", "",
              "| Window / foot | Duty | Pitch-roll raw disagreement | Contact speed m/s | Slip proxy m | Swing clearance P10/P50/P90 m | Mean swing / stance s |",
              "|---|---:|---:|---:|---:|---|---|"]
    for w, b in summary["windows"]["successful_carry"].items():
        if b is None:
            continue
        for side, f in b["feet"].items():
            clearance = f.get("swing_clearance", {})
            lines.append(f"| {w} / {side} | {f['contact_duty_factor']:.3f} | "
                         f"{f['pitch_roll_comparison']['raw']['disagreement_ratio']:.3f} | "
                         f"{fmt(f.get('contact_speed', {}).get('mean'))} | {f['slip_distance_proxy_m']:.4f} | "
                         + "/".join(fmt(clearance.get(f"p{p}")) for p in (10, 50, 90)) + " | "
                         + "/".join(fmt(f.get(k+'_duration', {}).get('mean')) for k in ("swing", "stance")) + " |")
    lines += ["", "## Tracking and preservation", "",
              "| Window | RMSE vx/vy/yaw | Physical bilateral ratio | Box tilt mean deg | Relative speed mean m/s | Posture error mean | Alternating touchdown pairs ratio |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for w, b in summary["windows"]["successful_carry"].items():
        if b is None:
            continue
        p = b["preservation"]
        lines.append(f"| {w} | " + "/".join(fmt(b["tracking_rmse"][a]) for a in ("vx", "vy", "yaw"))
                     + " | " + " | ".join(fmt(p.get(k, {}).get("mean")) for k in ("physical_bilateral", "tilt", "relative", "posture_error"))
                     + " | " + fmt(b["alternating_touchdown_pair_ratio"]) + " |")
    lines += ["", "Net-body contact disagreement does not independently prove which source represents the ground. "
              "Compare contact-phase clearance residuals and the rendered collision geometry before changing reward indices. "
              "The all_episodes cohort includes Teacher warmup and early failures; it is available in gait_summary.json. "
              "Per-joint stance/swing/liftoff limiter ratios are also in that file."]
    lines += ["", "## Measurement definitions", "", json.dumps(SETTINGS, indent=2), "",
              "Swing clearance percentiles are time-weighted valid-swing samples. Complete-bout peak-clearance "
              "percentiles are reported separately. Duration bouts crossing reporting boundaries are censored. "
              "A 1-second transition can leave student_0_1s empty because windows use elapsed time from first Student control.", "",
              "## Limitations", ""] + ["- " + x for x in summary["limitations"]]
    lines += ["", "## Experiment 1 decision", "",
              "First inspect pitch/roll and legacy/combined disagreement. If measurement mismatch is material, "
              "validate a corrected reward-contact source in a separate experiment before changing weights. "
              "If contact-phase slip dominates, test only the existing slip term using the validated source. "
              "If valid swings are consistently low while slip and limiting are small, test a small swing-clearance term. "
              "If limiting clusters at liftoff, a reward change is not yet justified; diagnose target bandwidth first. "
              "No reward modification or training was performed."]
    (directory / "gait_diagnostics.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    analyze_gait(parser.parse_args().directory)
