"""Offline plots for contact-qualified CarryBox evaluation runs."""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path


AXES = (("vx", "vx (m/s)"), ("vy", "vy (m/s)"),
        ("yaw_rate", "yaw rate (rad/s)"))


def _rows(path):
    with open(path, newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _float(row, key):
    try:
        value = float(row.get(key, "nan"))
        return value if math.isfinite(value) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _mean(values):
    good = [x for x in values if math.isfinite(x)]
    return sum(good) / len(good) if good else float("nan")


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def load_run(directory):
    directory = Path(directory).resolve()
    with open(directory / "run_metadata.json", encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata.get("protocol_version") != "contact_gated_v1":
        raise ValueError(f"Unsupported evaluation schema in {directory}")
    if metadata.get("status") != "completed":
        raise ValueError(f"Run is not complete: {directory}")
    return {"path": directory, "metadata": metadata,
            "attempts": _rows(directory / "summary.csv"),
            "segments": _rows(directory / "segment_summary.csv")}


def _comparison_signature(metadata):
    keys = ("protocol_version", "policy_dt", "frames", "readiness_config",
            "command_ranges", "warmup_steps", "measure_steps")
    return tuple(json.dumps(metadata.get(key), sort_keys=True) for key in keys)


def _trial_key(row):
    return tuple(row.get(key, "") for key in
                 ("protocol", "mode", "case_type", "seed", "carry_motion_id",
                  "carry_phase", "vx", "vy", "yaw_rate"))


def _segment_key(row):
    return _trial_key(row) + (row.get("segment_index", ""),)


def matched_comparison(first, second):
    if _comparison_signature(first["metadata"]) != _comparison_signature(second["metadata"]):
        return {"compatible": False, "reason": "protocol, dt, frames, gate, ranges or window differs"}
    a = {_trial_key(row): row for row in first["attempts"]}
    b = {_trial_key(row): row for row in second["attempts"]}
    shared = a.keys() & b.keys()
    sa = {_segment_key(row): row for row in first["segments"]}
    sb = {_segment_key(row): row for row in second["segments"]}
    shared_segments = sa.keys() & sb.keys()
    complete = [key for key in shared_segments if
                sa[key].get("segment_status") == "completed" and
                sb[key].get("segment_status") == "completed"]
    return {"compatible": True, "matched_attempts": len(shared),
            "matched_segments": len(shared_segments),
            "jointly_completed_segments": len(complete),
            "initialization_rate_first": _mean([_float(a[k], "initialization_success") for k in shared]),
            "initialization_rate_second": _mean([_float(b[k], "initialization_success") for k in shared]),
            "end_to_end_rate_first": _mean([_float(a[k], "trial_completed") for k in shared]),
            "end_to_end_rate_second": _mean([_float(b[k], "trial_completed") for k in shared]),
            "jointly_completed_rmse_difference_second_minus_first": {
                axis: _mean([_float(sb[k], axis + "_rmse") - _float(sa[k], axis + "_rmse")
                             for k in complete]) for axis, _ in AXES}}


def _save(fig, target):
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(str(target) + "." + extension, dpi=180)
    import matplotlib.pyplot as plt
    plt.close(fig)


def _mode_plot(runs, output):
    import matplotlib.pyplot as plt
    modes = ("stand", "vx", "vy", "yaw", "mixed")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    width = .78 / max(1, len(runs))
    for a, (axis, label) in zip(axes, AXES):
        for j, run in enumerate(runs):
            values, colors = [], []
            for mode in modes:
                selected = [row for row in run["segments"] if row["mode"] == mode
                            and row["segment_status"] == "completed"
                            and row["protocol"] == "constant"]
                values.append(_mean([_float(row, axis + "_rmse") for row in selected]))
                colors.append("tab:orange" if axis in {"vx": ("vx", "mixed"),
                            "vy": ("vy", "mixed"), "yaw_rate": ("yaw", "mixed")}[axis]
                            else "tab:blue")
            for i, (value, color) in enumerate(zip(values, colors)):
                if math.isfinite(value):
                    a.bar(i - .39 + width * (j + .5), value, width=width,
                          color=color, alpha=max(.35, 1.-j*.13))
        a.set_xticks(range(len(modes)), modes)
        a.set_title(label)
        a.set_ylabel("RMSE: orange=commanded, blue=zero-command leakage")
        a.grid(axis="y", alpha=.2)
    fig.suptitle("Constant commands, completed segments; within each mode bars follow input run order")
    _save(fig, output / "mode_tracking")


def _response_scatter(runs, output):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (axis, label) in zip(axes, AXES):
        for j, run in enumerate(runs):
            for case, marker in (("corner", "s"), ("interior", "o"),
                                 ("axis", "^"), ("stand", "x")):
                rows = [r for r in run["segments"] if r["protocol"] == "constant"
                        and r["segment_status"] == "completed" and r["case_type"] == case]
                x = [_float(r, axis) for r in rows]
                y = [_float(r, axis + "_actual_mean") for r in rows]
                ax.scatter(x, y, marker=marker, s=22, alpha=.65,
                           label=f"run {j+1} {case}")
        ax.axline((0, 0), slope=1, color="black", linewidth=1, linestyle="--")
        ax.set_xlabel("target " + label)
        ax.set_ylabel("actual " + label)
        ax.grid(alpha=.2)
    axes[-1].legend(fontsize=7, loc="best")
    fig.suptitle("Constant target versus observed mean; mixed corners are separate markers")
    _save(fig, output / "target_response")


def _step_plots(runs, output):
    import matplotlib.pyplot as plt
    for run_index, run in enumerate(runs, 1):
        for family in ("vx", "vy", "yaw", "mixed"):
            candidates = [r for r in run["attempts"] if
                          r["protocol"] == "step" and r["mode"] == family]
            candidate = next((r for r in candidates if int(float(r["initialization_success"]))),
                             candidates[0] if candidates else None)
            if candidate is None:
                continue
            trace = _rows(run["path"] / "traces" / (candidate["trial_id"] + ".csv"))
            trace = [r for r in trace if r["phase"] in ("WARMUP", "MEASURE")]
            if not trace:
                continue
            fig, plots = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
            for ax, (axis, label) in zip(plots, AXES):
                time = [_float(row, "time_s") for row in trace]
                ax.step(time, [_float(row, "command_" + axis) for row in trace],
                        where="post", label="command", linewidth=1.4)
                ax.plot(time, [_float(row, "actual_" + axis + "_training_frame")
                               for row in trace], label="actual", linewidth=1)
                for onset in sorted({_float(row, "command_onset_s") for row in trace}):
                    if math.isfinite(onset):
                        ax.axvline(onset, color="grey", alpha=.35, linestyle=":")
                ax.set_ylabel(label)
                ax.grid(alpha=.2)
            plots[0].legend()
            plots[-1].set_xlabel("time since reset (s); dotted lines = command switches")
            fig.suptitle(f"run {run_index}: {family}; {candidate['trial_id']} "
                         f"motion={candidate['carry_motion_id']} phase={candidate['carry_phase']} "
                         f"seed={candidate['seed']}")
            _save(fig, output / f"step_run{run_index}_{family}")


def _failures_plot(runs, output):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for i, run in enumerate(runs):
        rows = run["attempts"]
        initialized = [r for r in rows if _float(r, "initialization_success") == 1.]
        rates = (len(initialized)/len(rows) if rows else float("nan"),
                 _mean([_float(r, "trial_completed") for r in initialized]),
                 _mean([_float(r, "trial_completed") for r in rows]))
        axes[0].bar([i*4+j for j in range(3)], rates)
        waits = [_float(r, "settle_duration_s") for r in rows]
        axes[2].hist([v for v in waits if math.isfinite(v)], bins=12,
                     alpha=.5, label=f"run {i+1}")
    axes[0].set_xticks([i*4+j for i in range(len(runs)) for j in range(3)],
                       [f"r{i+1} {x}" for i in range(len(runs))
                        for x in ("init", "conditional", "end-to-end")], rotation=55)
    axes[0].set_ylim(0, 1.05)
    reasons = Counter(r.get("termination_phase", "") + ":" + r.get("termination_reason", "")
                      for run in runs for r in run["attempts"]
                      if r.get("trial_completed") != "1")
    axes[1].barh(list(reasons), list(reasons.values()))
    axes[1].set_title("Failed attempts by phase/reason")
    axes[2].set_title("Initialization wait (all attempts)")
    axes[2].set_xlabel("seconds")
    axes[2].legend()
    fig.suptitle("Initialization and tracking outcomes")
    _save(fig, output / "initialization_and_failures")


def analyze(run_directories, output_dir=None):
    runs = [load_run(directory) for directory in run_directories]
    if not runs:
        raise ValueError("Provide at least one run")
    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = runs[0]["path"] / "analysis" / stamp
    else:
        output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    import matplotlib
    matplotlib.use("Agg")
    comparison_groups = []
    if len(runs) == 1:
        chart_directory = output
        _mode_plot(runs, chart_directory)
        _response_scatter(runs, chart_directory)
        _step_plots(runs, chart_directory)
        _failures_plot(runs, chart_directory)
    else:
        for index, run in enumerate(runs, 1):
            chart_directory = output / f"run_{index}"
            chart_directory.mkdir()
            _mode_plot([run], chart_directory)
            _response_scatter([run], chart_directory)
            _step_plots([run], chart_directory)
            _failures_plot([run], chart_directory)
        groups = defaultdict(list)
        for index, run in enumerate(runs, 1):
            groups[_comparison_signature(run["metadata"])].append((index, run))
        for group_number, members in enumerate(groups.values(), 1):
            if len(members) < 2:
                continue
            common = set.intersection(*(
                {_trial_key(row) for row in run["attempts"]} for _, run in members))
            if not common:
                continue
            comparison_groups.append((group_number, [index for index, _ in members], len(common)))
            matched = []
            for _, run in members:
                matched.append({**run,
                    "attempts": [r for r in run["attempts"] if _trial_key(r) in common],
                    "segments": [r for r in run["segments"] if _trial_key(r) in common]})
            chart_directory = output / f"matched_group_{group_number}"
            chart_directory.mkdir()
            _mode_plot(matched, chart_directory)
            _response_scatter(matched, chart_directory)
            _failures_plot(matched, chart_directory)
    lines = ["# CarryBox evaluation analysis", "",
             "Runs are numbered in the input order. All rates include failed reset attempts.",
             "Command coverage is deterministic; these are descriptive results, not confidence intervals.",
             "RMSE on an inactive axis describes leakage. m/s and rad/s use separate panels.",
             "Multi-run figures live under each `run_N/`. `matched_group_N/` overlays only common "
             "planned contexts with identical dt, frames, gate, ranges and windows.", ""]
    for group_number, members, count in comparison_groups:
        lines.append(f"Matched group {group_number}: runs {members}, {count} common planned attempts.")
    if comparison_groups:
        lines.append("")
    for i, run in enumerate(runs, 1):
        rows = run["attempts"]
        lines.extend((f"## Run {i}: {run['path']}", "",
                      f"Checkpoint SHA256: `{run['metadata']['checkpoint_sha256']}`", "",
                      f"Attempts: {len(rows)}; initialized: "
                      f"{sum(_float(r, 'initialization_success') == 1. for r in rows)}; "
                      f"completed: {sum(_float(r, 'trial_completed') == 1. for r in rows)}.", ""))
        for protocol in ("constant", "step"):
            selected = [r for r in rows if r["protocol"] == protocol]
            if selected:
                lines.append(f"- {protocol}: attempts {len(selected)}, initialization "
                             f"{_mean([_float(r, 'initialization_success') for r in selected]):.3f}, "
                             f"end-to-end completion {_mean([_float(r, 'trial_completed') for r in selected]):.3f}.")
        lines.append("")
    for i in range(len(runs)):
        for j in range(i+1, len(runs)):
            result = matched_comparison(runs[i], runs[j])
            lines.extend((f"## Run {i+1} versus run {j+1}", "",
                          "```json", json.dumps(_json_safe(result), indent=2, allow_nan=False), "```", ""))
    lines.extend(("Figures: `mode_tracking`, `target_response`, "
                  "`initialization_and_failures`, and representative `step_run*` are saved as PNG and SVG.",
                  "", "A missing settling time means the measured segment did not remain in band; "
                  "terminated segments are censored, not assigned a numeric failure penalty.", ""))
    (output / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output_dir")
    args = parser.parse_args()
    print(analyze(args.runs, args.output_dir))


if __name__ == "__main__":
    main()
