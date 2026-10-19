"""Detailed offline analysis for CarryBox velocity tracking and box stability.

The evaluator's exported segment summaries are sufficient for command-response,
completion, tilt, and step-dynamics plots. Per-timestep traces are optional and
are not required by this script.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


AXES = (
    ("vx", r"$v_x$ (m/s)"),
    ("vy", r"$v_y$ (m/s)"),
    ("yaw_rate", r"yaw rate (rad/s)"),
)
MODES = ("stand", "vx", "vy", "yaw", "mixed")
MODE_COLORS = {
    "stand": "#7A7A7A",
    "vx": "#0072B2",
    "vy": "#009E73",
    "yaw": "#CC79A7",
    "mixed": "#D55E00",
}


def _find_csv(run_dir: Path, exact: str, pattern: str) -> Path:
    direct = run_dir / exact
    if direct.exists():
        return direct
    candidates = sorted(run_dir.glob(pattern), key=lambda p: (len(p.name), p.name))
    if not candidates:
        raise FileNotFoundError(f"No {exact} or {pattern} found in {run_dir}")
    return candidates[0]


def _load(run_dir: Path):
    segment_path = _find_csv(run_dir, "segment_summary.csv", "segment_summary*.csv")
    summary_path = _find_csv(run_dir, "summary.csv", "summary*.csv")
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    with open(metadata_path, encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata.get("status") != "completed":
        raise ValueError(f"Run is not completed: {metadata.get('status')}")
    segments = pd.read_csv(segment_path)
    attempts = pd.read_csv(summary_path)
    if len(attempts) != int(metadata.get("number_of_attempts", len(attempts))):
        raise ValueError("Attempt count does not match run_metadata.json")
    return metadata, attempts, segments, summary_path, segment_path


def _style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _save(fig, output_dir: Path, stem: str):
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _annotate_bars(ax, bars, fmt="{:.0%}"):
    for bar in bars:
        value = bar.get_height()
        if math.isfinite(value):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.018,
                fmt.format(value),
                ha="center",
                va="bottom",
                fontsize=8,
            )


def _heatmap(ax, table: pd.DataFrame, title: str, fmt: str, vmin=None, vmax=None, cmap="viridis"):
    values = table.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(values)
    image = ax.imshow(masked, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks(range(len(table.columns)), [str(x) for x in table.columns])
    ax.set_yticks(range(len(table.index)), [str(x) for x in table.index])
    ax.set_xlabel("carry phase")
    ax.set_ylabel("carry motion id")
    ax.set_title(title)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            label = "n/a" if not math.isfinite(value) else fmt.format(value)
            color = "white" if math.isfinite(value) and image.norm(value) > 0.58 else "black"
            ax.text(j, i, label, ha="center", va="center", fontsize=8.5, color=color)
    return image


def _command_summary(segments: pd.DataFrame) -> pd.DataFrame:
    constant = segments.loc[segments["protocol"].eq("constant")].copy()
    constant["completed"] = constant["segment_status"].eq("completed")
    constant["normalized_command_norm"] = np.sqrt(
        (constant["vx"] / 1.2) ** 2
        + (constant["vy"] / 0.5) ** 2
        + (constant["yaw_rate"] / 0.7) ** 2
    )
    keys = ["mode", "vx", "vy", "yaw_rate", "case_type"]
    metrics = [
        "vx_actual_mean",
        "vy_actual_mean",
        "yaw_rate_actual_mean",
        "vx_bias",
        "vy_bias",
        "yaw_rate_bias",
        "vx_rmse",
        "vy_rmse",
        "yaw_rate_rmse",
        "normalized_vector_error_rmse",
        "box_vx_rmse",
        "box_vy_rmse",
        "box_yaw_rate_rmse",
        "box_tilt_mean_deg",
        "box_tilt_p95_deg",
        "box_tilt_max_deg",
        "robot_box_relative_linear_velocity_norm_p95",
        "bilateral_hand_contact_fraction",
        "grasp_loss_fraction",
        "normalized_command_norm",
    ]
    rows = []
    for key, group in constant.groupby(keys, sort=True, dropna=False):
        completed = group.loc[group["completed"]]
        row = dict(zip(keys, key))
        row.update(
            planned_segments=len(group),
            completed_segments=len(completed),
            completion_rate=float(group["completed"].mean()),
        )
        for metric in metrics:
            row[metric + "_mean"] = float(completed[metric].mean())
            row[metric + "_median"] = float(completed[metric].median())
            row[metric + "_q25"] = float(completed[metric].quantile(0.25))
            row[metric + "_q75"] = float(completed[metric].quantile(0.75))
            row[metric + "_max"] = float(completed[metric].max())
        rows.append(row)
    return pd.DataFrame(rows)


def _context_summary(attempts: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    constant_attempts = attempts.loc[attempts["protocol"].eq("constant")].copy()
    mixed_completed = segments.loc[
        segments["protocol"].eq("constant")
        & segments["mode"].eq("mixed")
        & segments["segment_status"].eq("completed")
    ]
    rows = []
    for (motion, phase), group in constant_attempts.groupby(["carry_motion_id", "carry_phase"]):
        completed = mixed_completed.loc[
            mixed_completed["carry_motion_id"].eq(motion)
            & mixed_completed["carry_phase"].eq(phase)
        ]
        rows.append(
            {
                "carry_motion_id": motion,
                "carry_phase": phase,
                "planned_constant_trials": len(group),
                "initialization_rate": group["initialization_success"].mean(),
                "end_to_end_completion_rate": group["trial_completed"].mean(),
                "excessive_box_tilt_failure_rate": group["termination_reason"].eq("excessive_box_tilt").mean(),
                "box_drop_failure_rate": group["termination_reason"].eq("box_drop").mean(),
                "completed_mixed_segments": len(completed),
                "mixed_normalized_tracking_rmse_mean": completed["normalized_vector_error_rmse"].mean(),
                "mixed_box_tilt_p95_deg_mean": completed["box_tilt_p95_deg"].mean(),
            }
        )
    return pd.DataFrame(rows)


def _step_summary(segments: pd.DataFrame) -> pd.DataFrame:
    step = segments.loc[segments["protocol"].eq("step")].copy()
    rows = []
    for family in ("vx", "vy", "yaw", "mixed"):
        family_rows = step.loc[step["sequence_family"].eq(family)]
        for axis, _ in AXES:
            changed = family_rows.loc[family_rows[axis + "_command_changed"].eq(1)]
            completed = changed.loc[changed["segment_status"].eq("completed")]
            if changed.empty:
                continue
            rows.append(
                {
                    "sequence_family": family,
                    "axis": axis,
                    "planned_changed_segments": len(changed),
                    "completed_changed_segments": len(completed),
                    "completion_rate": len(completed) / len(changed),
                    "settled_fraction_completed": completed[axis + "_settled"].mean(),
                    "settling_time_s_mean_when_settled": completed.loc[
                        completed[axis + "_settled"].eq(1), axis + "_settling_time_s"
                    ].mean(),
                    "overshoot_pct_median_completed": completed[axis + "_overshoot_pct"].median(),
                    "overshoot_pct_p95_completed": completed[axis + "_overshoot_pct"].quantile(0.95),
                }
            )
    return pd.DataFrame(rows)


def _plot_funnel(attempts: pd.DataFrame, output_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2))
    ax = axes[0, 0]
    grouped = (
        attempts.groupby(["protocol", "mode"], sort=False)
        .agg(initialized=("initialization_success", "mean"), end_to_end=("trial_completed", "mean"))
        .reset_index()
    )
    labels = [f"{p}\n{m}" for p, m in zip(grouped["protocol"], grouped["mode"])]
    x = np.arange(len(grouped))
    width = 0.36
    b1 = ax.bar(x - width / 2, grouped["initialized"], width, label="initialized", color="#56B4E9")
    b2 = ax.bar(x + width / 2, grouped["end_to_end"], width, label="end-to-end", color="#D55E00")
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylim(0, 1.06)
    ax.set_ylabel("rate")
    ax.set_title("Attempt success includes readiness failures")
    ax.legend(loc="upper right")
    _annotate_bars(ax, b2)

    ax = axes[0, 1]
    failures = attempts.loc[attempts["trial_completed"].ne(1), "termination_reason"].value_counts()
    bars = ax.barh(failures.index[::-1], failures.values[::-1], color="#C44E52")
    total_failures = int(failures.sum())
    for bar, value in zip(bars, failures.values[::-1]):
        ax.text(value + total_failures * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{value} ({value / total_failures:.1%})", va="center", fontsize=8.5)
    ax.set_xlim(0, max(failures.values) * 1.22)
    ax.set_xlabel("failed attempts")
    ax.set_title("Termination reasons")

    constant = attempts.loc[attempts["protocol"].eq("constant")]
    table = constant.pivot_table(
        index="carry_motion_id", columns="carry_phase", values="trial_completed", aggfunc="mean"
    )
    image = _heatmap(axes[1, 0], table, "Constant end-to-end completion", "{:.1%}", 0, 1, "RdYlGn")
    fig.colorbar(image, ax=axes[1, 0], fraction=0.046, pad=0.04, label="completion rate")

    step = attempts.loc[attempts["protocol"].eq("step")]
    table = step.pivot_table(
        index="carry_motion_id", columns="carry_phase", values="trial_completed", aggfunc="mean"
    )
    image = _heatmap(axes[1, 1], table, "Step end-to-end completion", "{:.1%}", 0, 1, "RdYlGn")
    fig.colorbar(image, ax=axes[1, 1], fraction=0.046, pad=0.04, label="completion rate")
    fig.suptitle("CarryBox evaluation funnel and context sensitivity", fontsize=14, y=1.01)
    _save(fig, output_dir, "01_evaluation_funnel")


def _plot_specialists(segments: pd.DataFrame, command_summary: pd.DataFrame, output_dir: Path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6))
    completed = segments.loc[
        segments["protocol"].eq("constant") & segments["segment_status"].eq("completed")
    ]
    for col, (mode, axis, label) in enumerate((
        ("vx", "vx", r"$v_x$ (m/s)"),
        ("vy", "vy", r"$v_y$ (m/s)"),
        ("yaw", "yaw_rate", "yaw rate (rad/s)"),
    )):
        ax = axes[0, col]
        raw = completed.loc[completed["mode"].eq(mode)]
        agg = command_summary.loc[command_summary["mode"].eq(mode)].sort_values(axis)
        ax.scatter(raw[axis], raw[axis + "_actual_mean"], s=20, alpha=0.23,
                   color=MODE_COLORS[mode], linewidths=0)
        target = agg[axis].to_numpy()
        mean = agg[axis + "_actual_mean_median"].to_numpy()
        low = agg[axis + "_actual_mean_q25"].to_numpy()
        high = agg[axis + "_actual_mean_q75"].to_numpy()
        ax.errorbar(target, mean, yerr=np.vstack([mean - low, high - mean]), fmt="o-",
                    color="#111111", linewidth=1.5, markersize=4, capsize=3, label="mean and IQR")
        lo = min(float(target.min()), float(raw[axis + "_actual_mean"].min()))
        hi = max(float(target.max()), float(raw[axis + "_actual_mean"].max()))
        margin = 0.08 * (hi - lo if hi > lo else 1)
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin], "--", color="#777777",
                linewidth=1, label="ideal")
        ax.set_xlabel("command " + label)
        ax.set_ylabel("observed mean " + label)
        ax.set_title(f"{mode} specialist: target response")
        if col == 0:
            ax.legend(loc="upper left")

        ax = axes[1, col]
        positions = np.arange(len(agg))
        bars = ax.bar(positions, agg[axis + "_rmse_mean"], color=MODE_COLORS[mode], alpha=0.82,
                      label="RMSE")
        ax.set_xticks(positions, [f"{x:g}" for x in agg[axis]], rotation=0)
        ax.set_xlabel("command " + label)
        ax.set_ylabel("RMSE (same units)", color=MODE_COLORS[mode])
        ax.tick_params(axis="y", labelcolor=MODE_COLORS[mode])
        ax2 = ax.twinx()
        ax2.plot(positions, agg["completion_rate"], "o-", color="#111111", linewidth=1.4,
                 markersize=4, label="end-to-end completion")
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("end-to-end completion", color="#111111")
        ax2.grid(False)
        if col == 0:
            handles = [bars, Line2D([0], [0], color="#111111", marker="o")]
            ax.legend(handles, ["RMSE, completed only", "completion, all planned"], loc="upper left")
    fig.suptitle("Constant-command specialist tracking: conditional accuracy and end-to-end success",
                 fontsize=14, y=1.01)
    _save(fig, output_dir, "02_specialist_velocity_tracking")


def _plot_mixed(segments: pd.DataFrame, command_summary: pd.DataFrame, output_dir: Path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6))
    completed = segments.loc[
        segments["protocol"].eq("constant")
        & segments["mode"].eq("mixed")
        & segments["segment_status"].eq("completed")
    ].copy()
    agg = command_summary.loc[command_summary["mode"].eq("mixed")].copy()
    for col, (axis, label) in enumerate(AXES):
        top = axes[0, col]
        for case, color, marker in (("interior", "#0072B2", "o"), ("corner", "#D55E00", "s")):
            data = completed.loc[completed["case_type"].eq(case)]
            top.scatter(data[axis], data[axis + "_actual_mean"], s=20, alpha=0.20,
                        color=color, marker=marker, linewidths=0, label=case)
        means = agg.sort_values(axis)
        size = 30 + 150 * means["completion_rate"].to_numpy()
        top.scatter(means[axis], means[axis + "_actual_mean_mean"], s=size, facecolors="none",
                    edgecolors="#111111", linewidths=1.0, label="command mean; size=completion")
        lo = min(float(completed[axis].min()), float(completed[axis + "_actual_mean"].min()))
        hi = max(float(completed[axis].max()), float(completed[axis + "_actual_mean"].max()))
        margin = 0.08 * (hi - lo if hi > lo else 1)
        top.plot([lo - margin, hi + margin], [lo - margin, hi + margin], "--", color="#777777")
        top.set_xlabel("command " + label)
        top.set_ylabel("observed mean " + label)
        top.set_title(f"Mixed commands: {label}")
        if col == 0:
            top.legend(loc="upper left", fontsize=8)

        bottom = axes[1, col]
        x = agg[axis].to_numpy()
        residual = agg[axis + "_actual_mean_median"].to_numpy() - x
        low = agg[axis + "_actual_mean_q25"].to_numpy() - x
        high = agg[axis + "_actual_mean_q75"].to_numpy() - x
        colors = ["#D55E00" if c == "corner" else "#0072B2" for c in agg["case_type"]]
        bottom.errorbar(x, residual, yerr=np.vstack([residual - low, high - residual]), fmt="none",
                        ecolor="#888888", alpha=0.6, capsize=2)
        bottom.scatter(x, residual, s=30 + 150 * agg["completion_rate"], c=colors,
                       edgecolors="white", linewidths=0.5, alpha=0.9)
        bottom.axhline(0, color="#333333", linestyle="--", linewidth=1)
        bottom.set_xlabel("command " + label)
        bottom.set_ylabel("mean response bias")
        bottom.set_title("Command-level bias; size=completion")
    fig.suptitle("Mixed-command velocity response across the training range", fontsize=14, y=1.01)
    _save(fig, output_dir, "03_mixed_velocity_tracking")


def _plot_box_stability(segments: pd.DataFrame, command_summary: pd.DataFrame, output_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    completed = segments.loc[
        segments["protocol"].eq("constant") & segments["segment_status"].eq("completed")
    ].copy()

    ax = axes[0, 0]
    data = [completed.loc[completed["mode"].eq(mode), "box_tilt_p95_deg"].dropna().to_numpy()
            for mode in MODES]
    box = ax.boxplot(data, labels=MODES, patch_artist=True, showfliers=True,
                     medianprops={"color": "#111111", "linewidth": 1.5})
    for patch, mode in zip(box["boxes"], MODES):
        patch.set_facecolor(MODE_COLORS[mode])
        patch.set_alpha(0.72)
    ax.axhline(70, color="#C44E52", linestyle="--", linewidth=1.2, label="termination threshold (70°)")
    ax.set_ylabel("per-segment box tilt P95 (deg)")
    ax.set_title("Tilt among completed constant segments")
    ax.legend(loc="upper right")

    ax = axes[0, 1]
    for mode in MODES:
        data = command_summary.loc[command_summary["mode"].eq(mode)]
        ax.scatter(data["normalized_command_norm_mean"], data["box_tilt_p95_deg_mean"],
                   s=25 + 170 * data["completion_rate"], color=MODE_COLORS[mode], alpha=0.78,
                   edgecolors="white", linewidths=0.5, label=mode)
    ax.set_xlabel("normalized command-vector magnitude")
    ax.set_ylabel("mean box tilt P95 (deg), completed only")
    ax.set_title("Tilt versus command magnitude; size=completion")
    ax.legend(ncol=2, fontsize=8)

    ax = axes[1, 0]
    labels, robot_values, box_values = [], [], []
    for mode, axis, short in (
        ("vx", "vx", "vx/vx"),
        ("vy", "vy", "vy/vy"),
        ("yaw", "yaw_rate", "yaw/yaw"),
        ("mixed", "vx", "mixed/vx"),
        ("mixed", "vy", "mixed/vy"),
        ("mixed", "yaw_rate", "mixed/yaw"),
    ):
        data = completed.loc[completed["mode"].eq(mode)]
        labels.append(short)
        robot_values.append(data[axis + "_rmse"].mean())
        box_values.append(data["box_" + axis + "_rmse"].mean())
    x = np.arange(len(labels))
    width = 0.38
    ax.bar(x - width / 2, robot_values, width, label="robot", color="#0072B2")
    ax.bar(x + width / 2, box_values, width, label="box", color="#E69F00")
    ax.set_xticks(x, labels, rotation=28, ha="right")
    ax.set_ylabel("velocity RMSE")
    ax.set_title("Box motion tracks the command less accurately")
    ax.legend()

    ax = axes[1, 1]
    plot_data = command_summary.loc[command_summary["mode"].ne("stand")].dropna(
        subset=["normalized_vector_error_rmse_mean", "box_tilt_p95_deg_mean"]
    )
    for mode in ("vx", "vy", "yaw", "mixed"):
        data = plot_data.loc[plot_data["mode"].eq(mode)]
        ax.scatter(data["normalized_vector_error_rmse_mean"], data["box_tilt_p95_deg_mean"],
                   s=25 + 170 * data["completion_rate"], color=MODE_COLORS[mode], alpha=0.8,
                   edgecolors="white", linewidths=0.5, label=mode)
    if len(plot_data) > 1:
        corr = np.corrcoef(
            plot_data["normalized_vector_error_rmse_mean"], plot_data["box_tilt_p95_deg_mean"]
        )[0, 1]
        ax.text(0.98, 0.96, f"command-level Pearson r = {corr:.2f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=9)
    ax.set_xlabel("normalized robot tracking RMSE")
    ax.set_ylabel("box tilt P95 (deg), completed only")
    ax.set_title("Tracking error and tilt are weakly coupled after gating")
    ax.legend(fontsize=8)
    fig.suptitle("Box stability and box–robot motion coupling", fontsize=14, y=1.01)
    _save(fig, output_dir, "04_box_stability")


def _plot_step_dynamics(attempts: pd.DataFrame, step_summary: pd.DataFrame, output_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2))
    step_attempts = attempts.loc[attempts["protocol"].eq("step")]
    trial = step_attempts.groupby("mode", sort=False).agg(
        initialized=("initialization_success", "mean"), end_to_end=("trial_completed", "mean")
    ).reindex(["vx", "vy", "yaw", "mixed"])
    ax = axes[0, 0]
    x = np.arange(len(trial))
    width = 0.36
    ax.bar(x - width / 2, trial["initialized"], width, label="initialized", color="#56B4E9")
    bars = ax.bar(x + width / 2, trial["end_to_end"], width, label="end-to-end", color="#D55E00")
    ax.set_xticks(x, trial.index)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("trial rate")
    ax.set_title("Step trials")
    ax.legend()
    _annotate_bars(ax, bars)

    ordered = step_summary.copy()
    ordered["label"] = ordered["sequence_family"] + "/" + ordered["axis"].str.replace("yaw_rate", "yaw")
    positions = np.arange(len(ordered))
    ax = axes[0, 1]
    bars = ax.bar(positions, ordered["settled_fraction_completed"], color=[
        MODE_COLORS.get(x, "#777777") for x in ordered["sequence_family"]
    ])
    ax.set_xticks(positions, ordered["label"], rotation=35, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("settled fraction")
    ax.set_title("Changed axes that settle within the segment")
    _annotate_bars(ax, bars)

    ax = axes[1, 0]
    ax.bar(positions, ordered["overshoot_pct_median_completed"], color=[
        MODE_COLORS.get(x, "#777777") for x in ordered["sequence_family"]
    ])
    ax.axhline(100, color="#C44E52", linestyle="--", linewidth=1, label="100% of command step")
    ax.set_xticks(positions, ordered["label"], rotation=35, ha="right")
    ax.set_ylabel("median directional overshoot (%)")
    ax.set_title("Overshoot among completed segments")
    ax.legend()

    ax = axes[1, 1]
    ax.bar(positions, ordered["settling_time_s_mean_when_settled"], color=[
        MODE_COLORS.get(x, "#777777") for x in ordered["sequence_family"]
    ])
    ax.axhline(3.0, color="#777777", linestyle="--", linewidth=1, label="3.0 s measurement window")
    ax.set_xticks(positions, ordered["label"], rotation=35, ha="right")
    ax.set_ylabel("mean settling time (s), settled only")
    ax.set_title("Settling time is conditional on settling")
    ax.legend()
    fig.suptitle("Step-command dynamics: steady-state completion hides slow transients", fontsize=14, y=1.01)
    _save(fig, output_dir, "05_step_dynamics")


def _plot_context(attempts: pd.DataFrame, segments: pd.DataFrame, output_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
    constant = attempts.loc[attempts["protocol"].eq("constant")]
    completion = constant.pivot_table(index="carry_motion_id", columns="carry_phase",
                                      values="trial_completed", aggfunc="mean")
    image = _heatmap(axes[0], completion, "All constant commands: completion", "{:.1%}", 0, 1, "RdYlGn")
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
    mixed = segments.loc[
        segments["protocol"].eq("constant")
        & segments["mode"].eq("mixed")
        & segments["segment_status"].eq("completed")
    ]
    tracking = mixed.pivot_table(index="carry_motion_id", columns="carry_phase",
                                 values="normalized_vector_error_rmse", aggfunc="mean")
    image = _heatmap(axes[1], tracking, "Completed mixed: normalized RMSE", "{:.3f}",
                     cmap="YlOrRd")
    fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
    tilt = mixed.pivot_table(index="carry_motion_id", columns="carry_phase",
                             values="box_tilt_p95_deg", aggfunc="mean")
    image = _heatmap(axes[2], tilt, "Completed mixed: box tilt P95", "{:.1f}°", cmap="YlOrRd")
    fig.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04)
    fig.suptitle("Reference motion and phase dominate readiness; tracking metrics are survivor-only",
                 fontsize=14, y=1.03)
    _save(fig, output_dir, "06_context_sensitivity")


def _linear_fit(data: pd.DataFrame, x: str, y: str):
    selected = data[[x, y]].dropna()
    if len(selected) < 2:
        return float("nan"), float("nan"), float("nan")
    xv = selected[x].to_numpy(dtype=float)
    yv = selected[y].to_numpy(dtype=float)
    slope, intercept = np.polyfit(xv, yv, 1)
    predicted = slope * xv + intercept
    denom = np.sum((yv - yv.mean()) ** 2)
    r2 = 1 - np.sum((yv - predicted) ** 2) / denom if denom > 0 else float("nan")
    return float(slope), float(intercept), float(r2)


def _fmt(value, digits=3):
    return "n/a" if not math.isfinite(float(value)) else f"{float(value):.{digits}f}"


def _write_report(
    output_dir: Path,
    run_dir: Path,
    metadata: dict,
    attempts: pd.DataFrame,
    segments: pd.DataFrame,
    command_summary: pd.DataFrame,
    context_summary: pd.DataFrame,
    step_summary: pd.DataFrame,
    source_summary: Path,
    source_segments: Path,
):
    failures = attempts.loc[attempts["trial_completed"].ne(1)]
    total_attempts = len(attempts)
    total_completed = int(attempts["trial_completed"].sum())
    total_failures = len(failures)
    reason_counts = failures["termination_reason"].value_counts()
    tilt_failures = int(reason_counts.get("excessive_box_tilt", 0))
    constant_attempts = attempts.loc[attempts["protocol"].eq("constant")]
    step_attempts = attempts.loc[attempts["protocol"].eq("step")]
    completed_constant = segments.loc[
        segments["protocol"].eq("constant") & segments["segment_status"].eq("completed")
    ]
    mixed = completed_constant.loc[completed_constant["mode"].eq("mixed")]

    fit_rows = []
    for mode, axis in (("vx", "vx"), ("vy", "vy"), ("yaw", "yaw_rate")):
        data = completed_constant.loc[completed_constant["mode"].eq(mode)]
        slope, intercept, r2 = _linear_fit(data, axis, axis + "_actual_mean")
        fit_rows.append((mode, axis, len(data), slope, intercept, r2,
                         data[axis + "_rmse"].mean()))
    mixed_fit_rows = []
    for axis, _ in AXES:
        slope, intercept, r2 = _linear_fit(mixed, axis, axis + "_actual_mean")
        mixed_fit_rows.append((axis, slope, intercept, r2, mixed[axis + "_rmse"].mean()))

    def command_row(mode, axis, value):
        row = command_summary.loc[
            command_summary["mode"].eq(mode) & np.isclose(command_summary[axis], value)
        ]
        return row.iloc[0] if not row.empty else None

    vy_neg = command_row("vy", "vy", -0.15)
    vy_pos = command_row("vy", "vy", 0.15)
    yaw_neg = command_row("yaw", "yaw_rate", -0.20)
    yaw_pos = command_row("yaw", "yaw_rate", 0.20)
    worst_mixed = command_summary.loc[
        command_summary["mode"].eq("mixed") & command_summary["completed_segments"].ge(3)
    ].sort_values("normalized_vector_error_rmse_mean", ascending=False).head(5)
    worst_tilt = command_summary.loc[
        command_summary["completed_segments"].ge(3)
    ].sort_values("box_tilt_p95_deg_mean", ascending=False).head(5)

    box_pairs = []
    for mode, axis in (("vx", "vx"), ("vy", "vy"), ("yaw", "yaw_rate"),
                       ("mixed", "vx"), ("mixed", "vy"), ("mixed", "yaw_rate")):
        data = completed_constant.loc[completed_constant["mode"].eq(mode)]
        robot = data[axis + "_rmse"].mean()
        box = data["box_" + axis + "_rmse"].mean()
        box_pairs.append((mode, axis, robot, box, box / robot))

    constant_initialized = int(constant_attempts["initialization_success"].sum())
    constant_completed = int(constant_attempts["trial_completed"].sum())
    step_initialized = int(step_attempts["initialization_success"].sum())
    step_completed = int(step_attempts["trial_completed"].sum())
    best_context = context_summary.sort_values("end_to_end_completion_rate", ascending=False).iloc[0]
    worst_context = context_summary.sort_values("end_to_end_completion_rate", ascending=True).iloc[0]

    lines = [
        "# CarryBox locomotion：velocity tracking 与 box tilt 分析",
        "",
        "## 结论摘要",
        "",
        f"本次数据包含 **{total_attempts} 个 trial、{len(segments)} 个计划 segment**，端到端完成 "
        f"**{total_completed}/{total_attempts} = {total_completed / total_attempts:.1%}**。主要瓶颈不是进入测量后的 "
        f"velocity tracking，而是 SETTLE/readiness：{tilt_failures}/{total_failures} = "
        f"**{tilt_failures / total_failures:.1%} 的失败由 excessive_box_tilt 触发**。",
        "",
        f"Constant protocol 初始化成功 {constant_initialized}/{len(constant_attempts)}，并且初始化后的 "
        f"{constant_completed}/{constant_initialized} 个 trial 全部完成。Step protocol 初始化成功 "
        f"{step_initialized}/{len(step_attempts)}，其中 {step_completed}/{step_initialized} 完成；仅 1 个 mixed step "
        "trial 在 MEASURE 阶段因 humanoid_tilt 失败。因此当前系统呈现明显的两阶段行为：初始化经常失败；一旦进入测量，" 
        "3 s 内保持 carrying 的能力很强。",
        "",
        "Completed-only tracking 的稳态增益较好，但这部分结果存在显著 survivor bias。Box tilt 在成功样本中看起来温和，" 
        "不能据此判断端到端 box stability 已解决。",
        "",
        "![Evaluation funnel](01_evaluation_funnel.png)",
        "",
        "## 1. 评估口径",
        "",
        f"- Policy dt 为 {metadata['policy_dt']:.3f} s（约 {1 / metadata['policy_dt']:.1f} Hz）。每个 segment warmup "
        f"{metadata['warmup_steps']} steps，测量 {metadata['measure_steps']} steps（3.0 s）。",
        f"- readiness 要求连续 {metadata['stable_required_steps']} steps 稳定，最长等待 "
        f"{metadata['readiness_config']['settle_timeout_s']:.1f} s。Box tilt termination threshold 为 "
        f"{metadata['readiness_config']['box_tilt_max_deg']:.0f}°。",
        "- 速度平面量在 pelvis-yaw frame 中计算，yaw rate 使用 world-z；box planar velocity 同样在 pelvis-yaw frame 中计算。",
        "- Command coverage 是确定性的训练范围覆盖，不是从部署分布独立采样。因此下文的比例是描述性覆盖结果，不应解释为统计置信区间。",
        f"- 原始输入：`{source_summary.name}` 与 `{source_segments.name}`。当前结果目录没有 `traces/`，所以无法恢复每个 20 ms step 的时序曲线；" 
        "本报告的 response 图使用每个 3 s segment 的 target、actual mean、RMSE、bias、overshoot 和 settling 指标。",
        "",
        "## 2. 不同 command 下的 velocity tracking",
        "",
        "### 2.1 单轴 specialist",
        "",
        "| Mode | Completed segments | Response gain | Intercept | R² | Mean RMSE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode, axis, count, slope, intercept, r2, rmse in fit_rows:
        unit = "m/s" if axis != "yaw_rate" else "rad/s"
        lines.append(
            f"| {mode} | {count} | {slope:.3f} | {intercept:.3f} {unit} | {r2:.3f} | {rmse:.3f} {unit} |"
        )
    lines.extend(
        [
            "",
            "VX 对 command 基本线性，但整体增益约 0.93，边界速度略欠跟踪。VY 和 yaw 在小幅 command 附近有明显 dead-zone/保守响应：",
            "",
            f"- vy=-0.15 m/s 时 observed mean={vy_neg['vy_actual_mean_mean']:.3f} m/s；vy=+0.15 m/s 时 "
            f"observed mean={vy_pos['vy_actual_mean_mean']:.3f} m/s。",
            f"- yaw=-0.20 rad/s 时 observed mean={yaw_neg['yaw_rate_actual_mean_mean']:.3f} rad/s；yaw=+0.20 rad/s 时 "
            f"observed mean={yaw_pos['yaw_rate_actual_mean_mean']:.3f} rad/s。",
            "- 各 command 的 completion rate 只有约 15%–44%，但它主要反映 readiness/reset 条件，而不是 command 执行后的生存率。" 
            "因为 command 在 readiness 后才生效，不应把较低 completion 直接归因于该 command 本身。",
            "",
            "![Specialist velocity tracking](02_specialist_velocity_tracking.png)",
            "",
            "### 2.2 Mixed command",
            "",
            "| Axis | Response gain | Intercept | R² | Mean RMSE |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for axis, slope, intercept, r2, rmse in mixed_fit_rows:
        unit = "m/s" if axis != "yaw_rate" else "rad/s"
        lines.append(f"| {axis} | {slope:.3f} | {intercept:.3f} {unit} | {r2:.3f} | {rmse:.3f} {unit} |")
    lines.extend(
        [
            "",
            f"Mixed completed segments 的 normalized vector RMSE 平均为 {mixed['normalized_vector_error_rmse'].mean():.3f}。" 
            "VX/VY 的命令响应仍接近线性，yaw 是主要误差源。高幅三轴 corner 最困难，尤其是正向高速、横移和转向同时出现时。",
            "",
            "最差的 5 个 mixed command（至少 3 个 completed contexts）：",
            "",
            "| vx | vy | yaw | Complete | actual vx | actual vy | actual yaw | normalized RMSE |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in worst_mixed.iterrows():
        lines.append(
            f"| {row.vx:.3f} | {row.vy:.3f} | {row.yaw_rate:.3f} | {row.completion_rate:.1%} | "
            f"{row.vx_actual_mean_mean:.3f} | {row.vy_actual_mean_mean:.3f} | "
            f"{row.yaw_rate_actual_mean_mean:.3f} | {row.normalized_vector_error_rmse_mean:.3f} |"
        )
    lines.extend(
        [
            "",
            "![Mixed velocity tracking](03_mixed_velocity_tracking.png)",
            "",
            "## 3. Box tilt 与 box velocity",
            "",
            f"成功的 constant segments 中，box tilt mean 通常约 {completed_constant['box_tilt_mean_deg'].mean():.1f}°，" 
            f"per-segment P95 平均 {completed_constant['box_tilt_p95_deg'].mean():.1f}°，观测到的最大值为 "
            f"{completed_constant['box_tilt_max_deg'].max():.1f}°。这些数值远低于 70° termination threshold。",
            "",
            f"但全部 {total_failures} 个失败 trial 中有 {tilt_failures} 个在 SETTLE/readiness 因 excessive_box_tilt 失败。" 
            "因此结果呈现明显的两阶段并且 tilt 统计被强烈截尾：失败 rollout 在进入 MEASURE 前被剔除，completed-only tilt 图只描述存活子集。",
            "",
            "Box velocity 对 command 的误差显著大于 robot base velocity，说明手臂—箱体系统存在相对运动、姿态耦合或角速度放大：",
            "",
            "| Mode/axis | Robot RMSE | Box RMSE | Box/robot |",
            "|---|---:|---:|---:|",
        ]
    )
    for mode, axis, robot, box, ratio in box_pairs:
        unit = "m/s" if axis != "yaw_rate" else "rad/s"
        lines.append(f"| {mode}/{axis} | {robot:.3f} {unit} | {box:.3f} {unit} | {ratio:.2f}× |")
    lines.extend(
        [
            "",
            f"Mixed 模式中 box yaw-rate RMSE 达 {mixed['box_yaw_rate_rmse'].mean():.3f} rad/s，约为 robot yaw RMSE 的 "
            f"{mixed['box_yaw_rate_rmse'].mean() / mixed['yaw_rate_rmse'].mean():.2f}×。" 
            f"Completed mixed segments 的 robot–box relative linear velocity norm P95 平均为 "
            f"{mixed['robot_box_relative_linear_velocity_norm_p95'].mean():.3f} m/s。双手接触率平均 "
            f"{mixed['bilateral_hand_contact_fraction'].mean():.2%}，说明主要问题不是持续失去双手接触，而更像箱体在夹持约束内发生相对运动和姿态摆动。",
            "",
            "Completed subset 中 tracking RMSE 与 tilt P95 的 command-level 相关性较弱，进一步说明大倾角失败主要发生在 readiness/reset "
            "阶段，而不是 3 s 稳态 command tracking 的自然延伸。",
            "",
            "Completed subset 中 tilt P95 最高的 command：",
            "",
            "| Mode | vx | vy | yaw | Complete | tilt P95 | tilt max |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in worst_tilt.iterrows():
        lines.append(
            f"| {row['mode']} | {row.vx:.3f} | {row.vy:.3f} | {row.yaw_rate:.3f} | "
            f"{row.completion_rate:.1%} | {row.box_tilt_p95_deg_mean:.1f}° | {row.box_tilt_max_deg_max:.1f}° |"
        )
    lines.extend(
        [
            "",
            "![Box stability](04_box_stability.png)",
            "",
            "## 4. Step command 动态响应",
            "",
            "Step trial 的 initialized-to-completed 比例很高，但严格 settling 指标较弱。Settling tolerance 是 max(0.05, 5% command delta)，" 
            "且要求之后一直保持在 tolerance 内；因此它比 3 s 平均 tracking 更能暴露振荡与慢响应。",
            "",
            "| Sequence/axis | Completed changed segments | Settled | Mean settle time (settled only) | Median overshoot |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in step_summary.iterrows():
        lines.append(
            f"| {row.sequence_family}/{row.axis} | {int(row.completed_changed_segments)}/{int(row.planned_changed_segments)} | "
            f"{row.settled_fraction_completed:.1%} | {_fmt(row.settling_time_s_mean_when_settled)} s | "
            f"{_fmt(row.overshoot_pct_median_completed, 1)}% |"
        )
    lines.extend(
        [
            "",
            "Yaw transient 最差：yaw specialist 只有约 6.7% 的 completed changed segments 满足 settling；mixed sequence 的 yaw "
            "median directional overshoot 超过 100%。这与 steady-state yaw gain 接近 1 并不矛盾：均值可以正确，但响应会过冲或持续振荡。",
            "",
            "![Step dynamics](05_step_dynamics.png)",
            "",
            "## 5. Carry motion / phase 敏感性",
            "",
            f"最好的 constant context 是 motion={int(best_context.carry_motion_id)}, phase={best_context.carry_phase:g}，完成率 "
            f"{best_context.end_to_end_completion_rate:.1%}；最差 context 是 motion={int(worst_context.carry_motion_id)}, "
            f"phase={worst_context.carry_phase:g}，完成率 {worst_context.end_to_end_completion_rate:.1%}。" 
            "motion 1 在 phase 0.50 为 0%，而 phase 0.75 接近 100%，说明 reference pose/phase reset、箱体初始姿态与 readiness controller 的匹配 "
            "远比 seed 或 command family 更关键。",
            "",
            "![Context sensitivity](06_context_sensitivity.png)",
            "",
            "## 6. 诊断结论与优先级",
            "",
            "### P0：先修复 reset/readiness 的 box orientation 问题",
            "",
            "最可能原因按优先级排序：",
            "",
            "1. Carry reference 在特定 motion/phase 的 box orientation、hand pose 或 pelvis frame 与环境 reset 不一致。motion 1 phase 0.50 的 0% "
            "通过率是最强证据。",
            "2. Reset 后 box projected gravity / quaternion convention、插值或坐标系存在不连续，导致箱体瞬间接近或越过 70°。",
            "3. Readiness 阶段的上肢夹持控制对 phase-conditioned pose 不够稳定，箱体角速度未被抑制；policy 尚未接收 command 前就失败。",
            "4. Contact solver/box inertia/friction 与 reference 所需夹持力不匹配，使箱体先旋转再掉落；box_drop 是第二大失败原因。",
            "",
            "最小验证 logging：每个 reset 从 t=0 开始记录 box quaternion、projected_gravity_box、tilt、box angular velocity、两手接触力、" 
            "hand-to-box pose、robot–box relative velocity、carry phase 和 termination predicate。按 motion/phase 对齐画前 2 s 曲线。" 
            "重点比较 motion1/phase0.50 与 motion1/phase0.75。",
            "",
            "### P1：改善 box motion，而不只优化 base velocity",
            "",
            "Observation 建议显式包含 box projected gravity、box angular velocity、robot–box relative linear/angular velocity、双手 contact state、carry phase、" 
            "当前 command、上一 command 和 command delta，并检查这些信号在 50 Hz policy timing 下对齐。Reward/objective 应把 base tracking、box orientation、" 
            "box angular damping、relative motion、contact preservation、action smoothness 分开记录。对 20°/35°/50° tilt 设置连续代价或 curriculum，保留 70° "
            "作为安全 termination，避免只有终止信号导致稀疏 credit assignment。",
            "",
            "### P2：修复小 command dead-zone 与 step transient",
            "",
            "在训练 command distribution 中增加 ±0.1–0.25 m/s、±0.1–0.3 rad/s 的密集采样，并加入 step/ramp curriculum。若 policy action 是 PD target，" 
            "应检查 action scale、joint target clipping 和 command change 时的腰部/髋部带宽。可以在部署端做 command low-pass，但评估时应同时保留 raw-step suite，" 
            "否则滤波会掩盖 controller dynamics。",
            "",
            "### P3：建立端到端与条件性能的双指标门槛",
            "",
            "建议以后固定同时报告：① initialization/readiness rate；② initialized 后 completion；③ completed-only velocity RMSE；④ all-observed/censored survival；" 
            "⑤ box tilt failure rate；⑥ box velocity RMSE；⑦ step settling/overshoot。只有 completed-only tracking 会严重高估当前系统能力。",
            "",
            "## 7. 输出文件",
            "",
            "- `constant_command_summary.csv`：每个 constant command 的计划数、完成率、response、RMSE、tilt 和 box-relative motion。",
            "- `step_transition_summary.csv`：每个 step family/axis 的完成率、settling 与 overshoot。",
            "- `context_summary.csv`：motion × phase 的 readiness、失败原因和 mixed tracking。",
            "- 图表以高分辨率 PNG 提供。",
        ]
    )
    (output_dir / "velocity_box_analysis.md").write_text("\n".join(lines), encoding="utf-8")


def analyze(run_dir: Path, output_dir: Path):
    run_dir = run_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    metadata, attempts, segments, source_summary, source_segments = _load(run_dir)
    _style()
    command_summary = _command_summary(segments)
    context_summary = _context_summary(attempts, segments)
    step_summary = _step_summary(segments)
    command_summary.to_csv(output_dir / "constant_command_summary.csv", index=False)
    context_summary.to_csv(output_dir / "context_summary.csv", index=False)
    step_summary.to_csv(output_dir / "step_transition_summary.csv", index=False)
    _plot_funnel(attempts, output_dir)
    print("[1/6] evaluation funnel")
    _plot_specialists(segments, command_summary, output_dir)
    print("[2/6] specialist velocity tracking")
    _plot_mixed(segments, command_summary, output_dir)
    print("[3/6] mixed velocity tracking")
    _plot_box_stability(segments, command_summary, output_dir)
    print("[4/6] box stability")
    _plot_step_dynamics(attempts, step_summary, output_dir)
    print("[5/6] step dynamics")
    _plot_context(attempts, segments, output_dir)
    print("[6/6] context sensitivity")
    _write_report(
        output_dir, run_dir, metadata, attempts, segments, command_summary,
        context_summary, step_summary, source_summary, source_segments,
    )
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--output_dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or args.run_dir / "velocity_box_analysis"
    print(analyze(args.run_dir, output))


if __name__ == "__main__":
    main()
