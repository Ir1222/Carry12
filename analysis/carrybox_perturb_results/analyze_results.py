from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_RESULT_ROOT = Path(r"D:\D_Downloads\results")
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "v2"
V2_REQUIRED_COLUMNS = {
    "trial_id",
    "seed",
    "profile",
    "direction",
    "beta",
    "hold_duration",
    "humanoid_failure",
    "box_failure",
    "timeout",
    "carry_achieved",
    "force_scheduled",
    "contact_loss",
    "final_confirmed_carry",
}
TRACE_COLUMNS = {
    "phase",
    "policy_step",
    "peak_force_N",
    "actual_force_scale",
    "force_impulse_Ns",
    "left_hand_contact_proxy",
    "right_hand_contact_proxy",
    "confirmed_carry",
    "max_hand_box_relative_speed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze CarryBox evaluator CSV v2 results.")
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def discover_runs(result_root: Path) -> list[tuple[str, Path]]:
    if (result_root / "summary.csv").is_file():
        return [(result_root.parent.name or result_root.name, result_root)]

    runs = []
    if result_root.is_dir():
        for child in sorted(path for path in result_root.iterdir() if path.is_dir()):
            run_dir = child / "metrics_v2"
            if not (run_dir / "summary.csv").is_file():
                run_dir = child
            if (run_dir / "summary.csv").is_file():
                runs.append((child.name, run_dir))
    if not runs:
        raise FileNotFoundError(
            f"No CSV v2 runs found below {result_root}. Expected summary.csv either "
            "directly or under <run>/metrics_v2/."
        )
    return runs


def read_summaries(runs: list[tuple[str, Path]]) -> pd.DataFrame:
    frames = []
    for policy, run_dir in runs:
        path = run_dir / "summary.csv"
        frame = pd.read_csv(path)
        missing = sorted(V2_REQUIRED_COLUMNS - set(frame.columns))
        if missing:
            raise ValueError(
                f"{path} is not a CarryBox CSV v2 summary; missing columns={missing}"
            )
        frame.insert(0, "policy", policy)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined["condition_id"] = (
        combined["seed"].astype(str)
        + "|"
        + combined["profile"].astype(str)
        + "|"
        + combined["direction"].astype(str)
        + "|"
        + combined["beta"].astype(str)
        + "|"
        + combined["hold_duration"].astype(str)
    )
    return combined


def _safe_max(series: pd.Series) -> float:
    return float(series.max(skipna=True)) if not series.empty else math.nan


def _safe_mean(series: pd.Series) -> float:
    return float(series.mean(skipna=True)) if not series.empty else math.nan


def read_trace_metrics(runs: list[tuple[str, Path]]) -> pd.DataFrame:
    rows = []
    for policy, run_dir in runs:
        for path in sorted((run_dir / "traces").glob("T*.csv")):
            if path.stat().st_size == 0:
                rows.append({"policy": policy, "trial_id": path.stem, "trace_rows": 0})
                continue
            trace = pd.read_csv(path, usecols=lambda column: column in TRACE_COLUMNS)
            force = trace[trace["phase"].eq("force")]
            response = trace[trace["phase"].isin(["force", "post_force"])]
            force_norm = (
                force["peak_force_N"] * force["actual_force_scale"]
                if not force.empty
                else pd.Series(dtype=float)
            )
            both_contact = (
                response["left_hand_contact_proxy"].eq(1)
                & response["right_hand_contact_proxy"].eq(1)
            ) if not response.empty else pd.Series(dtype=bool)
            rows.append(
                {
                    "policy": policy,
                    "trial_id": path.stem,
                    "trace_rows": int(len(trace)),
                    "force_trace_present": int(not force.empty),
                    "response_rows": int(len(response)),
                    "peak_external_force_N": _safe_max(force_norm),
                    "final_trace_impulse_Ns": _safe_max(trace["force_impulse_Ns"]),
                    "both_contact_response_rate": (
                        _safe_mean(both_contact.astype(float))
                        if not response.empty
                        else math.nan
                    ),
                    "any_contact_loss_response": (
                        int((~both_contact).any()) if not response.empty else math.nan
                    ),
                    "peak_response_rel_speed_mps": (
                        _safe_max(response["max_hand_box_relative_speed"])
                        if not response.empty
                        else math.nan
                    ),
                    "mean_response_rel_speed_mps": (
                        _safe_mean(response["max_hand_box_relative_speed"])
                        if not response.empty
                        else math.nan
                    ),
                    "final_response_confirmed_carry": (
                        int(response["confirmed_carry"].iloc[-1])
                        if not response.empty
                        else math.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def make_key_tables(summary: pd.DataFrame, joined: pd.DataFrame) -> dict[str, pd.DataFrame]:
    aggregations = {
        "trials": ("trial_id", "count"),
        "humanoid_failure_rate": ("humanoid_failure", "mean"),
        "box_failure_rate": ("box_failure", "mean"),
        "timeout_rate": ("timeout", "mean"),
        "carry_achieved_rate": ("carry_achieved", "mean"),
        "force_scheduled_rate": ("force_scheduled", "mean"),
        "contact_response_trials": ("contact_loss", "count"),
        "contact_loss_rate": ("contact_loss", "mean"),
        "final_confirmed_carry_rate": ("final_confirmed_carry", "mean"),
        "median_max_rel_speed": ("max_hand_box_relative_speed", "median"),
    }
    overview = summary.groupby(["policy", "profile"]).agg(**aggregations).reset_index()
    by_beta = summary.groupby(["policy", "profile", "beta"]).agg(
        **aggregations
    ).reset_index()
    by_direction = summary.groupby(["policy", "profile", "direction"]).agg(
        **aggregations
    ).reset_index()

    trace_aggregations = {
        "trials": ("trial_id", "count"),
        "force_scheduled_rate": ("force_scheduled", "mean"),
        "humanoid_failure_rate": ("humanoid_failure", "mean"),
        "box_failure_rate": ("box_failure", "mean"),
        "contact_loss_rate": ("contact_loss", "mean"),
    }
    if "peak_external_force_N" in joined:
        trace_aggregations.update(
            median_peak_external_force_N=("peak_external_force_N", "median"),
            median_peak_rel_speed=("peak_response_rel_speed_mps", "median"),
            median_both_contact_response_rate=(
                "both_contact_response_rate",
                "median",
            ),
        )
    trace_overview = joined.groupby(["policy", "profile"]).agg(
        **trace_aggregations
    ).reset_index()
    return {
        "overview.csv": overview,
        "by_beta.csv": by_beta,
        "by_direction.csv": by_direction,
        "trace_overview.csv": trace_overview,
    }


def plot_outcome_rates(summary: pd.DataFrame, output_dir: Path) -> None:
    metrics = [
        ("humanoid_failure", "Humanoid failure"),
        ("box_failure", "Box failure"),
        ("contact_loss", "Contact loss | response"),
        ("carry_achieved", "Carry achieved"),
        ("final_confirmed_carry", "Final confirmed carry"),
    ]
    aggregate = summary.groupby("policy")[[metric for metric, _ in metrics]].mean()
    x = np.arange(len(aggregate.index))
    width = 0.16
    fig, axis = plt.subplots(figsize=(11, 5.8))
    for index, (metric, label) in enumerate(metrics):
        axis.bar(x + (index - 2) * width, aggregate[metric], width, label=label)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Rate")
    axis.set_title("CarryBox CSV v2 outcome rates")
    axis.set_xticks(x)
    axis.set_xticklabels(aggregate.index, rotation=12, ha="right")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "01_outcome_rates_v2.png", dpi=180)
    plt.close(fig)


def plot_beta_curves(summary: pd.DataFrame, output_dir: Path) -> None:
    metrics = [
        ("humanoid_failure", "Humanoid failure"),
        ("box_failure", "Box failure"),
        ("contact_loss", "Contact loss | response"),
        ("carry_achieved", "Carry achieved"),
        ("force_scheduled", "Force scheduled"),
        ("final_confirmed_carry", "Final confirmed carry"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    for axis, (metric, title) in zip(axes.flat, metrics):
        for policy, group in summary.groupby("policy"):
            curve = group.groupby("beta")[metric].mean()
            axis.plot(curve.index, curve.values, marker="o", label=policy)
        axis.set_title(title)
        axis.set_ylim(0, 1)
        axis.set_xlabel("beta")
        axis.grid(alpha=0.25)
    axes[0, 0].set_ylabel("Rate")
    axes[1, 0].set_ylabel("Rate")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("CarryBox CSV v2 outcomes vs perturbation strength")
    fig.tight_layout()
    fig.savefig(output_dir / "02_beta_curves_v2.png", dpi=180)
    plt.close(fig)


def plot_heatmaps(summary: pd.DataFrame, output_dir: Path) -> None:
    metrics = (
        ("humanoid_failure", "humanoid_failure"),
        ("box_failure", "box_failure"),
        ("contact_loss", "contact_loss_response"),
    )
    policies = list(summary["policy"].drop_duplicates())
    for metric, suffix in metrics:
        fig, axes = plt.subplots(
            1, len(policies), figsize=(6 * len(policies), 4.8), squeeze=False
        )
        image = None
        for axis, policy in zip(axes[0], policies):
            subset = summary[summary["policy"].eq(policy)]
            table = subset.pivot_table(
                index="direction", columns="beta", values=metric, aggfunc="mean"
            ).sort_index()
            image = axis.imshow(
                table.values, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1
            )
            axis.set_title(f"{policy}: {suffix}")
            axis.set_xticks(np.arange(len(table.columns)))
            axis.set_xticklabels(table.columns)
            axis.set_yticks(np.arange(len(table.index)))
            axis.set_yticklabels(table.index)
            axis.set_xlabel("beta")
            axis.set_ylabel("direction")
            for row in range(table.shape[0]):
                for column in range(table.shape[1]):
                    value = table.iat[row, column]
                    axis.text(
                        column,
                        row,
                        "" if pd.isna(value) else f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )
        if image is not None:
            fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.88, label="Rate")
        fig.savefig(
            output_dir / f"03_heatmap_{suffix}_v2.png", dpi=180, bbox_inches="tight"
        )
        plt.close(fig)


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for values in frame.itertuples(index=False, name=None):
        formatted = []
        for value in values:
            if isinstance(value, float):
                formatted.append("n/a" if pd.isna(value) else f"{value:.3f}")
            else:
                formatted.append(str(value))
        rows.append("| " + " | ".join(formatted) + " |")
    return "\n".join([header, divider, *rows])


def write_report(tables: dict[str, pd.DataFrame], output_dir: Path, result_root: Path) -> None:
    overview = tables["overview.csv"]
    report = [
        "# CarryBox evaluator CSV v2 analysis",
        "",
        f"Source: `{result_root}`",
        "",
        "The v2 schema separates humanoid failure, box failure, and timeout. "
        "Contact-loss rates exclude trials without a force-response window because "
        "their values are NaN.",
        "",
        "## Outcome overview",
        "",
        markdown_table(overview),
        "",
        "## Metric semantics",
        "",
        "- `humanoid_failure`: humanoid fall/contact termination; timeout is excluded.",
        "- `box_failure`: confirmed carry followed by a sustained ground drop, box "
        "speed instability, or enabled box-tilt termination.",
        "- `carry_achieved`: confirmed carry occurred at least once.",
        "- `contact_loss`: either hand proxy was absent during FORCE/POST_FORCE; "
        "undefined trials are excluded from its mean.",
    ]
    (output_dir / "carrybox_perturb_analysis_v2.md").write_text(
        "\n".join(report), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = discover_runs(args.result_root)
    summary = read_summaries(runs)
    trace_metrics = read_trace_metrics(runs)
    joined = (
        summary.merge(trace_metrics, on=["policy", "trial_id"], how="left")
        if not trace_metrics.empty
        else summary.copy()
    )

    summary.to_csv(args.output_dir / "combined_summary_v2.csv", index=False)
    trace_metrics.to_csv(args.output_dir / "trace_metrics_v2.csv", index=False)
    joined.to_csv(args.output_dir / "combined_with_trace_v2.csv", index=False)
    tables = make_key_tables(summary, joined)
    for name, table in tables.items():
        table.to_csv(args.output_dir / name, index=False)

    plot_outcome_rates(summary, args.output_dir)
    plot_beta_curves(summary, args.output_dir)
    plot_heatmaps(summary, args.output_dir)
    write_report(tables, args.output_dir, args.result_root)
    print(f"Wrote CarryBox CSV v2 analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
