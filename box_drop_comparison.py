from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(r"D:\D_Downloads")
OUT = Path(r"D:\AAAProject\0806\PhysHSI\box_drop_analysis")
OUT.mkdir(parents=True, exist_ok=True)

FILES = {
    "Aug20 policy": ROOT / "Aug20_full_sweep.csv",
    "Aug29 policy": ROOT / "Aug29_full_sweep.csv",
}

frames = []
for policy, path in FILES.items():
    df = pd.read_csv(path)
    df["policy"] = policy
    df["drop_flag"] = df["box_failure"].astype(int)
    frames.append(df)
data = pd.concat(frames, ignore_index=True)

summary = (
    data.groupby(["policy", "beta", "direction"], as_index=False)
    .agg(drop_count=("drop_flag", "sum"), trials=("drop_flag", "size"))
)
summary["drop_rate"] = summary["drop_count"] / summary["trials"]
summary.to_csv(OUT / "box_drop_summary_by_policy_beta_direction.csv", index=False)

drops = data[data["drop_flag"] == 1].copy()
drops[
    [
        "policy",
        "trial_id",
        "seed",
        "direction",
        "beta",
        "box_failure_reason",
        "peak_force_N",
        "impulse_Ns",
    ]
].to_csv(OUT / "box_drop_trials.csv", index=False)

overall_beta = (
    data.groupby(["policy", "beta"], as_index=False)
    .agg(drop_count=("drop_flag", "sum"), trials=("drop_flag", "size"))
)
overall_beta["drop_rate"] = overall_beta["drop_count"] / overall_beta["trials"]

direction = (
    data.groupby(["policy", "direction"], as_index=False)
    .agg(drop_count=("drop_flag", "sum"), trials=("drop_flag", "size"))
)
direction["drop_rate"] = direction["drop_count"] / direction["trials"]

sns.set_theme(style="whitegrid", context="talk")
colors = {"Aug20 policy": "#C44E52", "Aug29 policy": "#4C72B0"}
directions = ["+box_x", "-box_x", "+box_y", "-box_y"]
betas = sorted(data["beta"].unique())

# Standalone line chart: overall beta trend.
fig, ax = plt.subplots(figsize=(10, 6.5), dpi=160)
for policy, sub in overall_beta.groupby("policy"):
    sub = sub.sort_values("beta")
    ax.plot(
        sub["beta"],
        sub["drop_rate"] * 100,
        marker="o",
        linewidth=2.6,
        markersize=7,
        color=colors[policy],
        label=policy,
    )
    for _, row in sub.iterrows():
        ax.annotate(
            f"{row.drop_rate * 100:.0f}%",
            (row.beta, row.drop_rate * 100),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=9,
            color=colors[policy],
        )
ax.set_title("Overall box-failure rate by β", weight="bold")
ax.set_xlabel("β")
ax.set_ylabel("Box failure rate (%)")
ax.set_xticks(betas)
ax.set_ylim(0, 70)
ax.legend(frameon=True, loc="upper left")
ax.set_title(
    "Overall box-failure rate by β\n"
    "Failure = box_failure == 1; 40 trials per β",
    weight="bold",
)
fig.tight_layout()
fig.savefig(OUT / "box_drop_line.png", bbox_inches="tight")
plt.close(fig)

# Standalone bar chart: direction summary.
fig, ax = plt.subplots(figsize=(10, 6.5), dpi=160)
wide = direction.pivot(index="direction", columns="policy", values="drop_rate").reindex(directions)
wide = wide[["Aug20 policy", "Aug29 policy"]]
x = range(len(directions))
width = 0.36
for i, policy in enumerate(wide.columns):
    bars = ax.bar(
        [v + (i - 0.5) * width for v in x],
        wide[policy] * 100,
        width,
        label=policy,
        color=colors[policy],
        alpha=0.9,
    )
    for bar, value in zip(bars, wide[policy] * 100):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
ax.set_title("Failure rate by force direction", weight="bold")
ax.set_xlabel("Force direction")
ax.set_ylabel("Box failure rate (%)")
ax.set_xticks(list(x), directions)
ax.set_ylim(0, 55)
ax.legend(frameon=True, loc="upper left")
ax.set_title(
    "Box-failure rate by force direction\n"
    "Failure = box_failure == 1; 70 trials per direction",
    weight="bold",
)
fig.tight_layout()
fig.savefig(OUT / "box_drop_direction_bar.png", bbox_inches="tight")
plt.close(fig)

# Remove the previous combined/reason-only outputs so the folder keeps only the
# two requested standalone figures.
for legacy_name in ["box_drop_comparison.png", "box_drop_failure_reasons.png"]:
    legacy_path = OUT / legacy_name
    if legacy_path.exists():
        legacy_path.unlink()

print(f"Wrote {OUT / 'box_drop_line.png'}")
print(f"Wrote {OUT / 'box_drop_direction_bar.png'}")
print(f"Wrote {OUT / 'box_drop_summary_by_policy_beta_direction.csv'}")
print(f"Wrote {OUT / 'box_drop_trials.csv'}")
print("Overall:")
print(data.groupby("policy")["drop_flag"].agg(["sum", "count", "mean"]).to_string())
