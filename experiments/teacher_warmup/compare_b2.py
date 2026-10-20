"""Compare real A5/B2 evaluations with episode-level uncertainty; optionally export training curves."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "legged_gym/legged_gym/scripts"))
from analyze_gait_diagnostics import Bucket
from gait_diagnostics import SIDES


def episode_metrics(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError("Only completed physical evaluations can be compared")
    names = [n for n in manifest["dof_names"] if n.startswith(SIDES) and any(k in n for k in ("hip_", "knee_", "ankle_"))]
    buckets, terminal, eligible = {}, {}, {}
    rewards = {}
    with (directory / "steps.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            ep = int(row["episode"])
            eligible[ep] = eligible.get(ep, False) or bool(int(row["eligible"]))
            if int(row["done"]):
                terminal[ep] = int(row["reason"])
            if not int(row["student_valid"]) or int(row["done"]):
                continue
            for key, value in row.items():
                if key.startswith(("reward_raw/", "reward_weighted/")):
                    number = float(value)
                    if not math.isfinite(number):
                        raise ValueError("Nonfinite reward: " + key)
                    record = rewards.setdefault(key, [0., 0., 0])
                    record[0] += number
                    record[1] = max(record[1], abs(number))
                    record[2] += 1
            if row["gait_window"] == "student_3_15s":
                buckets.setdefault(ep, Bucket(manifest["dt"], names)).add(row)
    if len(terminal) != manifest["episodes"]:
        raise ValueError("Missing terminal episodes")
    metrics = {"end_to_end_completion": [float(terminal[e] == 4) for e in sorted(terminal)],
               "handoff_eligibility": [float(eligible[e]) for e in sorted(terminal)],
               "conditional_completion": [float(terminal[e] == 4) for e in sorted(terminal) if eligible[e]]}
    # Episode means / complete-bout medians avoid pretending 50-Hz frames are independent.
    for ep in sorted(buckets):
        r = buckets[ep].report()
        values = {"vx_rmse": r["tracking_rmse"]["vx"], "yaw_rmse": r["tracking_rmse"]["yaw"],
                  "step_frequency_hz": r["step_frequency_hz"],
                  "leg_limiter_activation": r["leg_limiter_joint_sample_ratio"]}
        for key, source in (("bilateral_contact", "physical_bilateral"), ("box_tilt_deg", "tilt"),
                            ("box_relative_velocity", "relative"), ("posture_error", "posture_error")):
            values[key] = r["preservation"].get(source, {}).get("mean")
        for side in SIDES:
            f = r["feet"][side]
            for key, value in {"swing_duration_s": f["swing_duration"]["mean"],
                               "swing_peak_clearance_p50_m": f["swing_peak_clearance"]["p50"],
                               "noncontact_scuff_rate": f["noncontact_scuff_rate"],
                               "contact_speed_m_s": f["contact_speed"]["mean"],
                               "contact_duty_factor": f["contact_duty_factor"]}.items():
                values[side + "_" + key] = value
        for key, value in values.items():
            if value is not None:
                metrics.setdefault(key, []).append(value)
    reward_summary = {key: dict(mean=value[0]/value[2], max_abs=value[1], samples=value[2])
                      for key, value in rewards.items()}
    return manifest, metrics, reward_summary


def compare(baseline, candidate, allow_small_sample=False):
    a, x, _ = episode_metrics(Path(baseline))
    b, y, rewards = episode_metrics(Path(candidate))
    synthetic = bool(a.get("synthetic_test_fixture") or b.get("synthetic_test_fixture"))
    if synthetic and not allow_small_sample:
        raise ValueError("Synthetic fixtures cannot be a final physical comparison")
    for key in ("teacher_sha256", "reference_sha256", "seed", "carry_seconds", "num_envs", "physical_reset_configuration"):
        if key not in a or key not in b or a[key] != b[key]:
            raise ValueError("Comparison condition differs or is missing: " + key)
    if a["handoff"] != b["handoff"] or a["gait_diagnostics"] != b["gait_diagnostics"]:
        raise ValueError("Handoff/contact/geometry diagnostics must match exactly")
    if not allow_small_sample and min(a["episodes"], b["episodes"]) < 100:
        raise ValueError("Final comparison requires >=100 episodes per checkpoint")
    rng = np.random.RandomState(1)
    rows = []
    for name in sorted(set(x) | set(y)):
        left, right = np.asarray(x.get(name, [])), np.asarray(y.get(name, []))
        row = dict(metric=name, a5_episodes=len(left), b2_episodes=len(right),
                   a5_mean=float(left.mean()) if len(left) else None,
                   b2_mean=float(right.mean()) if len(right) else None,
                   delta=None, delta_ci95=None, uncertain=True)
        if len(left) and len(right):
            delta = right[rng.randint(len(right), size=(4000, len(right)))].mean(-1) - left[
                rng.randint(len(left), size=(4000, len(left)))].mean(-1)
            lo, hi = np.quantile(delta, [.025, .975])
            row.update(delta=float(right.mean()-left.mean()), delta_ci95=[float(lo), float(hi)],
                       uncertain=bool(lo <= 0 <= hi))
            # Degenerate all-success Bernoulli bootstraps otherwise suggest false certainty.
            if name in ("end_to_end_completion", "handoff_eligibility", "conditional_completion"):
                def wilson(v):
                    n, p, z = len(v), v.mean(), 1.96
                    center = (p+z*z/(2*n))/(1+z*z/n)
                    radius = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
                    return [float(center-radius), float(center+radius)]
                ai, bi = wilson(left), wilson(right)
                row.update(a5_ci95=ai, b2_ci95=bi, delta_ci95=[bi[0]-ai[1], bi[1]-ai[0]],
                           uncertain=bool(bi[0]-ai[1] <= 0 <= bi[1]-ai[0]))
        rows.append(row)
    return dict(baseline=str(baseline), candidate=str(candidate), metrics=rows, reward_magnitudes=rewards,
                synthetic_test_fixture=synthetic,
                final_sample_size_satisfied=min(a["episodes"], b["episodes"]) >= 100,
                uncertainty="Independent episode bootstrap, 4000 resamples; Wilson envelopes for binary rates. "
                            "Gait uses all episodes reaching 3-15s, including later failures; inspect exposure/survivor bias. "
                            "Intervals are descriptive and do not account for checkpoint selection or multiple comparisons.",
                recommended_checkpoint=None,
                selection="Review completion, grasp, tracking and gait together. No automatic selection from gait rewards.")


def export_curves(run, output):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    accumulator = EventAccumulator(str(run), size_guidance={"scalars": 0}).Reload()
    tags = [t for t in accumulator.Tags()["scalars"] if t.startswith(
        ("TrackingRMSE/", "Episode/", "Physics/reward_", "Handoff/valid_ratio", "Handoff/skipped_update"))]
    with (output / "training_scalars.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["tag", "iteration_zero_based", "value"])
        for tag in tags:
            for event in accumulator.Scalars(tag):
                writer.writerow([tag, event.step, event.value])
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)
    for tag in tags:
        panel = (0 if tag.startswith("TrackingRMSE/") else 1 if tag.startswith("Episode/")
                 and any(k in tag for k in ("pickup", "handoff", "reason_carry_timeout")) else
                 2 if tag.startswith("Physics/reward_weighted/gait_") else None)
        if panel is not None:
            events = accumulator.Scalars(tag)
            axes[panel].plot([e.step+1 for e in events], [e.value for e in events], label=tag)
    for ax, title in zip(axes, ("Tracking RMSE", "Physical episode outcomes", "Weighted gait contributions per policy step")):
        ax.set(title=title, xlabel="Additional B2 iteration")
        if ax.lines:
            ax.legend(fontsize=7)
    fig.savefig(output / "training_curves.png", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--run", type=Path)
    p.add_argument("--allow_small_sample", action="store_true", help="Intermediate checks only; not a final comparison")
    args = p.parse_args()
    result = compare(args.baseline, args.candidate, args.allow_small_sample)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    if args.run:
        export_curves(args.run, args.output)
    print(args.output / "comparison.json")


if __name__ == "__main__":
    main()
