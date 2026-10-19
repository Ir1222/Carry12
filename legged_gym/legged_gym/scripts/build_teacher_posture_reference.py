"""Build waist/shoulder/elbow/wrist soft intervals from successful FULL rollouts."""
import argparse
import csv
import json
from pathlib import Path

import torch

from rsl_rl.teacher_handoff import file_sha256, joint_groups
from rsl_rl.teacher_handoff_reference import validate_reference


def build(directory, output, margin=0.05, softness=0.20):
    directory, output = Path(directory), Path(output)
    if output.exists():
        raise FileExistsError(output)
    if margin < 0 or softness <= 0:
        raise ValueError("Require margin>=0 and softness>0 (radians)")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["kind"] != "full_teacher_physical_pickup":
        raise ValueError("Use evaluator --collect_teacher; student samples are not references")
    # Two passes keep large per-joint CSVs out of RAM. Only selected pose/size
    # arrays from successful episodes are retained for exact quantiles.
    episodes = {}
    with (directory / "steps.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            episodes[row["episode"]] = (row["done"], row["reason"])
    successful = {episode for episode, status in episodes.items() if status == ("1", "4")}
    names = manifest["dof_names"]
    groups = joint_groups(names)
    upper = [names[i] for i in groups["waist"] + groups["arms"]]
    selected, sizes = [], []
    with (directory / "steps.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if (row["episode"] in successful and row["ready"] == "1" and row["student_valid"] == "1"
                    and float(row["handoff_relative_s"]) >= 1.0):
                selected.append([float(row["q:" + name]) for name in upper])
                sizes.append([float(row["box_" + axis]) for axis in "xyz"])
    if len(selected) < 100 or len(successful) < 3:
        raise ValueError("Need >=3 successful full episodes and >=100 stable samples")
    q = torch.tensor(selected)
    box_sizes = torch.tensor(sizes)
    if not torch.isfinite(q).all() or not torch.isfinite(box_sizes).all():
        raise ValueError("Nonfinite teacher rollout")
    lower, upper_q = torch.quantile(q, 0.05, dim=0) - margin, torch.quantile(q, 0.95, dim=0) + margin
    reference = {"schema": 1, "source_kind": manifest["kind"],
                 "teacher_sha256": manifest["teacher_sha256"],
                 "source_steps_sha256": file_sha256(directory / "steps.csv"),
                 "source_manifest": manifest, "sample_count": len(selected),
                 "successful_episodes": len(successful), "joint_names": upper,
                 "lower": lower.tolist(), "upper": upper_q.tolist(),
                 "softness": [softness] * len(upper),
                 "median": q.median(dim=0).values.tolist(),
                 "box_size_min": box_sizes.min(dim=0).values.tolist(),
                 "box_size_max": box_sizes.max(dim=0).values.tolist(),
                 "interval": "pooled P05/P95 + margin", "margin_rad": margin}
    validate_reference(reference, names)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(reference, indent=2), encoding="utf-8")
    print(f"[Reference] {output}: {len(selected)} samples / {len(successful)} successful episodes")
    return reference


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--softness", type=float, default=0.20)
    args = parser.parse_args()
    build(args.directory, args.output, args.margin, args.softness)
