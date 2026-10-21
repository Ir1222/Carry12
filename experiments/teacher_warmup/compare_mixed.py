"""Compare B2 and M1 evaluations produced by the same M1 evaluation harness."""
import argparse
import json
from pathlib import Path


def compare(baseline, candidate):
    baseline, candidate = Path(baseline), Path(candidate)
    manifests = [json.loads((p / "manifest.json").read_text(encoding="utf-8")) for p in (baseline, candidate)]
    for key in ("task", "seed", "episodes", "num_envs", "dt", "carry_seconds", "teacher_sha256",
                "dof_names", "action_scale", "reference_sha256", "rewards", "gait_configuration", "gait_geometry", "physical_reset_configuration", "evaluation_command_override"):
        if any(key not in manifest for manifest in manifests):
            raise ValueError("Missing evaluation condition: " + key)
        if manifests[0][key] != manifests[1][key]:
            raise ValueError("Evaluation conditions differ: " + key)
    if any(m.get("status") != "complete" or m["task"] != "carrybox_teacher_warmup_mixed" for m in manifests):
        raise ValueError("Comparison requires complete evaluations using the M1 harness")
    handoffs = [{k: v for k, v in m["handoff"].items() if k not in ("teacher_checkpoint", "posture_reference")}
                for m in manifests]
    if handoffs[0] != handoffs[1]:
        raise ValueError("Evaluation handoff/command-manager settings differ")
    result = {"baseline": str(baseline), "candidate": str(candidate), "reports": {}}
    for filename in ("summary.json", "mixed_summary.json"):
        rows = [json.loads((p / filename).read_text(encoding="utf-8")) for p in (baseline, candidate)]
        keys = sorted(set(rows[0]) | set(rows[1]))
        result["reports"][filename] = {key: dict(baseline=rows[0].get(key), candidate=rows[1].get(key),
            delta=(rows[1][key] - rows[0][key] if isinstance(rows[0].get(key), (int, float))
                   and isinstance(rows[1].get(key), (int, float)) else None)) for key in keys}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = compare(args.baseline, args.candidate)
    with Path(args.output).open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))
