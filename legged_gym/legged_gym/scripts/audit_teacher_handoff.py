"""CPU checkpoint audit; tensor shapes cannot establish historical observation semantics."""
import argparse
import json
from pathlib import Path

from rsl_rl.teacher_handoff import checkpoint_actor, file_sha256, read_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher_checkpoint", required=True)
    parser.add_argument("--student_checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    report = {}
    for kind, path in (("teacher", args.teacher_checkpoint), ("student", args.student_checkpoint)):
        actor, spec = checkpoint_actor(path)
        checkpoint = read_checkpoint(path)
        std = checkpoint["model_state_dict"]["std"]
        report[kind] = dict(spec, sha256=file_sha256(path), checkpoint_keys=sorted(checkpoint),
                            std_min=float(std.min()), std_max=float(std.max()),
                            parameter_count=sum(p.numel() for p in actor.parameters()))
    root = Path(__file__).resolve().parents[3]
    report["joint_order"] = (root / "legged_gym/resources/config/joint_id.txt").read_text().splitlines()
    report["source_contract"] = {
        "actor_frame": "108 proprio + 15 task; 6 frames = 738",
        "proprio": "ang_vel(3), gravity(3), q-default(29), dq(29), end_effectors(15), applied_action(29)",
        "task": "box_pos_local(3), box_rot_6d(6), box_size(3), raw command(3)",
        "velocity_frame": "pelvis yaw frame vx/vy; world vertical yaw rate",
        "action_scale": 0.25, "control_dt": 0.02,
        "target": "default_dof_pos + action_scale * clipped/applied_action",
        "limitation": "ELU, joint mapping and normalization are inferred from source, not encoded by tensor shapes"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
