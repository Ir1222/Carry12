"""Fixed-seed, full-pickup A0--A5 evaluation with per-joint CSV traces."""
import argparse
import csv
import json
from pathlib import Path
import sys


def parse_args():
    p = argparse.ArgumentParser(allow_abbrev=False, description=__doc__,
                               epilog="Also accepts Isaac Gym flags: --task --num_envs --seed --headless --rl_device")
    p.add_argument("--teacher_checkpoint", required=True)
    p.add_argument("--student_checkpoint", required=True)
    p.add_argument("--ablation", choices=[f"A{i}" for i in range(6)], default="A0")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--carry_seconds", type=float, default=45.0)
    p.add_argument("--command_suite", choices=["fixed", "mixed"], default="fixed")
    p.add_argument("--fixed_command", type=float, nargs=3, default=[0.4, 0, 0])
    p.add_argument("--posture_reference", default="")
    p.add_argument("--handoff_config")
    p.add_argument("--output", required=True)
    p.add_argument("--collect_teacher", action="store_true")
    own, remaining = p.parse_known_args()
    __import__("isaacgym")  # Must initialize before importing torch.
    # Match the existing entrypoints: envs must start registration before utils
    # imports task_registry, which itself imports a config from the envs package.
    __import__("legged_gym.envs")
    from legged_gym.utils import get_args
    saved = sys.argv
    try:
        sys.argv = [saved[0], *remaining]
        args = get_args()
    finally:
        sys.argv = saved
    if args.seed is None:
        args.seed = 1
    if args.num_envs is None:
        args.num_envs = 1
    if own.episodes < 1 or own.carry_seconds < 30:
        raise ValueError("Require episodes > 0 and carry_seconds >= 30 for 3/15/long evaluation")
    args.num_envs = min(args.num_envs, own.episodes)
    return args, own


def main():
    args, own = parse_args()
    import torch
    from legged_gym.envs import task_registry
    from legged_gym.utils.helpers import class_to_dict
    from rsl_rl.teacher_handoff import checkpoint_actor, file_sha256, read_checkpoint
    from train_teacher_warmup_loco import apply_configuration
    if args.task != "carrybox_teacher_warmup":
        raise ValueError("Use --task carrybox_teacher_warmup")
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    apply_configuration(env_cfg, train_cfg, own)
    env_cfg.handoff.student_timeout_s = own.carry_seconds
    env_cfg.env.episode_length_s = (env_cfg.handoff.warmup_timeout_s
                                   + env_cfg.handoff.transition_timeout_s + own.carry_seconds + 5)
    env_cfg.env.test = True
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=env_cfg)
    actor, spec = checkpoint_actor(own.student_checkpoint, env.device)
    if spec["dims"][0] != env.num_obs or spec["dims"][-1] != env.num_actions:
        raise ValueError("Student observation/action contract mismatch")
    saved_contract = read_checkpoint(own.student_checkpoint).get("handoff_contract")
    if saved_contract:
        if saved_contract["teacher_sha256"] != file_sha256(own.teacher_checkpoint):
            raise ValueError("Evaluation teacher differs from the fine-tuning teacher")
        current_handoff = class_to_dict(env.cfg.handoff)
        for key in ("adapter", "target_rate_arms", "target_rate_waist", "target_rate_legs", "joint_target_rates"):
            if saved_contract["handoff"][key] != current_handoff[key]:
                raise ValueError("Evaluation must use the training action adapter: " + key)
        if saved_contract["dof_names"] != list(env.dof_names) or saved_contract["action_scale"] != env.cfg.control.action_scale:
            raise ValueError("Evaluation joint order/action scaling differs from training")
    if own.collect_teacher:
        if env.cfg.handoff.command_suite != "fixed" or env.cfg.handoff.fixed_command != [0.4, 0, 0]:
            raise ValueError("Collect initial reference at fixed [0.4,0,0]")
        actor = env.teacher
    out = Path(own.output)
    out.mkdir(parents=True, exist_ok=False)
    manifest = {"seed": args.seed, "ablation": own.ablation, "episodes": own.episodes,
                "kind": "full_teacher_physical_pickup" if own.collect_teacher else "dual_policy",
                "teacher_sha256": file_sha256(own.teacher_checkpoint),
                "student_sha256": file_sha256(own.student_checkpoint),
                "dof_names": env.dof_names, "handoff": class_to_dict(env.cfg.handoff),
                "dt": env.dt, "actor_spec": spec,
                "action_scale": env.cfg.control.action_scale,
                "posture_reference": own.posture_reference or None,
                "carry_seconds": own.carry_seconds}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    base_fields = ["episode", "env", "time_s", "student_valid", "phase", "control_phase", "handoff_relative_s",
                   "ready", "pickup", "eligible", "warmup_s", "reason", "done",
                   "bilateral", "lift", "tilt", "relative", "omega", "posture_error",
                   "gap_arms", "gap_waist", "gap_legs", "raw_log_probability",
                   "command_vx", "command_vy", "command_yaw", "actual_vx", "actual_vy", "actual_yaw",
                   "box_x", "box_y", "box_z"]
    joint_fields = [f"{kind}:{name}" for kind in ("q", "dq", "raw_action", "applied_action", "pd_target")
                    for name in env.dof_names]
    slots = list(range(env.num_envs))
    ages = [0.0] * env.num_envs
    next_episode = env.num_envs
    completed = 0
    with (out / "steps.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=base_fields + joint_fields)
        writer.writeheader()
        with torch.no_grad():
            obs, _ = env.reset()
            # Include the physical initialization step performed by BaseTask.reset.
            pending = True
            while completed < own.episodes:
                if pending:
                    pending = False
                    dones = env.reset_buf
                    valid = env.control_mask
                    gaps = torch.zeros_like(env.actions)
                else:
                    student = actor(obs)
                    teacher = env.teacher(obs)
                    gaps = (student - teacher).square()
                    obs, _, _, dones, infos, _, _, _ = env.step(student)
                    valid = infos["student_valid"]
                snap = {k: v.cpu() for k, v in env.last_step_snapshot.items()}
                for e, episode in enumerate(slots):
                    if episode is None:
                        continue
                    ages[e] += env.dt
                    row = dict(episode=episode, env=e, time_s=ages[e], student_valid=int(valid[e]),
                               phase=int(snap["phase"][e]),
                               control_phase=int(snap["control_phase"][e]),
                               handoff_relative_s=float(snap["handoff_age"][e]) if bool(snap["eligible"][e]) else -1,
                               ready=int(snap["ready"][e]), pickup=int(snap["pickup"][e]),
                               eligible=int(snap["eligible"][e]), warmup_s=float(snap["warmup_elapsed"][e]),
                               reason=int(snap["failure_reason"][e]), done=int(dones[e]),
                               raw_log_probability="")  # deterministic evaluation: no sampled log probability
                    for key in ("bilateral", "lift", "tilt", "relative", "omega", "posture_error"):
                        row[key] = float(snap[key][e]) if key != "posture_error" or env.pose_reference is not None else ""
                    for group, ids in env.groups.items():
                        row["gap_" + group] = float(gaps[e, ids].mean().sqrt())
                    for i, axis in enumerate(("vx", "vy", "yaw")):
                        row["command_" + axis] = float(snap["command"][e, i])
                        row["actual_" + axis] = float(snap["actual"][e, i])
                    for i, axis in enumerate("xyz"):
                        row["box_" + axis] = float(snap["box_size"][e, i])
                    for kind in ("q", "dq", "raw_action", "applied_action", "pd_target"):
                        row.update({f"{kind}:{name}": float(snap[kind][e, j])
                                    for j, name in enumerate(env.dof_names)})
                    writer.writerow(row)
                    if bool(dones[e]):
                        completed += 1
                        print(f"[Eval] episode={episode} reason={row['reason']} pickup={row['pickup']} "
                              f"eligible={row['eligible']} carry={row['handoff_relative_s']:.2f}s")
                        slots[e] = next_episode if next_episode < own.episodes else None
                        next_episode += 1
                        ages[e] = 0.0
    from analyze_teacher_warmup import analyze
    analyze(out)


if __name__ == "__main__":
    main()
