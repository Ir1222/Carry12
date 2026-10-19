"""Independent training entrypoint; never use random episode ages."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys


def parse_args():
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__,
                                    epilog="Also accepts Isaac Gym flags: --task --num_envs --seed --max_iterations --headless --rl_device --resume --resume_path")
    parser.add_argument("--teacher_checkpoint", required=True)
    parser.add_argument("--student_checkpoint")
    parser.add_argument("--ablation", choices=["A2", "A3", "A4", "A5"], default="A3")
    parser.add_argument("--command_suite", choices=["fixed", "mixed"], default="fixed")
    parser.add_argument("--fixed_command", type=float, nargs=3, default=[0.4, 0.0, 0.0])
    parser.add_argument("--posture_reference", default="")
    parser.add_argument("--handoff_config", help="JSON overrides: handoff, algorithm, rewards")
    own, remaining = parser.parse_known_args()
    # Isaac Gym must be imported before PyTorch.
    __import__("isaacgym")  # Must initialize before importing torch.
    from legged_gym.utils import get_args
    saved = sys.argv
    try:
        sys.argv = [saved[0], *remaining]
        args = get_args()
    finally:
        sys.argv = saved
    return args, own


def apply_configuration(env_cfg, train_cfg, own):
    h = env_cfg.handoff
    h.teacher_checkpoint = str(Path(own.teacher_checkpoint).resolve())
    h.ablation = own.ablation
    h.command_suite = own.command_suite
    h.fixed_command = own.fixed_command
    h.posture_reference = str(Path(own.posture_reference).resolve()) if own.posture_reference else ""
    h.adapter = "rate_limited" if own.ablation in ("A1", "A5") else "hard"
    train_cfg.algorithm.handoff_imitation_coef = 0.0 if own.ablation in ("A0", "A1", "A2") else 0.5
    env_cfg.rewards.scales.carry_upper_body_pose = 0.5 if own.ablation in ("A4", "A5") else 0.0
    if own.handoff_config:
        with open(own.handoff_config, encoding="utf-8") as stream:
            overrides = json.load(stream)
        targets = {"handoff": h, "algorithm": train_cfg.algorithm,
                   "rewards": env_cfg.rewards.scales}
        for section, values in overrides.items():
            if section not in targets:
                raise ValueError("Unknown config section: " + section)
            for key, value in values.items():
                if not hasattr(targets[section], key):
                    raise ValueError("Unknown configuration key: " + key)
                setattr(targets[section], key, value)
    env_cfg.env.episode_length_s = (h.warmup_timeout_s + h.transition_timeout_s + h.student_timeout_s + 5)
    train_cfg.runner.run_name = own.ablation + "_" + h.command_suite


def main():
    args, own = parse_args()
    if args.task != "carrybox_teacher_warmup":
        raise ValueError("Use --task carrybox_teacher_warmup with this entrypoint")
    if args.init_policy_path or args.finetune_path:
        raise ValueError("Use --student_checkpoint for initialization")
    if args.resume and (not args.resume_path or own.student_checkpoint):
        raise ValueError("Resume requires --resume_path and no --student_checkpoint")
    if not args.resume and not own.student_checkpoint:
        raise ValueError("New runs require --student_checkpoint")
    from legged_gym import LEGGED_GYM_ROOT_DIR
    from legged_gym.envs import task_registry
    from legged_gym.utils.helpers import class_to_dict, update_cfg_from_args
    from rsl_rl.runners.teacher_warmup_on_policy_runner import TeacherWarmupOnPolicyRunner
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    apply_configuration(env_cfg, train_cfg, own)
    _, train_cfg = update_cfg_from_args(None, train_cfg, args)
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=env_cfg)
    cfg = class_to_dict(train_cfg)
    cfg["handoff_contract"] = class_to_dict(env.cfg.handoff)
    # Teacher path is recorded separately by content hash, so moving a run is OK.
    cfg["handoff_contract"].pop("teacher_checkpoint")
    cfg["handoff_contract"].pop("posture_reference")
    cfg["reward_contract"] = class_to_dict(env.cfg.rewards)
    cfg["environment_contract"] = {
        key: class_to_dict(getattr(env.cfg, key))
        for key in ("control", "normalization", "domain_rand", "asset", "commands", "dataset", "sim",
                    "noise", "terrain", "init_state")
    }
    root = Path(LEGGED_GYM_ROOT_DIR) / "logs" / train_cfg.runner.experiment_name
    log_dir = root / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + train_cfg.runner.run_name)
    log_dir.mkdir(parents=True, exist_ok=False)
    print(f"[Run] {log_dir}")
    source = args.resume_path if args.resume else own.student_checkpoint
    runner = TeacherWarmupOnPolicyRunner(env, cfg, source, str(log_dir), args.rl_device)
    if args.resume:
        runner.load(args.resume_path)
    else:
        runner.load_init_policy(own.student_checkpoint)
    runner.learn(train_cfg.runner.max_iterations, init_at_random_ep_len=False)


if __name__ == "__main__":
    main()
