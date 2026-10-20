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
    parser.add_argument("--handoff_config", help="JSON overrides: handoff, algorithm, rewards, gait, runner, env, train")
    own, remaining = parser.parse_known_args()
    # Isaac Gym must be imported before PyTorch.
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
                   "rewards": env_cfg.rewards.scales, "gait": env_cfg.gait,
                   "runner": train_cfg.runner, "env": env_cfg.env, "train": train_cfg}
        for section, values in overrides.items():
            if section not in targets:
                raise ValueError("Unknown config section: " + section)
            if not isinstance(values, dict):
                raise ValueError("Config section must be an object: " + section)
            for key, value in values.items():
                if (isinstance(value, dict) or key.startswith("_") or not hasattr(targets[section], key)
                        or isinstance(getattr(targets[section], key), type)
                        or callable(getattr(targets[section], key))):
                    raise ValueError("Unknown configuration key: " + key)
                setattr(targets[section], key, value)
    env_cfg.env.episode_length_s = (h.warmup_timeout_s + h.transition_timeout_s + h.student_timeout_s + 5)
    if train_cfg.runner.run_name != "A5_B2_gait_fixed_vx08":
        train_cfg.runner.run_name = own.ablation + "_" + h.command_suite


def validate_b2_initialization(env_cfg, train_cfg, own, args):
    """Pre-simulator audit, only for the explicit B2 experiment. Never substitute."""
    active = any(getattr(env_cfg.rewards.scales, key) != 0 for key in
                 ("gait_foot_slip", "gait_swing_clearance", "gait_foot_scuff", "gait_air_time"))
    if not active and train_cfg.runner.run_name != "A5_B2_gait_fixed_vx08":
        return None
    if train_cfg.runner.run_name != "A5_B2_gait_fixed_vx08":
        raise ValueError("B2 gait training requires run_name=A5_B2_gait_fixed_vx08")
    from rsl_rl.teacher_handoff import checkpoint_actor, file_sha256, read_checkpoint
    from gait_diagnostics import validate_a5_contract, validate_b2_reward_contract
    from legged_gym.utils.helpers import class_to_dict
    root = Path(__file__).resolve().parents[3]
    exact = {
        "teacher": root / "shared_artifacts/2026-10-19/carry_track/model_33999.pt",
        "student": root / "legged_gym/logs/carrybox_teacher_warmup/20261019_235711_985014_A5_fixed/model_5000.pt",
        "reference": root / "experiments/teacher_warmup/results/lab_20261019_233908/full_teacher_upper.json",
    }
    for name, path in exact.items():
        if not path.is_file():
            raise FileNotFoundError("Required exact B2 " + name + " file: " + str(path))
    if Path(own.teacher_checkpoint).resolve() != exact["teacher"].resolve():
        raise ValueError("B2 requires the exact Full Teacher checkpoint")
    if Path(own.posture_reference).resolve() != exact["reference"].resolve():
        raise ValueError("B2 requires the exact A5 posture reference")
    if not args.resume and Path(own.student_checkpoint).resolve() != exact["student"].resolve():
        raise ValueError("B2 requires the exact trained A5 model_5000.pt")
    h = class_to_dict(env_cfg.handoff)
    for key, value in {"ablation": "A5", "adapter": "rate_limited", "warmup_vx": .8,
                       "fixed_command": [.8, 0., 0.], "command_suite": "fixed", "student_timeout_s": 45.,
                       "target_rate_arms": 2., "target_rate_waist": 1.5, "target_rate_legs": 12.,
                       "joint_target_rates": None, "imitation_window_s": 2.}.items():
        if h.get(key) != value:
            raise ValueError("B2 effective handoff mismatch: " + key)
    for key, value in {"learning_rate": 5e-5, "min_valid_samples": 2048, "num_learning_epochs": 5,
                       "num_mini_batches": 4, "handoff_imitation_coef": .5, "imitation_arms_weight": 2.,
                       "imitation_waist_weight": 2., "imitation_legs_weight": .25}.items():
        if getattr(train_cfg.algorithm, key) != value:
            raise ValueError("B2 effective algorithm mismatch: " + key)
    if (train_cfg.runner.num_steps_per_env != 100 or train_cfg.runner.save_interval != 100
            or train_cfg.seed != 1 or args.rl_device != "cuda:0"):
        raise ValueError("B2 requires A5 rollout/save settings, seed=1, rl_device=cuda:0")
    checkpoint = read_checkpoint(exact["student"])
    _, spec = checkpoint_actor(exact["student"])
    if spec["dims"] != [738, 512, 256, 256, 29]:
        raise ValueError("B2 requires the trained 738-[512,256,256]-29 A5 Actor")
    names = [line.split()[1] for line in (root / "legged_gym/resources/config/joint_id.txt").read_text().splitlines() if line.strip()]
    teacher_hash, reference_hash = file_sha256(exact["teacher"]), file_sha256(exact["reference"])
    contract = validate_a5_contract(checkpoint, spec, teacher_hash, reference_hash, h, names,
                                    env_cfg.control.action_scale, env_cfg.sim.dt * env_cfg.control.decimation)
    if contract.get("critic_obs") != 160 or checkpoint["iter"] != 5000:
        raise ValueError("Expected trained A5 iteration 5000 with 160D critic")
    if class_to_dict(train_cfg.algorithm) != contract.get("algorithm"):
        raise ValueError("B2 PPO/imitation configuration differs from saved A5")
    policy = class_to_dict(train_cfg.policy)
    policy.update(actor_hidden_dims=spec["dims"][1:-1], activation="elu")
    if policy != contract.get("policy"):
        raise ValueError("B2 Actor/Critic architecture differs from saved A5")
    rewards = class_to_dict(env_cfg.rewards)
    validate_b2_reward_contract(contract["rewards"], rewards)
    current_environment = {key: class_to_dict(getattr(env_cfg, key)) for key in
                           ("control", "normalization", "domain_rand", "asset", "commands", "dataset", "sim",
                            "noise", "terrain", "init_state")}
    if contract.get("environment") != current_environment:
        raise ValueError("B2 environment/control differs from saved A5")
    result = dict(student_checkpoint=str(exact["student"].resolve()), student_sha256=file_sha256(exact["student"]),
                  teacher_sha256=teacher_hash, reference_sha256=reference_hash,
                  original_iteration=checkpoint["iter"], actor_dims=spec["dims"],
                  original_warmup_vx=contract["handoff"].get("warmup_vx"),
                  original_fixed_command=contract["handoff"].get("fixed_command"),
                  iteration_semantics="model_N.pt = N additional B2 iterations after A5 model_5000.pt")
    print("[B2 provenance] " + json.dumps(result, sort_keys=True))
    return result


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
    provenance = validate_b2_initialization(env_cfg, train_cfg, own, args)
    if provenance is not None:
        env_cfg.seed = train_cfg.seed
    print("[Effective handoff] " + json.dumps({key: getattr(env_cfg.handoff, key) for key in
          ("warmup_vx", "fixed_command", "command_suite", "ablation", "adapter")}, sort_keys=True))
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
    if getattr(env, "gait_enabled", False):
        # Runner persists this dict in both config.json and every checkpoint.
        cfg["environment_contract"].update(gait=class_to_dict(env.cfg.gait), gait_geometry=env.gait_geometry)
        cfg["environment_contract"]["b2_execution"] = dict(num_envs=env.num_envs, seed=env.cfg.seed,
            rl_device=args.rl_device, num_steps_per_env=train_cfg.runner.num_steps_per_env,
            save_interval=train_cfg.runner.save_interval)
        if provenance is not None:
            cfg["environment_contract"]["b2_initialization"] = provenance
        print(f"[B2 effective] num_envs={env.num_envs} dt={env.dt} obs={env.num_obs} "
              f"critic={env.num_privileged_obs} actions={env.num_actions} seed={train_cfg.seed}")
        if (env.dt != .02 or env.num_obs != 738 or env.num_actions != 29 or env.num_privileged_obs != 160
                or env.num_envs != 512 or env.cfg.seed != 1):
            raise ValueError("B2 runtime observation/action/control contract mismatch")
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
        if provenance is not None:
            import torch
            from rsl_rl.teacher_handoff import read_checkpoint
            source_state = read_checkpoint(own.student_checkpoint)["model_state_dict"]
            for key, value in runner.alg.actor_critic.state_dict().items():
                if key.startswith("actor.") or key == "std":
                    if not torch.equal(value, source_state[key].to(value.device)):
                        raise ValueError("A5 Actor/std initialization mismatch: " + key)
            if runner.alg.optimizer.state:
                raise ValueError("B2 optimizer must start fresh")
            print("[B2 init] Actor + std exactly equal A5; optimizer empty; critic newly initialized")
    runner.learn(train_cfg.runner.max_iterations, init_at_random_ep_len=False)


if __name__ == "__main__":
    main()
