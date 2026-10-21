"""Strict M1 contracts; source provenance is metadata, never a hardcoded model path."""
import math
from pathlib import Path

from rsl_rl.teacher_handoff import checkpoint_actor, file_sha256, read_checkpoint

MIXED_TASK = "carrybox_teacher_warmup_mixed"
MANAGER_KEYS = ("mixed_command_hold_s", "mixed_command_slew_rates", "mixed_command_manager")
ENV_KEYS = ("control", "normalization", "domain_rand", "asset", "commands", "dataset", "sim",
            "noise", "terrain", "init_state")


def validate_m1_contract(checkpoint, spec, expected, resume=False):
    """Compare semantic snapshots, allowing only the intentional B2 -> M1 delta."""
    c = checkpoint.get("handoff_contract")
    if not c or c.get("schema") != 1 or not isinstance(checkpoint.get("iter"), int) or checkpoint["iter"] < (0 if resume else 1):
        raise ValueError("M1 requires a trained checkpoint with a saved handoff contract")
    if (spec["dims"] != [738, 512, 256, 256, 29] or spec["critic_input"] != 160
            or spec["std_shape"] != [29]):
        raise ValueError("M1 requires compatible 738D Actor / 160D Critic / 29D actions")
    if not resume and (not checkpoint.get("optimizer_state_dict", {}).get("state")
            or checkpoint.get("timesteps", 0) <= 0):
        raise ValueError("M1 requires evidence of trained optimizer steps and physical rollout timesteps")
    state = checkpoint["model_state_dict"]
    import torch
    if not all(torch.isfinite(v).all() for k, v in state.items() if k.startswith("actor.") or k == "std"):
        raise ValueError("Nonfinite Actor/std checkpoint")
    if not torch.all(state["std"] > 0):
        raise ValueError("Invalid action std")
    for key in ("teacher_sha256", "reference_sha256", "dof_names", "actor_obs", "critic_obs",
                "dt", "action_scale", "rewards", "algorithm", "policy"):
        if c.get(key) != expected[key]:
            raise ValueError("M1 checkpoint contract mismatch: " + key)
    handoff = dict(expected["handoff"])
    if resume:
        if c.get("task") != MIXED_TASK:
            raise ValueError("M1 resume requires an M1 checkpoint")
    else:
        if c.get("task") not in (None, "carrybox_teacher_warmup") or not c.get("environment", {}).get("b2_initialization"):
            raise ValueError("M1 initialization requires a genuine B2 checkpoint with B2 provenance")
        provenance = c["environment"]["b2_initialization"]
        for key in ("teacher_sha256", "reference_sha256"):
            if provenance.get(key) != expected[key]:
                raise ValueError("B2 initialization provenance mismatch: " + key)
        if provenance.get("actor_dims") != spec["dims"] or provenance.get("original_iteration", 0) <= 0:
            raise ValueError("Missing trained A5 -> B2 initialization provenance")
        for key in MANAGER_KEYS:
            handoff.pop(key)
        handoff.update(command_suite="fixed", fixed_command=[.8, 0., 0.])
    if c.get("handoff") != handoff:
        raise ValueError("M1 checkpoint handoff/adapter contract mismatch")
    environment = c.get("environment") or {}
    current = expected["environment"]
    # Execution size and initialization filenames are provenance, not dynamics.
    for key in ENV_KEYS + ("gait",):
        saved = environment.get(key)
        wanted = current[key]
        if key == "domain_rand" and saved and "push_interval" in saved:
            wanted = dict(wanted, push_interval=math.ceil(wanted["push_interval_s"] / expected["dt"]))
        if saved != wanted:
            raise ValueError("M1 checkpoint environment mismatch: " + key)
    if not environment.get("gait_geometry"):
        raise ValueError("B2/M1 checkpoint must record calibrated gait geometry")
    return c


def validate_m1_training_settings(env_cfg, train_cfg, num_envs=None):
    """M1 execution validation is independent of the legacy fixed-size B2 audit."""
    count = env_cfg.env.num_envs if num_envs is None else num_envs
    for name, value in (("num_envs", count), ("max_iterations", train_cfg.runner.max_iterations)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("M1 requires a positive integer " + name)
    if (train_cfg.algorithm.learning_rate != 5e-5 or train_cfg.algorithm.min_valid_samples != 2048
            or train_cfg.runner.num_steps_per_env != 100 or train_cfg.runner.save_interval != 500):
        raise ValueError("M1 requires lr=5e-5, min_valid_samples=2048, rollout=100 and save_interval=500")


def validate_m1_runtime(env, train_cfg):
    validate_m1_training_settings(env.cfg, train_cfg, env.num_envs)
    if (not math.isclose(env.dt, .02, rel_tol=0., abs_tol=1e-9) or env.num_obs != 738
            or env.num_actions != 29 or env.num_privileged_obs != 160):
        raise ValueError("M1 runtime observation/action/control contract mismatch")


def validate_m1_initialization(env_cfg, train_cfg, own, args):
    from legged_gym.utils.helpers import class_to_dict
    validate_m1_training_settings(env_cfg, train_cfg, getattr(args, "num_envs", None))
    h = class_to_dict(env_cfg.handoff)
    # M1 is a fixed experiment contract. Explicit incompatible training overrides fail.
    for key, value in dict(ablation="A5", adapter="rate_limited", warmup_vx=.8,
                           fixed_command=[.8, 0., 0.], command_suite="mixed",
                           mixed_command_manager="hold_then_slew_v1").items():
        if h.get(key) != value:
            raise ValueError("M1 training configuration mismatch: " + key)
    rates = h["mixed_command_slew_rates"]
    if (not math.isfinite(h["mixed_command_hold_s"]) or h["mixed_command_hold_s"] < 0
            or len(rates) != 3 or any(not math.isfinite(v) or v <= 0 for v in rates)):
        raise ValueError("Invalid configurable M1 hold/slew parameters")
    if not h["posture_reference"]:
        raise ValueError("M1 requires --posture_reference for the saved A5 teacher prior")
    names = [line.split()[1] for line in (Path(__file__).resolve().parents[2] /
             "resources/config/joint_id.txt").read_text().splitlines() if line.strip()]
    teacher_hash = file_sha256(h.pop("teacher_checkpoint"))
    reference_hash = file_sha256(h.pop("posture_reference"))
    policy = class_to_dict(train_cfg.policy)
    policy.update(actor_hidden_dims=[512, 256, 256], activation="elu")
    expected = dict(teacher_sha256=teacher_hash, reference_sha256=reference_hash,
                    dof_names=names, actor_obs=738, critic_obs=160,
                    dt=env_cfg.sim.dt * env_cfg.control.decimation, action_scale=env_cfg.control.action_scale,
                    handoff=h, rewards=class_to_dict(env_cfg.rewards), algorithm=class_to_dict(train_cfg.algorithm),
                    policy=policy, environment={k: class_to_dict(getattr(env_cfg, k)) for k in ENV_KEYS + ("gait",)})
    path = args.resume_path if args.resume else own.student_checkpoint
    checkpoint = read_checkpoint(path)
    _, spec = checkpoint_actor(path)
    contract = validate_m1_contract(checkpoint, spec, expected, resume=args.resume)
    # Assert the task's validated B2 scales even if a legacy JSON was supplied.
    scales = expected["rewards"]["scales"]
    for key, value in dict(gait_foot_slip=-.08, gait_swing_clearance=.15, gait_foot_scuff=-.08,
                           gait_air_time=.05, feet_air_time=0., feet_slip=0., feet_clearance=0.,
                           carry_upper_body_pose=.5).items():
        if scales[key] != value:
            raise ValueError("M1 requires B2 gait scales: " + key)
    return dict(source_checkpoint=str(Path(path).resolve()), source_sha256=file_sha256(path),
                source_iteration=checkpoint["iter"], source_kind="M1_resume" if args.resume else "trained_B2",
                teacher_sha256=teacher_hash, reference_sha256=reference_hash,
                source_gait_geometry=contract["environment"]["gait_geometry"],
                actor_dims=spec["dims"], critic_initialization="restored" if args.resume else "fresh",
                optimizer_initialization="restored" if args.resume else "fresh",
                source_initialization=checkpoint.get("initialization_provenance"))
