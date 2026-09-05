"""CPU harnesses: execute repository definitions without importing Isaac Gym.

Only simulator-dependent module imports and unrelated class methods are omitted.
The selected method bodies (including super() calls) are compiled unchanged.
These checks exercise tensor logic, not PhysX or simulator initialization.
"""

import ast
import importlib.util
import inspect
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_DIR.parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))


def definitions(path, names, namespace=None, methods=None, base=object):
    path = REPO_ROOT / path
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    assert {node.name for node in nodes} == set(names)
    if methods is not None:
        for node in nodes:
            node.bases = [ast.Name(id="HarnessBase", ctx=ast.Load())]
            node.body = [item for item in node.body if getattr(item, "name", None) in methods]
            assert {item.name for item in node.body} == set(methods)
    scope = {"torch": torch, "np": np, "math": math, "inspect": inspect,
             "HarnessBase": base, **(namespace or {})}
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(path), "exec"), scope)
    return scope


def carrybox_configs():
    scope = definitions("legged_gym/legged_gym/envs/base/base_config.py", ["BaseConfig"])
    scope = definitions("legged_gym/legged_gym/envs/base/legged_robot_config.py",
                        ["LeggedRobotCfg", "LeggedRobotCfgPPO"], scope)
    scope = definitions("legged_gym/legged_gym/envs/g1/carrybox_config.py",
                        ["G1Cfg", "G1CfgPPO"], scope)
    return scope["G1Cfg"](), scope["G1CfgPPO"]()


def actor_critic_class():
    path = REPO_ROOT / "rsl_rl/rsl_rl/modules/actor_critic.py"
    spec = importlib.util.spec_from_file_location("nforce_test_actor_critic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ActorCritic


def command_env(command=(0.4, 0.0, 0.0)):
    from configs.nominal_clean_config import apply_nominal_clean_config

    scope = definitions("legged_gym/legged_gym/utils/math.py", ["wrap_to_pi"])
    base = definitions(
        "legged_gym/legged_gym/envs/g1/carrybox.py", ["LeggedRobot"], scope,
        methods={"_reset_task", "_update_carry_heading_commands", "_compute_is_stage_carry"},
    )["LeggedRobot"]
    nominal = definitions(
        "experiments/carrybox_clean_perturb/envs/carrybox_nominal_clean_env.py",
        ["LeggedRobot"], methods={"_reset_task", "_set_nominal_clean_command"}, base=base,
    )["LeggedRobot"]
    env = nominal()
    cfg, _ = carrybox_configs()
    env.cfg = apply_nominal_clean_config(cfg, command=command)
    env.device, env.num_envs, env.dt = "cpu", 1, 0.02
    env.commands = torch.zeros(1, 4)
    env.carry_policy_commands = torch.zeros(1, 3)
    env.carry_heading_initialized = torch.zeros(1, dtype=torch.bool)
    env.carry_heading_ref = torch.zeros(1)
    env.carry_heading_error = torch.zeros(1)
    env.carry_yaw_resample_time = torch.zeros(1)
    env.is_stage_carry = torch.zeros(1, dtype=torch.bool)
    env.yaw = torch.zeros(1)
    env.tar_platform_states = torch.zeros(1, 13)
    env.tar_platform_default_states = torch.zeros(1, 13)
    return env


_EvalCore = definitions(
    "experiments/carrybox_clean_perturb/envs/carrybox_perturb_env.py", ["LeggedRobot"],
    methods={"_update_confirmed_carry_detector", "_snapshot_clean_eval_for_summary",
             "summary_scalar", "summary_reason", "_apply_box_external_force",
             "_assert_no_force_inactive"},
)["LeggedRobot"]
_NForceCore = definitions(
    "experiments/carrybox_clean_perturb/evaluator_NForce.py", ["NForceCarryBoxEnv"],
    methods={"_snapshot_clean_eval_for_summary"}, base=_EvalCore,
)["NForceCarryBoxEnv"]


class RolloutEnv(_NForceCore):
    """Scripted post-step states with the actual carry detector and snapshots."""

    def __init__(self, contact=lambda step: True, terminal_step=None,
                 failure_step=None, timeout=False, command=(0.4, 0.0, 0.0)):
        self.contact = contact
        self.terminal_step, self.failure_step, self.timeout = terminal_step, failure_step, timeout
        self.command = command
        cmd_env = command_env(command)
        self.cfg = cmd_env.cfg
        self.cfg.clean_perturbation.enabled = False
        self.num_envs, self.num_actions, self.num_task_obs = 1, 29, 15
        self.actor_obs_length, self.max_episode_length, self.dt = 738, 1500, 0.02
        self.commands, self.carry_policy_commands = cmd_env.commands, cmd_env.carry_policy_commands
        self.base_lin_vel = torch.tensor([[0.3, 0.0, 0.0]])
        self.base_ang_vel = torch.zeros(1, 3)
        self.yaw, self.carry_heading_ref, self.carry_heading_error = (torch.zeros(1) for _ in range(3))
        self.nforce_terminal_base_yaw_rad = torch.full((1,), float("nan"))
        self.obs = torch.zeros(1, 738)
        self.device = "cpu"
        self.contact_forces = torch.zeros(1, 2, 3)
        self.left_hand_contact_proxy_index, self.right_hand_contact_proxy_index = 0, 1
        self.root_states, self.box_states = torch.zeros(1, 13), torch.zeros(1, 13)
        self.is_stage_carry = torch.ones(1, dtype=torch.bool)
        self.clean_eval_force_tensor, self.disturbance = torch.zeros(1, 2, 3), torch.zeros(1, 2, 3)
        self.clean_eval_remaining_physics_steps = torch.zeros(1, dtype=torch.long)
        self.clean_eval_actual_force_scale = torch.zeros(1)
        self.confirmed_carry_streak = torch.zeros(1, dtype=torch.long)
        for name in ("left_hand_contact_proxy", "right_hand_contact_proxy", "clean_carry_condition_buf",
                     "confirmed_carry_buf", "clean_eval_has_terminal_snapshot"):
            setattr(self, name, torch.zeros(1, dtype=torch.bool))
        self.scalar_names = (
            "peak_force_N", "impulse_Ns", "force_duration_s", "recovery_success_buf",
            "recovery_time_s", "carry_achieved_buf", "humanoid_failure_buf", "box_failure_buf",
            "timeout_buf", "event_count_buf",
        )
        for name in self.scalar_names:
            setattr(self, "clean_eval_" + name, torch.zeros(1))
            setattr(self, "clean_eval_terminal_" + name, torch.zeros(1))
        self.clean_eval_terminal_confirmed_carry_buf = torch.zeros(1, dtype=torch.bool)
        for prefix in ("clean_eval_", "clean_eval_terminal_"):
            for name in ("humanoid_failure_reason", "box_failure_reason"):
                setattr(self, prefix + name, [""])
        self.clean_eval_last_termination_reason = [""]
        self.extras = {}
        self.step_id = 0

    def reset_evaluation_trial_state(self, clear_actor_history=True):
        assert clear_actor_history
        self.obs.zero_()
        self.confirmed_carry_streak.zero_()
        self.clean_eval_has_terminal_snapshot.zero_()

    def reset(self):
        self.step_id = 0
        self.commands[:, :3] = torch.tensor(self.command)
        self.carry_policy_commands[:] = self.commands[:, :3]
        self.carry_policy_commands[:, 1] = 0.0
        self.obs[:, -3:] = self.carry_policy_commands
        return self.obs, None

    def clear_summary_snapshot(self, env_id=0):
        self.clean_eval_has_terminal_snapshot[env_id] = False

    def step(self, actions):
        assert actions.shape == (1, 29)
        self.step_id += 1
        self.contact_forces[:] = 2.0 if self.contact(self.step_id) else 0.0
        self._update_confirmed_carry_detector()
        self.clean_eval_carry_achieved_buf[:] = torch.maximum(
            self.clean_eval_carry_achieved_buf, self.confirmed_carry_buf.float())
        self._apply_box_external_force()  # Disabled force must never call the gym API.
        self.yaw[:] = 0.7
        done = self.step_id == self.terminal_step
        if self.step_id == self.failure_step:
            self.clean_eval_box_failure_buf[:] = 1
            self.clean_eval_box_failure_reason[0] = "dropped_to_ground"
        if done:
            self.clean_eval_timeout_buf[:] = self.timeout
            self.clean_eval_humanoid_failure_buf[:] = not self.timeout
            self.clean_eval_humanoid_failure_reason[0] = "" if self.timeout else "head_low"
            self.clean_eval_last_termination_reason[0] = "timeout" if self.timeout else "head_low"
            self._snapshot_clean_eval_for_summary(0)
            # Simulate the base environment's automatic reset after snapshotting.
            self.yaw[:] = -2.0
            self.confirmed_carry_buf[:] = False
            self.obs[:] = -999.0
            for name in self.scalar_names:
                getattr(self, "clean_eval_" + name).zero_()
            self.clean_eval_humanoid_failure_reason[0] = ""
        ids = torch.tensor([0], dtype=torch.long) if done else torch.empty(0, dtype=torch.long)
        return self.obs, None, None, torch.tensor([done]), {}, ids, None, None


def trial(env=None, warmup=0.2, duration=5.0):
    from evaluation.nforce_trial import run_trial

    env = env or RolloutEnv()
    seeds = []
    result = run_trial(env, lambda obs: torch.zeros(1, 29), "model.pt", 1, env.command,
                       SimpleNamespace(steady_carry_warmup=warmup, steady_duration=duration),
                       seed_fn=seeds.append)
    assert seeds == [1]
    return result
