import ast
import pathlib
import types
import unittest

import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / "legged_gym" / "legged_gym" / "envs" / "g1" / "carrybox.py"
CONFIG_PATH = ENV_PATH.with_name("carrybox_config.py")
FORCE_PATH = ENV_PATH.with_name("carrybox_force.py")


def _env_method(name, extra_namespace=None):
    tree = ast.parse(ENV_PATH.read_text(encoding="utf-8"))
    env_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LeggedRobot"
    )
    method = next(
        node for node in env_class.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace = {"torch": torch}
    if extra_namespace:
        namespace.update(extra_namespace)
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(ENV_PATH), "exec"), namespace)
    return namespace[name]


def _bind(obj, name, extra_namespace=None):
    method = _env_method(name, extra_namespace)
    setattr(obj, name, types.MethodType(method, obj))


def _gate_fixture():
    rewards = types.SimpleNamespace(
        thresh_robot2object=0.7,
        carry_tracking_enter_min_lift_height=0.10,
        carry_tracking_enter_stable_steps=10,
        carry_tracking_exit_min_lift_height=0.05,
        carry_tracking_exit_contact_loss_steps=5,
    )
    commands_cfg = types.SimpleNamespace(heading_kp=1.0, max_yaw_rate=0.4)
    env = types.SimpleNamespace(
        cfg=types.SimpleNamespace(rewards=rewards, commands=commands_cfg),
        box_states=torch.zeros((1, 13)),
        _box_size=torch.tensor([[0.4, 0.4, 0.4]]),
        platform_pos=torch.zeros((1, 3)),
        hand_contact_filt=torch.zeros((1, 2), dtype=torch.bool),
        carry_tracking_active=torch.zeros(1, dtype=torch.bool),
        entering_carry_tracking=torch.zeros(1, dtype=torch.bool),
        exiting_carry_tracking=torch.zeros(1, dtype=torch.bool),
        carry_tracking_enter_streak=torch.zeros(1, dtype=torch.long),
        carry_tracking_contact_loss_streak=torch.zeros(1, dtype=torch.long),
        carry_tracking_entry_count=torch.zeros(1),
        carry_tracking_exit_count=torch.zeros(1),
        approach_mask=torch.zeros(1, dtype=torch.bool),
        pickup_or_recovery_mask=torch.zeros(1, dtype=torch.bool),
        robot2object_dist=torch.tensor([2.0]),
        commands=torch.tensor([[0.6, 0.0, 0.2, 0.0]]),
        carry_policy_commands=torch.zeros((1, 4)),
        carry_heading_ref=torch.zeros(1),
        carry_heading_error=torch.zeros(1),
        carry_heading_initialized=torch.zeros(1, dtype=torch.bool),
        yaw=torch.tensor([0.4]),
        dt=0.02,
    )
    _bind(env, "_box_lift_height")
    _bind(env, "_update_carry_tracking_state")
    _bind(env, "_update_task_phase_masks")
    _bind(
        env,
        "_update_carry_heading_commands",
        {"wrap_to_pi": lambda value: torch.remainder(value + torch.pi, 2 * torch.pi) - torch.pi},
    )
    return env


class TestEpisodeCommandLifecycle(unittest.TestCase):
    def test_only_reset_task_samples_nominal_command(self):
        tree = ast.parse(ENV_PATH.read_text(encoding="utf-8"))
        env_class = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "LeggedRobot"
        )
        callers = []
        for method in (node for node in env_class.body if isinstance(node, ast.FunctionDef)):
            if any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_resample_carry_commands"
                for node in ast.walk(method)
            ):
                callers.append(method.name)
        self.assertEqual(callers, ["_reset_task"])

        combined_source = (
            ENV_PATH.read_text(encoding="utf-8")
            + FORCE_PATH.read_text(encoding="utf-8")
            + CONFIG_PATH.read_text(encoding="utf-8")
        )
        for removed_name in (
            "carry_resample_interval_s",
            "carry_command_resample_time",
            "_sample_carry_command_resample_time",
            "_update_nominal_commands_with_force_resample_deferral",
        ):
            self.assertNotIn(removed_name, combined_source)

    def test_approach_pickup_carry_drop_and_reacquire_keep_vx(self):
        env = _gate_fixture()
        nominal_vx = env.commands[:, 0].clone()

        env._update_carry_tracking_state()
        env._update_task_phase_masks()
        env._update_carry_heading_commands()
        self.assertTrue(env.approach_mask.item())
        self.assertEqual(env.carry_policy_commands[0, 2].item(), 0.0)

        env.robot2object_dist[:] = 0.5
        env._update_task_phase_masks()
        env._update_carry_heading_commands()
        self.assertTrue(env.pickup_or_recovery_mask.item())

        env.box_states[:, 2] = 0.2 + 0.11
        env.hand_contact_filt[:] = True
        for _ in range(10):
            env._update_carry_tracking_state()
        env._update_task_phase_masks()
        env._update_carry_heading_commands()
        self.assertTrue(env.carry_tracking_active.item())
        self.assertTrue(env.carry_heading_initialized.item())
        self.assertNotEqual(env.carry_policy_commands[0, 2].item(), 0.0)

        env.box_states[:, 2] = 0.2 + 0.04
        env._update_carry_tracking_state()
        env._update_task_phase_masks()
        env._update_carry_heading_commands()
        self.assertFalse(env.carry_tracking_active.item())
        self.assertTrue(env.pickup_or_recovery_mask.item())
        self.assertEqual(env.carry_policy_commands[0, 2].item(), 0.0)

        env.box_states[:, 2] = 0.2 + 0.11
        for _ in range(10):
            env._update_carry_tracking_state()
        env._update_carry_heading_commands()
        self.assertTrue(env.carry_tracking_active.item())
        self.assertEqual(env.carry_tracking_entry_count.item(), 2.0)
        self.assertEqual(env.carry_tracking_exit_count.item(), 1.0)
        torch.testing.assert_close(env.commands[:, 0], nominal_vx)
        torch.testing.assert_close(env.carry_policy_commands[:, 0], nominal_vx)

    def test_contact_loss_exit_is_debounced(self):
        env = _gate_fixture()
        env.carry_tracking_active[:] = True
        env.box_states[:, 2] = 0.2 + 0.11
        env.hand_contact_filt[:] = False
        for _ in range(4):
            env._update_carry_tracking_state()
            self.assertTrue(env.carry_tracking_active.item())
        env._update_carry_tracking_state()
        self.assertFalse(env.carry_tracking_active.item())


class TestRewardGating(unittest.TestCase):
    def test_velocity_rewards_are_off_during_pickup(self):
        env = types.SimpleNamespace(
            cfg=types.SimpleNamespace(
                rewards=types.SimpleNamespace(
                    tracking_sigma=0.25,
                    carry_lin_vel=2.0,
                    carry_yaw_vel=0.5,
                )
            ),
            commands=torch.tensor(
                [[0.6, 0.0, 0.0, 0.0]] * 3, dtype=torch.float
            ),
            carry_policy_commands=torch.tensor(
                [[0.6, 0.0, 0.0, 0.0]] * 3, dtype=torch.float
            ),
            robot2object_dir=torch.tensor([[1.0, 0.0]] * 3),
            approach_mask=torch.tensor([True, False, False]),
            pickup_or_recovery_mask=torch.tensor([False, True, False]),
            carry_tracking_active=torch.tensor([False, False, True]),
            carry_heading_ref=torch.zeros(3),
            rigid_body_states=torch.zeros((3, 1, 13)),
            upper_body_index=0,
            base_ang_vel=torch.zeros((3, 3)),
        )
        env.rigid_body_states[:, 0, 7] = 0.6
        _bind(env, "_approach_direction")
        _bind(env, "_reward_approach_velocity_tracking")
        _bind(env, "_reward_carry_velocity_task")

        approach_reward = env._reward_approach_velocity_tracking()
        carry_reward = env._reward_carry_velocity_task()
        self.assertGreater(approach_reward[0].item(), 0.0)
        self.assertEqual(approach_reward[1].item(), 0.0)
        self.assertEqual(carry_reward[1].item(), 0.0)
        self.assertGreater(carry_reward[2].item(), 0.0)


class TestConfigAndInterface(unittest.TestCase):
    def test_stage1a_values_and_actor_interface_are_explicit(self):
        source = CONFIG_PATH.read_text(encoding="utf-8")
        self.assertIn("episode_length_s = 30", source)
        self.assertIn("lin_vel_x = [0.4, 1.0]", source)
        self.assertIn("approach_velocity_tracking = 2.0", source)
        self.assertIn("carry_lin_vel = 2.0", source)
        self.assertNotIn("target_speed_loco", source)

        tree = ast.parse(source)
        cfg = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "G1Cfg"
        )
        env_cfg = next(
            node for node in cfg.body
            if isinstance(node, ast.ClassDef) and node.name == "env"
        )
        assignments = {
            target.id: ast.literal_eval(node.value)
            for node in env_cfg.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance((target := node.targets[0]), ast.Name)
            and target.id in {"num_actions", "num_task_obs"}
        }
        self.assertEqual(assignments, {"num_actions": 29, "num_task_obs": 15})


if __name__ == "__main__":
    unittest.main()
