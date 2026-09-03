import ast
import importlib.util
import math
import pathlib
import tempfile
import types
import unittest

import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
UTILS_PATH = (
    REPO_ROOT
    / "legged_gym"
    / "legged_gym"
    / "envs"
    / "g1"
    / "carrybox_force_utils.py"
)
SPEC = importlib.util.spec_from_file_location("carrybox_force_utils", UTILS_PATH)
UTILS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(UTILS)
compute_admittance_teacher = UTILS.compute_admittance_teacher
resolve_directional_beta_ranges = UTILS.resolve_directional_beta_ranges
sample_directional_beta = UTILS.sample_directional_beta
smooth_force_profile = UTILS.smooth_force_profile

CONFIG_PATH = (
    REPO_ROOT
    / "legged_gym"
    / "legged_gym"
    / "envs"
    / "g1"
    / "carrybox_force_config.py"
)
FORCE_ENV_PATH = (
    REPO_ROOT
    / "legged_gym"
    / "legged_gym"
    / "envs"
    / "g1"
    / "carrybox_force.py"
)
STAGE1_ENV_PATH = FORCE_ENV_PATH.with_name("carrybox.py")


def _class_assignments(class_node):
    assignments = {}
    for node in class_node.body:
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            continue
        try:
            assignments[node.targets[0].id] = ast.literal_eval(node.value)
        except ValueError:
            continue
    return assignments


def _formal_force_config():
    tree = ast.parse(CONFIG_PATH.read_text(encoding="utf-8"))
    env_cfg = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "G1Cfg"
    )
    external_force = next(
        node
        for node in env_cfg.body
        if isinstance(node, ast.ClassDef) and node.name == "external_force"
    )
    return _class_assignments(external_force)


def _load_force_schedule_method(quat_rotate):
    tree = ast.parse(FORCE_ENV_PATH.read_text(encoding="utf-8"))
    force_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LeggedRobot"
    )
    method_node = next(
        node
        for node in force_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "_schedule_force_event"
    )
    namespace = {
        "torch": torch,
        "quat_rotate": quat_rotate,
        "sample_directional_beta": sample_directional_beta,
    }
    method_module = ast.Module(body=[method_node], type_ignores=[])
    ast.fix_missing_locations(method_module)
    exec(compile(method_module, str(FORCE_ENV_PATH), "exec"), namespace)
    return namespace["_schedule_force_event"]


def _load_force_method(method_name):
    tree = ast.parse(FORCE_ENV_PATH.read_text(encoding="utf-8"))
    force_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LeggedRobot"
    )
    method_node = next(
        node
        for node in force_class.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    namespace = {"torch": torch}
    method_module = ast.Module(body=[method_node], type_ignores=[])
    ast.fix_missing_locations(method_module)
    exec(compile(method_module, str(FORCE_ENV_PATH), "exec"), namespace)
    return namespace[method_name]


def _load_stage1_method(method_name):
    tree = ast.parse(STAGE1_ENV_PATH.read_text(encoding="utf-8"))
    stage1_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LeggedRobot"
    )
    method_node = next(
        node
        for node in stage1_class.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    namespace = {"torch": torch}
    method_module = ast.Module(body=[method_node], type_ignores=[])
    ast.fix_missing_locations(method_module)
    exec(compile(method_module, str(STAGE1_ENV_PATH), "exec"), namespace)
    return namespace[method_name]


def _force_ready_fixture(lift_height, hand_contacts):
    lift_height = torch.as_tensor(lift_height, dtype=torch.float)
    hand_contacts = torch.as_tensor(hand_contacts, dtype=torch.bool)
    count = len(lift_height)
    box_size = torch.full((count, 3), 0.4)
    platform_pos = torch.zeros((count, 3))
    platform_pos[:, 2] = -5.0
    box_states = torch.zeros((count, 13))
    box_states[:, 2] = box_size[:, 2] / 2.0 + lift_height
    contact_forces = torch.zeros((count, 2, 3))
    contact_forces[:, :, 0] = hand_contacts.to(dtype=torch.float) * 2.0
    fixture = types.SimpleNamespace(
        cfg=types.SimpleNamespace(
            external_force=types.SimpleNamespace(force_ready_min_lift_height=0.10),
            rewards=types.SimpleNamespace(hand_contact_threshold=1.0),
        ),
        box_states=box_states,
        _box_size=box_size,
        platform_pos=platform_pos,
        _platform_height=0.02,
        env_origins=torch.zeros((count, 3)),
        contact_forces=contact_forces,
        hand_colli_indices=torch.tensor([0, 1]),
        force_last_hand_contacts=torch.zeros((count, 2), dtype=torch.bool),
    )
    fixture._box_lift_height = types.MethodType(
        _load_stage1_method("_box_lift_height"), fixture
    )
    return fixture


def _force_scheduler_fixture(rotated_directions):
    def quat_rotate(_box_quaternions, _direction_local):
        return rotated_directions.clone()

    schedule = _load_force_schedule_method(quat_rotate)
    count = len(rotated_directions)
    cfg = types.SimpleNamespace(
        force_directions=("+box_x",),
        curriculum_beta_ranges={1: {"+box_x": (0.2, 0.2)}},
        curriculum_stage=1,
        beta_range=None,
        force_hold_duration_range_s=(0.04, 0.04),
        force_ramp_up_duration_s=0.02,
        force_ramp_down_duration_s=0.02,
    )
    scheduler = types.SimpleNamespace(
        device="cpu",
        cfg=types.SimpleNamespace(external_force=cfg),
        _DIRECTION_LOCAL={"+box_x": (1.0, 0.0, 0.0)},
        box_states=torch.zeros((count, 7)),
        box_masses=torch.arange(1, count + 1, dtype=torch.float),
        external_force_direction_world=torch.full((count, 3), 9.0),
        external_force_beta=torch.zeros(count),
        external_force_box_mass=torch.zeros(count),
        external_force_peak_N=torch.zeros(count),
        force_hold_duration_s=torch.zeros(count),
        force_ramp_up_steps=torch.zeros(count, dtype=torch.long),
        force_hold_steps=torch.zeros(count, dtype=torch.long),
        force_ramp_down_steps=torch.zeros(count, dtype=torch.long),
        force_elapsed_physics_steps=torch.zeros(count, dtype=torch.long),
        force_remaining_physics_steps=torch.zeros(count, dtype=torch.long),
        force_phase=torch.zeros(count, dtype=torch.long),
        force_event_count=torch.zeros(count, dtype=torch.long),
        force_event_decision_made=torch.ones(count, dtype=torch.bool),
        force_stable_carry_streak=torch.full((count,), 10, dtype=torch.long),
        _force_start_pending=torch.zeros(count, dtype=torch.bool),
        FORCE_PHASE_RAMP_UP=1,
        sim_params=types.SimpleNamespace(dt=0.02),
        _uniform_sample=lambda value_range, sample_count: torch.full(
            (sample_count,), float(value_range[0])
        ),
        _seconds_to_physics_steps=lambda duration_s: int(round(duration_s / 0.02)),
    )
    return schedule, scheduler


class TestForceProfile(unittest.TestCase):
    def test_cosine_ramp_hold_and_down(self):
        elapsed = torch.tensor([0.0, 0.2, 0.4, 2.4, 2.6, 2.8, 2.81])
        ramp = torch.full_like(elapsed, 0.4)
        hold = torch.full_like(elapsed, 2.0)
        actual = smooth_force_profile(elapsed, ramp, hold, ramp)
        expected = torch.tensor([0.0, 0.5, 1.0, 1.0, 0.5, 0.0, 0.0])
        torch.testing.assert_close(actual, expected, atol=1.0e-6, rtol=0.0)


class TestForceOnsetEligibility(unittest.TestCase):
    def setUp(self):
        self.compute_ready = _load_force_method("_compute_force_ready_mask")
        self.update_scheduler = _load_force_method("_update_force_event_scheduler")

    def test_lifted_box_with_bilateral_contact_is_ready(self):
        scheduler = _force_ready_fixture([0.11], [[True, True]])

        self.assertEqual(self.compute_ready(scheduler).tolist(), [True])

    def test_missing_one_hand_expires_after_one_step_filter(self):
        scheduler = _force_ready_fixture([0.11], [[True, True]])
        self.assertEqual(self.compute_ready(scheduler).tolist(), [True])

        scheduler.contact_forces[0, 1] = 0.0
        self.assertEqual(self.compute_ready(scheduler).tolist(), [True])
        self.assertEqual(self.compute_ready(scheduler).tolist(), [False])

    def test_bilateral_contact_without_sufficient_lift_is_not_ready(self):
        scheduler = _force_ready_fixture([0.09], [[True, True]])

        self.assertEqual(self.compute_ready(scheduler).tolist(), [False])

    def test_readiness_loss_beyond_filter_tolerance_resets_streak(self):
        scheduler = _force_ready_fixture([0.11], [[True, True]])
        scheduler.cfg.external_force.stable_carry_policy_steps = 10
        scheduler.cfg.external_force.max_force_events_per_episode = 1
        scheduler.cfg.external_force.force_event_probability = 1.0
        scheduler.device = "cpu"
        scheduler.force_stable_carry_streak = torch.zeros(1, dtype=torch.long)
        scheduler.force_event_decision_made = torch.ones(1, dtype=torch.bool)
        scheduler.force_event_count = torch.zeros(1, dtype=torch.long)
        scheduler.force_remaining_physics_steps = torch.zeros(1, dtype=torch.long)
        scheduler._compute_force_ready_mask = types.MethodType(self.compute_ready, scheduler)

        for _ in range(3):
            self.update_scheduler(scheduler)
        self.assertEqual(scheduler.force_stable_carry_streak.tolist(), [3])

        scheduler.contact_forces[0, 1] = 0.0
        self.update_scheduler(scheduler)
        self.assertEqual(scheduler.force_stable_carry_streak.tolist(), [4])
        self.update_scheduler(scheduler)
        self.assertEqual(scheduler.force_stable_carry_streak.tolist(), [0])

    def test_active_event_is_not_cancelled_when_readiness_disappears(self):
        scheduler = _force_ready_fixture([0.0], [[False, False]])
        scheduler.cfg.external_force.stable_carry_policy_steps = 10
        scheduler.cfg.external_force.max_force_events_per_episode = 1
        scheduler.cfg.external_force.force_event_probability = 1.0
        scheduler.device = "cpu"
        scheduler.force_stable_carry_streak = torch.full((1,), 10, dtype=torch.long)
        scheduler.force_event_decision_made = torch.ones(1, dtype=torch.bool)
        scheduler.force_event_count = torch.ones(1, dtype=torch.long)
        scheduler.force_remaining_physics_steps = torch.tensor([12], dtype=torch.long)
        scheduler.force_phase = torch.tensor([1], dtype=torch.long)
        scheduler._compute_force_ready_mask = types.MethodType(self.compute_ready, scheduler)

        self.update_scheduler(scheduler)

        self.assertEqual(scheduler.force_stable_carry_streak.tolist(), [0])
        self.assertEqual(scheduler.force_remaining_physics_steps.tolist(), [12])
        self.assertEqual(scheduler.force_phase.tolist(), [1])

    def test_force_contact_memory_reset_is_scoped_to_env_ids(self):
        tree = ast.parse(FORCE_ENV_PATH.read_text(encoding="utf-8"))
        force_class = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LeggedRobot"
        )
        reset_node = next(
            node
            for node in force_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "reset_idx"
        )
        reset_source = ast.get_source_segment(
            FORCE_ENV_PATH.read_text(encoding="utf-8"), reset_node
        )
        self.assertIn('"force_last_hand_contacts"', reset_source)
        self.assertIn("getattr(self, name)[env_ids] = False", reset_source)
        self.assertNotIn("force_last_hand_contacts.zero_", reset_source)


class TestForceDirectionScheduling(unittest.TestCase):
    def test_mixed_valid_and_invalid_directions_are_filtered_per_environment(self):
        rotated_directions = torch.tensor(
            [
                [2.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, -3.0, 0.0],
                [float("nan"), 0.0, 0.0],
            ]
        )
        schedule, scheduler = _force_scheduler_fixture(rotated_directions)

        schedule(scheduler, torch.arange(4))

        torch.testing.assert_close(
            scheduler.external_force_direction_world,
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [9.0, 9.0, 9.0],
                    [0.0, -1.0, 0.0],
                    [9.0, 9.0, 9.0],
                ]
            ),
        )
        self.assertEqual(scheduler.force_event_count.tolist(), [1, 0, 1, 0])
        self.assertEqual(scheduler._force_start_pending.tolist(), [True, False, True, False])
        self.assertEqual(scheduler.force_event_decision_made.tolist(), [True, False, True, False])
        self.assertEqual(scheduler.force_stable_carry_streak.tolist(), [10, 0, 10, 0])
        self.assertEqual(scheduler.external_force_box_mass.tolist(), [1.0, 0.0, 3.0, 0.0])
        self.assertNotIn(
            "Box-frame force direction has a degenerate horizontal projection",
            FORCE_ENV_PATH.read_text(encoding="utf-8"),
        )

    def test_all_invalid_directions_leave_no_scheduled_event(self):
        schedule, scheduler = _force_scheduler_fixture(
            torch.tensor([[0.0, 0.0, 1.0], [float("inf"), 0.0, 0.0]])
        )

        schedule(scheduler, torch.arange(2))

        self.assertEqual(scheduler.force_event_count.tolist(), [0, 0])
        self.assertEqual(scheduler.force_event_decision_made.tolist(), [False, False])
        self.assertEqual(scheduler.force_stable_carry_streak.tolist(), [0, 0])
        self.assertTrue(torch.all(scheduler.external_force_direction_world == 9.0))


class TestManualForceCurriculum(unittest.TestCase):
    def setUp(self):
        cfg = _formal_force_config()
        self.directions = cfg["force_directions"]
        self.presets = cfg["curriculum_beta_ranges"]

    def test_formal_stage2a_defaults(self):
        cfg = _formal_force_config()
        self.assertTrue(cfg["enable_external_force"])
        self.assertEqual(cfg["force_event_probability"], 1.0)
        self.assertEqual(cfg["force_ready_min_lift_height"], 0.10)
        self.assertEqual(
            cfg["force_directions"],
            ("+box_x", "-box_x", "+box_y", "-box_y"),
        )
        self.assertEqual(cfg["curriculum_stage"], 1)
        self.assertIsNone(cfg["beta_range"])

    def test_exact_widening_presets_keep_backward_capped(self):
        expected = {
            1: {
                "-box_x": (0.2, 0.4),
                "+box_x": (0.2, 0.4),
                "+box_y": (0.2, 0.4),
                "-box_y": (0.2, 0.4),
            },
            2: {
                "-box_x": (0.2, 0.4),
                "+box_x": (0.2, 0.6),
                "+box_y": (0.2, 0.6),
                "-box_y": (0.2, 0.6),
            },
            3: {
                "-box_x": (0.2, 0.4),
                "+box_x": (0.2, 0.8),
                "+box_y": (0.2, 0.8),
                "-box_y": (0.2, 0.8),
            },
        }
        self.assertEqual(self.presets, expected)
        for stage in (1, 2, 3):
            self.assertEqual(self.presets[stage]["-box_x"][1], 0.4)

    def test_invalid_curriculum_stage_fails_clearly(self):
        for stage in (0, 4, -1):
            with self.subTest(stage=stage):
                with self.assertRaisesRegex(ValueError, "must be one of"):
                    resolve_directional_beta_ranges(self.presets, stage, self.directions)

    def test_mixed_batch_uses_each_selected_direction_range(self):
        direction_ids = torch.tensor([0, 1, 2, 3] * 128)
        beta = sample_directional_beta(
            self.directions,
            direction_ids,
            self.presets,
            curriculum_stage=3,
        )
        for direction_id, direction_name in enumerate(self.directions):
            selected = beta[direction_ids == direction_id]
            low, high = self.presets[3][direction_name]
            self.assertTrue(torch.all(selected >= low))
            self.assertTrue(torch.all(selected <= high))

    def test_fixed_beta_override_bypasses_directional_curriculum(self):
        direction_ids = torch.zeros(64, dtype=torch.long)
        beta = sample_directional_beta(
            ("-box_x",),
            direction_ids,
            self.presets,
            curriculum_stage=3,
            beta_range=(0.10, 0.10),
        )
        torch.testing.assert_close(beta, torch.full_like(beta, 0.10))

    def test_cli_direction_and_beta_install_fixed_debug_override(self):
        helpers_path = REPO_ROOT / "legged_gym" / "legged_gym" / "utils" / "helpers.py"
        tree = ast.parse(helpers_path.read_text(encoding="utf-8"))
        update_node = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "update_cfg_from_args"
        )
        namespace = {}
        method_module = ast.Module(body=[update_node], type_ignores=[])
        ast.fix_missing_locations(method_module)
        exec(compile(method_module, str(helpers_path), "exec"), namespace)

        external_force = types.SimpleNamespace(
            enable_external_force=True,
            force_directions=self.directions,
            beta_range=None,
            curriculum_stage=1,
            debug_logging=False,
            debug_draw_force=False,
        )
        env_cfg = types.SimpleNamespace(
            env=types.SimpleNamespace(num_envs=4096),
            seed=1,
            external_force=external_force,
        )
        args = types.SimpleNamespace(
            num_envs=None,
            seed=None,
            play_dataset=None,
            enable_external_force=False,
            force_direction="-box_x",
            force_beta=0.10,
            force_curriculum_stage=3,
            force_debug=False,
            force_debug_viz=False,
        )
        namespace["update_cfg_from_args"](env_cfg, None, args)
        self.assertEqual(external_force.force_directions, ("-box_x",))
        self.assertEqual(external_force.beta_range, (0.10, 0.10))
        self.assertEqual(external_force.curriculum_stage, 3)


class TestAdmittanceTeacher(unittest.TestCase):
    def _teacher(self, force_xy):
        force = torch.tensor([[force_xy[0], force_xy[1], 0.0]], dtype=torch.float64)
        mass = torch.tensor([2.5], dtype=torch.float64)
        nominal_vx = torch.tensor([0.5], dtype=torch.float64)
        nominal_heading = torch.tensor([0.0], dtype=torch.float64)
        robot_yaw = torch.tensor([0.0], dtype=torch.float64)
        nominal_heading_error = torch.tensor([0.0], dtype=torch.float64)
        nominal_raw_yaw = torch.tensor([0.0], dtype=torch.float64)
        nominal_yaw_target = torch.tensor([0.0], dtype=torch.float64)
        return compute_admittance_teacher(
            force,
            mass,
            nominal_vx,
            nominal_heading,
            robot_yaw,
            nominal_heading_error,
            nominal_raw_yaw,
            nominal_yaw_target,
            heading_kp=1.0,
            max_yaw_rate=0.4,
            admittance_d_bar=12.0,
            max_heading_offset=math.pi / 4.0,
            teacher_vx_min=0.0,
            teacher_vx_max=0.8,
        )

    def test_backward_force_reduces_speed_by_beta_g_over_dbar(self):
        mass = 2.5
        peak = 0.10 * mass * 9.81
        teacher = self._teacher((-peak, 0.0))
        self.assertAlmostEqual(float(teacher[0]), 0.5 - 0.10 * 9.81 / 12.0, places=12)
        self.assertAlmostEqual(float(teacher[1]), 0.0, places=12)
        self.assertAlmostEqual(float(teacher[3]), 0.0, places=12)

    def test_strong_pure_backward_force_stops_without_turning(self):
        teacher = self._teacher((-18.0, 0.0))
        self.assertAlmostEqual(float(teacher[0]), 0.0, places=12)
        self.assertAlmostEqual(float(teacher[1]), 0.0, places=12)
        self.assertAlmostEqual(float(teacher[3]), 0.0, places=12)

    def test_backward_force_before_zero_crossing_remains_longitudinal(self):
        teacher = self._teacher((-7.5, 0.0))
        self.assertAlmostEqual(float(teacher[0]), 0.25, places=12)
        self.assertAlmostEqual(float(teacher[1]), 0.0, places=12)

    def test_direction_signs(self):
        mass = 2.5
        peak = 0.10 * mass * 9.81
        forward = self._teacher((peak, 0.0))
        left = self._teacher((0.0, peak))
        right = self._teacher((0.0, -peak))
        self.assertGreater(float(forward[0]), 0.5)
        self.assertGreater(float(left[1]), 0.0)
        self.assertGreater(float(left[3]), 0.0)
        self.assertLess(float(right[1]), 0.0)
        self.assertLess(float(right[3]), 0.0)

    def test_zero_force_returns_exact_stage1_targets(self):
        nominal_vx = torch.tensor([0.37], dtype=torch.float64)
        nominal_heading = torch.tensor([0.21], dtype=torch.float64)
        robot_yaw = torch.tensor([0.11], dtype=torch.float64)
        nominal_heading_error = torch.tensor([0.1], dtype=torch.float64)
        nominal_yaw_target = torch.tensor([0.13], dtype=torch.float64)
        teacher = compute_admittance_teacher(
            torch.zeros((1, 3), dtype=torch.float64),
            torch.tensor([2.5], dtype=torch.float64),
            nominal_vx,
            nominal_heading,
            robot_yaw,
            nominal_heading_error,
            torch.tensor([0.03], dtype=torch.float64),
            nominal_yaw_target,
            heading_kp=1.0,
            max_yaw_rate=0.4,
            admittance_d_bar=12.0,
            max_heading_offset=math.pi / 4.0,
            teacher_vx_min=0.0,
            teacher_vx_max=0.8,
        )
        self.assertTrue(torch.equal(teacher[0], nominal_vx))
        self.assertTrue(torch.equal(teacher[1], nominal_heading))
        self.assertTrue(torch.equal(teacher[2], nominal_heading_error))
        self.assertTrue(torch.equal(teacher[3], nominal_yaw_target))


class TestActorObservationIsolation(unittest.TestCase):
    def test_force_and_teacher_names_are_absent_from_observation_builders(self):
        source_path = (
            REPO_ROOT / "legged_gym" / "legged_gym" / "envs" / "g1" / "carrybox_force.py"
        )
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        forbidden = {
            "external_force_world",
            "external_force_beta",
            "external_force_active",
            "teacher_vx",
            "teacher_heading",
            "teacher_yaw_rate",
            "force_parallel",
            "force_perp",
        }
        observation_methods = {
            "compute_observations",
            "compute_task_observations",
            "compute_termination_observations",
        }
        force_class = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LeggedRobot"
        )
        defined_methods = {
            node.name for node in force_class.body if isinstance(node, ast.FunctionDef)
        }
        self.assertTrue(
            observation_methods.isdisjoint(defined_methods),
            "carrybox_force must inherit Stage1 observation construction unchanged",
        )
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in observation_methods:
                names = {child.attr for child in ast.walk(node) if isinstance(child, ast.Attribute)}
                self.assertTrue(forbidden.isdisjoint(names), f"leaked names: {forbidden & names}")

    def test_nominal_command_buffers_have_no_force_or_teacher_rhs(self):
        source_path = (
            REPO_ROOT / "legged_gym" / "legged_gym" / "envs" / "g1" / "carrybox_force.py"
        )
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        protected = {"commands", "carry_policy_commands", "carry_heading_ref"}
        forbidden = {
            "external_force_world",
            "external_force_beta",
            "external_force_active",
            "teacher_vx",
            "teacher_heading",
            "teacher_yaw_rate",
            "force_parallel",
            "force_perp",
        }
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AugAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            target_attrs = {
                child.attr for target in targets for child in ast.walk(target)
                if isinstance(child, ast.Attribute)
            }
            if protected.isdisjoint(target_attrs):
                continue
            value_attrs = {
                child.attr for child in ast.walk(node.value) if isinstance(child, ast.Attribute)
            }
            self.assertTrue(forbidden.isdisjoint(value_attrs), f"command leakage: {forbidden & value_attrs}")


class TestArchitectureParity(unittest.TestCase):
    def test_force_config_inherits_actor_critic_interface(self):
        tree = ast.parse(CONFIG_PATH.read_text(encoding="utf-8"))
        env_cfg = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "G1Cfg")
        ppo_cfg = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "G1CfgPPO")
        self.assertEqual(ast.unparse(env_cfg.bases[0]), "CarryBoxCfg")
        self.assertEqual(ast.unparse(ppo_cfg.bases[0]), "CarryBoxCfgPPO")
        self.assertNotIn("env", {node.name for node in env_cfg.body if isinstance(node, ast.ClassDef)})
        self.assertNotIn("policy", {node.name for node in ppo_cfg.body if isinstance(node, ast.ClassDef)})
        algorithm = next(
            node
            for node in ppo_cfg.body
            if isinstance(node, ast.ClassDef) and node.name == "algorithm"
        )
        self.assertEqual(ast.unparse(algorithm.bases[0]), "CarryBoxCfgPPO.algorithm")
        self.assertEqual(
            _class_assignments(algorithm),
            {"learning_rate": 3.0e-4, "schedule": "adaptive", "desired_kl": 0.01},
        )

    def test_stage1_checkpoint_dimensions(self):
        checkpoint_path = REPO_ROOT / "legged_gym" / "resources" / "ckpt" / "carrybox.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint["model_state_dict"]
        self.assertEqual(tuple(state["actor.0.weight"].shape), (512, 738))
        self.assertEqual(tuple(state["actor.6.weight"].shape), (29, 256))
        self.assertEqual(tuple(state["critic.0.weight"].shape), (512, 126))


class _StateRecorder:
    def __init__(self):
        self.loaded = []

    def load_state_dict(self, state):
        self.loaded.append(state)


class TestCheckpointLoadingSemantics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        runner_path = (
            REPO_ROOT / "rsl_rl" / "rsl_rl" / "runners" / "him_on_policy_runner.py"
        )
        tree = ast.parse(runner_path.read_text(encoding="utf-8"))
        runner_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "HIMOnPolicyRunner"
        )
        load_node = next(
            node
            for node in runner_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "load"
        )
        namespace = {"math": math, "torch": torch}
        method_module = ast.Module(body=[load_node], type_ignores=[])
        ast.fix_missing_locations(method_module)
        exec(compile(method_module, str(runner_path), "exec"), namespace)
        cls.load_method = staticmethod(namespace["load"])

    @staticmethod
    def _optimizer(group_lrs):
        parameters = [torch.nn.Parameter(torch.tensor(float(index))) for index in range(len(group_lrs))]
        optimizer = torch.optim.Adam(
            [
                {"params": [parameter], "lr": learning_rate}
                for parameter, learning_rate in zip(parameters, group_lrs)
            ]
        )
        for parameter in parameters:
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        return optimizer

    def _runner(self, group_count=1):
        return types.SimpleNamespace(
            device="cpu",
            cfg={"use_muon_optim": False},
            alg=types.SimpleNamespace(
                actor_critic=_StateRecorder(),
                amp=_StateRecorder(),
                optimizer=self._optimizer((3.0e-4,) * group_count),
                learning_rate=3.0e-4,
            ),
            current_learning_iteration=0,
        )

    def _checkpoint(self, optimizer_lrs):
        checkpoint = {
            "model_state_dict": {"actor_critic": "weights"},
            "amp_state_dict": {"amp": "weights"},
            "optimizer_state_dict": self._optimizer(optimizer_lrs).state_dict(),
            "iter": 1234,
            "infos": {"source": "unit-test"},
        }
        temp_dir = tempfile.TemporaryDirectory()
        path = pathlib.Path(temp_dir.name) / "model.pt"
        torch.save(checkpoint, path)
        return temp_dir, path

    def test_finetune_loads_weights_only_and_starts_at_zero(self):
        runner = self._runner()
        initial_optimizer_state = runner.alg.optimizer.state_dict()
        temp_dir, path = self._checkpoint((1.2e-4,))
        try:
            infos = self.load_method(
                runner,
                path,
                load_optimizer=False,
                load_iteration=False,
            )
        finally:
            temp_dir.cleanup()
        self.assertEqual(runner.alg.actor_critic.loaded, [{"actor_critic": "weights"}])
        self.assertEqual(runner.alg.amp.loaded, [{"amp": "weights"}])
        self.assertEqual(runner.alg.optimizer.state_dict(), initial_optimizer_state)
        self.assertEqual(runner.alg.optimizer.param_groups[0]["lr"], 3.0e-4)
        self.assertEqual(runner.alg.learning_rate, 3.0e-4)
        self.assertEqual(runner.current_learning_iteration, 0)
        self.assertEqual(infos, {"source": "unit-test"})

    def test_resume_synchronizes_adaptive_lr_and_restores_iteration(self):
        runner = self._runner()
        temp_dir, path = self._checkpoint((1.333e-4,))
        try:
            self.load_method(runner, path)
        finally:
            temp_dir.cleanup()
        self.assertEqual(runner.alg.actor_critic.loaded, [{"actor_critic": "weights"}])
        self.assertEqual(runner.alg.amp.loaded, [{"amp": "weights"}])
        self.assertEqual(runner.alg.optimizer.param_groups[0]["lr"], 1.333e-4)
        self.assertEqual(runner.alg.learning_rate, 1.333e-4)
        self.assertEqual(runner.current_learning_iteration, 1234)

    def test_resume_accepts_identical_lr_across_optimizer_groups(self):
        runner = self._runner(group_count=3)
        temp_dir, path = self._checkpoint((1.5e-4, 1.5e-4, 1.5e-4))
        try:
            self.load_method(runner, path)
        finally:
            temp_dir.cleanup()
        self.assertEqual(
            [group["lr"] for group in runner.alg.optimizer.param_groups],
            [1.5e-4, 1.5e-4, 1.5e-4],
        )
        self.assertEqual(runner.alg.learning_rate, 1.5e-4)

    def test_resume_rejects_inconsistent_optimizer_group_lrs(self):
        runner = self._runner(group_count=2)
        temp_dir, path = self._checkpoint((1.0e-4, 3.0e-4))
        try:
            with self.assertRaisesRegex(ValueError, "single shared optimizer LR"):
                self.load_method(runner, path)
        finally:
            temp_dir.cleanup()

    def test_task_registry_dispatches_explicit_loading_modes(self):
        registry_path = (
            REPO_ROOT / "legged_gym" / "legged_gym" / "utils" / "task_registry.py"
        )
        tree = ast.parse(registry_path.read_text(encoding="utf-8"))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "runner"
            and node.func.attr == "load"
        ]
        self.assertEqual(len(calls), 2)
        resume_call = next(call for call in calls if ast.unparse(call.args[0]) == "resume_path")
        finetune_call = next(
            call for call in calls if ast.unparse(call.args[0]) == "finetune_path"
        )
        self.assertEqual(resume_call.keywords, [])
        self.assertEqual(
            {keyword.arg: ast.literal_eval(keyword.value) for keyword in finetune_call.keywords},
            {"load_optimizer": False, "load_iteration": False},
        )


if __name__ == "__main__":
    unittest.main()
