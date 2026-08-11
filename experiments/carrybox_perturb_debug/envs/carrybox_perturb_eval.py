import hashlib
import json
import math

import numpy as np
import torch
from isaacgym import gymapi, gymtorch

from evaluation.force_profiles import smooth_hold_total_duration
from envs.carrybox_perturb_debug import LeggedRobot as CarryBoxPerturbDebug


class LeggedRobot(CarryBoxPerturbDebug):
    """Controlled, quiet evaluator wrapper around the debug carrybox env."""

    _PROFILE_IDS = {
        "half_sine": 0,
        "smooth_hold": 1,
    }

    def _init_buffers(self):
        super()._init_buffers()
        self.box_perturb_profile_id = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.box_perturb_ramp_up_s = torch.zeros(self.num_envs, device=self.device)
        self.box_perturb_hold_duration_s = torch.zeros(self.num_envs, device=self.device)
        self.box_perturb_ramp_down_s = torch.zeros(self.num_envs, device=self.device)
        self.box_perturb_eval_impulse_Ns = torch.zeros(self.num_envs, device=self.device)

    def _evaluation_verbose(self):
        cfg = getattr(self.cfg.box_perturbation, "evaluation_verbose", False)
        return bool(cfg)

    def _debug_log_fixed_scene(self, env_ids):
        if self._evaluation_verbose():
            super()._debug_log_fixed_scene(env_ids)

    def _debug_log_carry_gate(self, debug):
        if self._evaluation_verbose():
            super()._debug_log_carry_gate(debug)

    def _debug_log_applied_force(self, phase):
        if self._evaluation_verbose():
            super()._debug_log_applied_force(phase)

    def _debug_log_force_viewer_check_once(
        self,
        env_id,
        force_world,
        draw_vector,
        box_com_world,
        box_com_draw_frame,
        env_origin,
        epsilon,
    ):
        if self._evaluation_verbose():
            super()._debug_log_force_viewer_check_once(
                env_id,
                force_world,
                draw_vector,
                box_com_world,
                box_com_draw_frame,
                env_origin,
                epsilon,
            )

    def schedule_evaluation_force(
        self,
        direction_name,
        beta,
        profile,
        env_id=0,
        pulse_duration_s=None,
        hold_duration_s=1.0,
        ramp_up_s=0.15,
        ramp_down_s=0.15,
    ):
        if direction_name not in self._DIRECTION_IDS:
            raise ValueError(f"Unknown perturbation direction: {direction_name}")
        if profile not in self._PROFILE_IDS:
            raise ValueError(f"Unknown force profile: {profile}")

        threshold = int(
            self.cfg.box_perturbation.stable_confirmed_carry_policy_steps
        )
        if int(self.confirmed_carry_streak[env_id].item()) < threshold:
            raise RuntimeError(
                "Perturbation requested before confirmed-carry gate: "
                f"streak={int(self.confirmed_carry_streak[env_id])}, threshold={threshold}"
            )
        if int(self.box_perturb_remaining_physics_steps[env_id].item()) != 0:
            raise RuntimeError("A box perturbation event is already active")

        if profile == "smooth_hold":
            total_duration_s = smooth_hold_total_duration(
                hold_duration_s, ramp_up_s, ramp_down_s
            )
        else:
            total_duration_s = float(pulse_duration_s)

        env_ids = torch.tensor([env_id], dtype=torch.long, device=self.device)
        direction_local = torch.zeros((1, 3), device=self.device)
        direction_is_world = torch.zeros(1, dtype=torch.bool, device=self.device)
        component = 0 if direction_name[-1] == "x" else 1
        direction_local[0, component] = 1.0 if direction_name[0] == "+" else -1.0
        beta_tensor = torch.tensor([float(beta)], device=self.device)
        direction_ids = torch.tensor(
            [self._DIRECTION_IDS[direction_name]],
            dtype=torch.long,
            device=self.device,
        )

        self.box_perturb_profile_id[env_ids] = self._PROFILE_IDS[profile]
        self.box_perturb_ramp_up_s[env_ids] = float(ramp_up_s)
        self.box_perturb_hold_duration_s[env_ids] = float(hold_duration_s)
        self.box_perturb_ramp_down_s[env_ids] = float(ramp_down_s)
        self.box_perturb_eval_impulse_Ns[env_ids] = 0.0
        self._freeze_force_trace_baseline()
        self.set_box_perturb_trace_phase("force")
        self._commit_box_perturbation(
            env_ids,
            direction_local,
            direction_is_world,
            beta_tensor,
            direction_ids,
            f"evaluation:{profile}:{direction_name}",
            pulse_duration_s=total_duration_s,
        )
        direction_world = self.box_perturb_direction_world[env_id].detach().clone()
        return {
            "direction_world": [
                float(value) for value in direction_world.detach().cpu().tolist()
            ],
            "box_mass_kg": float(self.box_masses[env_id].item()),
            "target_force_N": float(self.box_perturb_peak_force_N[env_id].item()),
            "total_duration_s": float(total_duration_s),
        }

    def _apply_box_perturbation_force(self):
        """Apply the active evaluation force profile at the box COM."""
        self.box_perturb_force_tensor.zero_()
        self.box_perturb_actual_force_scale.zero_()
        active = (
            bool(self.cfg.box_perturbation.enabled)
            & (self.box_perturb_remaining_physics_steps > 0)
        )
        if torch.any(active):
            active_ids = torch.nonzero(active, as_tuple=False).flatten()
            for env_id_tensor in active_ids:
                env_id = int(env_id_tensor.item())
                profile = self._profile_scale(env_id)
                force = (
                    profile
                    * self.box_perturb_peak_force_N[env_id]
                    * self.box_perturb_direction_world[env_id]
                )
                self.box_perturb_force_tensor[
                    env_id, int(self.box_net_contact_force_index), :
                ] = force
                self.box_perturb_actual_force_scale[env_id] = profile
                self.box_perturb_debug_draw_force_N[env_id] = torch.linalg.vector_norm(
                    force
                )
                self.box_perturb_eval_impulse_Ns[env_id] += (
                    torch.linalg.vector_norm(force) * float(self.sim_params.dt)
                )

        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            forceTensor=gymtorch.unwrap_tensor(self.box_perturb_force_tensor),
            space=gymapi.CoordinateSpace.GLOBAL_SPACE,
        )

        if torch.any(active):
            self.box_perturb_elapsed_physics_steps[active] += 1
            self.box_perturb_remaining_physics_steps[active] -= 1

    def _profile_scale(self, env_id):
        elapsed_s = (
            float(self.box_perturb_elapsed_physics_steps[env_id].item()) + 0.5
        ) * float(self.sim_params.dt)
        profile_id = int(self.box_perturb_profile_id[env_id].item())
        if profile_id == self._PROFILE_IDS["smooth_hold"]:
            return self._smooth_hold_scale(env_id, elapsed_s)
        return self._half_sine_scale(env_id, elapsed_s)

    def _half_sine_scale(self, env_id, elapsed_s):
        duration_s = max(float(self.box_perturb_pulse_duration_s[env_id].item()), 1.0e-9)
        tau = min(max(float(elapsed_s) / duration_s, 0.0), 1.0)
        return math.sin(math.pi * tau)

    def _smooth_hold_scale(self, env_id, elapsed_s):
        ramp_up_s = max(float(self.box_perturb_ramp_up_s[env_id].item()), 0.0)
        hold_duration_s = max(
            float(self.box_perturb_hold_duration_s[env_id].item()), 0.0
        )
        ramp_down_s = max(float(self.box_perturb_ramp_down_s[env_id].item()), 0.0)
        hold_start = ramp_up_s
        ramp_down_start = ramp_up_s + hold_duration_s
        total_s = ramp_down_start + ramp_down_s

        if elapsed_s < 0.0 or elapsed_s > total_s:
            return 0.0
        if ramp_up_s > 0.0 and elapsed_s < hold_start:
            tau = elapsed_s / ramp_up_s
            return 0.5 * (1.0 - math.cos(math.pi * tau))
        if elapsed_s < ramp_down_start:
            return 1.0
        if ramp_down_s > 0.0:
            tau = (elapsed_s - ramp_down_start) / ramp_down_s
            return 0.5 * (1.0 + math.cos(math.pi * min(max(tau, 0.0), 1.0)))
        return 0.0

    def evaluation_initial_state_signature(self, env_id=0):
        data = {
            "robot_root_pose": self._tensor_list(
                torch.cat((self.root_states[env_id, 0:3], self.root_states[env_id, 3:7]))
            ),
            "robot_root_linear_velocity": self._tensor_list(
                self.root_states[env_id, 7:10]
            ),
            "robot_root_angular_velocity": self._tensor_list(
                self.root_states[env_id, 10:13]
            ),
            "dof_position": self._tensor_list(self.dof_pos[env_id]),
            "dof_velocity": self._tensor_list(self.dof_vel[env_id]),
            "box_pose": self._tensor_list(
                torch.cat((self.box_states[env_id, 0:3], self.box_states[env_id, 3:7]))
            ),
            "box_linear_velocity": self._tensor_list(self.box_states[env_id, 7:10]),
            "box_angular_velocity": self._tensor_list(self.box_states[env_id, 10:13]),
            "goal": self._tensor_list(self.goal_pos[env_id]),
            "box_mass": float(self.box_masses[env_id].item()),
        }
        payload = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return {
            "data": data,
            "json": payload,
            "sha1": hashlib.sha1(payload.encode("utf-8")).hexdigest(),
        }

    @staticmethod
    def _tensor_list(tensor):
        return [
            round(float(value), 9)
            for value in tensor.detach().cpu().reshape(-1).tolist()
        ]

    def _draw_debug_vis(self):
        """Draw the real instantaneous physical external force only."""
        self.gym.clear_lines(self.viewer)
        cfg = self.cfg.box_perturbation
        if not bool(cfg.debug_draw_force):
            return

        scale = float(cfg.debug_force_draw_scale_m_per_N)
        line_count = max(1, int(cfg.debug_force_bundle_line_count))
        jitter = max(0.0, float(cfg.debug_force_bundle_jitter_m))
        max_envs = min(self.num_envs, int(cfg.debug_force_draw_max_envs))
        color = np.asarray([0.851, 0.144, 0.07], dtype=np.float32)

        for env_id in range(max_envs):
            force_t = self.box_perturb_force_tensor[
                env_id, int(self.box_net_contact_force_index), :
            ]
            magnitude = float(torch.linalg.vector_norm(force_t).item())
            if magnitude <= 1.0e-6:
                continue

            start = (self.box_states[env_id, 0:3] - self.env_origins[env_id]).detach().cpu().numpy()
            end = start + force_t.detach().cpu().numpy() * scale
            if jitter > 0.0:
                offset = (
                    np.random.random((line_count, 3)).astype(np.float32) - 0.5
                ) * jitter
            else:
                offset = np.zeros((line_count, 3), dtype=np.float32)
            starts = np.repeat(start.reshape(1, 3), line_count, axis=0) + offset
            ends = np.repeat(end.reshape(1, 3), line_count, axis=0) + offset
            vertices = np.concatenate((starts, ends), axis=1).astype(np.float32)
            colors = np.repeat(color.reshape(1, 3), line_count, axis=0)
            self.gym.add_lines(
                self.viewer,
                self.envs[env_id],
                line_count,
                vertices,
                colors,
            )
