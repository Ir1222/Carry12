"""Task-local adapter: no moving-carry experts and no cross-clip interpolation."""
import torch


class PickupStandMotionSampler:
    def __init__(self, library):
        self.library = library
        if set(library.skills) != {"pickUp", "loco"}:
            raise ValueError("Pickup-stand requires exactly pickUp and loco motions")

    def __getattr__(self, key):
        return getattr(self.library, key)

    def get_amp_hist_obs(self, skill, motion_ids, motion_times):
        lib = self.library
        offsets = torch.arange(lib.window_length - 1, -1, -1, device=lib.device) * (lib.fps / lib.env_fps)
        times = (motion_times[:, None] - offsets).clamp_min(0.)
        times = torch.minimum(times, (lib.motion_len[skill][motion_ids] - 1)[:, None])
        low = times.floor().long()
        high = torch.minimum(low + 1, (lib.motion_len[skill][motion_ids] - 1)[:, None])
        start = lib.motion_start_ids[skill][motion_ids, None]
        weight = (times - low).unsqueeze(-1)
        features = [lib.motion_base_height, lib.motion_dof_pos, lib.motion_end_effector_pos,
                    lib.motion_box_pos, lib.motion_base_lin_vel, lib.motion_base_ang_vel,
                    lib.motion_base_rot]
        return torch.cat([x[start + low] * (1 - weight) + x[start + high] * weight
                          for x in features], dim=-1).flatten(1)

    def get_expert_obs_for_stages(self, stages):
        if not torch.all((stages == 0) | (stages == 1)):
            raise ValueError("HOLD must never enter the dynamic discriminator")
        result = torch.empty(len(stages), 600, device=self.device)
        for stage, skill in [(0, "loco"), (1, "pickUp")]:
            mask = stages == stage
            count = int(mask.sum())
            if count:
                ids = self.sample_motions(skill, count)
                # Histories end at t, rather than stepping past a clip boundary.
                span = (self.window_length - 1) * self.fps / self.env_fps
                length = self.motion_len[skill][ids] - 1
                if torch.any(length <= span):
                    raise ValueError("Expert clip shorter than AMP history")
                times = span + torch.rand(count, device=self.device) * (length - span)
                result[mask] = self.get_amp_hist_obs(skill, ids, times)
        return result

    def object_velocity(self, skill, ids, times):
        # Match the position convention and Z correction of get_obj_motion_state.
        previous = (times - 0.5).clamp_min(0)
        following = torch.minimum(times + 0.5, self.motion_len[skill][ids] - 1)
        p0, q0, _, _ = self.get_obj_motion_state(skill, ids, previous)
        p1, q1, _, _ = self.get_obj_motion_state(skill, ids, following)
        duration = ((following - previous) / self.fps).clamp_min(1.e-6)[:, None]
        linear = (p1 - p0) / duration
        yaw0 = torch.atan2(2 * (q0[:, 3] * q0[:, 2] + q0[:, 0] * q0[:, 1]), 1 - 2 * (q0[:, 1] ** 2 + q0[:, 2] ** 2))
        yaw1 = torch.atan2(2 * (q1[:, 3] * q1[:, 2] + q1[:, 0] * q1[:, 1]), 1 - 2 * (q1[:, 1] ** 2 + q1[:, 2] ** 2))
        angular = torch.zeros_like(linear)
        angular[:, 2] = torch.atan2(torch.sin(yaw1 - yaw0), torch.cos(yaw1 - yaw0)) / duration[:, 0]
        return linear, angular
