"""Opt-in real CUDA timings for M1 smoke runs; no PPO or checkpoint policy changes."""
import json
import math
from pathlib import Path
import time


def summarize(rows):
    updates = [row for row in rows if not row["skipped_update"]]
    mixed_steps = sum(row["mixed_steps"] for row in rows)
    elapsed = sum(row["iteration_seconds"] for row in rows)
    update_elapsed = sum(row["iteration_seconds"] for row in updates)
    finite = all(row.get("losses_finite", True) for row in rows)
    return dict(
        status=("nonfinite_training" if not finite else "measured_student_ppo" if updates and mixed_steps
                else "incomplete_student_coverage"),
        iterations=len(rows), ppo_updates=len(updates),
        valid_student_samples=sum(row["valid_student_samples"] for row in rows),
        mixed_steps=mixed_steps, ramp_steps=sum(row["ramp_steps"] for row in rows),
        overall_env_steps_per_second=sum(row["env_steps"] for row in rows) / elapsed if elapsed else None,
        student_update_env_steps_per_second=(sum(row["env_steps"] for row in updates) / update_elapsed
                                             if update_elapsed else None),
        mean_rollout_seconds=sum(row["rollout_seconds"] for row in rows) / len(rows) if rows else None,
        mean_ppo_update_seconds=sum(row["update_seconds"] for row in updates) / len(updates) if updates else None,
        peak_torch_allocated_bytes=max((row["peak_torch_allocated_bytes"] for row in rows), default=None),
        peak_torch_reserved_bytes=max((row["peak_torch_reserved_bytes"] for row in rows), default=None),
        max_sampled_device_used_bytes=max((row["device_used_bytes"] for row in rows), default=None),
        min_sampled_device_free_bytes=min((row["device_free_bytes"] for row in rows), default=None))


class PerformanceSmokeRecorder:
    def __init__(self, path, device, num_envs, num_steps, save_interval):
        import torch
        self.cuda = torch.cuda
        if torch.device(device).type != "cuda" or not self.cuda.is_available():
            raise ValueError("Performance smoke recording requires real CUDA")
        self.device = device
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise FileExistsError(self.path)
        self.rows_path = self.path.with_suffix(".jsonl")
        self.rows_path.open("x", encoding="utf-8").close()
        props = self.cuda.get_device_properties(device)
        self.configuration = dict(num_envs=num_envs, num_steps_per_env=num_steps,
                                  save_interval=save_interval, device=str(device), gpu=props.name,
                                  total_device_memory_bytes=props.total_memory)
        self.rows = []

    def begin_iteration(self):
        self.cuda.synchronize(self.device)
        self.started = time.perf_counter()

    def rollout_done(self):
        self.cuda.synchronize(self.device)
        self.rollout_end = time.perf_counter()

    def update_done(self):
        self.cuda.synchronize(self.device)
        self.update_end = time.perf_counter()

    def record(self, iteration, diagnostics, mixed_values, checkpoint_seconds, losses):
        self.cuda.synchronize(self.device)
        free, total = self.cuda.mem_get_info(self.device)
        valid = diagnostics["valid_samples"]
        row = dict(iteration=iteration, valid_student_samples=valid,
            losses_finite=all(math.isfinite(float(value)) for value in losses),
            skipped_update=bool(diagnostics["skipped_update"]),
            mixed_steps=int(sum(mixed_values.get(mode + "/all/steps", 0) for mode in
                                ("stand", "vx", "vy", "yaw", "mixed"))),
            ramp_steps=round(mixed_values.get("ramp_ratio", 0) * valid),
            env_steps=self.configuration["num_envs"] * self.configuration["num_steps_per_env"],
            rollout_seconds=self.rollout_end - self.started,
            update_seconds=self.update_end - self.rollout_end,
            iteration_seconds=time.perf_counter() - self.started,
            checkpoint_seconds=checkpoint_seconds,
            torch_allocated_bytes=self.cuda.memory_allocated(self.device),
            torch_reserved_bytes=self.cuda.memory_reserved(self.device),
            peak_torch_allocated_bytes=self.cuda.max_memory_allocated(self.device),
            peak_torch_reserved_bytes=self.cuda.max_memory_reserved(self.device),
            device_free_bytes=free, device_used_bytes=total - free)
        self.rows.append(row)
        with self.rows_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")

    def finish(self, log_dir, final_checkpoint_seconds):
        result = dict(configuration=self.configuration, summary=summarize(self.rows),
                      run_dir=str(log_dir), final_checkpoint_seconds=final_checkpoint_seconds,
                      iteration_rows=str(self.rows_path),
                      memory_scope="Torch peaks include initialization and updates; device usage includes other processes")
        with self.path.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
        return result
