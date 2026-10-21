"""Run isolated 1024/2048/4096-env M1 CUDA memory/performance smoke cases.

Uses the real training entrypoint and real teacher pickup. No synthetic Student
samples, shortened handoff budgets, reward changes or full production run.
"""
import argparse
import csv
from datetime import datetime
import io
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_COUNTS = (1024, 2048, 4096)


def runtime_probe():
    runtime = dict(interpreter=sys.executable, python=platform.python_version(), platform=platform.platform())
    blockers = []
    # Required ordering even for this read-only probe.
    try:
        __import__("isaacgym")
        runtime["isaacgym_import"] = True
    except Exception as error:
        runtime.update(isaacgym_import=False, isaacgym_error=str(error))
        blockers.append("Isaac Gym import failed")
    try:
        import torch
        runtime.update(torch=torch.__version__, cuda_available=torch.cuda.is_available())
        if not runtime["cuda_available"]:
            blockers.append("CUDA is unavailable")
    except Exception as error:
        runtime.update(torch_error=str(error), cuda_available=False)
        blockers.append("PyTorch import failed")
    return runtime, blockers


def build_command(args, num_envs, report):
    return [sys.executable, "-u", str(ROOT / "legged_gym/legged_gym/scripts/train_teacher_warmup_loco.py"),
            "--task", "carrybox_teacher_warmup_mixed", "--teacher_checkpoint", str(args.teacher_checkpoint),
            "--posture_reference", str(args.posture_reference), "--init_policy_path", str(args.init_policy_path),
            "--num_envs", str(num_envs), "--max_iterations", str(args.iterations), "--seed", str(args.seed),
            "--run_name", "M1_smoke_" + str(num_envs), "--rl_device", args.rl_device, "--headless",
            "--smoke_report", str(report)]


def gpu_sample():
    """Device-wide sampling includes PhysX and concurrent GPU workloads."""
    try:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                timeout=5, creationflags=flags)
        if result.returncode:
            return None
        return [dict(index=int(row[0]), uuid=row[1].strip(), name=row[2].strip(),
                     total_MiB=float(row[3]), used_MiB=float(row[4]), utilization_percent=float(row[5]))
                for row in csv.reader(io.StringIO(result.stdout)) if row]
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def run_case(args, num_envs, directory):
    report = directory / (str(num_envs) + ".json")
    log = directory / (str(num_envs) + ".log")
    command = build_command(args, num_envs, report)
    print("[M1 smoke] starting " + str(num_envs) + " envs", flush=True)
    started = time.perf_counter()
    device_peaks = {}
    timed_out = False
    with log.open("x", encoding="utf-8") as stream:
        child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            while True:
                for gpu in gpu_sample() or []:
                    peak = device_peaks.setdefault(gpu["uuid"], dict(gpu, samples=0))
                    peak["used_MiB"] = max(peak["used_MiB"], gpu["used_MiB"])
                    peak["samples"] += 1
                if time.perf_counter() - started > args.timeout_s:
                    timed_out = True
                    child.kill()
                    child.wait()
                    break
                try:
                    child.wait(timeout=.5)
                    break
                except subprocess.TimeoutExpired:
                    pass
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
    result = dict(num_envs=num_envs, command=command, returncode=child.returncode,
                  wall_seconds=time.perf_counter() - started, log=str(log),
                  nvidia_smi_peak_device_usage=list(device_peaks.values()) or None,
                  nvidia_smi_scope="All visible physical GPUs; device-wide, includes other processes")
    if report.exists():
        result["measurement"] = json.loads(report.read_text(encoding="utf-8"))
    elif report.with_suffix(".jsonl").exists():
        # Preserve partial real measurements if an update OOMs after earlier iterations.
        rows = [json.loads(line) for line in report.with_suffix(".jsonl").read_text().splitlines() if line.strip()]
        sys.path.insert(0, str(ROOT / "rsl_rl"))
        from rsl_rl.m1_performance import summarize
        result["partial_measurement"] = summarize(rows)
    if timed_out:
        result["status"] = "timeout"
    elif child.returncode:
        tail = log.read_text(encoding="utf-8", errors="replace")[-16000:]
        result["status"] = ("out_of_memory" if any(term in tail.lower() for term in
                            ("out of memory", "cuda_error_out_of_memory", "failed to allocate")) else "failed")
        result["error_tail"] = tail[-3000:]
    elif "measurement" not in result:
        result["status"] = "missing_measurements"
    else:
        result["status"] = result["measurement"]["summary"]["status"]
    print("[M1 smoke] " + str(num_envs) + " envs: " + result["status"], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher_checkpoint", type=Path)
    parser.add_argument("--posture_reference", type=Path)
    parser.add_argument("--init_policy_path", type=Path)
    parser.add_argument("--num_envs", type=int, nargs="+", default=list(DEFAULT_ENV_COUNTS))
    parser.add_argument("--iterations", type=int, default=40,
                        help="Bounded smoke iterations per case; rollout remains 100 steps")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--rl_device", default="cuda:0")
    parser.add_argument("--timeout_s", type=float, default=1800.)
    parser.add_argument("--probe_only", action="store_true", help="Report availability; launch no simulation")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (args.iterations <= 0 or any(n <= 0 for n in args.num_envs)
            or len(set(args.num_envs)) != len(args.num_envs) or not 0 < args.timeout_s < float("inf")):
        parser.error("Require positive iterations, unique positive env counts and finite positive timeout")
    directory = args.output or ROOT / "outputs" / ("M1_smoke_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    directory.mkdir(parents=True, exist_ok=False)
    runtime, blockers = runtime_probe()
    for name in ("teacher_checkpoint", "posture_reference", "init_policy_path"):
        value = getattr(args, name)
        if value is None and args.probe_only:
            continue
        if value is None or not value.is_file():
            blockers.append("Missing input " + name + ": " + str(value))
        else:
            setattr(args, name, value.resolve())
    report = dict(runtime=runtime, blockers=blockers, probe_only=args.probe_only,
                  production_settings=dict(num_envs=4096, max_iterations=10000, num_steps_per_env=100,
                                           save_interval=500, learning_rate=5e-5, min_valid_samples=2048),
                  smoke_iterations_per_case=args.iterations, cases=[])
    if blockers or args.probe_only:
        report["status"] = "blocked" if blockers else "ready_not_executed"
        report["cases"] = [dict(num_envs=n, status="not_executed", peak_memory_bytes=None,
                                env_steps_per_second=None) for n in args.num_envs]
    else:
        for count in args.num_envs:
            report["cases"].append(run_case(args, count, directory))
            # A durable partial matrix remains reviewable between isolated cases.
            (directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        report["status"] = ("passed" if all(case["status"] == "measured_student_ppo" for case in report["cases"])
                            else "not_passed")
    (directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(dict(status=report["status"], report=str(directory / "report.json"), blockers=blockers), indent=2))
    return 0 if report["status"] in ("passed", "ready_not_executed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
