"""Read-only exact-input/runtime/geometry audit. Does not launch simulation or training."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rsl_rl"))
sys.path.insert(0, str(ROOT / "legged_gym/legged_gym/scripts"))


def audit():
    # Match Isaac Gym's required import order even in an audit process.
    runtime = dict(interpreter=sys.executable, python=platform.python_version(), platform=platform.platform())
    blockers = []
    try:
        __import__("isaacgym")
        runtime["isaacgym_import"] = True
    except ImportError as error:
        runtime["isaacgym_import"] = False
        runtime["isaacgym_error"] = str(error)
        blockers.append("Isaac Gym cannot be imported")
    import torch
    from rsl_rl.teacher_handoff import checkpoint_actor, file_sha256, read_checkpoint
    from gait_diagnostics import build_gait_geometry, validate_a5_contract, validate_b2_reward_contract
    runtime.update(torch=torch.__version__, cuda_available=torch.cuda.is_available())
    if not runtime["cuda_available"]:
        blockers.append("CUDA is unavailable")
    paths = {
        "teacher": ROOT / "shared_artifacts/2026-10-19/carry_track/model_33999.pt",
        "student": ROOT / "legged_gym/logs/carrybox_teacher_warmup/20261019_235711_985014_A5_fixed/model_5000.pt",
        "reference": ROOT / "experiments/teacher_warmup/results/lab_20261019_233908/full_teacher_upper.json",
    }
    files = {name: dict(path=str(path), exists=path.is_file(), sha256=file_sha256(path) if path.is_file() else None)
             for name, path in paths.items()}
    for name, info in files.items():
        if not info["exists"]:
            blockers.append("Missing exact " + name + ": " + info["path"])
    configuration = json.loads((ROOT / "experiments/teacher_warmup/config_B2_gait_vx08.json").read_text())
    provenance = None
    if all(info["exists"] for info in files.values()):
        try:
            checkpoint = read_checkpoint(paths["student"])
            _, spec = checkpoint_actor(paths["student"])
            saved = checkpoint["handoff_contract"]
            handoff = dict(saved["handoff"], **configuration["handoff"])
            names = [line.split()[1] for line in (ROOT / "legged_gym/resources/config/joint_id.txt").read_text().splitlines() if line.strip()]
            validate_a5_contract(checkpoint, spec, files["teacher"]["sha256"], files["reference"]["sha256"],
                                 handoff, names, .25, .02)
            if spec["dims"] != [738, 512, 256, 256, 29] or checkpoint["iter"] != 5000 or saved.get("critic_obs") != 160:
                raise ValueError("Not the required trained A5 architecture/iteration/critic")
            import copy
            rewards = copy.deepcopy(saved["rewards"])
            rewards["scales"].update(configuration["rewards"])
            validate_b2_reward_contract(saved["rewards"], rewards)
            provenance = dict(actor_spec=spec, original_iteration=checkpoint["iter"],
                              original_warmup_vx=saved["handoff"].get("warmup_vx"),
                              original_fixed_command=saved["handoff"].get("fixed_command"),
                              teacher_and_adapter_contract_verified=True)
        except (ValueError, KeyError) as error:
            blockers.append("A5 provenance audit failed: " + str(error))
    geometry = build_gait_geometry(ROOT / "legged_gym/resources/robots/g1/urdf/g1_29dof.urdf")
    return dict(experiment="B2", status="blocked_preflight" if blockers else "ready_for_physical_smoke",
                simulation_executed=False, training_executed=False, inputs=files, runtime=runtime,
                configuration=configuration, a5_provenance=provenance, geometry=geometry, blockers=blockers,
                uncertainty="Geometry errors are sampled full-sphere mesh support errors, not measured PhysX errors.",
                next_required_validation="Physical B2 reward smoke, Actor/std equality and fresh optimizer checks before training.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit()
    directory = args.output or ROOT / "experiments/teacher_warmup/results" / ("B2_preflight_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": result["status"], "report": str(directory / "report.json"),
                      "blockers": result["blockers"],
                      "geometry_errors": {k: v["error"] for k, v in result["geometry"]["bodies"].items()}}, indent=2))
    return 2 if result["blockers"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
