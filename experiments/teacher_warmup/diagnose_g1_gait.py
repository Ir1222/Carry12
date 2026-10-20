"""Experiment 0 preflight, then 3-episode smoke and 2 x 10-episode evaluation.

Run from any directory. No training entrypoint is invoked. Missing artifacts
produce an explicit blocked report; historical locomotion weights are never used
as an A5 substitute. An explicit student/reference can be supplied after copying
the actual artifacts from the training machine.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "legged_gym/legged_gym/scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "rsl_rl"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher_checkpoint", default=str(ROOT / "shared_artifacts/2026-10-19/carry_track/model_33999.pt"))
    parser.add_argument("--student_checkpoint", default="")
    parser.add_argument("--posture_reference", default="")
    parser.add_argument("--handoff_config", default=str(ROOT / "experiments/teacher_warmup/config_gait_diagnostics_vx08.json"))
    parser.add_argument("--output")
    parser.add_argument("--preflight_only", action="store_true")
    own = parser.parse_args()
    out = Path(own.output) if own.output else ROOT / "experiments/teacher_warmup/results" / (
        "gait_diagnostics_vx08_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    out.mkdir(parents=True, exist_ok=False)
    from gait_diagnostics import foot_geometry, GaitDiagnostics, SETTINGS
    from rsl_rl.teacher_handoff import read_checkpoint, file_sha256, checkpoint_actor
    import torch
    inventory, blockers = [], []
    teacher = Path(own.teacher_checkpoint)
    teacher_hash = file_sha256(teacher) if teacher.is_file() else None
    if not teacher_hash:
        blockers.append("Full Teacher checkpoint missing: " + str(teacher))
    paths = sorted(set(ROOT.rglob("model_*.pt")))
    if own.student_checkpoint:
        paths = sorted(set(paths + [Path(own.student_checkpoint).resolve()]))
    candidates = []
    for path in paths:
        try:
            ckpt = read_checkpoint(path)
            contract = ckpt.get("handoff_contract") or {}
            state = ckpt.get("model_state_dict", {})
            weights = [v for k, v in state.items() if k.startswith("actor.") and k.endswith(".weight")]
            item = dict(path=str(path), bytes=path.stat().st_size, iteration=ckpt.get("iter"),
                        sha256=file_sha256(path),
                        actor_input=int(weights[0].shape[1]) if weights else None,
                        actor_output=int(weights[-1].shape[0]) if weights else None,
                        ablation=contract.get("handoff", {}).get("ablation"),
                        teacher_sha256=contract.get("teacher_sha256"),
                        reference_sha256=contract.get("reference_sha256"),
                        has_handoff_contract=bool(contract), modified_ns=path.stat().st_mtime_ns)
            inventory.append(item)
            if item["ablation"] == "A5" and item["iteration"] and item["iteration"] > 0:
                candidates.append(item)
            del ckpt, state, weights
        except Exception as error:
            inventory.append(dict(path=str(path), error=str(error)))
    if not own.student_checkpoint and candidates:
        own.student_checkpoint = max(candidates, key=lambda x: x["modified_ns"])["path"]
    if not own.student_checkpoint:
        blockers.append("No trained A5 checkpoint with handoff_contract found in repository (including ignored logs)")
    student_exists = bool(own.student_checkpoint and Path(own.student_checkpoint).is_file())
    if own.student_checkpoint and not student_exists:
        blockers.append("Explicit A5 checkpoint does not exist: " + own.student_checkpoint)
    if student_exists and not own.posture_reference:
        contract = read_checkpoint(own.student_checkpoint).get("handoff_contract") or {}
        wanted = contract.get("reference_sha256")
        if wanted:
            for path in ROOT.rglob("*.json"):
                if path.stat().st_size < 10_000_000 and file_sha256(path) == wanted:
                    own.posture_reference = str(path)
                    break
    if not own.posture_reference or not Path(own.posture_reference).is_file():
        blockers.append("Exact A5 Teacher posture reference JSON unavailable; no reference is synthesized")
    # Probe in a fresh interpreter, with Isaac Gym before torch, just as runtime requires.
    probe = subprocess.run([sys.executable, "-c", "import isaacgym; import torch; print(torch.cuda.is_available())"],
                           capture_output=True, text=True, cwd=str(ROOT))
    if probe.returncode:
        blockers.append("Isaac Gym import unavailable in selected interpreter: " + (probe.stderr.strip().splitlines()[-1] if probe.stderr.strip() else "unknown import failure"))
    if not torch.cuda.is_available():
        blockers.append("CUDA unavailable in selected interpreter (torch=" + torch.__version__ + ")")
    geometry = foot_geometry(ROOT / "legged_gym/resources/robots/g1/urdf/g1_29dof.urdf")
    geometry_report = {name: dict(bounds=g["bounds"], sole_centroid_local=g["sole"].tolist(),
                                  vertex_count=len(g["vertices"]), meshes=g["meshes"])
                       for name, g in geometry.items()}
    # Full strict contract/config validation is also performed before stepping
    # physics by the evaluator. Here validate the available artifacts on CPU.
    evaluation_overrides = {}
    if student_exists and own.posture_reference and Path(own.posture_reference).is_file():
        try:
            from gait_diagnostics import validate_a5_contract, evaluation_handoff_overrides
            from rsl_rl.teacher_handoff_reference import validate_reference
            ckpt = read_checkpoint(own.student_checkpoint)
            contract = ckpt.get("handoff_contract") or {}
            actor, spec = checkpoint_actor(own.student_checkpoint)
            reference = json.loads(Path(own.posture_reference).read_text(encoding="utf-8"))
            names = [line.split()[1] for line in (ROOT / "legged_gym/resources/config/joint_id.txt").read_text().splitlines()]
            validate_reference(reference, names)
            if reference["teacher_sha256"] != teacher_hash:
                raise ValueError("Posture reference Teacher hash mismatch")
            requested = dict(contract.get("handoff", {}))
            requested.update(json.loads(Path(own.handoff_config).read_text())["handoff"])
            validate_a5_contract(ckpt, spec, teacher_hash, file_sha256(own.posture_reference), requested,
                                 names, .25, .02, contract.get("rewards"))
            evaluation_overrides = evaluation_handoff_overrides(contract["handoff"], requested)
            del actor, ckpt
        except Exception as error:
            blockers.append("A5 artifact validation failed: " + str(error))
    manifest = dict(status="blocked_preflight" if blockers else "preflight_passed", experiment=0,
                    simulation_executed=False, training_executed=False, requested_episodes=20,
                    teacher_checkpoint=str(teacher), teacher_sha256=teacher_hash,
                    student_checkpoint=own.student_checkpoint or None, posture_reference=own.posture_reference or None,
                    configuration=json.loads(Path(own.handoff_config).read_text()),
                    interpreter=sys.executable, python=sys.version, platform=platform.platform(),
                    torch_version=torch.__version__, torch_cuda=torch.version.cuda,
                    cuda_available=torch.cuda.is_available(), isaacgym_probe=dict(returncode=probe.returncode,
                    stdout=probe.stdout, stderr=probe.stderr), checkpoint_inventory=inventory,
                    checkpoint_search_root=str(ROOT), geometry=geometry_report, settings=SETTINGS, blockers=blockers,
                    evaluation_handoff_overrides=evaluation_overrides)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True, text=True)
    manifest["git_revision"] = revision.stdout.strip() if revision.returncode == 0 else None
    sources = [SCRIPTS / f for f in ("gait_diagnostics.py", "analyze_gait_diagnostics.py", "eval_teacher_warmup_loco.py")]
    sources += [ROOT / "legged_gym/legged_gym/envs/g1/carrybox_teacher_warmup.py", Path(__file__)]
    manifest["source_sha256"] = {str(path.relative_to(ROOT)): file_sha256(path) for path in sources}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # Always leave reviewable results. Empty CSV is a schema, never rollout data.
    import csv
    names = [line.split()[1] for line in (ROOT / "legged_gym/resources/config/joint_id.txt").read_text().splitlines()]
    leg_names = [n for n in names if any(k in n for k in ("hip_", "knee_", "ankle_"))]
    with (out / "steps.csv").open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerow(["episode", "env", "time_s", "student_valid", "control_phase", "handoff_relative_s"]
                                   + GaitDiagnostics.fields(leg_names)
                                   + [f"{kind}:{name}" for kind in ("q", "dq", "raw_action", "applied_action", "pd_target") for name in names])
    commands = []
    for stage, seed, episodes in (("smoke_seed1", 1, 3), ("eval_seed1", 1, 10), ("eval_seed2", 2, 10)):
        command = [sys.executable, str(SCRIPTS / "eval_teacher_warmup_loco.py"), "--task", "carrybox_teacher_warmup",
                   "--teacher_checkpoint", str(teacher), "--student_checkpoint", own.student_checkpoint or "/path/to/verified_A5/model_N.pt",
                   "--posture_reference", own.posture_reference or "/path/to/exact_A5_posture_reference.json",
                   "--ablation", "A5", "--gait_diagnostics", "--handoff_config", str(Path(own.handoff_config).resolve()),
                   "--command_suite", "fixed", "--fixed_command", "0.8", "0", "0", "--num_envs", "1" if episodes == 3 else "4",
                   "--episodes", str(episodes), "--carry_seconds", "30" if episodes == 3 else "45", "--seed", str(seed),
                   "--headless", "--rl_device", "cuda:0", "--output", str(out / stage)]
        commands.append(command)
    (out / "commands.json").write_text(json.dumps(commands, indent=2), encoding="utf-8")
    report = ["# Experiment 0: local preflight", "", "Status: " + manifest["status"], "",
              "No physical rollout data were generated by preflight. steps.csv contains column headers only. "
              "No training, fine-tuning, checkpoint modification or Teacher policy change was performed.", "",
              "## Blockers", ""] + ["- " + b for b in blockers]
    report += ["", "## Confirmed code findings", "",
               "- Existing asset.foot_name selects ankle_pitch_link. feet_pos/feet_vel are that link's state, not the sole.",
               "- Existing filter is norm(force)>1 N, current OR previous; air-time increments before reward and clears after reward. "
               "The original reset does not clear last_contacts/contact_filt; diagnostic history is isolated and clears on episode/owner changes.",
               "- Both ankle links have independent active collision meshes. Commented spheres in the URDF are inactive.",
               "- Robot is created after three single-body actors. Diagnostic indices explicitly use DOMAIN_ENV; "
               "runtime index validation remains unverified without Gym.",
               "- Baseline leg target rate is 12 rad/s. At dt=0.02 s and action_scale=0.25, the normalized step limit is 0.96 "
               "(target change 0.24 rad); actual A5 contract must match before evaluation.",
               "- feet_clearance and no_fly scales are zero; feet_air_time=0.2, feet_slip=-0.2, action_rate=-0.03. "
               "Hip roll/yaw, foot heading, width and upper-body priors are active. Their causal role remains unmeasured.", "",
               "## Collision geometry (meters, link-local)", "", "```json", json.dumps(geometry_report, indent=2), "```", "",
               "## Unresolved questions and Experiment 1", "",
               "Pitch-vs-roll accuracy, alternating swing, support ratios, slip, swing clearance, limiter activation, "
               "handoff-versus-steady gait, tracking/grasp/posture agreement are all unverified without simulation. "
               "Mesh geometry suggests roll is a necessary comparison source; it does not prove measured force fidelity. "
               "First obtain exact A5 artifacts and physical traces. Prioritize validating the reward contact source. "
               "Only then select one minimal change: correct slip/contact source if dragging dominates, or add a small "
               "swing-clearance term if low valid swings dominate with little slip/limiting. No new reward is implemented.", "",
               "## Execution", "", "commands.json contains exact argument arrays for smoke and two seeded evaluations. "
               "Paths containing /path/to are explicitly unresolved artifacts; supply the actual files to this runner. "
               "Run with PYTHONPATH containing repository legged_gym and rsl_rl directories. "
               "The runner sets these paths for its child processes. It never uses --collect_teacher."]
    (out / "gait_diagnostics.md").write_text("\n".join(report)+"\n", encoding="utf-8")
    summary = dict(status=manifest["status"], simulation_executed=False, measured_episodes=0,
                   gait_metrics=None, blockers=blockers)
    (out / "gait_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(dict(output=str(out), blockers=blockers), indent=2))
    if blockers or own.preflight_only:
        return 2 if blockers else 0
    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "legged_gym"), str(ROOT / "rsl_rl"), child_env.get("PYTHONPATH", "")])
    for command in commands:
        stage = Path(command[-1])
        with (out / (stage.name + ".log")).open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=str(ROOT), env=child_env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            manifest.update(status="evaluation_failed", failed_stage=stage.name)
            break
        required = ["manifest.json", "steps.csv", "summary.json", "gait_summary.json", "gait_diagnostics.md",
                    "contact_timeline.png", "swing_clearance.png", "pitch_roll_contacts.png", "leg_limiter_activation.png"]
        if any(not (stage / f).is_file() for f in required) or json.loads((stage / "manifest.json").read_text())["status"] != "complete":
            manifest.update(status="evaluation_incomplete", failed_stage=stage.name)
            break
    else:
        manifest.update(status="complete", simulation_executed=True, stage_directories=[str(Path(c[-1])) for c in commands])
    stage_summaries = {}
    for command in commands:
        stage = Path(command[-1])
        if (stage / "gait_summary.json").is_file():
            stage_summaries[stage.name] = json.loads((stage / "gait_summary.json").read_text())
    manifest["simulation_executed"] = bool(stage_summaries)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary.update(status=manifest["status"], simulation_executed=bool(stage_summaries),
                   measured_episodes=sum(s["episodes"] for name, s in stage_summaries.items() if name.startswith("eval_")),
                   stages=stage_summaries, gait_metrics={name: s["windows"] for name, s in stage_summaries.items()})
    (out / "gait_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report += ["", "## Evaluation execution", "", "Final status: " + manifest["status"], "",
               "Preflight root steps.csv is a schema only. Each stage has its own full steps.csv, gait_summary.json, "
               "gait_diagnostics.md and four diagnostic plots. Seeded evaluation summaries are also embedded in root gait_summary.json."]
    report += ["- " + str(Path(command[-1])) for command in commands]
    (out / "gait_diagnostics.md").write_text("\n".join(report)+"\n", encoding="utf-8")
    return 0 if manifest["status"] == "complete" else 3


if __name__ == "__main__":
    sys.exit(main())
