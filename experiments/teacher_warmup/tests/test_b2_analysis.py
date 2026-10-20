"""Synthetic data for the comparison/curve exporter; never treated as evaluation evidence."""
import csv
import importlib.util
import json
from pathlib import Path

import pytest

from test_teacher_warmup import ROOT
from test_gait_diagnostics import trace_rows, NAMES


def module():
    path = ROOT / "experiments/teacher_warmup/compare_b2.py"
    spec = importlib.util.spec_from_file_location("compare_b2", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def data(path):
    _, diag, rows = trace_rows()
    path.mkdir()
    manifest = dict(status="complete", episodes=2, dt=.02, dof_names=NAMES, gait_diagnostics=diag.metadata(),
                    teacher_sha256="synthetic", reference_sha256="synthetic", seed=1, carry_seconds=45.,
                    num_envs=1, physical_reset_configuration={"synthetic": True}, handoff={"warmup_vx": .8},
                    synthetic_test_fixture=True)
    (path / "manifest.json").write_text(json.dumps(manifest))
    for row in rows:
        row["reward_raw/gait_foot_slip"] = .5 if int(row["student_valid"]) else 0.
        row["reward_weighted/gait_foot_slip"] = row["reward_raw/gait_foot_slip"] * -.08 * .02
    with (path / "steps.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def test_episode_uncertainty_and_fair_conditions(tmp_path):
    m = module()
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    data(baseline)
    manifest = data(candidate)
    report = m.compare(baseline, candidate, allow_small_sample=True)
    assert report["synthetic_test_fixture"]
    assert report["final_sample_size_satisfied"] is False
    assert report["recommended_checkpoint"] is None
    by_name = {v["metric"]: v for v in report["metrics"]}
    assert by_name["end_to_end_completion"]["delta"] == 0.
    assert by_name["end_to_end_completion"]["uncertain"]
    assert by_name["end_to_end_completion"]["a5_ci95"][0] < .5
    assert report["reward_magnitudes"]["reward_weighted/gait_foot_slip"]["mean"] == pytest.approx(-.0008)
    with pytest.raises(ValueError, match="Synthetic fixtures"):
        m.compare(baseline, candidate)
    manifest["seed"] = 2
    (candidate / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="seed"):
        m.compare(baseline, candidate, allow_small_sample=True)


def test_curve_export_from_synthetic_event_stream(tmp_path):
    from torch.utils.tensorboard import SummaryWriter
    run = tmp_path / "events"
    writer = SummaryWriter(str(run))
    writer.add_scalar("TrackingRMSE/vx", .1, 0)
    writer.add_scalar("Episode/pickup_success", 1., 0)
    writer.add_scalar("Physics/reward_weighted/gait_foot_slip", -.0008, 0)
    writer.close()
    out = tmp_path / "out"
    out.mkdir()
    module().export_curves(run, out)
    assert (out / "training_curves.png").stat().st_size > 100
    assert "Physics/reward_weighted/gait_foot_slip" in (out / "training_scalars.csv").read_text()
