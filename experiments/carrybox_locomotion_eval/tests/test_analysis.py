import csv
import json

from experiments.carrybox_locomotion_eval.analyze import analyze, load_run, matched_comparison


def _csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _run(directory, *, threshold=1., completed=True, step=False):
    directory.mkdir()
    (directory / "traces").mkdir()
    metadata = {
        "protocol_version": "contact_gated_v1", "status": "completed",
        "checkpoint_sha256": "test", "policy_dt": .02,
        "frames": {"planar": "pelvis_yaw", "yaw_rate": "pelvis_world_z"},
        "readiness_config": {"contact_force_n": threshold},
        "command_ranges": {"vx": [-.6, 1.2]},
        "warmup_steps": 10, "measure_steps": 150,
    }
    (directory / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    attempts = [{"trial_id": "T1", "protocol": "constant",
                                      "mode": "vx", "case_type": "axis", "seed": 1,
                                      "carry_motion_id": 0, "carry_phase": .5,
                                      "vx": .4, "vy": 0., "yaw_rate": 0.,
                                      "initialization_success": 1,
                                      "trial_completed": int(completed),
                                      "termination_reason": "completed" if completed else "box_drop",
                                      "termination_phase": "END" if completed else "MEASURE",
                                      "settle_duration_s": .22}]
    segments = [{"trial_id": "T1", "protocol": "constant",
                                              "mode": "vx", "case_type": "axis", "seed": 1,
                                              "carry_motion_id": 0, "carry_phase": .5,
                                              "vx": .4, "vy": 0., "yaw_rate": 0.,
                                              "segment_index": 0, "segment_status":
                                              "completed" if completed else "terminated",
                                              "vx_rmse": .1, "vy_rmse": .03,
                                              "yaw_rate_rmse": .04,
                                              "vx_actual_mean": .35,
                                              "vy_actual_mean": .01,
                                              "yaw_rate_actual_mean": .01}]
    if step:
        attempts.append({**attempts[0], "trial_id": "T2", "protocol": "step",
                         "case_type": "sequence", "seed": 2})
        segments.append({**segments[0], "trial_id": "T2", "protocol": "step",
                         "case_type": "sequence", "seed": 2})
        _csv(directory / "traces" / "T2.csv", [
            {"phase": "WARMUP" if index == 0 else "MEASURE",
             "time_s": index * .02 + .2, "command_onset_s": .2,
             "command_vx": .4, "command_vy": 0., "command_yaw_rate": 0.,
             "actual_vx_training_frame": .1 + .1 * index,
             "actual_vy_training_frame": 0.,
             "actual_yaw_rate_training_frame": 0.}
            for index in range(5)])
    _csv(directory / "summary.csv", attempts)
    _csv(directory / "segment_summary.csv", segments)
    return directory


def test_analyzer_writes_readable_png_svg_and_matched_counts(tmp_path):
    first = _run(tmp_path / "run_a", step=True)
    second = _run(tmp_path / "run_b", completed=False)
    comparison = matched_comparison(load_run(first), load_run(second))
    assert comparison["compatible"]
    assert comparison["matched_attempts"] == 1
    assert comparison["jointly_completed_segments"] == 0
    output = analyze((first, second), tmp_path / "analysis")
    assert (output / "matched_group_1" / "mode_tracking.png").stat().st_size > 1000
    assert (output / "matched_group_1" / "target_response.svg").stat().st_size > 1000
    assert (output / "run_1" / "initialization_and_failures.png").stat().st_size > 1000
    assert (output / "run_1" / "step_run1_vx.svg").stat().st_size > 1000
    assert "matched_attempts" in (output / "analysis_report.md").read_text(encoding="utf-8")


def test_analyzer_rejects_cross_gate_metric_comparison(tmp_path):
    first = _run(tmp_path / "run_a")
    second = _run(tmp_path / "run_b", threshold=2.)
    assert not matched_comparison(load_run(first), load_run(second))["compatible"]
    output = analyze((first, second), tmp_path / "analysis")
    assert (output / "run_1" / "mode_tracking.png").exists()
    assert (output / "run_2" / "mode_tracking.png").exists()
    assert not list(output.glob("matched_group_*"))
