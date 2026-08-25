import importlib.util
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "analysis" / "carrybox_perturb_results" / "analyze_results.py"
SPEC = importlib.util.spec_from_file_location("carrybox_analysis_v2", SCRIPT_PATH)
ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)


def test_v2_aggregation_separates_failures_and_excludes_nan_contact_trials():
    summary = pd.DataFrame(
        {
            "policy": ["pi", "pi"],
            "profile": ["smooth_hold", "smooth_hold"],
            "trial_id": ["T0001", "T0002"],
            "beta": [0.1, 0.1],
            "direction": ["+box_x", "+box_x"],
            "humanoid_failure": [1, 0],
            "box_failure": [0, 1],
            "timeout": [0, 1],
            "carry_achieved": [0, 1],
            "force_scheduled": [0, 1],
            "contact_loss": [float("nan"), 1.0],
            "final_confirmed_carry": [0, 1],
            "max_hand_box_relative_speed": [0.5, 0.7],
        }
    )
    tables = ANALYSIS.make_key_tables(summary, summary.copy())
    overview = tables["overview.csv"].iloc[0]
    assert overview["humanoid_failure_rate"] == 0.5
    assert overview["box_failure_rate"] == 0.5
    assert overview["contact_response_trials"] == 1
    assert overview["contact_loss_rate"] == 1.0
