import csv
import sys
import tempfile
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))

from evaluation.logger import EvaluationCsvLogger  # noqa: E402


def test_logger_rejects_existing_non_v2_header():
    with tempfile.TemporaryDirectory(dir=EXPERIMENT_DIR) as temp_dir:
        output_dir = Path(temp_dir)
        summary_path = output_dir / "summary.csv"
        summary_path.write_text(
            "trial_id,physical_failure\nT0001,1\n", encoding="utf-8"
        )
        logger = EvaluationCsvLogger(output_dir)
        try:
            logger.append_summary({"trial_id": "T0002", "humanoid_failure": 0})
        except ValueError as error:
            assert "schema does not match" in str(error)
        else:
            raise AssertionError("Expected old summary schema to be rejected")


def test_logger_appends_only_identical_schema():
    with tempfile.TemporaryDirectory(dir=EXPERIMENT_DIR) as temp_dir:
        output_dir = Path(temp_dir)
        logger = EvaluationCsvLogger(output_dir)
        logger.append_summary({"trial_id": "T0001", "humanoid_failure": 0})
        logger.append_summary({"trial_id": "T0002", "humanoid_failure": 1})
        with (output_dir / "summary.csv").open(newline="") as file:
            rows = list(csv.DictReader(file))
        assert [row["trial_id"] for row in rows] == ["T0001", "T0002"]
