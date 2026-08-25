"""Dependency-free runner for the evaluator's focused CPU tests."""

from test_logger import (
    test_logger_appends_only_identical_schema,
    test_logger_rejects_existing_non_v2_header,
)
from test_analysis_v2 import (
    test_v2_aggregation_separates_failures_and_excludes_nan_contact_trials,
)
from test_metrics import (
    test_contact_loss_only_uses_force_response_window,
    test_no_force_response_metrics_are_nan_and_nominal_phase_is_used,
    test_sample_policy_metrics_uses_fixed_start_yaw,
)
from test_outcomes import (
    test_box_drop_requires_confirmed_carry_and_five_steps,
    test_box_instability_classification,
    test_humanoid_failure_thresholds_are_strict,
)


TESTS = (
    test_box_drop_requires_confirmed_carry_and_five_steps,
    test_humanoid_failure_thresholds_are_strict,
    test_box_instability_classification,
    test_no_force_response_metrics_are_nan_and_nominal_phase_is_used,
    test_contact_loss_only_uses_force_response_window,
    test_sample_policy_metrics_uses_fixed_start_yaw,
    test_logger_rejects_existing_non_v2_header,
    test_logger_appends_only_identical_schema,
    test_v2_aggregation_separates_failures_and_excludes_nan_contact_trials,
)


if __name__ == "__main__":
    for test in TESTS:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS {len(TESTS)} evaluator CPU tests")
