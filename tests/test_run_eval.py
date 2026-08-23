from eval.run_eval import HOLDOUT_MIN_HIT_RATE, TUNING_MIN_HIT_RATE, check_gate


def test_check_gate_passes_when_both_rates_meet_minimum():
    assert check_gate(TUNING_MIN_HIT_RATE, HOLDOUT_MIN_HIT_RATE) == []


def test_check_gate_fails_on_tuning_regression():
    failures = check_gate(TUNING_MIN_HIT_RATE - 0.01, HOLDOUT_MIN_HIT_RATE)
    assert len(failures) == 1
    assert "Tuning" in failures[0]


def test_check_gate_fails_on_holdout_regression():
    failures = check_gate(TUNING_MIN_HIT_RATE, HOLDOUT_MIN_HIT_RATE - 0.01)
    assert len(failures) == 1
    assert "Holdout" in failures[0]


def test_check_gate_reports_both_failures_independently():
    failures = check_gate(0.0, 0.0)
    assert len(failures) == 2
