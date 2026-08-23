from eval.run_eval import (
    HOLDOUT_MIN_HIT_RATE,
    MIN_ABSTENTION_RATE,
    TUNING_MIN_HIT_RATE,
    check_gate,
)


def test_check_gate_passes_when_all_rates_meet_minimum():
    assert check_gate(TUNING_MIN_HIT_RATE, HOLDOUT_MIN_HIT_RATE, MIN_ABSTENTION_RATE) == []


def test_check_gate_fails_on_tuning_regression():
    failures = check_gate(TUNING_MIN_HIT_RATE - 0.01, HOLDOUT_MIN_HIT_RATE, MIN_ABSTENTION_RATE)
    assert len(failures) == 1
    assert "Tuning" in failures[0]


def test_check_gate_fails_on_holdout_regression():
    failures = check_gate(TUNING_MIN_HIT_RATE, HOLDOUT_MIN_HIT_RATE - 0.01, MIN_ABSTENTION_RATE)
    assert len(failures) == 1
    assert "Holdout" in failures[0]


def test_check_gate_fails_on_abstention_regression():
    failures = check_gate(TUNING_MIN_HIT_RATE, HOLDOUT_MIN_HIT_RATE, MIN_ABSTENTION_RATE - 0.01)
    assert len(failures) == 1
    assert "Abstention" in failures[0]


def test_check_gate_reports_all_failures_independently():
    # -0.01 statt 0.0 fuer die Abstention-Rate: MIN_ABSTENTION_RATE selbst
    # ist der gemessene Boden 0.0 (siehe eval/run_eval.py), 0.0 faellt also
    # nicht durchs Gate -- fuer den Test brauchen wir einen echten Regressionswert.
    failures = check_gate(0.0, 0.0, -0.01)
    assert len(failures) == 3
