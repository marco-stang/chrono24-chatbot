import pytest

from eval.stats import format_rate, wilson_interval


def test_wilson_matches_known_values_for_the_current_eval_sets():
    """Referenzwerte fuer die drei Zahlen, mit denen das Projekt wirbt --
    nachgerechnet gegen die Wilson-Formel (z=1.96)."""
    assert wilson_interval(30, 33) == pytest.approx((0.7643, 0.9686), abs=5e-4)
    assert wilson_interval(15, 15) == pytest.approx((0.7961, 1.0), abs=5e-4)
    assert wilson_interval(7, 14) == pytest.approx((0.2680, 0.7320), abs=5e-4)


def test_wilson_stays_inside_zero_and_one_at_the_extremes():
    """Anders als das naive Normal-Intervall darf Wilson nie aus [0, 1] laufen --
    genau dafuer wird es hier benutzt, weil 15/15 und 0/14 real vorkommen."""
    lo, hi = wilson_interval(15, 15)
    assert hi == 1.0
    assert 0.0 < lo < 1.0

    lo, hi = wilson_interval(0, 14)
    assert lo == 0.0
    assert 0.0 < hi < 1.0


def test_wilson_narrows_as_the_sample_grows():
    """Der eigentliche Zweck: sichtbar machen, dass dieselbe Trefferquote bei
    mehr Fragen mehr aussagt. 91 % bei n=33 gegen 91 % bei n=150."""
    small = wilson_interval(30, 33)
    large = wilson_interval(136, 150)
    assert (large[1] - large[0]) < (small[1] - small[0]) / 2


def test_wilson_returns_full_range_for_empty_sample():
    """Ein leeres Set weiss nichts -- nicht 0 %, sondern 'keine Aussage'."""
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_wilson_rejects_more_hits_than_questions():
    with pytest.raises(ValueError):
        wilson_interval(5, 3)


def test_format_rate_shows_the_interval_next_to_the_number():
    assert format_rate(136, 150) == "91% (136/150, 95%-KI [85%, 94%])"


def test_format_rate_flags_a_sample_too_small_to_carry_the_number():
    """Ein Intervall von 46 Punkten ist keine Messung mehr, sondern eine
    Richtungsangabe -- das soll im CI-Log auffallen, nicht im Kleingedruckten
    stehen. Schwelle: 20 Punkte Breite."""
    assert "Stichprobe zu klein" in format_rate(7, 14)
    # Auch das Tuning-Set faellt darunter: 30/33 ist 20,4 Punkte breit.
    assert "Stichprobe zu klein" in format_rate(30, 33)
    assert "Stichprobe zu klein" not in format_rate(136, 150)
