from app.textproc import tokenize


def test_tokenize_lowercases_and_keeps_umlauts():
    assert tokenize("Käuferschutz greift SOFORT!") == ["käuferschutz", "greift", "sofort"]


def test_tokenize_splits_on_punctuation_and_digits_stay():
    assert tokenize("Artikel 14 Tage Rückgabe.") == ["artikel", "14", "tage", "rückgabe"]
