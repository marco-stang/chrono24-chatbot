from app.textproc import looks_german, tokenize


def test_looks_german_umlaut_is_instant_german():
    assert looks_german("Wie funktioniert der Käuferschutz?")


def test_looks_german_stopwords_without_umlauts():
    assert looks_german("Wie stelle ich eine Uhr zum Verkauf ein?")


def test_looks_german_rejects_english():
    assert not looks_german("What exactly is the Certified program on Chrono24?")
    assert not looks_german("How much does selling a watch cost?")


def test_looks_german_defaults_to_german_on_tie():
    assert looks_german("Chrono24 Trusted Checkout")


def test_tokenize_lowercases_and_keeps_umlauts():
    assert tokenize("Käuferschutz greift SOFORT!") == ["käuferschutz", "greift", "sofort"]


def test_tokenize_splits_on_punctuation_and_digits_stay():
    assert tokenize("Artikel 14 Tage Rückgabe.") == ["artikel", "14", "tage", "rückgabe"]
