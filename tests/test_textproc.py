from app.textproc import expand_query, looks_german, tokenize


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


def test_expand_query_adds_synonyms_after_original_tokens():
    tokens = expand_query("Muss ich etwas bezahlen?")
    assert tokens[:4] == ["muss", "ich", "etwas", "bezahlen"]
    assert "kosten" in tokens and "kostet" in tokens


def test_expand_query_maps_zurueckschicken_to_rueckgabe():
    assert "rückgabe" in expand_query("Kann ich die Uhr zurückschicken?")


def test_expand_query_without_synonym_hits_is_plain_tokenize():
    q = "Wie funktioniert der Käuferschutz?"
    assert expand_query(q) == tokenize(q)
