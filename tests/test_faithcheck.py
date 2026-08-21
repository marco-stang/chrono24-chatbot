"""Tests für den deterministischen Faithfulness-Check (Token-Overlap)."""

from app.faithcheck import check_answer, score_overlap, split_sentences, validate_claims

# --- Satz-Splitting ---

def test_split_two_sentences():
    assert split_sentences("Erster Satz. Zweiter Satz.") == [
        "Erster Satz.", "Zweiter Satz."]


def test_split_keeps_german_abbreviations():
    text = "Das gilt z. B. für Uhren. Zweiter Satz."
    assert split_sentences(text) == ["Das gilt z. B. für Uhren.", "Zweiter Satz."]


def test_split_attaches_trailing_citation_to_previous_sentence():
    # Das LLM setzt Zitate oft NACH den Satzpunkt: "… ab. [1]"
    text = "Der Käuferschutz sichert deine Zahlung ab. [1] Danach kommt mehr."
    assert split_sentences(text) == [
        "Der Käuferschutz sichert deine Zahlung ab. [1]",
        "Danach kommt mehr."]


def test_split_strips_markdown_markers():
    text = "**Wichtig:** Der Schutz gilt immer.\n- Erstens gilt A.\n- Zweitens gilt B."
    assert split_sentences(text) == [
        "Wichtig: Der Schutz gilt immer.", "Erstens gilt A.", "Zweitens gilt B."]


def test_split_keeps_capitalized_sentence_initial_abbreviation():
    # Abkürzung am Satzanfang ist großgeschrieben — darf trotzdem kein Satzende sein.
    text = "Ca. 500 Euro werden fällig. Der Rest folgt später."
    assert split_sentences(text) == [
        "Ca. 500 Euro werden fällig.", "Der Rest folgt später."]


# --- Overlap-Score ---

def test_score_overlap_identical_text_is_one():
    assert score_overlap("Der Schutz gilt", "Der Schutz gilt") == 1.0


def test_score_overlap_empty_claim_is_zero():
    assert score_overlap("", "Quelle") == 0.0


def test_score_overlap_ignores_citation_markers():
    # [1] darf den Score nicht drücken — Marker sind kein Inhalt.
    assert score_overlap("Der Schutz gilt. [1]", "Der Schutz gilt") == 1.0


# --- check_answer: Statuslogik ---

DOCS = ["Der Käuferschutz sichert deine Zahlung vollständig ab.",
        "Die Rückgabefrist beträgt vierzehn Tage nach Erhalt der Uhr."]


def test_cited_sentence_with_high_overlap_is_pass():
    checks = check_answer("Der Käuferschutz sichert deine Zahlung ab. [1]", DOCS)
    assert len(checks) == 1
    assert checks[0].status == "PASS"
    assert checks[0].score >= 0.5
    assert checks[0].sources == [1]


def test_cited_sentence_with_low_overlap_is_weak():
    checks = check_answer("Erstattungen dauern gewöhnlich mehrere Zahlung Wochen. [1]", DOCS)
    assert checks[0].status == "WEAK"
    assert 0 < checks[0].score < 0.5


def test_cited_sentence_with_zero_overlap_is_fail():
    checks = check_answer("Bitcoin Kurse steigen gerade enorm. [1]", DOCS)
    assert checks[0].status == "FAIL"
    assert checks[0].score == 0.0


def test_citation_to_missing_doc_is_fail():
    checks = check_answer("Der Käuferschutz sichert deine Zahlung ab. [7]", DOCS)
    assert checks[0].status == "FAIL"


def test_uncited_factual_sentence_is_weak_without_sources():
    checks = check_answer("Die Erstattung dauert einige Werktage danach.", DOCS)
    assert checks[0].status == "WEAK"
    assert checks[0].sources == []


def test_sentence_in_skip_collection_is_omitted():
    refusal = "Dazu finde ich nichts in den Chrono24-Hilfeseiten."
    assert check_answer(refusal, DOCS, skip={refusal}) == []


def test_short_sentence_is_omitted():
    assert check_answer("Gern geschehen.", DOCS) == []


def test_multiple_citations_use_concatenated_sources():
    answer = "Der Käuferschutz sichert deine Zahlung ab und die Rückgabefrist beträgt vierzehn Tage. [1] [2]"
    checks = check_answer(answer, DOCS)
    assert checks[0].sources == [1, 2]
    assert checks[0].status == "PASS"


# --- validate_claims (Handover-Briefing) ---

LINES = {"M01": "Meine Rolex Daytona ist nach zwei Wochen immer noch nicht angekommen.",
         "M02": "Der Käuferschutz sichert deine Zahlung vollständig ab."}


def test_claim_covered_by_cited_line_is_pass():
    claims = [{"text": "Rolex Daytona ist nicht angekommen", "source_lines": ["M01"]}]
    [check] = validate_claims(claims, LINES)
    assert check.status == "PASS"
    assert check.score >= 0.5
    assert check.sources == ["M01"]


def test_claim_with_low_overlap_is_weak():
    claims = [{"text": "Das Paket kommt nach Wochen nicht an",
               "source_lines": ["M01"]}]
    [check] = validate_claims(claims, LINES)
    assert check.status == "WEAK"
    assert 0 < check.score < 0.5


def test_claim_with_zero_overlap_is_fail():
    claims = [{"text": "Bitcoin Kurs steigt enorm", "source_lines": ["M02"]}]
    [check] = validate_claims(claims, LINES)
    assert check.status == "FAIL"
    assert check.score == 0.0


def test_claim_without_source_lines_is_fail():
    claims = [{"text": "Rolex Daytona ist nicht angekommen", "source_lines": []}]
    [check] = validate_claims(claims, LINES)
    assert check.status == "FAIL"


def test_claim_with_unknown_line_id_is_fail():
    claims = [{"text": "Rolex Daytona ist nicht angekommen", "source_lines": ["M99"]}]
    [check] = validate_claims(claims, LINES)
    assert check.status == "FAIL"


def test_tokenizer_ignores_line_id_tokens():
    # "M01" im Claim-Text darf den Score nicht verfälschen (analog [n]-Marker).
    claims = [{"text": "M01 Rolex Daytona ist nicht angekommen", "source_lines": ["M01"]}]
    [check] = validate_claims(claims, LINES)
    assert check.status == "PASS"


def test_claims_spanning_multiple_lines_concatenate_sources():
    claims = [{"text": "Rolex nicht angekommen, Käuferschutz sichert Zahlung ab",
               "source_lines": ["M01", "M02"]}]
    [check] = validate_claims(claims, LINES)
    assert check.status == "PASS"
