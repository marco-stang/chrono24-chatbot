from eval.run_llm_reranker_experiment import (
    KNOWN_MISS_IDS,
    OFFTOPIC_SAMPLE_INDICES,
    _known_miss_cases,
    _offtopic_sample,
)

_TUNING_QUESTIONS = [
    {"question": "Frage 1", "expected_doc_id": "faq-0001"},
    {"question": "Frage 2", "expected_doc_id": "faq-0098"},
    {"question": "Frage 3", "expected_doc_id": "faq-0033"},
]

_OFFTOPIC_QUESTIONS = [{"question": f"Off-Topic {i}"} for i in range(14)]


def test_known_miss_ids_matches_the_four_documented_problem_cases():
    assert KNOWN_MISS_IDS == {"faq-0098", "info-escrow-0007", "faq-0033", "faq-0162"}


def test_known_miss_cases_filters_only_documented_ids():
    cases = _known_miss_cases(_TUNING_QUESTIONS)
    assert {c["expected_doc_id"] for c in cases} == {"faq-0098", "faq-0033"}


def test_offtopic_sample_selects_configured_indices():
    sample = _offtopic_sample(_OFFTOPIC_QUESTIONS)
    assert sample == [_OFFTOPIC_QUESTIONS[i] for i in OFFTOPIC_SAMPLE_INDICES]
