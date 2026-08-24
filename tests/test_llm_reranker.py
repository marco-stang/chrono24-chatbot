from app.retrieval import RetrievedDoc
from eval.llm_reranker import (
    _parse_response,
    _system_prompt,
    build_llm_rerank_prompt,
    union_candidates,
)

DOCS = [
    RetrievedDoc(id="faq-0001", type="faq", title="Frage A",
                 url="https://www.chrono24.de/info/faqs.htm", text="Antwort A", score=0.1),
    RetrievedDoc(id="faq-0002", type="faq", title="Frage B",
                 url="https://www.chrono24.de/info/faqs.htm", text="Antwort B", score=0.1),
]


def test_system_prompt_names_exact_candidate_count():
    prompt = _system_prompt(3)
    assert "genau 3 nummerierte" in prompt
    assert "0 bis 2" in prompt


def test_system_prompt_includes_full_example_array():
    prompt = _system_prompt(3)
    assert '"ranking": [2, 1, 0]' in prompt


def test_build_llm_rerank_prompt_numbers_candidates_from_zero():
    prompt = build_llm_rerank_prompt("Wie funktioniert der Käuferschutz?", DOCS)
    assert "[0] Frage A" in prompt
    assert "[1] Frage B" in prompt
    assert "Antwort A" in prompt
    assert "Wie funktioniert der Käuferschutz?" in prompt


def test_parse_response_accepts_valid_json():
    ranking, confidence = _parse_response('{"ranking": [1, 0], "top1_confidence": 8}', n=2)
    assert ranking == [1, 0]
    assert confidence == 8.0


def test_parse_response_falls_back_to_identity_ranking_on_incomplete_array():
    ranking, _ = _parse_response('{"ranking": [0], "top1_confidence": 7}', n=2)
    assert ranking == [0, 1]


def test_parse_response_falls_back_to_zero_confidence_when_missing():
    ranking, confidence = _parse_response('{"ranking": [1, 0]}', n=2)
    assert confidence == 0.0
    assert ranking == [1, 0]


def test_parse_response_falls_back_to_zero_confidence_when_out_of_range():
    _, confidence = _parse_response('{"ranking": [1, 0], "top1_confidence": 15}', n=2)
    assert confidence == 0.0


def test_parse_response_falls_back_to_zero_confidence_on_boolean_value():
    # bool ist in Python eine int-Subklasse -- ohne expliziten Ausschluss
    # würde True fälschlich zu 1.0 statt zum konservativen Fallback.
    _, confidence = _parse_response('{"ranking": [1, 0], "top1_confidence": true}', n=2)
    assert confidence == 0.0


def test_parse_response_falls_back_completely_on_malformed_json():
    ranking, confidence = _parse_response("not json at all", n=2)
    assert ranking == [0, 1]
    assert confidence == 0.0


def test_union_candidates_dedupes_keeping_first_occurrence():
    result = union_candidates(["a", "b"], ["b", "c"])
    assert result == ["a", "b", "c"]
