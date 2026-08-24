import json

from app.retrieval import RetrievedDoc, Retriever
from eval.llm_reranker import (
    _parse_response,
    _system_prompt,
    build_llm_rerank_prompt,
    llm_two_signal_rerank,
    two_signal_candidates,
    union_candidates,
)
from pipeline.index import build_index

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
    ranking, confidence, used_fallback = _parse_response(
        '{"ranking": [1, 0], "top1_confidence": 8}', n=2)
    assert ranking == [1, 0]
    assert confidence == 8.0
    assert used_fallback is False


def test_parse_response_falls_back_to_identity_ranking_on_incomplete_array():
    ranking, _, used_fallback = _parse_response('{"ranking": [0], "top1_confidence": 7}', n=2)
    assert ranking == [0, 1]
    assert used_fallback is True


def test_parse_response_falls_back_to_zero_confidence_when_missing():
    ranking, confidence, used_fallback = _parse_response('{"ranking": [1, 0]}', n=2)
    assert confidence == 0.0
    assert ranking == [1, 0]
    assert used_fallback is True


def test_parse_response_falls_back_to_zero_confidence_when_out_of_range():
    _, confidence, used_fallback = _parse_response(
        '{"ranking": [1, 0], "top1_confidence": 15}', n=2)
    assert confidence == 0.0
    assert used_fallback is True


def test_parse_response_falls_back_to_zero_confidence_on_boolean_value():
    # bool ist in Python eine int-Subklasse -- ohne expliziten Ausschluss
    # würde True fälschlich zu 1.0 statt zum konservativen Fallback.
    _, confidence, used_fallback = _parse_response(
        '{"ranking": [1, 0], "top1_confidence": true}', n=2)
    assert confidence == 0.0
    assert used_fallback is True


def test_parse_response_falls_back_completely_on_malformed_json():
    ranking, confidence, used_fallback = _parse_response("not json at all", n=2)
    assert ranking == [0, 1]
    assert confidence == 0.0
    assert used_fallback is True


def test_parse_response_falls_back_on_mixed_type_ranking():
    # [0, "1"] darf nicht crashen (sorted() auf gemischten Typen wirft
    # TypeError) und muss konservativ auf die Identitaet zurueckfallen.
    # top1_confidence selbst ist hier valide (8) und bleibt unangetastet --
    # nur used_fallback markiert, dass die Rangfolge nicht vom Modell kam.
    ranking, confidence, used_fallback = _parse_response(
        '{"ranking": [0, "1"], "top1_confidence": 8}', n=2)
    assert ranking == [0, 1]
    assert confidence == 8.0
    assert used_fallback is True


def test_parse_response_falls_back_on_float_ranking():
    # [0.0, 1.0] besteht sorted(ranking) == identity (Gleichheit mit ints),
    # crasht aber danach bei docs[ranking[0]] mit "list indices must be
    # integers" -- muss deshalb schon hier als ungueltig erkannt werden.
    ranking, confidence, used_fallback = _parse_response(
        '{"ranking": [0.0, 1.0], "top1_confidence": 8}', n=2)
    assert ranking == [0, 1]
    assert confidence == 8.0
    assert used_fallback is True


def test_parse_response_falls_back_on_boolean_in_ranking():
    # bool ist eine int-Subklasse -- [true, 0] darf nicht als [True, 0]
    # durchgehen, sonst inkonsistent zum expliziten bool-Ausschluss bei
    # confidence zwei Zeilen darunter.
    ranking, confidence, used_fallback = _parse_response(
        '{"ranking": [true, 0], "top1_confidence": 8}', n=2)
    assert ranking == [0, 1]
    assert confidence == 8.0
    assert used_fallback is True


def test_parse_response_used_fallback_distinguishes_parse_failure_from_genuine_low_confidence():
    genuine_ranking, genuine_confidence, genuine_fallback = _parse_response(
        '{"ranking": [1, 0], "top1_confidence": 0}', n=2)
    broken_ranking, broken_confidence, broken_fallback = _parse_response(
        "not json at all", n=2)
    # Beide liefern confidence 0.0 -- ohne used_fallback nicht
    # unterscheidbar, mit used_fallback schon.
    assert genuine_confidence == broken_confidence == 0.0
    assert genuine_fallback is False
    assert broken_fallback is True
    assert genuine_ranking != broken_ranking or genuine_fallback != broken_fallback


def test_union_candidates_dedupes_keeping_first_occurrence():
    result = union_candidates(["a", "b"], ["b", "c"])
    assert result == ["a", "b", "c"]


class _FakeTextBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class _FakeResponse:
    def __init__(self, text):
        self.content = [_FakeTextBlock(text)]


class _FakeMessages:
    def __init__(self, response_text):
        self._response_text = response_text
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse(self._response_text)


class _FakeClient:
    def __init__(self, response_text):
        self.messages = _FakeMessages(response_text)


async def test_llm_two_signal_rerank_returns_parsed_ranking_and_confidence():
    client = _FakeClient('{"ranking": [1, 0], "top1_confidence": 9}')
    ranking, confidence, used_fallback = await llm_two_signal_rerank("Frage?", DOCS, client)
    assert ranking == [1, 0]
    assert confidence == 9.0
    assert used_fallback is False


async def test_llm_two_signal_rerank_pins_temperature_and_token_limit():
    client = _FakeClient('{"ranking": [0, 1], "top1_confidence": 5}')
    await llm_two_signal_rerank("Frage?", DOCS, client)
    call = client.messages.calls[0]
    assert call["temperature"] == 0
    assert call["max_tokens"] == 400
    assert f"genau {len(DOCS)} nummerierte" in call["system"]


async def test_llm_two_signal_rerank_falls_back_on_malformed_response():
    client = _FakeClient("kaputte Antwort, kein JSON")
    ranking, confidence, used_fallback = await llm_two_signal_rerank("Frage?", DOCS, client)
    assert ranking == [0, 1]
    assert confidence == 0.0
    assert used_fallback is True


_CORPUS_DOCS = [
    {"id": "faq-0001", "type": "faq", "question": "Wie funktioniert der Käuferschutz?",
     "answer": "Der Käuferschutz sichert deine Zahlung ab.", "category": "Kaufen",
     "url": "https://www.chrono24.de/info/faqs.htm"},
    {"id": "faq-0002", "type": "faq", "question": "Wie verkaufe ich eine Uhr?",
     "answer": "Über ein Verkäuferkonto.", "category": "Verkaufen",
     "url": "https://www.chrono24.de/info/faqs.htm"},
    {"id": "info-shipping-0001", "type": "page_chunk", "title": "Versand",
     "heading": "Versicherter Versand", "text": "Uhren werden versichert verschickt.",
     "url": "https://www.chrono24.de/info/shipping.htm"},
]

_DOC_VECS = {"Käuferschutz": [1.0, 0.0, 0.0], "verkaufe": [0.0, 1.0, 0.0],
             "Versand": [0.0, 0.0, 1.0]}


def _encode_one(text):
    for key, vec in _DOC_VECS.items():
        if key in text:
            return vec
    return [-1.0, 0.0, 0.0]


def _build_retriever(tmp_path):
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps({"scraped_at": "2026-08-24", "documents": _CORPUS_DOCS}),
                           encoding="utf-8")
    index_dir = tmp_path / "index"
    build_index(corpus_path, index_dir, encoder=lambda texts: [_encode_one(t) for t in texts])
    return Retriever(index_dir, corpus_path, encoder=_encode_one, reranker=False,
                     bm25_threshold=1.0)


def test_two_signal_candidates_returns_union_when_gate_open(tmp_path):
    retriever = _build_retriever(tmp_path)
    docs, gate_open = two_signal_candidates(retriever, "Wie funktioniert der Käuferschutz?", None)
    assert gate_open is True
    assert {d.id for d in docs} == {"faq-0001", "faq-0002", "info-shipping-0001"}


def test_two_signal_candidates_closes_gate_for_offtopic(tmp_path):
    retriever = _build_retriever(tmp_path)
    docs, gate_open = two_signal_candidates(retriever, "Gedicht über Katzen bitte", None)
    assert gate_open is False
    assert docs == []


def test_two_signal_candidates_union_vs_rrf_cut(tmp_path, monkeypatch):
    """Verify union (not RRF-top-n-cut) with divergent vector/bm25 rankings.

    The fix: union_candidates([vector_top_2], [bm25_top_2]) yields 3 docs
    when rankings diverge, while naive RRF-fuse-then-cut-to-2 would yield only 2.
    _vector_candidates/_bm25_candidates are mocked directly (both ignore `n`),
    so TOP_K_CANDIDATES is irrelevant here -- only the divergent rankings matter.
    """
    retriever = _build_retriever(tmp_path)

    # Mock rankings to have a split: vector=[A, B], bm25=[A, C]
    # This forces union=[A, B, C] but RRF-top-2-cut=[A, ...one of B/C]
    def mock_vector_candidates(query, n, total, audience):
        return (["faq-0001", "faq-0002"], 0.9)  # Top 2: faq-0001, faq-0002

    def mock_bm25_candidates(query, n, audience):
        return (["faq-0001", "info-shipping-0001"], 6.0)  # Top 2: faq-0001, info-shipping-0001

    monkeypatch.setattr(retriever, "_vector_candidates", mock_vector_candidates)
    monkeypatch.setattr(retriever, "_bm25_candidates", mock_bm25_candidates)

    docs, gate_open = two_signal_candidates(retriever, "test query", None)

    assert gate_open is True
    # Union of [faq-0001, faq-0002] + [faq-0001, info-shipping-0001]
    # should yield [faq-0001, faq-0002, info-shipping-0001] (3 docs)
    # If implementation used RRF-fuse-then-cut-to-2, it would return only 2.
    assert len(docs) == 3
    assert {d.id for d in docs} == {"faq-0001", "faq-0002", "info-shipping-0001"}
