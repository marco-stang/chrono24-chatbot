import json

from app.retrieval import Retriever
from eval.run_llm_reranker_stufe2 import (
    abstention_rate_two_signal,
    collect_confidences,
    hit_rate_at_k_two_signal,
    two_signal_result,
)
from pipeline.index import build_index

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


class _FakeTextBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class _FakeResponse:
    def __init__(self, text):
        self.content = [_FakeTextBlock(text)]


class _FakeMessages:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        text = self._responses.pop(0) if self._responses else '{"ranking": [], "top1_confidence": 0}'
        return _FakeResponse(text)


class _FakeClient:
    def __init__(self, responses):
        self.messages = _FakeMessages(responses)


async def test_two_signal_result_returns_none_when_gate_closed(tmp_path):
    retriever = _build_retriever(tmp_path)
    client = _FakeClient([])
    result = await two_signal_result(retriever, "Gedicht über Katzen bitte", client, None)
    assert result is None


async def test_two_signal_result_returns_full_tuple_when_gate_open(tmp_path):
    retriever = _build_retriever(tmp_path)
    client = _FakeClient(['{"ranking": [0, 1, 2], "top1_confidence": 7}'])
    result = await two_signal_result(retriever, "Wie funktioniert der Käuferschutz?", client, None)
    assert result is not None
    docs, ranking, confidence, used_fallback = result
    assert {d.id for d in docs} == {"faq-0001", "faq-0002", "info-shipping-0001"}
    assert ranking == [0, 1, 2]
    assert confidence == 7.0
    assert used_fallback is False


async def test_collect_confidences_excludes_fallback_and_gate_closed(tmp_path):
    retriever = _build_retriever(tmp_path)
    questions = [
        {"question": "Wie funktioniert der Käuferschutz?"},  # gate open, valid response
        {"question": "Wie verkaufe ich eine Uhr?"},  # gate open, malformed -> fallback
        {"question": "Gedicht über Katzen bitte"},  # gate closed
    ]
    client = _FakeClient([
        '{"ranking": [0, 1, 2], "top1_confidence": 8}',
        "kaputte Antwort, kein JSON",
    ])
    confidences, fallback_count, gate_closed_count = await collect_confidences(
        retriever, client, questions, query_fn=lambda item: item["question"]
    )
    assert confidences == [8.0]
    assert fallback_count == 1
    assert gate_closed_count == 1


async def test_hit_rate_at_k_two_signal_counts_hit_when_confidence_meets_threshold(tmp_path):
    retriever = _build_retriever(tmp_path)
    questions = [{"question": "Wie funktioniert der Käuferschutz?", "expected_doc_id": "faq-0001"}]
    client = _FakeClient(['{"ranking": [0, 1, 2], "top1_confidence": 9}'])
    rate, misses = await hit_rate_at_k_two_signal(retriever, client, questions, threshold=5.0)
    assert rate == 1.0
    assert misses == []


async def test_hit_rate_at_k_two_signal_misses_when_confidence_below_threshold(tmp_path):
    retriever = _build_retriever(tmp_path)
    questions = [{"question": "Wie funktioniert der Käuferschutz?", "expected_doc_id": "faq-0001"}]
    client = _FakeClient(['{"ranking": [0, 1, 2], "top1_confidence": 3}'])
    rate, misses = await hit_rate_at_k_two_signal(retriever, client, questions, threshold=5.0)
    assert rate == 0.0
    assert misses[0]["reason"] == "low_confidence"


async def test_hit_rate_at_k_two_signal_misses_when_expected_doc_not_in_ranking(tmp_path):
    retriever = _build_retriever(tmp_path)
    questions = [{"question": "Wie funktioniert der Käuferschutz?", "expected_doc_id": "does-not-exist"}]
    client = _FakeClient(['{"ranking": [0, 1, 2], "top1_confidence": 9}'])
    rate, misses = await hit_rate_at_k_two_signal(retriever, client, questions, threshold=5.0)
    assert rate == 0.0
    assert "got" in misses[0]


async def test_abstention_rate_two_signal_abstains_on_gate_closed_and_low_confidence(tmp_path):
    retriever = _build_retriever(tmp_path)
    questions = [
        {"question": "Gedicht über Katzen bitte"},  # gate closed
        {"question": "Wie verkaufe ich eine Uhr?"},  # gate open, low confidence
    ]
    client = _FakeClient(['{"ranking": [0, 1, 2], "top1_confidence": 2}'])
    rate, false_hits = await abstention_rate_two_signal(retriever, client, questions, threshold=5.0)
    assert rate == 1.0
    assert false_hits == []


async def test_abstention_rate_two_signal_counts_false_hit_on_high_confidence(tmp_path):
    retriever = _build_retriever(tmp_path)
    questions = [{"question": "Wie verkaufe ich eine Uhr?"}]
    client = _FakeClient(['{"ranking": [0, 1, 2], "top1_confidence": 9}'])
    rate, false_hits = await abstention_rate_two_signal(retriever, client, questions, threshold=5.0)
    assert rate == 0.0
    assert false_hits[0]["confidence"] == 9.0
