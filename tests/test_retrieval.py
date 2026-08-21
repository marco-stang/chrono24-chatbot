import json

import pytest

from app.retrieval import RetrievedDoc, Retriever, rrf_fuse
from pipeline.index import build_index

DOCS = [
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

DOC_VECS = {"Käuferschutz": [1.0, 0.0, 0.0], "verkaufe": [0.0, 1.0, 0.0],
            "Versand": [0.0, 0.0, 1.0]}


def encode_one(text):
    for key, vec in DOC_VECS.items():
        if key in text:
            return vec
    return [-1.0, 0.0, 0.0]


def neutral_reranker(query, texts):
    return [1.0 for _ in texts]


def make_retriever(tmp_path, reranker):
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps({"scraped_at": "2026-08-20", "documents": DOCS}),
                           encoding="utf-8")
    index_dir = tmp_path / "index"
    build_index(corpus_path, index_dir, encoder=lambda texts: [encode_one(t) for t in texts])
    return Retriever(index_dir, corpus_path, encoder=encode_one, reranker=reranker)


@pytest.fixture()
def retriever(tmp_path):
    return make_retriever(tmp_path, reranker=neutral_reranker)


def test_rrf_fuse_rewards_docs_in_both_rankings():
    scores = rrf_fuse([["a", "b"], ["b", "c"]], k=60)
    assert scores["b"] > scores["a"]
    assert scores["b"] > scores["c"]
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)


def test_retrieve_finds_matching_faq(retriever):
    docs = retriever.retrieve("Wie funktioniert der Käuferschutz?")
    assert docs
    assert isinstance(docs[0], RetrievedDoc)
    assert docs[0].id == "faq-0001"
    assert docs[0].title == "Wie funktioniert der Käuferschutz?"
    assert "sichert deine Zahlung" in docs[0].text


def test_retrieve_returns_empty_for_offtopic(retriever):
    docs = retriever.retrieve("Gedicht über Katzen bitte")
    assert docs == []


def test_reranker_reorders_candidates(tmp_path):
    def prefer_selling(query, texts):
        return [2.0 if "Verkäuferkonto" in t else 1.0 for t in texts]

    retriever = make_retriever(tmp_path, reranker=prefer_selling)
    docs = retriever.retrieve("Wie funktioniert der Käuferschutz?")
    assert docs[0].id == "faq-0002"
    assert docs[0].rerank_score == 2.0
    assert docs[1].rerank_score == 1.0


def test_reranker_false_keeps_rrf_order(tmp_path):
    retriever = make_retriever(tmp_path, reranker=False)
    docs = retriever.retrieve("Wie funktioniert der Käuferschutz?")
    assert docs[0].id == "faq-0001"
    assert docs[0].rerank_score is None


def test_gate_fires_before_reranker(tmp_path):
    def exploding_reranker(query, texts):
        raise AssertionError("Reranker darf bei Off-Topic nicht laufen")

    retriever = make_retriever(tmp_path, reranker=exploding_reranker)
    assert retriever.retrieve("Gedicht über Katzen bitte") == []
