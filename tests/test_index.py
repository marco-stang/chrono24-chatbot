import json
import pickle

from pipeline.index import build_index, doc_embed_text, doc_search_text

FAQ = {"id": "faq-0001", "type": "faq", "question": "Wie funktioniert der Käuferschutz?",
       "answer": "Er sichert Zahlungen ab.", "category": "Kaufen",
       "url": "https://www.chrono24.de/info/faqs.htm"}
CHUNK = {"id": "info-buyer-protection-0001", "type": "page_chunk", "title": "Käuferschutz",
         "heading": "Ablauf", "text": "Der Ablauf ist einfach.",
         "url": "https://www.chrono24.de/info/buyer-protection.htm"}


def fake_encoder(texts):
    return [[1.0, 0.0] if "Käuferschutz?" in t else [0.0, 1.0] for t in texts]


def test_embed_text_uses_question_for_faq():
    assert doc_embed_text(FAQ) == "Wie funktioniert der Käuferschutz?"
    assert doc_embed_text(CHUNK) == "Ablauf\nDer Ablauf ist einfach."


def test_search_text_includes_answer():
    assert "sichert Zahlungen" in doc_search_text(FAQ)


def test_build_index_writes_chroma_and_bm25(tmp_path):
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps({"scraped_at": "2026-08-20", "documents": [FAQ, CHUNK]}),
                           encoding="utf-8")
    index_dir = tmp_path / "index"
    build_index(corpus_path, index_dir, encoder=fake_encoder)

    import chromadb
    coll = chromadb.PersistentClient(path=str(index_dir / "chroma")).get_collection("docs")
    assert coll.count() == 2
    res = coll.query(query_embeddings=[[1.0, 0.0]], n_results=1)
    assert res["ids"][0] == ["faq-0001"]

    with open(index_dir / "bm25.pkl", "rb") as f:
        data = pickle.load(f)
    assert data["doc_ids"] == ["faq-0001", "info-buyer-protection-0001"]
