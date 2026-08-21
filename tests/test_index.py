import json
import pickle

from pipeline.index import build_index, dedupe_docs, doc_embed_text, doc_search_text

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


CHUNK2 = {"id": "info-buyer-protection-0002", "type": "page_chunk", "title": "Käuferschutz",
          "heading": "Ablauf", "text": "Der Ablauf ist wirklich einfach.",
          "url": "https://www.chrono24.de/info/buyer-protection.htm"}


def test_dedupe_drops_near_duplicate_chunks():
    docs = [FAQ, CHUNK, CHUNK2]
    embeddings = [[1.0, 0.0], [0.0, 1.0], [0.02, 0.999]]
    kept, kept_embeddings = dedupe_docs(docs, embeddings, threshold=0.95)
    assert [d["id"] for d in kept] == ["faq-0001", "info-buyer-protection-0001"]
    assert kept_embeddings == [[1.0, 0.0], [0.0, 1.0]]


def test_dedupe_keeps_distinct_chunks():
    docs = [CHUNK, CHUNK2]
    embeddings = [[0.0, 1.0], [0.9, 0.436]]
    kept, _ = dedupe_docs(docs, embeddings, threshold=0.95)
    assert [d["id"] for d in kept] == ["info-buyer-protection-0001", "info-buyer-protection-0002"]


def test_dedupe_never_drops_faq_docs():
    faq2 = {**FAQ, "id": "faq-0002"}
    docs = [FAQ, faq2]
    embeddings = [[1.0, 0.0], [1.0, 0.0]]
    kept, _ = dedupe_docs(docs, embeddings, threshold=0.95)
    assert [d["id"] for d in kept] == ["faq-0001", "faq-0002"]


def test_build_index_drops_duplicate_chunk_from_both_indexes(tmp_path):
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(
        json.dumps({"scraped_at": "2026-08-20", "documents": [FAQ, CHUNK, CHUNK2]}),
        encoding="utf-8")
    index_dir = tmp_path / "index"

    def encoder(texts):
        vecs = {"Käuferschutz?": [1.0, 0.0], "wirklich": [0.02, 0.999]}
        return [next((v for k, v in vecs.items() if k in t), [0.0, 1.0]) for t in texts]

    build_index(corpus_path, index_dir, encoder=encoder)

    import chromadb
    coll = chromadb.PersistentClient(path=str(index_dir / "chroma")).get_collection("docs")
    assert coll.count() == 2
    with open(index_dir / "bm25.pkl", "rb") as f:
        data = pickle.load(f)
    assert data["doc_ids"] == ["faq-0001", "info-buyer-protection-0001"]


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
