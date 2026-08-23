import json
import sqlite3

import pytest
import sqlite_vec

from pipeline.index import build_index, dedupe_docs, doc_embed_text, doc_search_text

FAQ = {"id": "faq-0001", "type": "faq", "question": "Wie funktioniert der Käuferschutz?",
       "answer": "Er sichert Zahlungen ab.", "category": "Kaufen",
       "url": "https://www.chrono24.de/info/buyer-protection.htm"}
CHUNK = {"id": "info-buyer-protection-0001", "type": "page_chunk", "title": "Käuferschutz",
         "heading": "Ablauf", "text": "Der Ablauf ist einfach.",
         "url": "https://www.chrono24.de/info/buyer-protection.htm"}


def fake_encoder(texts):
    return [[1.0, 0.0] if "Käuferschutz?" in t else [0.0, 1.0] for t in texts]


def _open_db(index_dir):
    db = sqlite3.connect(str(index_dir / "hybrid.db"))
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    return db


def _vector_count(db) -> int:
    return db.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]


def _bm25_doc_ids(db) -> list[str]:
    rows = db.execute("SELECT doc_id FROM bm25_docs ORDER BY rowid").fetchall()
    return [r[0] for r in rows]


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

    db = _open_db(index_dir)
    assert _vector_count(db) == 2
    assert _bm25_doc_ids(db) == ["faq-0001", "info-buyer-protection-0001"]


def test_build_index_writes_vectors_and_bm25(tmp_path):
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps({"scraped_at": "2026-08-20", "documents": [FAQ, CHUNK]}),
                           encoding="utf-8")
    index_dir = tmp_path / "index"
    build_index(corpus_path, index_dir, encoder=fake_encoder)

    db = _open_db(index_dir)
    assert _vector_count(db) == 2
    res = db.execute(
        "SELECT doc_id FROM vectors WHERE embedding MATCH ? AND k = 1 ORDER BY distance",
        (sqlite_vec.serialize_float32([1.0, 0.0]),),
    ).fetchall()
    assert res == [("faq-0001",)]

    assert _bm25_doc_ids(db) == ["faq-0001", "info-buyer-protection-0001"]


def test_build_index_stores_canonical_id_and_audience_metadata(tmp_path):
    """Nur canonical_id und audience -- die FAQ-Kategorie bleibt bewusst draussen
    (kein Abnehmer), audience dagegen wird gebraucht (harter Pre-Filter,
    corpus-storage-rethink-design.md Schritt 1). Dokumente ohne audience-Feld
    (wie FAQ/CHUNK hier) fallen auf "neutral" zurück."""
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps({"scraped_at": "2026-08-20", "documents": [FAQ, CHUNK]}),
                           encoding="utf-8")
    index_dir = tmp_path / "index"
    build_index(corpus_path, index_dir, encoder=fake_encoder)

    db = _open_db(index_dir)
    rows = db.execute(
        "SELECT doc_id, canonical_id, audience FROM vectors "
        "WHERE doc_id IN ('faq-0001', 'info-buyer-protection-0001') ORDER BY doc_id"
    ).fetchall()
    assert rows == [
        ("faq-0001", "faq-0001", "neutral"),
        ("info-buyer-protection-0001", "info-buyer-protection-0001", "neutral"),
    ]


def test_build_index_embeds_variants_pointing_to_canonical_faq(tmp_path):
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps({"scraped_at": "2026-08-20", "documents": [FAQ, CHUNK]}),
                           encoding="utf-8")
    variants_path = tmp_path / "variants.json"
    variants_path.write_text(json.dumps({"faq-0001": ["Wie läuft der Käuferschutz ab?"]}),
                             encoding="utf-8")
    index_dir = tmp_path / "index"

    build_index(corpus_path, index_dir, encoder=fake_encoder, variants_path=variants_path)

    db = _open_db(index_dir)
    assert _vector_count(db) == 3  # 2 Original-Docs + 1 Variante
    row = db.execute(
        "SELECT canonical_id, audience FROM vectors WHERE doc_id = 'faq-0001#v1'"
    ).fetchone()
    assert row == ("faq-0001", "neutral")


def test_build_index_ignores_variants_for_faq_ids_not_in_corpus(tmp_path):
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps({"scraped_at": "2026-08-20", "documents": [FAQ, CHUNK]}),
                           encoding="utf-8")
    variants_path = tmp_path / "variants.json"
    variants_path.write_text(json.dumps({"faq-9999": ["Verwaiste Variante"]}), encoding="utf-8")
    index_dir = tmp_path / "index"

    build_index(corpus_path, index_dir, encoder=fake_encoder, variants_path=variants_path)

    db = _open_db(index_dir)
    assert _vector_count(db) == 2


def test_build_index_without_variants_path_behaves_as_before(tmp_path):
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps({"scraped_at": "2026-08-20", "documents": [FAQ, CHUNK]}),
                           encoding="utf-8")
    index_dir = tmp_path / "index"

    build_index(corpus_path, index_dir, encoder=fake_encoder)

    db = _open_db(index_dir)
    assert _vector_count(db) == 2


def test_build_index_variant_inherits_canonical_audience(tmp_path):
    """Der harte Rollenfilter (Schritt 2, SQL-WHERE auf der audience-Partition)
    braucht die audience direkt auf der Varianten-Zeile -- sonst würde eine
    Käufer-Variante fälschlich als 'neutral' durchgelassen oder blockiert."""
    seller_faq = {**FAQ, "id": "faq-seller", "audience": "verkaeufer"}
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(
        json.dumps({"scraped_at": "2026-08-20", "documents": [seller_faq, CHUNK]}),
        encoding="utf-8")
    variants_path = tmp_path / "variants.json"
    variants_path.write_text(json.dumps({"faq-seller": ["Wie melde ich mich als Verkäufer an?"]}),
                             encoding="utf-8")
    index_dir = tmp_path / "index"

    build_index(corpus_path, index_dir, encoder=fake_encoder, variants_path=variants_path)

    db = _open_db(index_dir)
    row = db.execute(
        "SELECT canonical_id, audience FROM vectors WHERE doc_id = 'faq-seller#v1'"
    ).fetchone()
    assert row == ("faq-seller", "verkaeufer")


def test_build_index_bm25_remains_variant_free(tmp_path):
    """BM25-Index darf nie Varianten enthalten — nur kanonische Dokumente."""
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps({"scraped_at": "2026-08-20", "documents": [FAQ, CHUNK]}),
                           encoding="utf-8")
    variants_path = tmp_path / "variants.json"
    variants_path.write_text(json.dumps({"faq-0001": ["Wie läuft der Käuferschutz ab?"]}),
                             encoding="utf-8")
    index_dir = tmp_path / "index"

    build_index(corpus_path, index_dir, encoder=fake_encoder, variants_path=variants_path)

    db = _open_db(index_dir)
    # vectors sollte 3 Eintraege haben (2 Canonical + 1 Variante)
    assert _vector_count(db) == 3
    # BM25 sollte nur 2 kanonische Dokumente haben
    assert _bm25_doc_ids(db) == ["faq-0001", "info-buyer-protection-0001"]


def test_build_index_rejects_non_list_variant_entry(tmp_path):
    """Ein Bare-String statt einer Liste würde enumerate() über einzelne
    Zeichen iterieren lassen und den Index unbemerkt mit einem Eintrag pro
    Buchstaben vergiften. Statt eines Silent-Skips muss der Build laut
    scheitern."""
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps({"scraped_at": "2026-08-20", "documents": [FAQ, CHUNK]}),
                           encoding="utf-8")
    variants_path = tmp_path / "variants.json"
    variants_path.write_text(json.dumps({"faq-0001": "Wie läuft der Käuferschutz ab?"}),
                             encoding="utf-8")
    index_dir = tmp_path / "index"

    with pytest.raises(ValueError, match="faq-0001"):
        build_index(corpus_path, index_dir, encoder=fake_encoder, variants_path=variants_path)


def test_build_index_missing_variants_file_builds_normally_and_warns(tmp_path, capsys):
    """Fehlende Varianten-Datei führt zu Warnung, Index wird trotzdem gebaut."""
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps({"scraped_at": "2026-08-20", "documents": [FAQ, CHUNK]}),
                           encoding="utf-8")
    variants_path = tmp_path / "nonexistent.json"
    index_dir = tmp_path / "index"

    build_index(corpus_path, index_dir, encoder=fake_encoder, variants_path=variants_path)

    # Index sollte normal mit 2 kanonischen Dokumenten gebaut werden
    db = _open_db(index_dir)
    assert _vector_count(db) == 2

    # Die Warnung sollte gedruckt worden sein
    captured = capsys.readouterr()
    assert "Varianten-Datei nicht vorhanden" in captured.out
    assert str(variants_path) in captured.out
