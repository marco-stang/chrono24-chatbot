"""Baut den Hybrid-Index (FTS5 fuer BM25 + sqlite-vec fuer Vektoren) aus data/corpus.json.

Eine einzelne SQLite-Datei ersetzt den fruaeheren Chroma-Ordner + bm25.pkl (siehe
docs/superpowers/specs/2026-08-23-corpus-storage-rethink-design.md, Schritt 2).
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import sqlite_vec

from app.config import settings

# Ausgabedimension von settings.embed_model (paraphrase-multilingual-MiniLM-L12-v2).
# Nur Fallback fuer den (praktisch nie vorkommenden) Fall eines leeren Corpus --
# normalerweise wird die Dimension aus der ersten Embedding-Zeile abgeleitet.
DEFAULT_EMBED_DIM = 384

# unicode61 mit remove_diacritics 0: Umlaute (ä/ö/ü) bleiben eigene Zeichen statt
# auf a/o/u zusammengefaltet zu werden -- FTS5s Default wuerde "Käufer" und
# "Kaufer" gleich behandeln, was im Deutschen echte Wortpaare zusammenwirft
# (z. B. "waere"/"ware").
FTS5_TOKENIZER = "unicode61 remove_diacritics 0"


def doc_embed_text(doc: dict) -> str:
    if doc["type"] == "faq":
        return doc["question"]
    return f"{doc['heading']}\n{doc['text']}"


def doc_search_text(doc: dict) -> str:
    if doc["type"] == "faq":
        return f"{doc['question']}\n{doc['answer']}"
    return f"{doc['heading']}\n{doc['text']}"


# Ab dieser Cosine-Similarity gelten zwei page_chunks als inhaltsgleich.
DEDUPE_THRESHOLD = 0.95


def dedupe_docs(
    docs: list[dict], embeddings: list[list[float]], threshold: float = DEDUPE_THRESHOLD
) -> tuple[list[dict], list[list[float]]]:
    """Entfernt near-duplicate page_chunks (Embeddings sind normalisiert, Dot = Cosine).

    FAQ-Dokumente bleiben immer erhalten — ähnliche Fragen sind legitime eigene Einträge.
    Der jeweils erste Chunk gewinnt; spätere Duplikate verdrängen sonst im Retrieval
    das eigentlich beste Dokument aus den Top-5.
    """
    kept_docs: list[dict] = []
    kept_embeddings: list[list[float]] = []
    chunk_embeddings: list[list[float]] = []
    for doc, emb in zip(docs, embeddings):
        if doc["type"] == "page_chunk":
            similarity = max(
                (sum(a * b for a, b in zip(emb, other)) for other in chunk_embeddings),
                default=0.0,
            )
            if similarity >= threshold:
                print(f"Duplikat entfernt: {doc['id']} (Similarity {similarity:.3f})")
                continue
            chunk_embeddings.append(emb)
        kept_docs.append(doc)
        kept_embeddings.append(emb)
    return kept_docs, kept_embeddings


def _default_encoder(texts: list[str]) -> list[list[float]]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(settings.embed_model)
    return model.encode(texts, normalize_embeddings=True).tolist()


def _variant_entries(
    docs: list[dict], variants_path: Path, encoder
) -> tuple[list[str], list[list[float]], list[str], list[str]]:
    """Zusätzliche Vektor-Einträge für LLM-generierte FAQ-Umformulierungen.

    Zeigen per canonical_id auf denselben Antwort-Chunk zurück; die FTS5-Tabelle
    bleibt unangetastet -- Varianten adressieren gezielt den Embedding-Pfad
    (siehe Architektur-Begründung in variants.py). Jede Variante übernimmt die
    audience ihres kanonischen Dokuments, sonst würde der harte Rollenfilter
    (SQL WHERE auf der audience-Partition) Varianten rollenspezifischer FAQs
    unabhängig von ihrer eigentlichen Rolle durchlassen oder blockieren.
    """
    if not variants_path.exists():
        print(f"Varianten-Datei nicht vorhanden: {variants_path} (Index ohne Varianten)")
        return [], [], [], []
    variants: dict[str, list[str]] = json.loads(variants_path.read_text(encoding="utf-8"))
    docs_by_id = {d["id"]: d for d in docs}

    ids: list[str] = []
    texts: list[str] = []
    canonical_ids: list[str] = []
    audiences: list[str] = []
    for faq_id, questions in variants.items():
        if faq_id not in docs_by_id:
            continue  # Variante zu entfallenem FAQ (nicht in aktuellem Corpus).
        # Kein Silent-Skip: ein Bare-String statt einer Liste würde
        # enumerate() über einzelne Zeichen iterieren lassen und den Index
        # unbemerkt mit einem Eintrag pro Buchstaben vergiften.
        if not isinstance(questions, list) or not all(isinstance(q, str) for q in questions):
            raise ValueError(
                f"variants.json: Eintrag für {faq_id!r} muss eine Liste von Strings sein, "
                f"ist aber {type(questions).__name__!r} ({questions!r})."
            )
        audience = docs_by_id[faq_id].get("audience", "neutral")
        for i, question in enumerate(questions, 1):
            ids.append(f"{faq_id}#v{i}")
            texts.append(question)
            canonical_ids.append(faq_id)
            audiences.append(audience)

    if not ids:
        return [], [], [], []
    embeddings = encoder(texts)
    return ids, embeddings, canonical_ids, audiences


def _connect(db_path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(str(db_path))
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    return db


def build_index(
    corpus_path: Path, index_dir: Path, encoder=None, variants_path: Path | None = None
) -> None:
    encoder = encoder or _default_encoder
    docs = json.loads(corpus_path.read_text(encoding="utf-8"))["documents"]
    index_dir.mkdir(parents=True, exist_ok=True)
    db_path = Path(index_dir) / "hybrid.db"
    db_path.unlink(missing_ok=True)  # idempotenter Rebuild statt gewachsener Altlast.

    docs, embeddings = dedupe_docs(docs, encoder([doc_embed_text(d) for d in docs]))
    dim = len(embeddings[0]) if embeddings else DEFAULT_EMBED_DIM

    db = _connect(db_path)
    db.execute(
        f"""
        CREATE VIRTUAL TABLE vectors USING vec0(
            embedding float[{dim}] distance_metric=cosine,
            doc_id TEXT PRIMARY KEY,
            +canonical_id TEXT,
            audience TEXT PARTITION KEY
        )
        """
    )
    db.execute(
        f"""
        CREATE VIRTUAL TABLE bm25_docs USING fts5(
            search_text,
            doc_id UNINDEXED,
            audience UNINDEXED,
            tokenize = '{FTS5_TOKENIZER}'
        )
        """
    )

    for doc, emb in zip(docs, embeddings):
        audience = doc.get("audience", "neutral")
        db.execute(
            "INSERT INTO vectors(doc_id, embedding, canonical_id, audience) VALUES (?, ?, ?, ?)",
            (doc["id"], sqlite_vec.serialize_float32(emb), doc["id"], audience),
        )
        db.execute(
            "INSERT INTO bm25_docs(doc_id, search_text, audience) VALUES (?, ?, ?)",
            (doc["id"], doc_search_text(doc), audience),
        )

    if variants_path is not None:
        variant_ids, variant_embeddings, variant_canonicals, variant_audiences = _variant_entries(
            docs, variants_path, encoder
        )
        for vid, emb, canonical_id, audience in zip(
            variant_ids, variant_embeddings, variant_canonicals, variant_audiences
        ):
            db.execute(
                "INSERT INTO vectors(doc_id, embedding, canonical_id, audience) VALUES (?, ?, ?, ?)",
                (vid, sqlite_vec.serialize_float32(emb), canonical_id, audience),
            )

    db.commit()
    db.close()


if __name__ == "__main__":
    build_index(settings.corpus_path, settings.index_dir, variants_path=settings.variants_path)
    print(f"Index nach {settings.index_dir} geschrieben")
