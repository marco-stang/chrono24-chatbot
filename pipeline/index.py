"""Baut Vektor- (Chroma) und Keyword-Index (BM25) aus data/corpus.json."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import chromadb
from chromadb.errors import NotFoundError
from rank_bm25 import BM25Okapi

from app.config import settings
from app.textproc import tokenize


def doc_embed_text(doc: dict) -> str:
    if doc["type"] == "faq":
        return doc["question"]
    return f"{doc['heading']}\n{doc['text']}"


def doc_search_text(doc: dict) -> str:
    if doc["type"] == "faq":
        return f"{doc['question']}\n{doc['answer']}"
    return f"{doc['heading']}\n{doc['text']}"


def _doc_metadata(doc: dict) -> dict:
    """Extrahiert Chroma-Metadaten aus einem Corpus-Dokument.

    Alle Einträge erhalten canonical_id; FAQs zusätzlich category.
    """
    meta = {"canonical_id": doc["id"]}
    if doc["type"] == "faq":
        meta["category"] = doc["category"]
    return meta


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
) -> tuple[list[str], list[list[float]], list[dict]]:
    """Zusätzliche Chroma-Einträge für LLM-generierte FAQ-Umformulierungen.

    Zeigen per canonical_id-Metadatum auf denselben Antwort-Chunk zurück;
    BM25 bleibt unangetastet -- Varianten adressieren gezielt den
    Embedding-Pfad (siehe Architektur-Begründung in variants.py).
    """
    if not variants_path.exists():
        print(f"Varianten-Datei nicht vorhanden: {variants_path} (Index ohne Varianten)")
        return [], [], []
    variants: dict[str, list[str]] = json.loads(variants_path.read_text(encoding="utf-8"))
    faq_ids = {d["id"] for d in docs if d["type"] == "faq"}

    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict] = []
    for faq_id, questions in variants.items():
        if faq_id not in faq_ids:
            continue  # Variante zu entfallenem FAQ (nicht in aktuellem Corpus).
        # Kein Silent-Skip: ein Bare-String statt einer Liste würde
        # enumerate() über einzelne Zeichen iterieren lassen und den Index
        # unbemerkt mit einem Chroma-Eintrag pro Buchstaben vergiften.
        if not isinstance(questions, list) or not all(isinstance(q, str) for q in questions):
            raise ValueError(
                f"variants.json: Eintrag für {faq_id!r} muss eine Liste von Strings sein, "
                f"ist aber {type(questions).__name__!r} ({questions!r})."
            )
        for i, question in enumerate(questions, 1):
            ids.append(f"{faq_id}#v{i}")
            texts.append(question)
            metadatas.append({"canonical_id": faq_id})

    if not ids:
        return [], [], []
    embeddings = encoder(texts)
    return ids, embeddings, metadatas


def build_index(
    corpus_path: Path, index_dir: Path, encoder=None, variants_path: Path | None = None
) -> None:
    encoder = encoder or _default_encoder
    docs = json.loads(corpus_path.read_text(encoding="utf-8"))["documents"]
    index_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(index_dir / "chroma"))
    try:
        client.delete_collection("docs")
    except NotFoundError:
        pass  # Idempotentes Aufräumen: beim ersten Lauf existiert die Collection noch nicht.
    coll = client.create_collection("docs", metadata={"hnsw:space": "cosine"})
    docs, embeddings = dedupe_docs(docs, encoder([doc_embed_text(d) for d in docs]))
    ids = [d["id"] for d in docs]
    metadatas = [_doc_metadata(d) for d in docs]

    if variants_path is not None:
        variant_ids, variant_embeddings, variant_metadatas = _variant_entries(
            docs, variants_path, encoder
        )
        ids += variant_ids
        embeddings += variant_embeddings
        metadatas += variant_metadatas

    coll.add(ids=ids, embeddings=embeddings, metadatas=metadatas)

    bm25 = BM25Okapi([tokenize(doc_search_text(d)) for d in docs])
    with open(index_dir / "bm25.pkl", "wb") as f:
        pickle.dump({"doc_ids": [d["id"] for d in docs], "bm25": bm25}, f)


if __name__ == "__main__":
    build_index(settings.corpus_path, settings.index_dir, variants_path=settings.variants_path)
    print(f"Index nach {settings.index_dir} geschrieben")
