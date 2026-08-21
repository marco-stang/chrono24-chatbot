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


def build_index(corpus_path: Path, index_dir: Path, encoder=None) -> None:
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
    coll.add(ids=[d["id"] for d in docs], embeddings=embeddings)

    bm25 = BM25Okapi([tokenize(doc_search_text(d)) for d in docs])
    with open(index_dir / "bm25.pkl", "wb") as f:
        pickle.dump({"doc_ids": [d["id"] for d in docs], "bm25": bm25}, f)


if __name__ == "__main__":
    build_index(settings.corpus_path, settings.index_dir)
    print(f"Index nach {settings.index_dir} geschrieben")
