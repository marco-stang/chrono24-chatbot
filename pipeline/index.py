"""Baut Vektor- (Chroma) und Keyword-Index (BM25) aus data/corpus.json."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import chromadb
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
    except Exception:
        pass
    coll = client.create_collection("docs", metadata={"hnsw:space": "cosine"})
    coll.add(ids=[d["id"] for d in docs], embeddings=encoder([doc_embed_text(d) for d in docs]))

    bm25 = BM25Okapi([tokenize(doc_search_text(d)) for d in docs])
    with open(index_dir / "bm25.pkl", "wb") as f:
        pickle.dump({"doc_ids": [d["id"] for d in docs], "bm25": bm25}, f)


if __name__ == "__main__":
    build_index(settings.corpus_path, settings.index_dir)
    print(f"Index nach {settings.index_dir} geschrieben")
