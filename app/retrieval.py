"""Hybrid-Retrieval: BM25 + Vektorsuche, fusioniert per Reciprocal Rank Fusion."""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import chromadb

from app.config import settings
from app.textproc import tokenize

TOP_K_CANDIDATES = 10
RRF_K = 60
# Konfidenz-Gate: liegen BEIDE Signale unter ihrer Schwelle, gilt die Frage als
# themenfremd und es gibt keinen LLM-Call. Nach Task 10 (Eval) nachjustieren.
SIM_THRESHOLD = 0.35
BM25_THRESHOLD = 4.0


@dataclass
class RetrievedDoc:
    id: str
    type: str
    title: str
    url: str
    text: str
    score: float


def rrf_fuse(rankings: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


def _default_encoder():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(settings.embed_model)
    return lambda text: model.encode([text], normalize_embeddings=True)[0].tolist()


class Retriever:
    def __init__(self, index_dir: Path, corpus_path: Path, encoder=None):
        self.encoder = encoder or _default_encoder()
        client = chromadb.PersistentClient(path=str(index_dir / "chroma"))
        self.collection = client.get_collection("docs")
        with open(index_dir / "bm25.pkl", "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.doc_ids: list[str] = data["doc_ids"]
        corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
        self.docs = {d["id"]: d for d in corpus["documents"]}

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDoc]:
        n = min(TOP_K_CANDIDATES, len(self.doc_ids))
        res = self.collection.query(query_embeddings=[list(self.encoder(query))], n_results=n)
        vector_ranking = res["ids"][0]
        best_sim = 1.0 - res["distances"][0][0] if res["distances"][0] else 0.0

        bm25_scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
        bm25_ranking = [self.doc_ids[i] for i in order[:n] if bm25_scores[i] > 0]
        best_bm25 = bm25_scores[order[0]] if len(order) else 0.0

        if best_sim < SIM_THRESHOLD and best_bm25 < BM25_THRESHOLD:
            return []

        fused = rrf_fuse([vector_ranking, bm25_ranking])
        top = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [self._to_doc(doc_id, score) for doc_id, score in top]

    def _to_doc(self, doc_id: str, score: float) -> RetrievedDoc:
        doc = self.docs[doc_id]
        if doc["type"] == "faq":
            title, text = doc["question"], doc["answer"]
        else:
            title, text = f"{doc['title']} — {doc['heading']}", doc["text"]
        return RetrievedDoc(id=doc_id, type=doc["type"], title=title, url=doc["url"],
                            text=text, score=round(score, 4))
