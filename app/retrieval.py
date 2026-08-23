"""Hybrid-Retrieval: BM25 + Vektorsuche, fusioniert per Reciprocal Rank Fusion."""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import chromadb

from app.config import settings
from app.textproc import expand_query

TOP_K_CANDIDATES = 10
RRF_K = 60
# Konfidenz-Gate, ODER-verknüpft: reicht EIN Signal eindeutig nicht, gilt die
# Frage als themenfremd und es gibt keinen LLM-Call. Die frühere UND-Logik
# (sim<0.35 UND bm25<4.0) feuerte auf dem Off-Topic-Set nie (0/14), weil das
# multilinguale MiniLM schon für reine Fragesatzform hohe Cosine-Similarity
# liefert: "Wie backe ich einen Hefezopf?" erreicht 0.742, mehr als acht echte
# Fachfragen. Gemessen auf 48 on-topic- und 14 Off-Topic-Fragen
# (eval/questions*.json, Stand 2026-08-23):
#   Cosine-Sim   on-topic min 0.607   off-topic max 0.773
#   BM25         on-topic min 6.36    off-topic max 20.96
#   Rerank-Max   on-topic min -5.6    off-topic max -0.19 (Median -5.2)
# Kein Signal trennt allein; die drei Schwellen liegen je unter dem on-topic-
# Minimum und fangen zusammen 7/14 Off-Topic-Fragen bei 0 verlorenen Treffern.
# Die restlichen 7 sind absichtlich nah am Domänenvokabular (Omega-Wert,
# eBay-Vergleich) -- dort muss der LLM-Prompt bzw. der Faithcheck greifen.
# Margen sind dünn (BM25 5.5 vs. 6.36); eval-gate in CI misst die Rate.
SIM_THRESHOLD = 0.40
# Absolutwert, hängt an Korpusgröße und IDF -- gilt für data/corpus.json
# (313 Dokumente). Kleine Testkorpora übergeben eine eigene Schwelle.
BM25_THRESHOLD = 5.5
RERANK_THRESHOLD = -6.0
# Varianten teilen sich sonst die n Rohplätze: bei bis zu MAX_VARIANTS_PER_DOC
# Umformulierungen je FAQ kollabieren sie nach der canonical_id-Dedupe wieder
# auf einen Kandidaten. Deshalb überfetchen und erst nach der Dedupe kappen —
# der Vektorpfad liefert damit wieder n *verschiedene* Dokumente wie vor den
# Varianten.
MAX_VARIANTS_PER_DOC = 5


@dataclass
class RetrievedDoc:
    id: str
    type: str
    title: str
    url: str
    text: str
    score: float
    rerank_score: float | None = None


def rrf_fuse(rankings: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


def _dedupe_ranking(ids: list[str]) -> list[str]:
    """Erster (bester) Treffer pro kanonischer ID gewinnt -- Varianten-Duplikate raus."""
    seen: set[str] = set()
    result: list[str] = []
    for doc_id in ids:
        if doc_id not in seen:
            seen.add(doc_id)
            result.append(doc_id)
    return result


def _default_encoder():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(settings.embed_model)
    return lambda text: model.encode([text], normalize_embeddings=True)[0].tolist()


def _default_reranker():
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(settings.rerank_model)
    return lambda query, texts: [float(s) for s in model.predict([(query, t) for t in texts])]


class Retriever:
    def __init__(self, index_dir: Path, corpus_path: Path, encoder=None, reranker=None,
                 sim_threshold: float = SIM_THRESHOLD,
                 bm25_threshold: float = BM25_THRESHOLD,
                 rerank_threshold: float = RERANK_THRESHOLD):
        """reranker: Callable[(query, texts) -> scores] | None (Default-Modell) | False (aus)."""
        self.sim_threshold = sim_threshold
        self.bm25_threshold = bm25_threshold
        self.rerank_threshold = rerank_threshold
        self.encoder = encoder or _default_encoder()
        if reranker is False:
            self.reranker = None
        else:
            self.reranker = reranker or _default_reranker()
        client = chromadb.PersistentClient(path=str(index_dir / "chroma"))
        self.collection = client.get_collection("docs")
        with open(index_dir / "bm25.pkl", "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.doc_ids: list[str] = data["doc_ids"]
        corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
        self.docs = {d["id"]: d for d in corpus["documents"]}

    def retrieve(self, query: str, top_k: int = 5,
                 audience: str | None = None) -> list[RetrievedDoc]:
        n = min(TOP_K_CANDIDATES, len(self.doc_ids))
        # Überfetchen: die Collection enthält jetzt auch Varianten-Einträge,
        # von denen bis zu MAX_VARIANTS_PER_DOC pro FAQ auf denselben
        # canonical_id-Kandidaten zurückfallen. Erst nach der Dedupe unten
        # auf n kappen, sonst verdrängen sich Varianten derselben FAQ
        # gegenseitig aus den n Rohplätzen.
        fetch_n = min(n * (1 + MAX_VARIANTS_PER_DOC), self.collection.count())
        res = self.collection.query(
            query_embeddings=[list(self.encoder(query))], n_results=fetch_n,
            include=["metadatas", "distances"],
        )
        # Ältere Indexstände tragen keine Metadaten (Chroma liefert dann None).
        # Dort ist die Dokument-ID selbst schon die kanonische.
        vector_ranking = _dedupe_ranking([
            (meta or {}).get("canonical_id", doc_id)
            for meta, doc_id in zip(res["metadatas"][0], res["ids"][0])
        ])[:n]
        best_sim = 1.0 - res["distances"][0][0] if res["distances"][0] else 0.0

        # Synonym-Expansion nur hier: BM25 braucht exakte Wortformen, die
        # Embeddings matchen Bedeutung auch ohne Hilfe.
        bm25_scores = self.bm25.get_scores(expand_query(query))
        order = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
        bm25_ranking = [self.doc_ids[i] for i in order[:n] if bm25_scores[i] > 0]
        best_bm25 = bm25_scores[order[0]] if len(order) else 0.0

        # Harter Pre-Filter vor der RRF-Fusion (kein Score-Abzug): Kandidaten
        # der falschen Rolle fliegen aus beiden Rankings komplett raus, statt
        # nur schlechter bewertet zu werden -- Unterschied zum verworfenen
        # weichen Rollen-Malus (siehe README). Nur bei eindeutiger
        # Klassifikation aktiv; "neutral" (Default bei Dokumenten ohne Feld)
        # passiert den Filter für beide Rollen.
        if audience in ("kaeufer", "verkaeufer"):
            def _matches_audience(doc_id: str) -> bool:
                return self.docs[doc_id].get("audience", "neutral") in (audience, "neutral")

            vector_ranking = [doc_id for doc_id in vector_ranking if _matches_audience(doc_id)]
            bm25_ranking = [doc_id for doc_id in bm25_ranking if _matches_audience(doc_id)]

        # Stufe 1 des Gates: billig, vor dem Reranker.
        if best_sim < self.sim_threshold or best_bm25 < self.bm25_threshold:
            return []

        fused = rrf_fuse([vector_ranking, bm25_ranking])
        candidates = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:n]
        docs = [self._to_doc(doc_id, score) for doc_id, score in candidates]
        if self.reranker is not None:
            scores = self.reranker(query, [f"{d.title}\n{d.text}" for d in docs])
            for doc, score in zip(docs, scores):
                doc.rerank_score = round(float(score), 4)
            docs.sort(key=lambda d: d.rerank_score, reverse=True)
            # Stufe 2: passt selbst der beste Kandidat laut Cross-Encoder
            # eindeutig nicht, lieber leer als raten.
            if docs and docs[0].rerank_score < self.rerank_threshold:
                return []
        return docs[:top_k]

    def _to_doc(self, doc_id: str, score: float) -> RetrievedDoc:
        doc = self.docs[doc_id]
        if doc["type"] == "faq":
            title, text = doc["question"], doc["answer"]
        else:
            title, text = f"{doc['title']} — {doc['heading']}", doc["text"]
        return RetrievedDoc(id=doc_id, type=doc["type"], title=title, url=doc["url"],
                            text=text, score=round(score, 4))
