"""Hybrid-Retrieval: FTS5-BM25 + sqlite-vec, fusioniert per Reciprocal Rank Fusion."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import sqlite_vec

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
#
# sqlite-vec liefert mit distance_metric=cosine direkt 1 - Cosine-Similarity
# fuer normalisierte Vektoren (gemessen, siehe corpus-storage-rethink-design.md
# Schritt 2) -- SIM_THRESHOLD behaelt seine Bedeutung unveraendert.
SIM_THRESHOLD = 0.40
# FTS5s eingebautes bm25() hat eine andere Skala/Vorzeichen als
# rank_bm25.BM25Okapi.get_scores() (kleiner/negativer = relevanter). Um die
# "hoeher = relevanter, Schwelle ist eine Untergrenze"-Semantik ueberall im
# Code beizubehalten, negiert der Retriever den FTS5-Score bei der Abfrage
# (best_bm25 = -bm25(...)). BM25_THRESHOLD ist deshalb neu kalibriert, siehe
# README ("Warum Hybrid-RAG", Engine-Wechsel-Eintrag) fuer die Messung.
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


def _connect(db_path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(str(db_path))
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    return db


def _fts5_match_query(tokens: list[str]) -> str | None:
    """Baut eine FTS5-MATCH-Abfrage mit ODER-Semantik (wie BM25Okapi.get_scores).

    Jedes Token wird gequotet -- verhindert, dass ein Token als FTS5-Operator
    (AND/OR/NOT/NEAR, gross geschrieben) oder Spaltenfilter interpretiert wird;
    tokenize() liefert ohnehin nur [a-zäöüß0-9]+, also nie ein Anfuehrungszeichen.
    """
    if not tokens:
        return None
    return " OR ".join(f'"{t}"' for t in tokens)


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
        self.db = _connect(Path(index_dir) / "hybrid.db")
        corpus = json.loads(Path(corpus_path).read_text(encoding="utf-8"))
        self.docs = {d["id"]: d for d in corpus["documents"]}

    def _vector_candidates(
        self, query: str, n: int, total: int, audience: str | None
    ) -> tuple[list[str], float]:
        if not total or not n:
            return [], 0.0
        # Überfetchen: die Tabelle enthält jetzt auch Varianten-Einträge, von
        # denen bis zu MAX_VARIANTS_PER_DOC pro FAQ auf denselben canonical_id-
        # Kandidaten zurückfallen. Erst nach der Dedupe unten auf n kappen,
        # sonst verdrängen sich Varianten derselben FAQ gegenseitig aus den n
        # Rohplätzen.
        fetch_n = min(n * (1 + MAX_VARIANTS_PER_DOC), total)
        query_vec = sqlite_vec.serialize_float32(list(self.encoder(query)))
        # Harter Pre-Filter direkt als SQL-WHERE auf der audience-Partition
        # (Schritt 1 des Rollenfilters, jetzt in der DB statt in Python) --
        # "neutral" passiert für beide Rollen, siehe Kommentar an retrieve().
        if audience in ("kaeufer", "verkaeufer"):
            rows = self.db.execute(
                "SELECT canonical_id, distance FROM vectors "
                "WHERE embedding MATCH ? AND k = ? AND audience IN (?, 'neutral') "
                "ORDER BY distance",
                (query_vec, fetch_n, audience),
            ).fetchall()
        else:
            rows = self.db.execute(
                "SELECT canonical_id, distance FROM vectors "
                "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                (query_vec, fetch_n),
            ).fetchall()
        ranking = _dedupe_ranking([r[0] for r in rows])[:n]
        best_sim = 1.0 - rows[0][1] if rows else 0.0
        return ranking, best_sim

    def _bm25_candidates(
        self, query: str, n: int, audience: str | None
    ) -> tuple[list[str], float]:
        # Synonym-Expansion nur hier: BM25 braucht exakte Wortformen, die
        # Embeddings matchen Bedeutung auch ohne Hilfe.
        match_query = _fts5_match_query(expand_query(query))
        if match_query is None or not n:
            return [], 0.0
        if audience in ("kaeufer", "verkaeufer"):
            rows = self.db.execute(
                "SELECT doc_id, bm25(bm25_docs) FROM bm25_docs "
                "WHERE bm25_docs MATCH ? AND audience IN (?, 'neutral') "
                "ORDER BY bm25(bm25_docs) LIMIT ?",
                (match_query, audience, n),
            ).fetchall()
        else:
            rows = self.db.execute(
                "SELECT doc_id, bm25(bm25_docs) FROM bm25_docs "
                "WHERE bm25_docs MATCH ? ORDER BY bm25(bm25_docs) LIMIT ?",
                (match_query, n),
            ).fetchall()
        ranking = [r[0] for r in rows]
        # FTS5-Vorzeichen umgedreht (siehe Kommentar an BM25_THRESHOLD oben):
        # kleiner/negativer roher bm25()-Wert wird zu einem groesseren,
        # positiveren best_bm25 -- "hoeher = relevanter" bleibt ueberall gueltig.
        best_bm25 = -rows[0][1] if rows else 0.0
        return ranking, best_bm25

    def retrieve(self, query: str, top_k: int = 5,
                 audience: str | None = None) -> list[RetrievedDoc]:
        total = self.db.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
        n = min(TOP_K_CANDIDATES, total)

        # Harter Pre-Filter vor der RRF-Fusion (kein Score-Abzug): Kandidaten
        # der falschen Rolle fliegen aus beiden Rankings komplett raus, statt
        # nur schlechter bewertet zu werden -- Unterschied zum verworfenen
        # weichen Rollen-Malus (siehe README). Nur bei eindeutiger
        # Klassifikation aktiv; "neutral" (Default bei Dokumenten ohne Feld)
        # passiert den Filter für beide Rollen. Jetzt als SQL-WHERE in beiden
        # Teil-Queries statt Python-seitigem Filtern (Schritt 2, siehe
        # corpus-storage-rethink-design.md).
        vector_ranking, best_sim = self._vector_candidates(query, n, total, audience)
        bm25_ranking, best_bm25 = self._bm25_candidates(query, n, audience)

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
