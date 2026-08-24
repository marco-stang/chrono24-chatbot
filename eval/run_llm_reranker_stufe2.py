"""Stufe 2 des Zwei-Signal-LLM-Reranker-Experiments: voller Eval-Lauf plus
Schwellen-Rekalibrierung fuer top1_confidence, nach Stufe 1 (siehe
eval/run_llm_reranker_experiment.py und HANDOVER-llm-reranker.md).
Kalibrierungsmethodik wie fuer RERANK_THRESHOLD in app/retrieval.py
(on-topic-Minimum vs. off-topic-Maximum). Siehe
docs/superpowers/specs/2026-08-24-two-signal-reranker-design.md.

Kostet echte Anthropic-API-Calls (62 Fragen statt Stufe 1s 8) -- nur manuell
ausfuehren:
    python -m eval.run_llm_reranker_stufe2
"""
from __future__ import annotations

import asyncio
import json

from app.config import settings
from app.llm import get_client
from app.llm_reranker import llm_two_signal_rerank, union_candidates
from app.retrieval import TOP_K_CANDIDATES, Retriever
from app.textproc import classify_audience
from eval.run_eval import (
    HOLDOUT_QUESTIONS_PATH,
    OFFTOPIC_QUESTIONS_PATH,
    QUESTIONS_PATH,
    eval_query,
)

# Puffer unter dem gemessenen on-topic-Minimum, analog zu den bestehenden
# Schwellen in app/retrieval.py (z.B. RERANK_THRESHOLD, Puffer fuer
# Einzelfrage-Rauschen bei einer 0-10-Skala mit meist ganzzahligen Werten).
THRESHOLD_BUFFER = 0.5


def _two_signal_candidates(retriever, query, audience):
    total = retriever.db.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
    n = min(TOP_K_CANDIDATES, total)
    vector_ranking, best_sim = retriever._vector_candidates(query, n, total, audience)
    bm25_ranking, best_bm25 = retriever._bm25_candidates(query, n, audience)
    if best_sim < retriever.sim_threshold or best_bm25 < retriever.bm25_threshold:
        return [], False
    ids = union_candidates(vector_ranking, bm25_ranking)
    docs = [retriever._to_doc(doc_id, 0.0) for doc_id in ids]
    return docs, True


async def two_signal_result(
    retriever: Retriever, query: str, client, audience: str | None
):
    """None, wenn Stufe-1-Gate (sim/bm25) schon keine Kandidaten durchlaesst.
    Sonst (docs, ranking, confidence, used_fallback)."""
    docs, gate_open = _two_signal_candidates(retriever, query, audience)
    if not gate_open or not docs:
        return None
    ranking, confidence, used_fallback, _tokens = await llm_two_signal_rerank(query, docs, client)
    return docs, ranking, confidence, used_fallback


async def collect_confidences(
    retriever: Retriever, client, questions: list[dict], query_fn
) -> tuple[list[float], int, int]:
    """Gibt (echte confidence-Werte, Fallback-Anzahl, Gate-zu-Anzahl) zurueck.
    Fallback- und Gate-zu-Faelle fliessen nicht in die confidence-Liste ein --
    sonst wuerde ein Parse-Fehler (confidence 0.0) das on-topic-Minimum
    kuenstlich nach unten ziehen."""
    confidences: list[float] = []
    fallback_count = 0
    gate_closed_count = 0
    for item in questions:
        query = query_fn(item)
        audience = classify_audience(query)
        result = await two_signal_result(retriever, query, client, audience)
        if result is None:
            gate_closed_count += 1
            continue
        _, _, confidence, used_fallback = result
        if used_fallback:
            fallback_count += 1
            continue
        confidences.append(confidence)
    return confidences, fallback_count, gate_closed_count


async def hit_rate_at_k_two_signal(
    retriever: Retriever, client, questions: list[dict], threshold: float, k: int = 5
) -> tuple[float, list[dict]]:
    hits = 0
    misses = []
    for item in questions:
        query = eval_query(item)
        audience = classify_audience(query)
        result = await two_signal_result(retriever, query, client, audience)
        if result is None:
            misses.append({**item, "reason": "gate_closed"})
            continue
        docs, ranking, confidence, used_fallback = result
        if used_fallback or confidence < threshold:
            misses.append({**item, "reason": "low_confidence", "confidence": confidence})
            continue
        top_ids = [docs[i].id for i in ranking[:k]]
        if item["expected_doc_id"] in top_ids:
            hits += 1
        else:
            misses.append({**item, "got": top_ids})
    return hits / len(questions), misses


async def abstention_rate_two_signal(
    retriever: Retriever, client, questions: list[dict], threshold: float
) -> tuple[float, list[dict]]:
    abstained = 0
    false_hits = []
    for item in questions:
        query = item["question"]
        audience = classify_audience(query)
        result = await two_signal_result(retriever, query, client, audience)
        if result is None:
            abstained += 1
            continue
        docs, ranking, confidence, used_fallback = result
        if used_fallback or confidence < threshold:
            abstained += 1
            continue
        false_hits.append({**item, "got_id": docs[ranking[0]].id, "confidence": confidence})
    return abstained / len(questions), false_hits


async def main() -> None:
    retriever = Retriever(settings.index_dir, settings.corpus_path, reranker=False,
                          use_llm_reranker=False)
    client = get_client()

    tuning = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    holdout = json.loads(HOLDOUT_QUESTIONS_PATH.read_text(encoding="utf-8"))
    offtopic = json.loads(OFFTOPIC_QUESTIONS_PATH.read_text(encoding="utf-8"))

    print("-- Phase A: Rohmessung (ungegatet) --")
    on_topic_confidences, on_topic_fallbacks, on_topic_gate_closed = await collect_confidences(
        retriever, client, tuning + holdout, query_fn=eval_query
    )
    off_topic_confidences, off_topic_fallbacks, off_topic_gate_closed = await collect_confidences(
        retriever, client, offtopic, query_fn=lambda item: item["question"]
    )
    print(f"  on-topic:  {len(on_topic_confidences)} Werte, "
          f"{on_topic_fallbacks} Fallback, {on_topic_gate_closed} Gate-zu")
    print(f"  off-topic: {len(off_topic_confidences)} Werte, "
          f"{off_topic_fallbacks} Fallback, {off_topic_gate_closed} Gate-zu")

    if not on_topic_confidences:
        print("  Keine on-topic-Werte -- Schwellen-Vorschlag nicht moeglich.")
        return

    on_topic_min = min(on_topic_confidences)
    off_topic_max = max(off_topic_confidences) if off_topic_confidences else None
    print(f"  on-topic-Minimum:  {on_topic_min}")
    print(f"  off-topic-Maximum: {off_topic_max}")

    suggested_threshold = on_topic_min - THRESHOLD_BUFFER
    print(f"  Vorgeschlagene Schwelle: {suggested_threshold} "
          f"(on-topic-Minimum - Puffer {THRESHOLD_BUFFER})")

    print("\n-- Phase B: voller Eval mit vorgeschlagener Schwelle --")
    tuning_rate, tuning_misses = await hit_rate_at_k_two_signal(
        retriever, client, tuning, threshold=suggested_threshold
    )
    holdout_rate, _ = await hit_rate_at_k_two_signal(
        retriever, client, holdout, threshold=suggested_threshold
    )
    abstain_rate, false_hits = await abstention_rate_two_signal(
        retriever, client, offtopic, threshold=suggested_threshold
    )
    print(f"  Tuning-Hit-Rate@5:  {tuning_rate:.0%} ({len(tuning) - len(tuning_misses)}/{len(tuning)})")
    print(f"  Holdout-Hit-Rate@5: {holdout_rate:.0%}")
    print(f"  Abstention-Rate:    {abstain_rate:.0%}")
    for miss in tuning_misses:
        print(f"  TUNING MISS: {miss['question']!r} erwartet {miss['expected_doc_id']} "
              f"({miss.get('reason', 'wrong_top_k')})")
    for false_hit in false_hits:
        print(f"  FALSE HIT: {false_hit['question']!r} -> {false_hit['got_id']} "
              f"(confidence {false_hit['confidence']})")


if __name__ == "__main__":
    asyncio.run(main())
