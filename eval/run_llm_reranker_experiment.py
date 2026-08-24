"""Stufe 1 des Zwei-Signal-LLM-Reranker-Experiments: die 4 bekannten
Problemfaelle plus eine Off-Topic-Stichprobe, isoliert und billig, bevor
der volle Eval-Lauf (Stufe 2) folgt. Siehe
docs/superpowers/specs/2026-08-24-two-signal-reranker-design.md.

Kostet echte Anthropic-API-Calls -- nur manuell ausfuehren:
    python -m eval.run_llm_reranker_experiment
"""
from __future__ import annotations

import asyncio
import json

from app.config import settings
from app.llm import get_client
from app.llm_reranker import llm_two_signal_rerank, union_candidates
from app.retrieval import TOP_K_CANDIDATES, Retriever
from app.textproc import classify_audience
from eval.run_eval import OFFTOPIC_QUESTIONS_PATH, QUESTIONS_PATH, eval_query

KNOWN_MISS_IDS = {"faq-0098", "info-escrow-0007", "faq-0033", "faq-0162"}
# Indizes in eval/questions_offtopic.json: 0/1 klar themenfremd, 6/8
# absichtlich domaennah (Omega-Wert, eBay-Vergleich) -- genau die
# Fragetypen, die im Handover als Durchrutscher genannt sind.
OFFTOPIC_SAMPLE_INDICES = [0, 1, 6, 8]


def _known_miss_cases(questions: list[dict]) -> list[dict]:
    return [q for q in questions if q["expected_doc_id"] in KNOWN_MISS_IDS]


def _offtopic_sample(questions: list[dict]) -> list[dict]:
    return [questions[i] for i in OFFTOPIC_SAMPLE_INDICES]


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


async def _run_known_misses(retriever: Retriever, client) -> None:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    cases = _known_miss_cases(questions)
    print(f"-- {len(cases)} bekannte Problemfaelle --")
    for item in cases:
        query = eval_query(item)
        audience = classify_audience(query)
        docs, gate_open = _two_signal_candidates(retriever, query, audience)
        if not gate_open or not docs:
            print(f"  GATE ZU (Stufe 1): {item['question']!r}")
            continue
        ranking, confidence, used_fallback, _tokens = await llm_two_signal_rerank(query, docs, client)
        top1_id = docs[ranking[0]].id
        if used_fallback:
            print(f"  PARSE-FALLBACK: {item['question']!r} erwartet "
                  f"{item['expected_doc_id']}, top1 {top1_id} (Fallback-Ranking, "
                  "keine echte confidence)")
            continue
        status = "OK" if top1_id == item["expected_doc_id"] else "MISS"
        print(f"  {status}: {item['question']!r} erwartet {item['expected_doc_id']}, "
              f"top1 {top1_id}, confidence {confidence}")


async def _run_offtopic_sample(retriever: Retriever, client) -> None:
    questions = json.loads(OFFTOPIC_QUESTIONS_PATH.read_text(encoding="utf-8"))
    sample = _offtopic_sample(questions)
    print(f"-- {len(sample)} Off-Topic-Stichprobe --")
    for item in sample:
        query = item["question"]
        audience = classify_audience(query)
        docs, gate_open = _two_signal_candidates(retriever, query, audience)
        if not gate_open or not docs:
            print(f"  GATE ZU (Stufe 1, korrekt): {query!r}")
            continue
        ranking, confidence, used_fallback, _tokens = await llm_two_signal_rerank(query, docs, client)
        top1_id = docs[ranking[0]].id
        if used_fallback:
            print(f"  PARSE-FALLBACK: {query!r} -> top1 {top1_id} "
                  "(Fallback-Ranking, keine echte confidence)")
            continue
        print(f"  confidence {confidence}: {query!r} -> top1 {top1_id}")


async def main() -> None:
    retriever = Retriever(settings.index_dir, settings.corpus_path, reranker=False,
                          use_llm_reranker=False)
    client = get_client()
    await _run_known_misses(retriever, client)
    await _run_offtopic_sample(retriever, client)


if __name__ == "__main__":
    asyncio.run(main())
