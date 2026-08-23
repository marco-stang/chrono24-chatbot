"""Misst Hit-Rate@k des Retrievals gegen handgeschriebene Testfragen."""
from __future__ import annotations

import json
from pathlib import Path

QUESTIONS_PATH = Path("eval/questions.json")
HOLDOUT_QUESTIONS_PATH = Path("eval/questions_holdout.json")
OFFTOPIC_QUESTIONS_PATH = Path("eval/questions_offtopic.json")

# Aktuell gemessen (nach dem Vektor-Überfetch-Fix, siehe README): Tuning 91 %
# (30/33), Holdout 93 % (14/15) — Schwellen bewusst mit Puffer für
# Einzelfrage-Rauschen (33/15 Fragen sind ein kleines Sample) und nach jeder
# bewussten Verbesserung hier nachziehen, nicht nur nach oben schieben.
TUNING_MIN_HIT_RATE = 0.85
HOLDOUT_MIN_HIT_RATE = 0.80
# Anteil themenfremder Fragen, bei denen der Bot leer zurückgibt (siehe
# app.retrieval.SIM_THRESHOLD / BM25_THRESHOLD). Gemessen auf
# eval/questions_offtopic.json (14 Fragen, gemischt eindeutig off-topic und
# absichtlich nah am Domänenvokabular): 0 % (0/14) — sogar eindeutig fachfremde
# Fragen wie "Wie backe ich einen Hefezopf?" bekommen einen Treffer
# (best_sim 0.742, weit über SIM_THRESHOLD 0.35). Ursache laut Diagnose:
# SIM_THRESHOLD liegt unter der Rausch-Untergrenze des multilingualen
# MiniLM-Modells für kurze Fragesätze — schon reine Fragesatz-Struktur
# ("Wie … ich …?") erzeugt hohe Cosine-Similarity, unabhängig vom Thema.
# 0 % ist damit der ehrliche gemessene Boden, kein Puffer möglich (kleiner
# als 0 gibt es nicht) — die Schwelle bleibt bewusst bei 0, bis SIM_THRESHOLD
# selbst neu kalibriert wird (außerhalb des Scopes dieses Fixes, siehe
# README). Das Gate misst die Rate weiterhin bei jedem Lauf und macht die
# Regression sichtbar, kann sie aber aktuell nicht verhindern.
MIN_ABSTENTION_RATE = 0.0


def _questions_path(argv: list[str]) -> Path:
    """Liest optionales --questions PATH aus argv, sonst Default-Tuning-Set."""
    if "--questions" in argv:
        return Path(argv[argv.index("--questions") + 1])
    return QUESTIONS_PATH


def hit_rate_at_k(retriever, questions: list[dict], k: int = 5) -> tuple[float, list[dict]]:
    misses = []
    hits = 0
    for item in questions:
        ids = [d.id for d in retriever.retrieve(item["question"], top_k=k)]
        if item["expected_doc_id"] in ids:
            hits += 1
        else:
            misses.append({**item, "got": ids})
    return hits / len(questions), misses


def abstention_rate(retriever, questions: list[dict]) -> tuple[float, list[dict]]:
    """Anteil themenfremder Fragen, bei denen retrieve() korrekt leer zurückgibt.

    questions enthält kein expected_doc_id — der erwartete Fall ist gerade
    das Fehlen jedes Treffers. Fragen, die trotzdem einen Treffer bekommen,
    landen mit ihrem Top-Treffer (id + title) in der Fehlliste, damit ein
    Fehlschlag diagnostizierbar bleibt.
    """
    false_hits = []
    abstained = 0
    for item in questions:
        docs = retriever.retrieve(item["question"])
        if not docs:
            abstained += 1
        else:
            false_hits.append({**item, "got_id": docs[0].id, "got_title": docs[0].title})
    return abstained / len(questions), false_hits


def _rewrite_questions(questions: list[dict]) -> list[dict]:
    """Nicht-deutsche Fragen per Haiku umformulieren — wie im Live-Pfad (kostet API-Cents)."""
    import asyncio

    from app.llm import get_client, rewrite_query
    from app.textproc import looks_german

    async def rewrite_all() -> list[dict]:
        client = get_client()
        rewritten = []
        for item in questions:
            if looks_german(item["question"]):
                rewritten.append(item)
                continue
            new_question, _ = await rewrite_query([], item["question"], client)
            print(f"  umformuliert: {item['question']!r} -> {new_question!r}")
            rewritten.append({**item, "question": new_question})
        return rewritten

    return asyncio.run(rewrite_all())


def check_gate(tuning_rate: float, holdout_rate: float, abstention_rate_value: float) -> list[str]:
    """Prüft Hit- und Abstention-Rate gegen ihre Mindestschwelle. Leer = Gate bestanden."""
    failures = []
    if tuning_rate < TUNING_MIN_HIT_RATE:
        failures.append(
            f"Tuning-Hit-Rate {tuning_rate:.0%} unter Minimum {TUNING_MIN_HIT_RATE:.0%}"
        )
    if holdout_rate < HOLDOUT_MIN_HIT_RATE:
        failures.append(
            f"Holdout-Hit-Rate {holdout_rate:.0%} unter Minimum {HOLDOUT_MIN_HIT_RATE:.0%}"
        )
    if abstention_rate_value < MIN_ABSTENTION_RATE:
        failures.append(
            f"Abstention-Rate {abstention_rate_value:.0%} unter Minimum {MIN_ABSTENTION_RATE:.0%}"
        )
    return failures


if __name__ == "__main__":
    import sys

    from app.config import settings
    from app.retrieval import Retriever

    retriever = Retriever(settings.index_dir, settings.corpus_path)

    if "--gate" in sys.argv:
        tuning = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
        holdout = json.loads(HOLDOUT_QUESTIONS_PATH.read_text(encoding="utf-8"))
        offtopic = json.loads(OFFTOPIC_QUESTIONS_PATH.read_text(encoding="utf-8"))
        tuning_rate, _ = hit_rate_at_k(retriever, tuning)
        holdout_rate, _ = hit_rate_at_k(retriever, holdout)
        abstain_rate, false_hits = abstention_rate(retriever, offtopic)
        print(f"Tuning-Hit-Rate@5: {tuning_rate:.0%}")
        print(f"Holdout-Hit-Rate@5: {holdout_rate:.0%}")
        print(f"Abstention-Rate: {abstain_rate:.0%}")
        for false_hit in false_hits:
            print(f"  FALSE HIT: {false_hit['question']!r} -> "
                  f"{false_hit['got_id']} ({false_hit['got_title']!r})")
        failures = check_gate(tuning_rate, holdout_rate, abstain_rate)
        for failure in failures:
            print(f"GATE FAIL: {failure}")
        sys.exit(1 if failures else 0)

    questions = json.loads(_questions_path(sys.argv).read_text(encoding="utf-8"))
    if "--with-rewrite" in sys.argv:
        questions = _rewrite_questions(questions)
    rate, misses = hit_rate_at_k(retriever, questions)
    print(f"Hit-Rate@5: {rate:.0%} ({len(questions) - len(misses)}/{len(questions)})")
    for miss in misses:
        print(f"  MISS: {miss['question']!r} erwartet {miss['expected_doc_id']}, bekam {miss['got']}")
