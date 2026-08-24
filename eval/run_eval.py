"""Misst Hit-Rate@k des Retrievals gegen handgeschriebene Testfragen."""
from __future__ import annotations

import json
from pathlib import Path

from eval.stats import format_rate

QUESTIONS_PATH = Path("eval/questions.json")
HOLDOUT_QUESTIONS_PATH = Path("eval/questions_holdout.json")
OFFTOPIC_QUESTIONS_PATH = Path("eval/questions_offtopic.json")

# ACHTUNG (Stand 2026-08-24): diese drei Schwellen wurden gegen den
# Cross-Encoder-Pfad kalibriert (Werte/Messungen unten). Das eval-gate misst
# seit der Umstellung auf den LLM-Reranker (Task 9) aber den LLM-Pfad, der in
# einem frueheren Experiment ~100 % Tuning/Holdout-Hit-Rate und ~93 %
# Abstention gemessen hat (siehe HANDOVER-llm-reranker.md) -- deutlich ueber
# diesen Schwellen. Die Zahlen unten beschreiben also nicht mehr den Pfad, den
# das Gate aktuell prueft; das Gate wuerde selbst einen deutlichen Rueckgang
# der LLM-Pfad-Qualitaet (z.B. Abstention 93 % -> 40 %) noch durchwinken.
# Neukalibrierung gegen echte, aktuelle LLM-Pfad-Messungen steht aus.
#
# Historisch (Cross-Encoder-Pfad), gemessen: Tuning 91 % (30/33), Holdout
# 100 % (15/15) — letzteres seit die Eval nicht-deutsche Fragen umgeschrieben
# misst (siehe eval_query), also denselben Pfad wie der Live-Bot. Schwellen
# waren ein Regressionsboden mit rund zwei Fragen Puffer für
# Einzelfrage-Rauschen, keine Bestmarke: bei 33 bzw. 15 Fragen wiegt eine
# Frage 3 bzw. 6,7 Prozentpunkte.
TUNING_MIN_HIT_RATE = 0.85
HOLDOUT_MIN_HIT_RATE = 0.86
# Anteil themenfremder Fragen, bei denen der Bot leer zurückgibt (siehe
# app.retrieval: SIM_THRESHOLD / BM25_THRESHOLD / RERANK_THRESHOLD bzw.
# LLM_CONFIDENCE_THRESHOLD im LLM-Pfad, ODER-verknüpft). Historisch (Cross-
# Encoder-Pfad) gemessen auf eval/questions_offtopic.json (14 Fragen, gemischt
# eindeutig off-topic und absichtlich nah am Domänenvokabular): 50 % (7/14)
# bei 0 verlorenen on-topic-Treffern. Die 7 Durchrutscher sind die gewollt
# domänennahen Fragen (Omega-Wert, eBay-Vergleich, Versicherung) -- kein
# Retrieval-Signal trennt die vom Korpus, dort greifen LLM-Prompt und
# Faithcheck. Vor der Umstellung auf ODER-Logik (2026-08-23) lag die Rate bei
# 0 % (0/14). Schwelle mit Puffer für Einzelfrage-Rauschen (eine Frage = 7 %).
MIN_ABSTENTION_RATE = 0.35


def _questions_path(argv: list[str]) -> Path:
    """Liest optionales --questions PATH aus argv, sonst Default-Tuning-Set."""
    if "--questions" in argv:
        return Path(argv[argv.index("--questions") + 1])
    return QUESTIONS_PATH


def eval_query(item: dict) -> str:
    """Die Frage so, wie das Retrieval sie im Live-Betrieb sieht.

    Nicht-deutsche Fragen laufen dort erst durch rewrite_query (BM25 arbeitet auf
    deutschem Korpus). Die Umformulierung ist offline erzeugt und liegt als Feld
    `rewritten` im Fragen-JSON -- wie data/variants.json, damit der CI-Job ohne
    API-Kosten denselben Pfad misst, den Nutzer treffen.
    """
    return item.get("rewritten") or item["question"]


async def hit_rate_at_k(retriever, questions: list[dict], k: int = 5, client=None) -> tuple[float, list[dict]]:
    misses = []
    hits = 0
    for item in questions:
        docs, _ = await retriever.retrieve(eval_query(item), top_k=k, client=client)
        ids = [d.id for d in docs]
        if item["expected_doc_id"] in ids:
            hits += 1
        else:
            misses.append({**item, "got": ids})
    return hits / len(questions), misses


async def hit_rate_at_k_with_audience(
    retriever, questions: list[dict], k: int = 5, client=None
) -> tuple[float, list[dict]]:
    """Wie hit_rate_at_k, wendet aber vorher den harten audience-Filter aus
    Retriever.retrieve() an (Schritt 1, corpus-storage-rethink-design.md) --
    misst den tatsächlichen Live-Pfad, der in app/main.py nach dem Rewrite
    ebenfalls textproc.classify_audience() auf die Standalone-Frage anwendet.

    Additive eigene Funktion statt hit_rate_at_k selbst zu erweitern: so
    bleibt hit_rate_at_k als einfachere, audience-lose Variante fuer
    Aufrufer erhalten, die keinen Rollenfilter brauchen (siehe Docstring in
    Retriever.retrieve zur Rückwärtskompatibilität bestehender Aufrufer).
    """
    from app.textproc import classify_audience

    misses = []
    hits = 0
    for item in questions:
        query = eval_query(item)
        audience = classify_audience(query)
        docs, _ = await retriever.retrieve(query, top_k=k, audience=audience, client=client)
        ids = [d.id for d in docs]
        if item["expected_doc_id"] in ids:
            hits += 1
        else:
            misses.append({**item, "got": ids, "audience": audience})
    return hits / len(questions), misses


async def abstention_rate(retriever, questions: list[dict], client=None) -> tuple[float, list[dict]]:
    """Anteil themenfremder Fragen, bei denen retrieve() korrekt leer zurückgibt.

    questions enthält kein expected_doc_id — der erwartete Fall ist gerade
    das Fehlen jedes Treffers. Fragen, die trotzdem einen Treffer bekommen,
    landen mit ihrem Top-Treffer (id + title) in der Fehlliste, damit ein
    Fehlschlag diagnostizierbar bleibt.
    """
    false_hits = []
    abstained = 0
    for item in questions:
        docs, _ = await retriever.retrieve(item["question"], client=client)
        if not docs:
            abstained += 1
        else:
            false_hits.append({**item, "got_id": docs[0].id, "got_title": docs[0].title})
    return abstained / len(questions), false_hits


async def _rewrite_questions(questions: list[dict]) -> list[dict]:
    """Nicht-deutsche Fragen per Haiku umformulieren — wie im Live-Pfad (kostet API-Cents)."""
    from app.llm import get_client, rewrite_query
    from app.textproc import looks_german

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


async def _run_gate(retriever, client) -> int:
    tuning = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    holdout = json.loads(HOLDOUT_QUESTIONS_PATH.read_text(encoding="utf-8"))
    offtopic = json.loads(OFFTOPIC_QUESTIONS_PATH.read_text(encoding="utf-8"))
    tuning_rate, tuning_misses = await hit_rate_at_k_with_audience(retriever, tuning, client=client)
    holdout_rate, _ = await hit_rate_at_k_with_audience(retriever, holdout, client=client)
    abstain_rate, false_hits = await abstention_rate(retriever, offtopic, client=client)
    print(f"Tuning-Hit-Rate@5:  {format_rate(round(tuning_rate * len(tuning)), len(tuning))}")
    print(f"Holdout-Hit-Rate@5: {format_rate(round(holdout_rate * len(holdout)), len(holdout))}")
    print(f"Abstention-Rate:    {format_rate(len(offtopic) - len(false_hits), len(offtopic))}")
    for miss in tuning_misses:
        print(f"  TUNING MISS ({miss['audience']}): {miss['question']!r} erwartet "
              f"{miss['expected_doc_id']}, bekam {miss['got']}")
    for false_hit in false_hits:
        print(f"  FALSE HIT: {false_hit['question']!r} -> "
              f"{false_hit['got_id']} ({false_hit['got_title']!r})")
    failures = check_gate(tuning_rate, holdout_rate, abstain_rate)
    for failure in failures:
        print(f"GATE FAIL: {failure}")
    return 1 if failures else 0


async def _run_plain(retriever, argv: list[str], client) -> None:
    questions = json.loads(_questions_path(argv).read_text(encoding="utf-8"))
    if "--with-rewrite" in argv:
        questions = await _rewrite_questions(questions)
    _, misses = await hit_rate_at_k(retriever, questions, client=client)
    print(f"Hit-Rate@5: {format_rate(len(questions) - len(misses), len(questions))}")
    for miss in misses:
        print(f"  MISS: {miss['question']!r} erwartet {miss['expected_doc_id']}, bekam {miss['got']}")


if __name__ == "__main__":
    import asyncio
    import sys

    from app.config import settings
    from app.llm import get_client
    from app.retrieval import Retriever

    async def _main() -> int:
        retriever = Retriever(settings.index_dir, settings.corpus_path)
        client = get_client() if settings.use_llm_reranker else None
        if "--gate" in sys.argv:
            return await _run_gate(retriever, client)
        await _run_plain(retriever, sys.argv, client)
        return 0

    sys.exit(asyncio.run(_main()))
