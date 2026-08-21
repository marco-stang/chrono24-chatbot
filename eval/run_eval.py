"""Misst Hit-Rate@k des Retrievals gegen handgeschriebene Testfragen."""
from __future__ import annotations

import json
from pathlib import Path

QUESTIONS_PATH = Path("eval/questions.json")


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


if __name__ == "__main__":
    import sys

    from app.config import settings
    from app.retrieval import Retriever

    questions = json.loads(_questions_path(sys.argv).read_text(encoding="utf-8"))
    if "--with-rewrite" in sys.argv:
        questions = _rewrite_questions(questions)
    retriever = Retriever(settings.index_dir, settings.corpus_path)
    rate, misses = hit_rate_at_k(retriever, questions)
    print(f"Hit-Rate@5: {rate:.0%} ({len(questions) - len(misses)}/{len(questions)})")
    for miss in misses:
        print(f"  MISS: {miss['question']!r} erwartet {miss['expected_doc_id']}, bekam {miss['got']}")
