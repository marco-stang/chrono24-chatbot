"""Dedupe-Logik fuer Retrieval-Rankings, in eigenem Modul, damit app.retrieval
und app.llm_reranker sie beide importieren koennen ohne zirkulaeren Import
(app.retrieval importiert llm_two_signal_rerank/union_candidates aus
app.llm_reranker)."""
from __future__ import annotations


def dedupe_ranking(ids: list[str]) -> list[str]:
    """Erster (bester) Treffer pro kanonischer ID gewinnt -- Varianten-Duplikate raus."""
    seen: set[str] = set()
    result: list[str] = []
    for doc_id in ids:
        if doc_id not in seen:
            seen.add(doc_id)
            result.append(doc_id)
    return result
