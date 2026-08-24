"""Zwei-Signal-LLM-Reranker (Rangfolge + top1_confidence) -- Experiment,
noch nicht integriert. Siehe
docs/superpowers/specs/2026-08-24-two-signal-reranker-design.md.
app/retrieval.py bleibt unveraendert, solange nur gemessen wird."""
from __future__ import annotations

import json

from app.config import settings
from app.retrieval import RetrievedDoc, _dedupe_ranking

MAX_LLM_RERANK_TOKENS = 400


def _system_prompt(n: int) -> str:
    example_ranking = list(range(n - 1, -1, -1))
    example_json = '{"ranking": [' + ", ".join(str(i) for i in example_ranking) + '], "top1_confidence": 0}'
    return (
        "Du bist ein Reranker fuer Chrono24-Hilfeseiten. Du bekommst eine "
        f"Nutzerfrage und genau {n} nummerierte Kandidaten-Dokumente (Index "
        f"0 bis {n - 1}). Sortiere die Kandidaten nach Relevanz fuer die "
        "Frage, absteigend -- der am besten passende Kandidat zuerst. "
        "Antworte NUR mit einem JSON-Objekt, keine Erklaerung, kein "
        "Markdown drumherum, Beispiel-Format: " + example_json + "\n\n"
        f'"ranking" muss alle {n} Indizes genau einmal enthalten, in deiner '
        'Rangfolge. "top1_confidence" bewertet NUR den an Position 0 deiner '
        "Rangfolge platzierten Kandidaten, unabhaengig von den anderen: "
        "0 = beantwortet die Frage ueberhaupt nicht, auch nicht ansatzweise; "
        "10 = beantwortet die Frage vollstaendig und eindeutig. Werte "
        "dazwischen nach eigenem Ermessen."
    )


def build_llm_rerank_prompt(query: str, docs: list[RetrievedDoc]) -> str:
    candidates = "\n\n".join(f"[{i}] {doc.title}\n{doc.text}" for i, doc in enumerate(docs))
    return f"Frage: {query}\n\nKandidaten:\n{candidates}"


def _parse_response(text: str, n: int) -> tuple[list[int], float]:
    """Konservativer Fallback bei jedem Abweichen vom erwarteten Format --
    ein Malformed-Value darf nie zu einem falsch-positiven Treffer fuehren
    (Spec, Abschnitt Fehlerbehandlung)."""
    identity = list(range(n))
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return identity, 0.0
    if not isinstance(data, dict):
        return identity, 0.0

    ranking = data.get("ranking")
    if not isinstance(ranking, list) or sorted(ranking) != identity:
        ranking = identity

    confidence = data.get("top1_confidence")
    is_plain_number = isinstance(confidence, (int, float)) and not isinstance(confidence, bool)
    if not is_plain_number or not (0 <= confidence <= 10):
        confidence = 0.0

    return ranking, float(confidence)


def union_candidates(vector_ranking: list[str], bm25_ranking: list[str]) -> list[str]:
    """Vereinigung statt RRF-Top-n-Cut -- Kandidaten-Union-Fix aus der Spec:
    ohne ihn sieht kein Reranker Kandidaten, die nur in einer der beiden
    Top-10-Listen weit vorn liegen."""
    return _dedupe_ranking(vector_ranking + bm25_ranking)


async def llm_two_signal_rerank(
    query: str, docs: list[RetrievedDoc], client
) -> tuple[list[int], float]:
    """Gibt (Rangfolge als 0-indexierte Positionsliste, top1_confidence)
    zurueck. ranking[0] ist der Index des nach Rangfolge bestplatzierten
    Kandidaten in docs."""
    response = await client.messages.create(
        model=settings.model,
        max_tokens=MAX_LLM_RERANK_TOKENS,
        system=_system_prompt(len(docs)),
        temperature=0,
        messages=[{"role": "user", "content": build_llm_rerank_prompt(query, docs)}],
    )
    text = next((b.text for b in response.content if b.type == "text"), "").strip()
    return _parse_response(text, len(docs))
