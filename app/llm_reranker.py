"""Zwei-Signal-LLM-Reranker (Rangfolge + top1_confidence) fuer
Retriever.retrieve(). Siehe
docs/superpowers/specs/2026-08-24-llm-reranker-integration-design.md."""
from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

import anthropic

from app.config import settings
from app.ranking import dedupe_ranking

if TYPE_CHECKING:
    from app.retrieval import RetrievedDoc

MAX_LLM_RERANK_TOKENS = 400

_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL)


def _strip_code_fence(text: str) -> str:
    """Strips markdown code fence (```json or ```) from LLM response if present.

    Handles variants: ```json\n{...}\n``` and ```\n{...}\n```
    If no fence is present, returns the text unchanged.
    """
    match = _FENCE_RE.match(text.strip())
    return match.group(1) if match else text


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


def _parse_response(text: str, n: int) -> tuple[list[int], float, bool]:
    """Konservativer Fallback bei jedem Abweichen vom erwarteten Format --
    ein Malformed-Value darf nie zu einem falsch-positiven Treffer fuehren.
    Dritter Rueckgabewert `used_fallback`: True, wenn ranking und/oder
    confidence nicht valide geparst werden konnten -- sonst waere eine
    echte confidence von 0.0 nicht von einem Parse-Fehler zu
    unterscheiden."""
    identity = list(range(n))
    try:
        stripped_text = _strip_code_fence(text)
        data = json.loads(stripped_text)
    except (json.JSONDecodeError, TypeError):
        return identity, 0.0, True
    if not isinstance(data, dict):
        return identity, 0.0, True

    ranking = data.get("ranking")
    # bool ist eine int-Subklasse -- ohne den Ausschluss wuerde z.B.
    # [True, 0] als gueltige Indexliste durchgehen. sorted() faellt bei
    # gemischten Typen (z.B. [0, "1"]) sonst mit TypeError um, deshalb erst
    # der Typ-Check, dann erst sorted().
    is_index_list = isinstance(ranking, list) and all(
        isinstance(i, int) and not isinstance(i, bool) for i in ranking
    )
    ranking_fallback = not is_index_list or sorted(ranking) != identity
    if ranking_fallback:
        ranking = identity

    confidence = data.get("top1_confidence")
    is_plain_number = isinstance(confidence, (int, float)) and not isinstance(confidence, bool)
    confidence_fallback = not is_plain_number or not (0 <= confidence <= 10)
    if confidence_fallback:
        confidence = 0.0

    return ranking, float(confidence), ranking_fallback or confidence_fallback


def union_candidates(vector_ranking: list[str], bm25_ranking: list[str]) -> list[str]:
    """Vereinigung statt RRF-Top-n-Cut -- Kandidaten-Union-Fix: ohne ihn
    sieht kein Reranker Kandidaten, die nur in einer der beiden Top-10-
    Listen weit vorn liegen. dedupe_ranking lebt in app.ranking statt in
    app.retrieval, damit beide Module es importieren koennen ohne
    zirkulaeren Import (app.retrieval importiert umgekehrt
    llm_two_signal_rerank/union_candidates aus diesem Modul)."""
    return dedupe_ranking(vector_ranking + bm25_ranking)


async def llm_two_signal_rerank(
    query: str, docs: list[RetrievedDoc], client
) -> tuple[list[int], float, bool, int]:
    """Gibt (Rangfolge als 0-indexierte Positionsliste, top1_confidence,
    used_fallback, tokens) zurueck. ranking[0] ist der Index des nach
    Rangfolge bestplatzierten Kandidaten in docs. used_fallback ist True,
    wenn die Antwort nicht valide geparst werden konnte (siehe
    _parse_response) ODER der API-Call selbst fehlgeschlagen ist (Rate-Limit,
    Timeout, Overload -- nach den SDK-eigenen Retries). Aufrufer sollten das
    nicht mit einer echten confidence von 0.0 verwechseln. Vor dieser
    Integration war Retrieval rein lokal und konnte so nie fehlschlagen; ein
    API-Fehler wird hier bewusst wie ein Parse-Fehler behandelt (leeres
    Ergebnis statt Absturz, siehe Spec "Fehlerbehandlung") -- der Aufrufer
    (Retriever.retrieve) braucht dafuer keine eigene Fehlerbehandlung.
    tokens = verbrauchte Input- + Output-Tokens, fuers Tagesbudget-Tracking;
    0 bei einem API-Fehler, da dann keine Antwort mit usage-Feld existiert."""
    identity = list(range(len(docs)))
    try:
        response = await client.messages.create(
            model=settings.model,
            max_tokens=MAX_LLM_RERANK_TOKENS,
            system=_system_prompt(len(docs)),
            temperature=0,
            messages=[{"role": "user", "content": build_llm_rerank_prompt(query, docs)}],
        )
    except anthropic.APIError:
        return identity, 0.0, True, 0
    text = next((b.text for b in response.content if b.type == "text"), "").strip()
    ranking, confidence, used_fallback = _parse_response(text, len(docs))
    tokens = response.usage.input_tokens + response.usage.output_tokens
    return ranking, confidence, used_fallback, tokens
