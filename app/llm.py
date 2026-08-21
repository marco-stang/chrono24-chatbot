"""Anbindung an Claude: Kontextbau, Query-Rewriting, gestreamte Antwort."""
from __future__ import annotations

from collections.abc import AsyncIterator

from anthropic import AsyncAnthropic

from app.config import settings
from app.retrieval import RetrievedDoc
from app.textproc import looks_german

SYSTEM_PROMPT = (
    "Du bist ein Assistent für Fragen zu den Hilfeseiten von Chrono24. "
    "Beantworte die Frage AUSSCHLIESSLICH mit Informationen aus dem gelieferten Kontext. "
    "Antworte auf Deutsch, kurz und präzise. "
    "Formatiere mit einfachem Markdown: **fett** für Wichtiges, '- ' für Aufzählungen. "
    "Keine Überschriften, keine Tabellen, keine Links. "
    "Belege Aussagen mit den Quellennummern in eckigen Klammern, z. B. [1] oder [2]. "
    "Steht die Antwort nicht im Kontext, sage ehrlich: "
    "'Dazu finde ich nichts in den Chrono24-Hilfeseiten.' Erfinde nichts."
)

REWRITE_SYSTEM = (
    "Du erhältst einen Chatverlauf und eine Folgefrage. "
    "Formuliere die Folgefrage als eigenständige, vollständige Frage auf Deutsch um, "
    "sodass sie ohne den Verlauf verständlich ist. "
    "Antworte NUR mit der umformulierten Frage, ohne Erklärung."
)

MAX_ANSWER_TOKENS = 1024
MAX_REWRITE_TOKENS = 200

_client: AsyncAnthropic | None = None


def get_client() -> AsyncAnthropic:
    global _client
    if _client is None:
        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY ist nicht gesetzt")
        _client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    return _client


def build_context(docs: list[RetrievedDoc]) -> str:
    parts = [f"[{i}] {doc.title}\nURL: {doc.url}\n{doc.text}" for i, doc in enumerate(docs, 1)]
    return "\n\n".join(parts)


def build_rewrite_prompt(history: list[dict], question: str) -> str:
    lines = [f"{m['role']}: {m['content']}" for m in history]
    return "Chatverlauf:\n" + "\n".join(lines) + f"\n\nFolgefrage: {question}"


async def rewrite_query(history: list[dict], question: str, client) -> str:
    # Ohne Verlauf nur umformulieren, wenn die Frage nicht deutsch aussieht —
    # BM25 arbeitet auf deutschem Korpus, englische Queries matchen sonst kaum.
    if not history and looks_german(question):
        return question
    response = await client.messages.create(
        model=settings.model,
        max_tokens=MAX_REWRITE_TOKENS,
        system=REWRITE_SYSTEM,
        messages=[{"role": "user", "content": build_rewrite_prompt(history, question)}],
    )
    text = next((b.text for b in response.content if b.type == "text"), "").strip()
    return text or question


async def stream_answer(
    question: str, docs: list[RetrievedDoc], history: list[dict], client
) -> AsyncIterator[dict]:
    context = build_context(docs)
    messages = history + [
        {"role": "user", "content": f"Kontext:\n{context}\n\nFrage: {question}"}
    ]
    async with client.messages.stream(
        model=settings.model,
        max_tokens=MAX_ANSWER_TOKENS,
        system=SYSTEM_PROMPT,
        messages=messages,
    ) as stream:
        async for text in stream.text_stream:
            yield {"type": "token", "text": text}
        final = await stream.get_final_message()
    yield {
        "type": "usage",
        "input_tokens": final.usage.input_tokens,
        "output_tokens": final.usage.output_tokens,
    }
