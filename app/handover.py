"""Handover-Briefing: Chat-History → geprüftes Übergabe-Briefing (Stufe B).

Portiert aus dem Handover Brief Generator (src/extract.py, src/orchestrator.py):
das LLM extrahiert ein Briefing mit Zeilen-Zitaten, der deterministische
Validator aus app/faithcheck.py prüft jede Aussage per Token-Overlap."""
from __future__ import annotations

import json

from app import faithcheck
from app.config import settings

MAX_ATTEMPTS = 2
MAX_BRIEFING_TOKENS = 1024

SYSTEM_PROMPT = """Du extrahierst ein strukturiertes Übergabe-Briefing aus einem \
Chatverlauf zwischen einem Kunden und dem FAQ-Bot eines Luxusuhren-Marktplatzes. \
Antworte ausschließlich mit JSON in exakt diesem Schema:

{
  "situation": {"text": "...", "source_lines": ["M01"]},
  "history": {"text": "...", "source_lines": ["M01", "M02"]},
  "sentiment": {"label": "...", "quote": "wörtliches Zitat aus dem Chat", "source_lines": ["M01"]},
  "open_question": {"text": "...", "source_lines": ["M01"]},
  "claims": [{"text": "...", "source_lines": ["M01"]}]
}

Jedes Feld muss über source_lines exakt die Zeilen-IDs referenzieren, aus denen \
die Aussage stammt. Erfinde nichts, das nicht durch die referenzierten Zeilen \
gedeckt ist. Falls zwei Aussagen im Chat widersprüchlich sind, gib beide als \
separate Claims mit ihren jeweiligen Quellzeilen an, statt sie zu glätten."""

_ACTOR = {"user": "Kunde", "assistant": "Bot"}
_REQUIRED_FIELDS = {"situation", "history", "sentiment", "open_question", "claims"}


class HandoverError(Exception):
    """Briefing-Erzeugung fehlgeschlagen; trägt die bereits verbrannten Tokens,
    damit der Endpoint das Budget auch im Fehlerpfad korrekt belastet."""

    def __init__(self, tokens: int):
        self.tokens = tokens
        super().__init__("Briefing-Erzeugung fehlgeschlagen")


def build_lines(messages: list[dict]) -> list[dict]:
    return [{"id": f"M{i:02d}", "actor": _ACTOR[m["role"]], "text": m["content"]}
            for i, m in enumerate(messages, 1)]


def build_prompt(lines: list[dict], previous_failure_note: str | None = None) -> str:
    lines_text = "\n".join(f"{l['id']} [{l['actor']}]: {l['text']}" for l in lines)
    prompt = f"Chatverlauf:\n{lines_text}\n\nErzeuge das Handover-Briefing als JSON."
    if previous_failure_note:
        prompt += f"\n\nHinweis: {previous_failure_note}"
    return prompt


def _strip_markdown_fence(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
    return text.strip()


def parse_response(raw_text: str) -> dict:
    data = json.loads(_strip_markdown_fence(raw_text))
    missing = _REQUIRED_FIELDS - data.keys()
    if missing:
        raise ValueError(f"Briefing-Antwort ohne Pflichtfelder: {sorted(missing)}")
    return data


def normalize_briefing(briefing: dict) -> list[dict]:
    claims = []
    for field in ("situation", "history"):
        entry = briefing[field]
        claims.append({"text": entry["text"], "source_lines": entry["source_lines"]})
    sentiment = briefing["sentiment"]
    claims.append({"text": sentiment["quote"], "source_lines": sentiment["source_lines"]})
    entry = briefing["open_question"]
    claims.append({"text": entry["text"], "source_lines": entry["source_lines"]})
    for claim in briefing["claims"]:
        claims.append({"text": claim["text"], "source_lines": claim["source_lines"]})
    return claims


async def generate_briefing(messages: list[dict], client) -> dict:
    """Extract → Validierung, bei FAIL ein Retry mit Fehlerhinweis, dann rejected.

    Exceptions (kaputtes JSON, API-Fehler) propagieren zum Aufrufer —
    kein stiller Fallback."""
    lines = build_lines(messages)
    lines_by_id = {line["id"]: line["text"] for line in lines}
    tokens = 0
    failure_note = None
    briefing = None
    validation: list = []
    failed: list = []

    for _ in range(MAX_ATTEMPTS):
        try:
            response = await client.messages.create(
                model=settings.model,
                max_tokens=MAX_BRIEFING_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": build_prompt(lines, failure_note)}],
            )
            tokens += response.usage.input_tokens + response.usage.output_tokens
            raw = next((b.text for b in response.content if b.type == "text"), "")
            briefing = parse_response(raw)
            validation = faithcheck.validate_claims(normalize_briefing(briefing), lines_by_id)
        except Exception as exc:
            raise HandoverError(tokens) from exc
        failed = [c for c in validation if c.status == "FAIL"]
        if not failed:
            return {"status": "ok", "briefing": briefing, "validation": validation,
                    "lines": lines, "tokens": tokens}
        failure_note = ("Vorherige Antwort hatte unbelegte Aussage(n): "
                        + "; ".join(c.text for c in failed))

    return {"status": "rejected", "briefing": briefing, "validation": validation,
            "failed_claims": [c.text for c in failed], "lines": lines, "tokens": tokens}
