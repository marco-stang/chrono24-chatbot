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
