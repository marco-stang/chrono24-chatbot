"""Tests für die Handover-Briefing-Erzeugung (Stufe B)."""
import json

import pytest

from app.handover import (
    SYSTEM_PROMPT,
    build_lines,
    build_prompt,
    normalize_briefing,
    parse_response,
)

MESSAGES = [
    {"role": "user", "content": "Meine Rolex ist nicht angekommen."},
    {"role": "assistant", "content": "Der Käuferschutz sichert deine Zahlung ab. [1]"},
]


def test_build_lines_assigns_ids_and_actors():
    lines = build_lines(MESSAGES)
    assert lines == [
        {"id": "M01", "actor": "Kunde", "text": "Meine Rolex ist nicht angekommen."},
        {"id": "M02", "actor": "Bot",
         "text": "Der Käuferschutz sichert deine Zahlung ab. [1]"},
    ]


def test_build_prompt_contains_lines_and_failure_note():
    lines = build_lines(MESSAGES)
    prompt = build_prompt(lines, previous_failure_note="Aussage X war unbelegt")
    assert "M01 [Kunde]: Meine Rolex ist nicht angekommen." in prompt
    assert "Aussage X war unbelegt" in prompt


def test_system_prompt_is_german_chat_schema():
    assert "Chatverlauf" in SYSTEM_PROMPT
    assert "source_lines" in SYSTEM_PROMPT
    assert "claims" in SYSTEM_PROMPT


def test_parse_response_strips_markdown_fence():
    briefing = {"situation": {"text": "s", "source_lines": ["M01"]},
                "history": {"text": "h", "source_lines": ["M01"]},
                "sentiment": {"label": "frustriert", "quote": "q", "source_lines": ["M01"]},
                "open_question": {"text": "o", "source_lines": ["M01"]},
                "claims": []}
    raw = "```json\n" + json.dumps(briefing) + "\n```"
    assert parse_response(raw) == briefing


def test_parse_response_missing_field_raises():
    with pytest.raises(ValueError, match="claims"):
        parse_response(json.dumps({"situation": {}, "history": {},
                                   "sentiment": {}, "open_question": {}}))


def test_normalize_briefing_flattens_all_fields_to_claims():
    briefing = {"situation": {"text": "s", "source_lines": ["M01"]},
                "history": {"text": "h", "source_lines": ["M01", "M02"]},
                "sentiment": {"label": "frustriert", "quote": "q", "source_lines": ["M01"]},
                "open_question": {"text": "o", "source_lines": ["M02"]},
                "claims": [{"text": "c1", "source_lines": ["M01"]}]}
    claims = normalize_briefing(briefing)
    assert claims == [
        {"text": "s", "source_lines": ["M01"]},
        {"text": "h", "source_lines": ["M01", "M02"]},
        {"text": "q", "source_lines": ["M01"]},
        {"text": "o", "source_lines": ["M02"]},
        {"text": "c1", "source_lines": ["M01"]},
    ]
