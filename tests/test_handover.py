"""Tests für die Handover-Briefing-Erzeugung (Stufe B)."""
import json

import pytest

from app.handover import (
    SYSTEM_PROMPT,
    HandoverError,
    build_lines,
    build_prompt,
    normalize_briefing,
    parse_response,
)

MESSAGES = [
    {"role": "user", "content": "Meine Rolex ist nicht angekommen."},
    {"role": "assistant", "content": "Der Käuferschutz sichert deine Zahlung ab. [1]"},
]


def test_build_lines_maps_agent_role_to_support():
    lines = build_lines([{"role": "agent", "content": "Hier Tier-1-Support, ich übernehme."}])
    assert lines == [{"id": "M01", "actor": "Support",
                      "text": "Hier Tier-1-Support, ich übernehme."}]


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


# --- generate_briefing (Orchestrator) ---
from types import SimpleNamespace

from app.handover import generate_briefing

LINES_MESSAGES = [
    {"role": "user",
     "content": "Meine Rolex Daytona ist nach zwei Wochen immer noch nicht angekommen."},
    {"role": "assistant",
     "content": "Der Käuferschutz sichert deine Zahlung vollständig ab. [1]"},
]

VALID_BRIEFING = {
    "situation": {"text": "Rolex Daytona ist nach zwei Wochen nicht angekommen",
                  "source_lines": ["M01"]},
    "history": {"text": "Käuferschutz sichert die Zahlung vollständig ab",
                "source_lines": ["M02"]},
    "sentiment": {"label": "besorgt",
                  "quote": "nach zwei Wochen immer noch nicht angekommen",
                  "source_lines": ["M01"]},
    "open_question": {"text": "Wo ist die Rolex Daytona nach zwei Wochen",
                      "source_lines": ["M01"]},
    "claims": [],
}

BAD_BRIEFING = {**VALID_BRIEFING,
                "claims": [{"text": "Kunde verlangt sofortige Kontosperrung wegen Betrug",
                            "source_lines": ["M02"]}]}


def _response(briefing, input_tokens=100, output_tokens=50):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=json.dumps(briefing, ensure_ascii=False))],
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens))


class FakeClient:
    def __init__(self, responses):
        self.calls = []
        self._responses = list(responses)
        self.messages = self

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses[len(self.calls) - 1]


@pytest.mark.asyncio
async def test_valid_briefing_returns_ok_with_one_call():
    client = FakeClient([_response(VALID_BRIEFING)])
    result = await generate_briefing(LINES_MESSAGES, client)
    assert result["status"] == "ok"
    assert len(client.calls) == 1
    assert result["tokens"] == 150
    assert all(c.status in ("PASS", "WEAK") for c in result["validation"])
    assert result["lines"][0]["id"] == "M01"


@pytest.mark.asyncio
async def test_fail_then_valid_retries_with_failure_note():
    client = FakeClient([_response(BAD_BRIEFING), _response(VALID_BRIEFING)])
    result = await generate_briefing(LINES_MESSAGES, client)
    assert result["status"] == "ok"
    assert len(client.calls) == 2
    second_prompt = client.calls[1]["messages"][0]["content"]
    assert "unbelegte Aussage" in second_prompt
    assert "Kontosperrung" in second_prompt
    assert result["tokens"] == 300


@pytest.mark.asyncio
async def test_two_fails_returns_rejected_with_failed_claims():
    client = FakeClient([_response(BAD_BRIEFING), _response(BAD_BRIEFING)])
    result = await generate_briefing(LINES_MESSAGES, client)
    assert result["status"] == "rejected"
    assert len(client.calls) == 2
    assert any("Kontosperrung" in text for text in result["failed_claims"])


@pytest.mark.asyncio
async def test_invalid_json_on_retry_raises_handover_error_with_burned_tokens():
    # Erster Versuch: valides JSON, aber FAIL-Claim -> erzwingt den Retry.
    # Zweiter Versuch: kaputtes JSON -> generate_briefing muss die bereits
    # verbrannten Tokens als HandoverError weiterreichen, statt sie zu verlieren.
    invalid_json_response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="das ist kein JSON")],
        usage=SimpleNamespace(input_tokens=0, output_tokens=0))
    client = FakeClient([_response(BAD_BRIEFING), invalid_json_response])
    with pytest.raises(HandoverError) as exc_info:
        await generate_briefing(LINES_MESSAGES, client)
    assert exc_info.value.tokens == 150
