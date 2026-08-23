from pipeline.variants import (
    VARIANTS_SYSTEM,
    build_variants,
    generate_variants,
    parse_variants,
)


def test_parse_variants_reads_json_array():
    text = '["Wie lange dauert der Versand?", "Was kostet der Versand?"]'
    assert parse_variants(text) == ["Wie lange dauert der Versand?", "Was kostet der Versand?"]


def test_parse_variants_strips_code_fence():
    text = '```json\n["Frage A", "Frage B"]\n```'
    assert parse_variants(text) == ["Frage A", "Frage B"]


def test_parse_variants_returns_empty_on_invalid_json():
    assert parse_variants("Das ist kein JSON.") == []


def test_parse_variants_returns_empty_for_non_list_json():
    assert parse_variants('{"question": "x"}') == []


def test_parse_variants_ignores_non_string_and_blank_items():
    text = '["Frage A", "", 42, "  ", "Frage B"]'
    assert parse_variants(text) == ["Frage A", "Frage B"]


class FakeVariantMessages:
    def __init__(self, response_text):
        self.response_text = response_text
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return type("R", (), {
            "content": [type("B", (), {"type": "text", "text": self.response_text})()]
        })()


class FakeVariantClient:
    def __init__(self, response_text):
        self.messages = FakeVariantMessages(response_text)


async def test_generate_variants_calls_llm_with_question_and_parses_result():
    client = FakeVariantClient('["Wie lange dauert der Versand?", "Was kostet der Versand?"]')
    result = await generate_variants("Wie funktioniert der Versand?", client, model="claude-haiku-4-5")
    assert result == ["Wie lange dauert der Versand?", "Was kostet der Versand?"]
    call = client.messages.calls[0]
    assert call["model"] == "claude-haiku-4-5"
    assert call["system"] == VARIANTS_SYSTEM
    assert call["messages"] == [{"role": "user", "content": "Wie funktioniert der Versand?"}]


class FlakyVariantMessages:
    """Wirft beim zweiten Call, um Fehlertoleranz pro FAQ zu pruefen."""

    def __init__(self, responses):
        self.responses = responses
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        response = self.responses[self.calls - 1]
        if response is None:
            raise RuntimeError("API-Fehler")
        return type("R", (), {"content": [type("B", (), {"type": "text", "text": response})()]})()


class FlakyVariantClient:
    def __init__(self, responses):
        self.messages = FlakyVariantMessages(responses)


async def test_build_variants_skips_empty_results_and_survives_errors():
    faqs = [
        {"id": "faq-0001", "question": "Frage eins"},
        {"id": "faq-0002", "question": "Frage zwei"},
        {"id": "faq-0003", "question": "Frage drei"},
    ]
    client = FlakyVariantClient([
        '["Umformuliert eins"]',
        None,  # faq-0002: API-Fehler
        "[]",  # faq-0003: leere Antwort
    ])
    result = await build_variants(faqs, client, model="claude-haiku-4-5")
    assert result == {"faq-0001": ["Umformuliert eins"]}
