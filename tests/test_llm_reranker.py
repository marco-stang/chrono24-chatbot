import anthropic
import httpx
import pytest

from app.llm_reranker import (
    _parse_response,
    _system_prompt,
    build_llm_rerank_prompt,
    llm_two_signal_rerank,
    union_candidates,
)
from app.retrieval import RetrievedDoc

DOCS = [
    RetrievedDoc(id="faq-0001", type="faq", title="Frage A",
                 url="https://www.chrono24.de/info/faqs.htm", text="Antwort A", score=0.1),
    RetrievedDoc(id="faq-0002", type="faq", title="Frage B",
                 url="https://www.chrono24.de/info/faqs.htm", text="Antwort B", score=0.1),
]


def test_system_prompt_names_exact_candidate_count():
    prompt = _system_prompt(3)
    assert "genau 3 nummerierte" in prompt
    assert "0 bis 2" in prompt


def test_system_prompt_includes_full_example_array():
    prompt = _system_prompt(3)
    assert '"ranking": [2, 1, 0]' in prompt


def test_build_llm_rerank_prompt_numbers_candidates_from_zero():
    prompt = build_llm_rerank_prompt("Wie funktioniert der Käuferschutz?", DOCS)
    assert "[0] Frage A" in prompt
    assert "[1] Frage B" in prompt
    assert "Antwort A" in prompt
    assert "Wie funktioniert der Käuferschutz?" in prompt


def test_parse_response_accepts_valid_json():
    ranking, confidence, used_fallback = _parse_response(
        '{"ranking": [1, 0], "top1_confidence": 8}', n=2)
    assert ranking == [1, 0]
    assert confidence == 8.0
    assert used_fallback is False


def test_parse_response_falls_back_to_identity_ranking_on_incomplete_array():
    ranking, _, used_fallback = _parse_response('{"ranking": [0], "top1_confidence": 7}', n=2)
    assert ranking == [0, 1]
    assert used_fallback is True


def test_parse_response_falls_back_to_zero_confidence_when_missing():
    ranking, confidence, used_fallback = _parse_response('{"ranking": [1, 0]}', n=2)
    assert confidence == 0.0
    assert ranking == [1, 0]
    assert used_fallback is True


def test_parse_response_falls_back_to_zero_confidence_when_out_of_range():
    _, confidence, used_fallback = _parse_response(
        '{"ranking": [1, 0], "top1_confidence": 15}', n=2)
    assert confidence == 0.0
    assert used_fallback is True


def test_parse_response_falls_back_to_zero_confidence_on_boolean_value():
    # bool ist in Python eine int-Subklasse -- ohne expliziten Ausschluss
    # würde True fälschlich zu 1.0 statt zum konservativen Fallback.
    _, confidence, used_fallback = _parse_response(
        '{"ranking": [1, 0], "top1_confidence": true}', n=2)
    assert confidence == 0.0
    assert used_fallback is True


def test_parse_response_falls_back_completely_on_malformed_json():
    ranking, confidence, used_fallback = _parse_response("not json at all", n=2)
    assert ranking == [0, 1]
    assert confidence == 0.0
    assert used_fallback is True


def test_parse_response_falls_back_on_mixed_type_ranking():
    # [0, "1"] darf nicht crashen (sorted() auf gemischten Typen wirft
    # TypeError) und muss konservativ auf die Identitaet zurueckfallen.
    # top1_confidence selbst ist hier valide (8) und bleibt unangetastet --
    # nur used_fallback markiert, dass die Rangfolge nicht vom Modell kam.
    ranking, confidence, used_fallback = _parse_response(
        '{"ranking": [0, "1"], "top1_confidence": 8}', n=2)
    assert ranking == [0, 1]
    assert confidence == 8.0
    assert used_fallback is True


def test_parse_response_falls_back_on_float_ranking():
    # [0.0, 1.0] besteht sorted(ranking) == identity (Gleichheit mit ints),
    # crasht aber danach bei docs[ranking[0]] mit "list indices must be
    # integers" -- muss deshalb schon hier als ungueltig erkannt werden.
    ranking, confidence, used_fallback = _parse_response(
        '{"ranking": [0.0, 1.0], "top1_confidence": 8}', n=2)
    assert ranking == [0, 1]
    assert confidence == 8.0
    assert used_fallback is True


def test_parse_response_falls_back_on_boolean_in_ranking():
    # bool ist eine int-Subklasse -- [true, 0] darf nicht als [True, 0]
    # durchgehen, sonst inkonsistent zum expliziten bool-Ausschluss bei
    # confidence zwei Zeilen darunter.
    ranking, confidence, used_fallback = _parse_response(
        '{"ranking": [true, 0], "top1_confidence": 8}', n=2)
    assert ranking == [0, 1]
    assert confidence == 8.0
    assert used_fallback is True


def test_parse_response_used_fallback_distinguishes_parse_failure_from_genuine_low_confidence():
    genuine_ranking, genuine_confidence, genuine_fallback = _parse_response(
        '{"ranking": [1, 0], "top1_confidence": 0}', n=2)
    broken_ranking, broken_confidence, broken_fallback = _parse_response(
        "not json at all", n=2)
    # Beide liefern confidence 0.0 -- ohne used_fallback nicht
    # unterscheidbar, mit used_fallback schon.
    assert genuine_confidence == broken_confidence == 0.0
    assert genuine_fallback is False
    assert broken_fallback is True
    assert genuine_ranking != broken_ranking or genuine_fallback != broken_fallback


def test_parse_response_strips_json_code_fence():
    """Verifies that markdown code fence ```json\n...\n``` is stripped."""
    fenced_json = '```json\n{"ranking": [1, 0], "top1_confidence": 9}\n```'
    ranking, confidence, used_fallback = _parse_response(fenced_json, n=2)
    assert ranking == [1, 0]
    assert confidence == 9.0
    assert used_fallback is False


def test_parse_response_strips_code_fence_without_language_tag():
    """Verifies that plain code fence ```\n...\n``` (without json tag) is stripped."""
    fenced_json = '```\n{"ranking": [1, 0], "top1_confidence": 8}\n```'
    ranking, confidence, used_fallback = _parse_response(fenced_json, n=2)
    assert ranking == [1, 0]
    assert confidence == 8.0
    assert used_fallback is False


def test_parse_response_strips_fence_with_leading_trailing_whitespace():
    """Verifies that whitespace around fence markers is handled correctly."""
    fenced_json = '  ```json\n{"ranking": [1, 0], "top1_confidence": 7}\n```  '
    ranking, confidence, used_fallback = _parse_response(fenced_json, n=2)
    assert ranking == [1, 0]
    assert confidence == 7.0
    assert used_fallback is False


def test_parse_response_unfenced_json_still_works():
    """Verifies that plain JSON without code fence is still parsed correctly."""
    plain_json = '{"ranking": [1, 0], "top1_confidence": 6}'
    ranking, confidence, used_fallback = _parse_response(plain_json, n=2)
    assert ranking == [1, 0]
    assert confidence == 6.0
    assert used_fallback is False


def test_parse_response_malformed_text_still_falls_back():
    """Verifies that genuinely malformed text (fenced or not) still falls back."""
    malformed = '```json\nnot valid json at all\n```'
    ranking, confidence, used_fallback = _parse_response(malformed, n=2)
    assert ranking == [0, 1]
    assert confidence == 0.0
    assert used_fallback is True


def test_union_candidates_dedupes_keeping_first_occurrence():
    result = union_candidates(["a", "b"], ["b", "c"])
    assert result == ["a", "b", "c"]


class _FakeTextBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class _FakeUsage:
    input_tokens = 80
    output_tokens = 15


class _FakeResponse:
    def __init__(self, text):
        self.content = [_FakeTextBlock(text)]
        self.usage = _FakeUsage()


class _FakeMessages:
    def __init__(self, response_text):
        self._response_text = response_text
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse(self._response_text)


class _FakeClient:
    def __init__(self, response_text):
        self.messages = _FakeMessages(response_text)


async def test_llm_two_signal_rerank_returns_parsed_ranking_and_confidence():
    client = _FakeClient('{"ranking": [1, 0], "top1_confidence": 9}')
    ranking, confidence, used_fallback, tokens = await llm_two_signal_rerank("Frage?", DOCS, client)
    assert ranking == [1, 0]
    assert confidence == 9.0
    assert used_fallback is False
    assert tokens == 95  # _FakeUsage: input_tokens=80, output_tokens=15


async def test_llm_two_signal_rerank_pins_temperature_and_token_limit():
    client = _FakeClient('{"ranking": [0, 1], "top1_confidence": 5}')
    await llm_two_signal_rerank("Frage?", DOCS, client)
    call = client.messages.calls[0]
    assert call["temperature"] == 0
    assert call["max_tokens"] == 400
    assert f"genau {len(DOCS)} nummerierte" in call["system"]


async def test_llm_two_signal_rerank_falls_back_on_malformed_response():
    client = _FakeClient("kaputte Antwort, kein JSON")
    ranking, confidence, used_fallback, tokens = await llm_two_signal_rerank("Frage?", DOCS, client)
    assert ranking == [0, 1]
    assert confidence == 0.0
    assert used_fallback is True
    assert tokens == 95


class _RaisingMessages:
    """Simuliert einen API-Fehler (Rate-Limit/Timeout/Overload) nach den
    SDK-eigenen Retries -- client.messages.create() wirft statt eine Antwort
    zurueckzugeben."""

    def __init__(self, error: BaseException):
        self._error = error

    async def create(self, **kwargs):
        raise self._error


class _RaisingClient:
    def __init__(self, error: BaseException):
        self.messages = _RaisingMessages(error)


def _make_api_error() -> anthropic.APIError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.APIError("upstream overloaded", request, body=None)


async def test_llm_two_signal_rerank_treats_api_error_like_parse_fallback():
    """Vor dieser Integration war Retrieval rein lokal und konnte nie an
    einem Netzwerkfehler scheitern. anthropic.APIError (Basisklasse fuer
    retrybare wie nicht-retrybare Fehler des SDKs) muss deshalb genauso
    abstinieren wie ein Parse-Fallback -- kein unbehandelter Absturz bis in
    Retriever.retrieve() hinein."""
    client = _RaisingClient(_make_api_error())
    ranking, confidence, used_fallback, tokens = await llm_two_signal_rerank("Frage?", DOCS, client)
    assert ranking == [0, 1]
    assert confidence == 0.0
    assert used_fallback is True
    assert tokens == 0


async def test_llm_two_signal_rerank_does_not_catch_non_api_errors():
    """Nur anthropic.APIError (und Subklassen) wird abgefangen -- ein
    Programmierfehler (z.B. TypeError durch falsch verdrahtete kwargs) soll
    weiterhin sichtbar crashen statt still als Abstention maskiert zu
    werden."""
    client = _RaisingClient(TypeError("nicht abgefangen"))
    with pytest.raises(TypeError):
        await llm_two_signal_rerank("Frage?", DOCS, client)
