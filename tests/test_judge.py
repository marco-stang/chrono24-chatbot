from app.retrieval import RetrievedDoc
from eval.judge import (
    JUDGE_SYSTEM,
    MIN_FAITHFUL_RATE,
    aggregate,
    build_judge_prompt,
    check_gate,
    judge_one,
    parse_verdict,
)

DOCS = [
    RetrievedDoc(id="faq-0001", type="faq", title="Wie funktioniert der Käuferschutz?",
                 url="https://www.chrono24.de/info/faqs.htm",
                 text="Der Käuferschutz sichert deine Zahlung ab.", score=0.05),
]


def test_build_judge_prompt_contains_question_context_answer():
    prompt = build_judge_prompt(
        "Wie funktioniert der Käuferschutz?",
        "[1] Käuferschutz\nDer Käuferschutz sichert deine Zahlung ab.",
        "Der Käuferschutz sichert deine Zahlung ab [1].",
    )
    assert "Wie funktioniert der Käuferschutz?" in prompt
    assert "Der Käuferschutz sichert deine Zahlung ab." in prompt
    assert "Der Käuferschutz sichert deine Zahlung ab [1]." in prompt


def test_judge_system_is_german_and_mentions_json():
    assert isinstance(JUDGE_SYSTEM, str)
    assert "JSON" in JUDGE_SYSTEM
    assert "faithful" in JUDGE_SYSTEM
    assert "answered" in JUDGE_SYSTEM


def test_parse_verdict_clean_json():
    text = '{"faithful": true, "answered": "voll", "begruendung": "Alles gedeckt."}'
    verdict = parse_verdict(text)
    assert verdict == {"faithful": True, "answered": "voll", "begruendung": "Alles gedeckt."}


def test_parse_verdict_with_json_fence():
    text = '```json\n{"faithful": false, "answered": "teilweise", "begruendung": "Teils frei erfunden."}\n```'
    verdict = parse_verdict(text)
    assert verdict["faithful"] is False
    assert verdict["answered"] == "teilweise"
    assert verdict["begruendung"] == "Teils frei erfunden."


def test_parse_verdict_with_plain_fence_no_language_tag():
    text = '```\n{"faithful": true, "answered": "nein", "begruendung": "x"}\n```'
    verdict = parse_verdict(text)
    assert verdict["faithful"] is True
    assert verdict["answered"] == "nein"


def test_parse_verdict_garbage_returns_parse_error():
    text = "Das ist kein JSON, sondern nur Text."
    verdict = parse_verdict(text)
    assert verdict["faithful"] is None
    assert verdict["answered"] == "parse_error"
    assert verdict["begruendung"] == text[:200]


def test_aggregate_counts_faithful_rate_and_answered_categories():
    results = [
        {"question": "q1", "answer": "a1", "verdict": {"faithful": True, "answered": "voll"}},
        {"question": "q2", "answer": "a2", "verdict": {"faithful": False, "answered": "teilweise"}},
        {"question": "q3", "answer": "a3", "verdict": {"faithful": True, "answered": "verweigert"}},
        {"question": "q4", "answer": "a4", "verdict": {"faithful": None, "answered": "parse_error"}},
        {"question": "q5", "answer": "a5", "verdict": {"faithful": None, "answered": "error"}},
    ]
    summary = aggregate(results)
    assert summary["n"] == 5
    # 2 von 3 mit faithful != None sind True
    assert summary["faithful_rate"] == 2 / 3
    assert summary["answered_counts"] == {
        "voll": 1,
        "teilweise": 1,
        "verweigert": 1,
        "parse_error": 1,
        "error": 1,
    }


def test_aggregate_empty_results_has_zero_rate_and_empty_counts():
    summary = aggregate([])
    assert summary["n"] == 0
    assert summary["faithful_rate"] == 0.0
    assert summary["answered_counts"] == {}


class FakeRetriever:
    def __init__(self, docs):
        self.docs = docs

    def retrieve(self, query, top_k=5):
        return self.docs


class FakeRewriteClient:
    """Nicht benutzt, da rewrite_query ohne Verlauf bei deutscher Frage keinen Call macht."""


class FakeJudgeMessages:
    def __init__(self, response_text):
        self.response_text = response_text
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return type("R", (), {
            "content": [type("B", (), {"type": "text", "text": self.response_text})()]
        })()


class FakeJudgeClient:
    def __init__(self, response_text):
        self.messages = FakeJudgeMessages(response_text)


async def fake_answer_fn(question, docs, history, client):
    return "Der Käuferschutz sichert deine Zahlung ab [1]."


async def test_judge_one_uses_injected_answer_fn_and_fake_client_no_real_api_call():
    retriever = FakeRetriever(DOCS)
    judge_client = FakeJudgeClient(
        '{"faithful": true, "answered": "voll", "begruendung": "Alles gedeckt."}'
    )

    result = await judge_one(
        "Wie funktioniert der Käuferschutz?",
        retriever,
        judge_client,
        answer_fn=fake_answer_fn,
    )

    assert result["question"] == "Wie funktioniert der Käuferschutz?"
    assert result["answer"] == "Der Käuferschutz sichert deine Zahlung ab [1]."
    assert result["verdict"] == {
        "faithful": True,
        "answered": "voll",
        "begruendung": "Alles gedeckt.",
    }
    # Judge-Call mit erwarteten Parametern
    call = judge_client.messages.calls[0]
    assert call["max_tokens"] == 300
    assert call["system"] == JUDGE_SYSTEM


async def test_judge_one_empty_retrieval_uses_refusal_answer_without_answer_fn_call():
    retriever = FakeRetriever([])
    judge_client = FakeJudgeClient(
        '{"faithful": true, "answered": "verweigert", "begruendung": "Ehrliche Absage."}'
    )
    calls = []

    async def tracking_answer_fn(question, docs, history, client):
        calls.append(question)
        return "sollte nicht aufgerufen werden"

    result = await judge_one(
        "Frage ohne Treffer im Korpus",
        retriever,
        judge_client,
        answer_fn=tracking_answer_fn,
    )

    assert calls == []
    assert result["answer"] == "Dazu finde ich nichts in den Chrono24-Hilfeseiten."
    assert result["verdict"]["answered"] == "verweigert"


def test_check_gate_passes_at_minimum_faithful_rate():
    assert check_gate({"faithful_rate": MIN_FAITHFUL_RATE, "n": 10, "answered_counts": {}}) == []


def test_check_gate_fails_below_minimum_faithful_rate():
    failures = check_gate({"faithful_rate": MIN_FAITHFUL_RATE - 0.01, "n": 10, "answered_counts": {}})
    assert len(failures) == 1
    assert "Faithful" in failures[0]


async def test_judge_call_pins_temperature_to_zero():
    """Der Judge ist ein Messinstrument, kein Autor. Ohne temperature=0 laeuft er
    auf dem API-Default 1.0 und liefert fuer identischen Code unterschiedliche
    Urteile -- drei Laeufe am 23.08.2026 ergaben 100 %, 94 % und 88 %
    Faithful-Rate, ohne dass sich an der Pipeline etwas geaendert hatte."""
    retriever = FakeRetriever(DOCS)
    judge_client = FakeJudgeClient(
        '{"faithful": true, "answered": "voll", "begruendung": "Alles gedeckt."}'
    )

    await judge_one(
        "Wie funktioniert der Käuferschutz?",
        retriever,
        judge_client,
        answer_fn=fake_answer_fn,
    )

    assert judge_client.messages.calls[0]["temperature"] == 0
