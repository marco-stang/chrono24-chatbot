from app.llm import build_context, build_rewrite_prompt, rewrite_query
from app.retrieval import RetrievedDoc

DOCS = [
    RetrievedDoc(id="faq-0001", type="faq", title="Wie funktioniert der Käuferschutz?",
                 url="https://www.chrono24.de/info/faqs.htm",
                 text="Der Käuferschutz sichert deine Zahlung ab.", score=0.05),
    RetrievedDoc(id="info-shipping-0001", type="page_chunk", title="Versand — Versichert",
                 url="https://www.chrono24.de/info/shipping.htm",
                 text="Uhren werden versichert verschickt.", score=0.03),
]


def test_build_context_numbers_docs_with_urls():
    context = build_context(DOCS)
    assert "[1] Wie funktioniert der Käuferschutz?" in context
    assert "[2] Versand — Versichert" in context
    assert "https://www.chrono24.de/info/shipping.htm" in context
    assert "sichert deine Zahlung" in context


def test_build_rewrite_prompt_contains_history_and_question():
    history = [{"role": "user", "content": "Wie kaufe ich eine Uhr?"},
               {"role": "assistant", "content": "Über die Plattform."}]
    prompt = build_rewrite_prompt(history, "und beim Verkauf?")
    assert "Wie kaufe ich eine Uhr?" in prompt
    assert "und beim Verkauf?" in prompt


async def test_rewrite_query_without_history_returns_question():
    result, tokens = await rewrite_query([], "Wie funktioniert der Käuferschutz?", client=None)
    assert result == "Wie funktioniert der Käuferschutz?"
    assert tokens == 0


class FakeTextBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class FakeUsage:
    input_tokens = 80
    output_tokens = 15


class FakeResponse:
    def __init__(self, text):
        self.content = [FakeTextBlock(text)]
        self.usage = FakeUsage()


class FakeMessages:
    async def create(self, **kwargs):
        return FakeResponse("Wie verkaufe ich eine Uhr auf Chrono24?")


class FakeClient:
    messages = FakeMessages()


async def test_rewrite_query_with_history_calls_llm():
    history = [{"role": "user", "content": "Wie kaufe ich eine Uhr?"}]
    result, tokens = await rewrite_query(history, "und beim Verkauf?", client=FakeClient())
    assert result == "Wie verkaufe ich eine Uhr auf Chrono24?"
    assert tokens == 95


async def test_rewrite_query_translates_english_first_question():
    result, tokens = await rewrite_query([], "How do I sell a watch on Chrono24?", client=FakeClient())
    assert result == "Wie verkaufe ich eine Uhr auf Chrono24?"
    assert tokens == 95
