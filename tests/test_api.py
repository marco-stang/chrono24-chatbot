import json

from fastapi.testclient import TestClient

from app.guards import TokenBudget
from app.main import create_app
from app.retrieval import RetrievedDoc

DOC = RetrievedDoc(id="faq-0001", type="faq", title="Wie funktioniert der Käuferschutz?",
                   url="https://www.chrono24.de/info/faqs.htm",
                   text="Der Käuferschutz sichert deine Zahlung ab.", score=0.05)


class FakeRetriever:
    def __init__(self, docs):
        self.docs = docs

    def retrieve(self, query, top_k=5):
        return self.docs


async def fake_answer_fn(question, docs, history, client):
    yield {"type": "token", "text": "Der Käuferschutz "}
    yield {"type": "token", "text": "sichert deine Zahlung ab. [1]"}
    yield {"type": "usage", "input_tokens": 100, "output_tokens": 20}


async def fake_rewrite_fn(history, question, client):
    return question


def make_client(tmp_path, docs, budget=None):
    app = create_app(retriever=FakeRetriever(docs),
                     budget=budget or TokenBudget(tmp_path / "b.sqlite3", daily_limit=1000),
                     answer_fn=fake_answer_fn, rewrite_fn=fake_rewrite_fn,
                     llm_client=object())
    return TestClient(app)


def parse_sse(text):
    return [json.loads(line.removeprefix("data: "))
            for line in text.splitlines() if line.startswith("data: ")]


def test_health(tmp_path):
    response = make_client(tmp_path, [DOC]).get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_streams_tokens_sources_done(tmp_path):
    response = make_client(tmp_path, [DOC]).post("/api/chat", json={"messages": [
        {"role": "user", "content": "Wie funktioniert der Käuferschutz?"}]})
    assert response.status_code == 200
    events = parse_sse(response.text)
    types = [e["type"] for e in events]
    assert types == ["retrieval", "token", "token", "sources", "done"]
    sources = next(e for e in events if e["type"] == "sources")
    assert sources["items"][0]["url"] == "https://www.chrono24.de/info/faqs.htm"


def test_chat_offtopic_returns_notfound_message(tmp_path):
    response = make_client(tmp_path, []).post("/api/chat", json={"messages": [
        {"role": "user", "content": "Gedicht über Katzen"}]})
    events = parse_sse(response.text)
    assert any("finde ich nichts" in e.get("text", "") for e in events)
    assert events[-1]["type"] == "done"


def test_chat_spends_budget(tmp_path):
    budget = TokenBudget(tmp_path / "b3.sqlite3", daily_limit=1000)
    make_client(tmp_path, [DOC], budget=budget).post("/api/chat", json={"messages": [
        {"role": "user", "content": "Käuferschutz?"}]})
    assert budget.used_today() == 120


def test_chat_rejects_when_budget_empty(tmp_path):
    budget = TokenBudget(tmp_path / "b4.sqlite3", daily_limit=10)
    budget.spend(10)
    response = make_client(tmp_path, [DOC], budget=budget).post("/api/chat", json={
        "messages": [{"role": "user", "content": "Käuferschutz?"}]})
    assert response.status_code == 429
    assert "Demo-Budget" in response.json()["detail"]


def test_chat_validates_input(tmp_path):
    client = make_client(tmp_path, [DOC])
    assert client.post("/api/chat", json={"messages": []}).status_code == 422
    assert client.post("/api/chat", json={"messages": [
        {"role": "user", "content": "x" * 501}]}).status_code == 422
    too_many = [{"role": "user", "content": "hi"}] * 21
    assert client.post("/api/chat", json={"messages": too_many}).status_code == 422
    assert client.post("/api/chat", json={"messages": [
        {"role": "assistant", "content": "hi"}]}).status_code == 422
