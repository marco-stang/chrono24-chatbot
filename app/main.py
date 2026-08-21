"""FastAPI-Service: Chat-Endpoint mit SSE, Healthcheck, statisches Frontend."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app import faithcheck, llm
from app.config import settings
from app.guards import TokenBudget
from app.retrieval import Retriever

logger = logging.getLogger("chrono24-chatbot")

MAX_QUESTION_CHARS = 500
MAX_MESSAGE_CHARS = 4000
MAX_HISTORY_MESSAGES = 20
HISTORY_TURNS_FOR_LLM = 6
NOT_FOUND_ANSWER = "Dazu finde ich nichts in den Chrono24-Hilfeseiten."


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=MAX_HISTORY_MESSAGES)

    @field_validator("messages")
    @classmethod
    def last_message_must_be_user(cls, v: list[ChatMessage]) -> list[ChatMessage]:
        if v[-1].role != "user":
            raise ValueError("letzte Nachricht muss vom Nutzer sein")
        if len(v[-1].content) > MAX_QUESTION_CHARS:
            raise ValueError(f"Frage länger als {MAX_QUESTION_CHARS} Zeichen")
        return v


def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def create_app(retriever=None, budget=None, answer_fn=None, rewrite_fn=None,
               llm_client=None) -> FastAPI:
    app = FastAPI(title="Chrono24-FAQ-Chatbot")
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # Fail fast: ohne Index wirft Retriever() beim Start (statt leerer Antworten).
    app.state.retriever = retriever or Retriever(settings.index_dir, settings.corpus_path)
    app.state.budget = budget or TokenBudget(settings.budget_db, settings.daily_token_budget)

    if llm_client is None and not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY ist nicht gesetzt — Service startet nicht")

    app.state.answer_fn = answer_fn or llm.stream_answer
    app.state.rewrite_fn = rewrite_fn or llm.rewrite_query
    app.state.llm_client = llm_client

    @app.get("/api/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.post("/api/chat")
    @limiter.limit("10/minute;50/day")
    async def chat(request: Request, body: ChatRequest) -> StreamingResponse:
        if app.state.budget.remaining() <= 0:
            raise HTTPException(status_code=429, detail="Demo-Budget für heute erschöpft")

        history = [m.model_dump() for m in body.messages[:-1]][-HISTORY_TURNS_FOR_LLM * 2:]
        question = body.messages[-1].content

        async def event_stream():
            try:
                client = app.state.llm_client or llm.get_client()
                standalone, rewrite_tokens = await app.state.rewrite_fn(history, question, client)
                if rewrite_tokens:
                    app.state.budget.spend(rewrite_tokens)
                docs = app.state.retriever.retrieve(standalone)
                yield sse({"type": "retrieval",
                           "docs": [{"id": d.id, "title": d.title, "score": d.score,
                                     "rerank": d.rerank_score}
                                    for d in docs]})
                if not docs:
                    yield sse({"type": "token", "text": NOT_FOUND_ANSWER})
                    yield sse({"type": "done"})
                    return
                answer_text = ""
                async for event in app.state.answer_fn(standalone, docs, history, client):
                    if event["type"] == "usage":
                        app.state.budget.spend(event["input_tokens"] + event["output_tokens"])
                    else:
                        if event["type"] == "token":
                            answer_text += event["text"]
                        yield sse(event)
                yield sse({"type": "sources",
                           "items": [{"n": i, "title": d.title, "url": d.url}
                                     for i, d in enumerate(docs, 1)]})
                checks = faithcheck.check_answer(answer_text, [d.text for d in docs],
                                                 skip={NOT_FOUND_ANSWER})
                yield sse({"type": "validation",
                           "sentences": [{"text": c.text, "status": c.status,
                                          "score": c.score, "sources": c.sources}
                                         for c in checks]})
                yield sse({"type": "done"})
            except Exception:
                logger.exception("Chat-Anfrage fehlgeschlagen")
                yield sse({"type": "error",
                           "message": "Antwort gerade nicht möglich, versuch es gleich nochmal."})

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    static_dir = Path("static")
    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

    return app


# Beim Import durch uvicorn (Deployment) App sofort bauen; Tests nutzen create_app()
# direkt — der pytest-Guard verhindert, dass beim Test-Import der echte Index lädt.
app = create_app() if "pytest" not in sys.modules else None
