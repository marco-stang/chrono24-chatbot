"""LLM-as-Judge: bewertet Antwortqualität (Treue, Beantwortungsgrad) der Live-Pipeline."""
from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path

from app.config import settings
from app.llm import build_context, rewrite_query, stream_answer

logger = logging.getLogger("chrono24-chatbot.judge")

QUESTIONS_PATH = Path("eval/questions.json")
RESULTS_PATH = Path("eval/judge_results.json")
NOT_FOUND_ANSWER = "Dazu finde ich nichts in den Chrono24-Hilfeseiten."

MAX_JUDGE_TOKENS = 300

# Sechs Laeufe ueber dieselben 33 Fragen am 21./23.08.2026 ergaben 100 %, 94 %,
# 88 %, 94 %, 91 %, 94 % -- die letzten drei mit temperature=0 auf Judge und Bot.
# Die Pipeline war dabei unveraendert; die Spanne ist reines Sampling-Rauschen.
# temperature=0 senkt es, beseitigt es aber nicht: die API garantiert keine
# Determinismus, und bei 33 Fragen sind das schon 3 Prozentpunkte pro Frage.
# Die Schwelle liegt deshalb zwei Fragen unter dem gemessenen Boden (29/33):
# 27/33 = 82 %. Ein Gate bei 90 % war auf diesem Sample nachweislich flaky --
# ein Lauf von sechs faellt darunter, ohne dass sich am Code etwas geaendert hat.
MIN_FAITHFUL_RATE = 0.82

JUDGE_SYSTEM = (
    "Du bist ein strenger Prüfer für die Antworten eines RAG-Chatbots zu "
    "Chrono24-Hilfeseiten. Du bekommst eine Frage, den Kontext, den der Chatbot "
    "gesehen hat, und die Antwort des Chatbots. Bewerte ausschließlich anhand "
    "des gegebenen Kontexts, nicht anhand deines eigenen Wissens.\n\n"
    "Bewerte zwei Dimensionen:\n"
    "1. faithful (true/false): Sind ALLE Tatsachenaussagen der Antwort durch den "
    "Kontext gedeckt? Schon eine einzige nicht belegte oder erfundene Aussage "
    "macht die Antwort nicht treu (false).\n"
    "2. answered: einer der Werte \"voll\", \"teilweise\", \"nein\", \"verweigert\".\n"
    "   - \"voll\": die Frage ist vollständig beantwortet.\n"
    "   - \"teilweise\": ein Teil der Frage ist beantwortet, ein Teil fehlt.\n"
    "   - \"nein\": die Antwort verfehlt die Frage, obwohl der Kontext eine "
    "Antwort hergegeben hätte.\n"
    "   - \"verweigert\": die Antwort ist die ehrliche Absage 'Dazu finde ich "
    "nichts in den Chrono24-Hilfeseiten.' (oder sinngemäß), weil der Kontext "
    "keine Antwort hergibt.\n\n"
    "Antworte NUR mit einem JSON-Objekt, ohne Fließtext davor oder danach, "
    "in genau dieser Form:\n"
    '{"faithful": true oder false, "answered": "voll"|"teilweise"|"nein"|"verweigert", '
    '"begruendung": "ein Satz auf Deutsch"}'
)

_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL)


def build_judge_prompt(question: str, context: str, answer: str) -> str:
    return (
        f"Frage:\n{question}\n\n"
        f"Kontext (was der Chatbot gesehen hat):\n{context}\n\n"
        f"Antwort des Chatbots:\n{answer}"
    )


def parse_verdict(text: str) -> dict:
    stripped = text.strip()
    match = _FENCE_RE.match(stripped)
    candidate = match.group(1).strip() if match else stripped
    try:
        parsed = json.loads(candidate)
        if not isinstance(parsed, dict):
            raise TypeError("kein JSON-Objekt")
        return parsed
    except (json.JSONDecodeError, TypeError):
        return {"faithful": None, "answered": "parse_error", "begruendung": text[:200]}


async def _default_answer_fn(question: str, docs: list, history: list, client) -> str:
    parts = []
    async for event in stream_answer(question, docs, history, client):
        if event["type"] == "token":
            parts.append(event["text"])
    return "".join(parts)


async def judge_one(question: str, retriever, client, answer_fn=None) -> dict:
    answer_fn = answer_fn or _default_answer_fn
    standalone, _ = await rewrite_query([], question, client)
    docs = retriever.retrieve(standalone, top_k=5)

    if not docs:
        answer = NOT_FOUND_ANSWER
        context = ""
    else:
        answer = await answer_fn(standalone, docs, [], client)
        context = build_context(docs)

    judge_prompt = build_judge_prompt(question, context, answer)
    response = await client.messages.create(
        model=settings.model,
        max_tokens=MAX_JUDGE_TOKENS,
        system=JUDGE_SYSTEM,
        # Der Judge ist ein Messinstrument. Ohne diese Zeile laeuft er auf dem
        # API-Default 1.0 und wuerfelt: drei Laeufe ueber dieselbe unveraenderte
        # Pipeline ergaben am 23.08.2026 100 %, 94 % und 88 % Faithful-Rate.
        temperature=0,
        messages=[{"role": "user", "content": judge_prompt}],
    )
    judge_text = next((b.text for b in response.content if b.type == "text"), "")
    verdict = parse_verdict(judge_text)

    return {"question": question, "answer": answer, "verdict": verdict}


def aggregate(results: list[dict]) -> dict:
    faithful_values = [
        r["verdict"].get("faithful")
        for r in results
        if r["verdict"].get("faithful") is not None
    ]
    faithful_rate = (
        sum(1 for v in faithful_values if v is True) / len(faithful_values)
        if faithful_values
        else 0.0
    )

    answered_counts: dict[str, int] = {}
    for r in results:
        category = r["verdict"].get("answered", "unknown")
        answered_counts[category] = answered_counts.get(category, 0) + 1

    return {"n": len(results), "faithful_rate": faithful_rate, "answered_counts": answered_counts}


def check_gate(summary: dict) -> list[str]:
    """Prüft die Faithful-Rate gegen die Mindestschwelle. Leer = Gate bestanden."""
    failures = []
    if summary["faithful_rate"] < MIN_FAITHFUL_RATE:
        failures.append(
            f"Faithful-Rate {summary['faithful_rate']:.0%} unter Minimum {MIN_FAITHFUL_RATE:.0%}"
        )
    return failures


async def _run_all(questions: list[dict], retriever, client) -> list[dict]:
    results = []
    for item in questions:
        question = item["question"]
        try:
            result = await judge_one(question, retriever, client)
        except Exception:  # Fehlertoleranz: ein Fehler pro Frage stoppt den Lauf nicht.
            logger.exception("Judge-Lauf für Frage fehlgeschlagen: %r", question)
            result = {
                "question": question,
                "answer": "",
                "verdict": {"faithful": None, "answered": "error", "begruendung": ""},
            }
        results.append(result)
    return results


def _print_report(results: list[dict], summary: dict) -> None:
    print(f"\nLLM-as-Judge-Ergebnis ({summary['n']} Fragen)")
    print(f"Faithful-Rate: {summary['faithful_rate']:.0%}")
    print("Beantwortungsgrad:")
    for category, count in sorted(summary["answered_counts"].items()):
        print(f"  {category}: {count}")

    unfaithful = [r for r in results if r["verdict"].get("faithful") is False]
    if unfaithful:
        print(f"\nNicht-treue Antworten ({len(unfaithful)}):")
        for r in unfaithful:
            begruendung = r["verdict"].get("begruendung", "")
            print(f"  Q: {r['question']!r}")
            print(f"  A: {r['answer']!r}")
            print(f"  Begründung: {begruendung}")
            print()


def main() -> None:
    import sys

    from app.llm import get_client
    from app.retrieval import Retriever

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    retriever = Retriever(settings.index_dir, settings.corpus_path)
    client = get_client()

    results = asyncio.run(_run_all(questions, retriever, client))
    summary = aggregate(results)
    _print_report(results, summary)

    RESULTS_PATH.write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nRohergebnisse geschrieben nach {RESULTS_PATH}")

    if "--gate" in sys.argv:
        failures = check_gate(summary)
        for failure in failures:
            print(f"GATE FAIL: {failure}")
        sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
