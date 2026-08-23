"""Query-Varianten: pro FAQ-Frage 3-5 Umformulierungen generieren (offline, einmalig).

Cached in data/variants.json -- wie der Scraper ein einmaliger, lokaler Lauf,
der Online-Service liest zur Laufzeit nie das LLM fuer diesen Zweck an.
"""
from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger("chrono24-chatbot.variants")

VARIANTS_SYSTEM = (
    "Du erhältst eine FAQ-Frage von Chrono24. Formuliere sie auf 3 bis 5 "
    "verschiedene Arten um, wie eine Nutzerin oder ein Nutzer sie im Chat "
    "stellen könnte -- umgangssprachlicher, mit anderen Wortformen, teils "
    "kürzer. Die Bedeutung darf sich nicht ändern. Antworte NUR mit einem "
    "JSON-Array von Strings, ohne Erklärung."
)

MAX_VARIANT_TOKENS = 300

_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL)


def parse_variants(text: str) -> list[str]:
    stripped = text.strip()
    match = _FENCE_RE.match(stripped)
    candidate = match.group(1).strip() if match else stripped
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [v.strip() for v in parsed if isinstance(v, str) and v.strip()]


async def generate_variants(question: str, client, model: str) -> list[str]:
    response = await client.messages.create(
        model=model,
        max_tokens=MAX_VARIANT_TOKENS,
        system=VARIANTS_SYSTEM,
        messages=[{"role": "user", "content": question}],
    )
    text = next((b.text for b in response.content if b.type == "text"), "")
    return parse_variants(text)


async def build_variants(faq_docs: list[dict], client, model: str) -> dict[str, list[str]]:
    """Ein LLM-Call pro FAQ; ein Fehler pro Frage stoppt den Gesamtlauf nicht
    (analog eval/judge.py::_run_all). FAQs ohne Varianten fehlen im Ergebnis."""
    result: dict[str, list[str]] = {}
    for doc in faq_docs:
        try:
            variants = await generate_variants(doc["question"], client, model)
        except Exception:
            logger.exception("Varianten-Generierung fehlgeschlagen fuer %s", doc["id"])
            variants = []
        if variants:
            result[doc["id"]] = variants
    return result


def main() -> None:
    import asyncio

    from app.config import settings
    from app.llm import get_client

    corpus = json.loads(settings.corpus_path.read_text(encoding="utf-8"))
    faq_docs = [d for d in corpus["documents"] if d["type"] == "faq"]
    client = get_client()

    variants = asyncio.run(build_variants(faq_docs, client, settings.model))

    settings.variants_path.write_text(
        json.dumps(variants, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    total = sum(len(v) for v in variants.values())
    print(
        f"{total} Varianten für {len(variants)}/{len(faq_docs)} FAQs nach "
        f"{settings.variants_path} geschrieben"
    )


if __name__ == "__main__":
    main()
