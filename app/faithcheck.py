"""Deterministischer Faithfulness-Check: Token-Overlap zwischen Antwortsätzen
und den zitierten Quellen. Portiert aus dem Handover Brief Generator
(src/validate.py) — gleiche Tokenisierung, gleiche 0.5-Schwelle, statt
L-Zeilen-IDs werden hier [n]-Zitatmarker gefiltert."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Collection

PASS_THRESHOLD = 0.5
# Sätze mit weniger Wörtern gelten als Floskel und werden nicht bewertet.
MIN_CONTENT_WORDS = 4

_CITATION_RE = re.compile(r"\[(\d+)\]")
# Mehrbuchstabige Abkürzungen (kleingeschrieben, Vergleich case-insensitiv),
# nach denen ein Punkt kein Satzende ist.
_ABBREVIATIONS = ("bzw.", "ggf.", "ca.", "inkl.", "nr.", "usw.", "evtl.", "vgl.")
# Einzelbuchstabe + Punkt am Teilende ("z.", "B.", "d.", "h.") — Teil einer
# Abkürzung wie "z. B." oder "d. h.", nie ein echtes Satzende.
_SINGLE_LETTER_END = re.compile(r"(?:^|\s)[a-zäöüA-ZÄÖÜ]\.$")
# Satzgrenze: Punkt/Frage/Ausruf + Whitespace, aber nicht vor einem Zitatmarker.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?!\[)")
# Nachgestellte Zitate abtrennen: "] " gefolgt von Großbuchstabe = neuer Satz.
_AFTER_CITATION_SPLIT = re.compile(r"(?<=\])\s+(?=[A-ZÄÖÜ])")


@dataclass(frozen=True)
class SentenceCheck:
    text: str
    status: str  # "PASS" | "WEAK" | "FAIL"
    score: float
    sources: list[int] | list[str]


def _tokenize(text: str) -> set[str]:
    text = _CITATION_RE.sub(" ", text)
    tokens = set(re.findall(r"[a-zäöüß0-9]+", text.lower()))
    # Zeilen-IDs (m01) analog zu L-IDs im Schwesterprojekt herausfiltern.
    return {t for t in tokens if not re.fullmatch(r"m\d+", t)}


def score_overlap(claim_text: str, source_text: str) -> float:
    """Anteil der Claim-Tokens, die auch im zitierten Quelltext vorkommen."""
    claim_tokens = _tokenize(claim_text)
    if not claim_tokens:
        return 0.0
    return len(claim_tokens & _tokenize(source_text)) / len(claim_tokens)


def _ends_with_abbreviation(part: str) -> bool:
    # lower(): Abkürzungen am Satzanfang sind großgeschrieben ("Ca. 500 Euro").
    return part.lower().endswith(_ABBREVIATIONS) or bool(_SINGLE_LETTER_END.search(part))


def split_sentences(text: str) -> list[str]:
    """Zerlegt eine Bot-Antwort (Markdown-Subset) in Sätze.

    Markdown-Marker werden gestrippt, nachgestellte [n]-Zitate bleiben beim
    vorangehenden Satz, deutsche Abkürzungen erzeugen kein Satzende."""
    sentences: list[str] = []
    for line in text.splitlines():
        line = re.sub(r"^\s*[-*]\s+", "", line.strip()).replace("**", "")
        if not line:
            continue
        parts: list[str] = []
        for chunk in _SENTENCE_SPLIT.split(line):
            parts.extend(p for p in _AFTER_CITATION_SPLIT.split(chunk) if p)
        merged: list[str] = []
        for part in parts:
            if merged and _ends_with_abbreviation(merged[-1]):
                merged[-1] += " " + part
            else:
                merged.append(part)
        sentences.extend(merged)
    return sentences


def check_answer(answer: str, doc_texts: list[str],
                 skip: Collection[str] = ()) -> list[SentenceCheck]:
    """Prüft jeden Satz der Antwort deterministisch gegen die zitierten Quellen.

    Zitierter Satz → Overlap gegen die zitierten Dokumente (PASS/WEAK/FAIL).
    Satz ohne Zitat → WEAK. Sätze aus `skip` und Floskeln entfallen."""
    checks: list[SentenceCheck] = []
    for sentence in split_sentences(answer):
        if sentence in skip:
            continue
        cited = [int(n) for n in _CITATION_RE.findall(sentence)]
        if not cited and len(_tokenize(sentence)) < MIN_CONTENT_WORDS:
            continue
        if not cited:
            checks.append(SentenceCheck(sentence, "WEAK", 0.0, []))
            continue
        if any(n < 1 or n > len(doc_texts) for n in cited):
            checks.append(SentenceCheck(sentence, "FAIL", 0.0, cited))
            continue
        source_text = " ".join(doc_texts[n - 1] for n in cited)
        score = score_overlap(sentence, source_text)
        if score == 0.0:
            status = "FAIL"
        elif score < PASS_THRESHOLD:
            status = "WEAK"
        else:
            status = "PASS"
        checks.append(SentenceCheck(sentence, status, round(score, 3), cited))
    return checks


def validate_claims(claims: list[dict], lines_by_id: dict[str, str]) -> list[SentenceCheck]:
    """Prüft Briefing-Claims gegen die referenzierten Chat-Zeilen (Stufe B).

    Fehlende source_lines oder unbekannte Zeilen-IDs → FAIL, sonst
    Token-Overlap gegen den konkatenierten Zeilentext."""
    checks: list[SentenceCheck] = []
    for claim in claims:
        text = claim["text"]
        source_ids = claim.get("source_lines") or []
        if not source_ids or any(sid not in lines_by_id for sid in source_ids):
            checks.append(SentenceCheck(text, "FAIL", 0.0, source_ids))
            continue
        source_text = " ".join(lines_by_id[sid] for sid in source_ids)
        score = score_overlap(text, source_text)
        if score == 0.0:
            status = "FAIL"
        elif score < PASS_THRESHOLD:
            status = "WEAK"
        else:
            status = "PASS"
        checks.append(SentenceCheck(text, status, round(score, 3), source_ids))
    return checks
