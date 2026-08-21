import re

TOKEN_RE = re.compile(r"[a-zäöüß0-9]+")

# Kleine Stoppwort-Heuristik statt Sprachdetektions-Bibliothek: für die
# Entscheidung "Query vor BM25 ins Deutsche umformulieren?" reicht sie,
# und bei Gleichstand gewinnt Deutsch (kein unnötiger LLM-Call).
GERMAN_HINTS = frozenset(
    ["der", "die", "das", "und", "ich", "wie", "was", "ist", "eine", "einen", "nicht", "mit", "für", "auf", "bei", "kann", "muss", "wenn", "wird", "werden", "zu", "vom", "im", "meine", "mein", "ein", "dem", "den", "es", "sich", "als", "aus", "nach"]
)
ENGLISH_HINTS = frozenset(
    ["the", "what", "how", "are", "a", "an", "do", "does", "i", "my", "to", "of", "on", "for", "can", "it", "in", "with", "about", "their", "out", "is", "you", "your", "when", "where", "much", "sell", "selling", "buy"]
)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def looks_german(text: str) -> bool:
    if any(c in text.lower() for c in "äöüß"):
        return True
    tokens = set(tokenize(text))
    return len(tokens & GERMAN_HINTS) >= len(tokens & ENGLISH_HINTS)
