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


# Handkuratierte Synonym-Expansion für den BM25-Pfad — wie ein
# Elasticsearch-Synonym-Filter, bewusst klein gehalten. Schließt die
# Alltagswort-Lücke zwischen Nutzerfragen ("bezahlen", "zurückschicken")
# und FAQ-Titeln ("Was kostet …", "Rückgabebedingungen"). Gemessen:
# Tuning-Set 88 % → 91 %, Held-out unverändert (siehe README).
QUERY_SYNONYMS: dict[str, tuple[str, ...]] = {
    "bezahlen": ("kosten", "kostet"),
    "gebühr": ("kosten", "kostet"),
    "gebühren": ("kosten", "kostet"),
    "preis": ("kosten", "kostet"),
    "kostet": ("gebühr",),
    "kosten": ("gebühr",),
    "zurückschicken": ("rückgabe", "zurückgeben"),
    "zurücksenden": ("rückgabe", "zurückgeben"),
    "retoure": ("rückgabe",),
}


def expand_query(text: str) -> list[str]:
    """Query-Tokens plus Synonyme (nur für BM25 — Embeddings bleiben roh)."""
    tokens = tokenize(text)
    extra = [syn for t in tokens for syn in QUERY_SYNONYMS.get(t, ())]
    return tokens + extra


def looks_german(text: str) -> bool:
    if any(c in text.lower() for c in "äöüß"):
        return True
    tokens = set(tokenize(text))
    return len(tokens & GERMAN_HINTS) >= len(tokens & ENGLISH_HINTS)
