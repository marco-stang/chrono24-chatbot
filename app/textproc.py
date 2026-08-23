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


# Wortstämme statt exakter Formen, weil beide Rollen zusammengesetzte Wörter
# bilden ("Privatverkäufer", "Käuferschutz"), die als ganzes Token nie exakt
# in einer Wortliste stünden. Kurze Stämme fangen sie per Teilstring-Test.
# ACHTUNG geteiltes Vokabular: "kauf"/"käuf" stecken als Teilstring in
# "verkauf"/"verkäuf" ("ver" + "kaufen" ist ein trennbares Verb) -- deshalb
# prüft classify_audience() den Verkäufer-Stamm zuerst und zählt ein Token
# nur einmal, sonst würde jedes "verkaufen" fälschlich auch als
# Käufer-Treffer zählen.
BUYER_HINTS = frozenset(["kauf", "käuf", "erwerb"])
SELLER_HINTS = frozenset(["verkauf", "verkäuf", "anbiet", "inserier"])


def classify_audience(text: str) -> str:
    """Käufer/Verkäufer/neutral per Wortstamm-Mehrheit, kein LLM-Call.

    Reine Keyword-Heuristik wie looks_german() oben: pro Token erst auf den
    Verkäufer-Stamm prüfen (siehe Kommentar an SELLER_HINTS), sonst auf den
    Käufer-Stamm. Nur bei eindeutigem Übergewicht eines Wortfelds wird eine
    Rolle zurückgegeben -- bei Gleichstand oder ganz ohne Treffer "neutral",
    denn ein falsch klassifiziertes "neutral" filtert nichts weg, ein falsch
    klassifiziertes "kaeufer"/"verkaeufer" kann dagegen einen echten Treffer
    aus der Kandidatenmenge werfen (harter Pre-Filter in Retriever.retrieve).
    """
    buyer_hits = 0
    seller_hits = 0
    for token in tokenize(text):
        if any(stem in token for stem in SELLER_HINTS):
            seller_hits += 1
        elif any(stem in token for stem in BUYER_HINTS):
            buyer_hits += 1
    if buyer_hits > seller_hits:
        return "kaeufer"
    if seller_hits > buyer_hits:
        return "verkaeufer"
    return "neutral"
