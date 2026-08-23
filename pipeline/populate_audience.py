"""Schreibt das audience-Feld (kaeufer|verkaeufer|neutral) in data/corpus.json.

Einmalig laufen lassen, wenn sich der Corpus ändert (neuer Scrape) oder die
Heuristik/Korrekturliste angepasst wird -- siehe
docs/superpowers/specs/2026-08-23-corpus-storage-rethink-design.md, Schritt 1.
Population per app.textproc.classify_audience() auf Frage+Antwort (FAQ) bzw.
Überschrift+Text (page_chunk).

MANUAL_CORRECTIONS unten (Prinzip wie QUERY_SYNONYMS in app/textproc.py):
die reine Wortstamm-Heuristik verzählt sich systematisch bei Texten, die die
*Gegenpartei* öfter nennen als die eigene Rolle -- z. B. sagt ein Verkäufer-
Text "teilen Sie dem Käufer die Tracking-ID mit" und nennt "Käufer" damit
öfter als irgendeinen Verkäufer-Stamm. Gefunden durch Stichprobe (alle 9
info-escrow-*-Chunks von Hand gelesen: die Seite hat einen Käufer- und einen
Verkäufer-Abschnitt) und durch einen automatisierten Abgleich Frage-Rolle vs.
Volltext-Rolle über den ganzen Corpus (3 Treffer: faq-0115, faq-0121,
faq-0123 -- alle mit "Verkäufer"/"Verkauf" in der Frage, aber mehr
"Käufer"-Nennungen in der Antwort).

Symmetrisches Gegenstück beim gezielten Nachmessen der beiden bekannten
Miss-Fälle gefunden: faq-0019 und faq-0027 sind laut README-Diagnose die
falschen Konkurrenz-Dokumente, die faq-0098 bzw. info-escrow-0007 in der
RRF-Fusion schlagen -- beide klar aus Käufersicht (Kategorie "Uhren kaufen
– Von Privatverkäufern kaufen" / "Uhren kaufen – Bezahlung"), aber von der
Heuristik als "verkaeufer" eingestuft, weil ihr eigentliches Thema die
Gegenseite ist ("Privatverkäufer"/"Verkäufer" kommt als Diskussions-
gegenstand öfter vor als das eigene "kaufen"). Ohne diese Korrektur bleibt
der harte Filter wirkungslos: er lässt die falschen Konkurrenten durch,
weil sie sich selbst als "verkaeufer" ausgeben.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.config import settings
from app.textproc import classify_audience

MANUAL_CORRECTIONS: dict[str, str] = {
    # info-escrow-*: Sicherheitshinweise-Seite mit klarem Käufer-Abschnitt
    # (0001-0003) und Verkäufer-Abschnitt (0004-0009) -- per Hand anhand
    # Überschrift+Inhalt zugeordnet, die Heuristik verwechselt beide, weil
    # der jeweils andere Part als Gegenpartei genannt wird.
    "info-escrow-0001": "kaeufer",
    "info-escrow-0002": "kaeufer",
    "info-escrow-0004": "verkaeufer",
    "info-escrow-0005": "verkaeufer",
    "info-escrow-0006": "verkaeufer",
    "info-escrow-0007": "verkaeufer",  # bekannter Miss-Fall, siehe README
    "info-escrow-0008": "verkaeufer",
    "info-escrow-0009": "verkaeufer",
    # FAQ-Fragen mit "Verkäufer"/"Verkauf" in der Frage, deren Antwort aber
    # den Käufer öfter nennt als die Heuristik verträgt.
    "faq-0115": "verkaeufer",
    "faq-0121": "verkaeufer",  # bekannter Miss-Fall (README-Ablationstabelle)
    "faq-0123": "verkaeufer",
    # Falsche Konkurrenz-Dokumente der beiden bekannten Miss-Faelle (README-
    # Diagnosetabelle "Platz 1 stattdessen") -- klar Kaeufersicht, aber von
    # der Heuristik als verkaeufer eingestuft (siehe Docstring oben).
    "faq-0019": "kaeufer",
    "faq-0027": "kaeufer",
}


def _doc_text(doc: dict) -> str:
    if doc["type"] == "faq":
        return f"{doc['question']}\n{doc['answer']}"
    return f"{doc.get('heading', '')}\n{doc.get('text', '')}"


def populate_audience(corpus_path: Path) -> dict[str, int]:
    data = json.loads(corpus_path.read_text(encoding="utf-8"))
    counts = {"kaeufer": 0, "verkaeufer": 0, "neutral": 0}
    for doc in data["documents"]:
        audience = MANUAL_CORRECTIONS.get(doc["id"]) or classify_audience(_doc_text(doc))
        doc["audience"] = audience
        counts[audience] += 1
    # indent=1, kein Trailing-Newline: exakt das bestehende Formatierung von
    # data/corpus.json, damit der Diff nur die neuen audience-Zeilen zeigt.
    corpus_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return counts


if __name__ == "__main__":
    result = populate_audience(settings.corpus_path)
    print(f"audience-Feld geschrieben nach {settings.corpus_path}: {result}")
