"""Parser: Roh-HTML der Hilfeseiten → strukturierte Korpus-Dokumente.

Echte Struktur von data/raw/info__faqs.htm.html (siehe Task-2-Report für
Details): keine simplen <h2>/<h3>-Frage-Antwort-Paare, sondern Akkordeons.
Jede Kategorie ist ein `div.js-faq-chapter` mit einem `<h2>`-Titel. Darin
liegen entweder direkt Akkordeon-Items (`div.js-accordion-item`) oder,
gruppiert nach Unterkategorie, `div.js-sub-chapter`-Blöcke mit eigenem
`<h3>`-Titel. Jedes Akkordeon-Item enthält Frage
(`span.js-accordion-title`) und Antwort (`div.js-accordion-body`).
"""
from __future__ import annotations

from bs4 import BeautifulSoup

FAQ_CHAPTER_CLASS = "js-faq-chapter"
SUB_CHAPTER_CLASS = "js-sub-chapter"
ACCORDION_ITEM_CLASS = "js-accordion-item"
ACCORDION_TITLE_CLASS = "js-accordion-title"
ACCORDION_BODY_CLASS = "js-accordion-body"


class CorpusValidationError(Exception):
    """Korpus unvollständig oder strukturell kaputt — laut scheitern statt leise."""


def _category_for(item, chapter_name: str) -> str:
    sub_chapter = item.find_parent("div", class_=SUB_CHAPTER_CLASS)
    if sub_chapter is None:
        return chapter_name
    h3 = sub_chapter.find("h3")
    sub_name = h3.get_text(strip=True) if h3 else ""
    if not sub_name:
        return chapter_name
    return f"{chapter_name} – {sub_name}"


def parse_faq_page(html: str, url: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    docs: list[dict] = []

    for chapter in soup.find_all("div", class_=FAQ_CHAPTER_CLASS):
        h2 = chapter.find("h2")
        chapter_name = h2.get_text(strip=True) if h2 else ""

        for item in chapter.find_all("div", class_=ACCORDION_ITEM_CLASS):
            title_el = item.find("span", class_=ACCORDION_TITLE_CLASS)
            body_el = item.find("div", class_=ACCORDION_BODY_CLASS)
            if title_el is None or body_el is None:
                continue
            question = title_el.get_text(" ", strip=True)
            answer = body_el.get_text(" ", strip=True)
            if not question or not answer:
                continue
            docs.append(
                {
                    "id": f"faq-{len(docs) + 1:04d}",
                    "type": "faq",
                    "question": question,
                    "answer": answer,
                    "category": _category_for(item, chapter_name),
                    "url": url,
                }
            )

    if not docs:
        raise CorpusValidationError(
            "Keine FAQ-Einträge gefunden — Selektoren pruefen (js-faq-chapter/"
            "js-accordion-item), Seitenstruktur koennte sich geaendert haben."
        )

    return docs
