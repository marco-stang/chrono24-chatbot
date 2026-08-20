"""Parser: Roh-HTML der Hilfeseiten → strukturierte Korpus-Dokumente.

Echte Struktur von data/raw/info__faqs.htm.html (siehe Task-2-Report für
Details): keine simplen <h2>/<h3>-Frage-Antwort-Paare, sondern Akkordeons.
Jede Kategorie ist ein `div.js-faq-chapter` mit einem `<h2>`-Titel. Darin
liegen entweder direkt Akkordeon-Items (`div.js-accordion-item`) oder,
gruppiert nach Unterkategorie, `div.js-sub-chapter`-Blöcke mit eigenem
`<h3>`-Titel. Jedes Akkordeon-Item enthält Frage
(`span.js-accordion-title`) und Antwort (`div.js-accordion-body`).

Echte Struktur der Info-Unterseiten (siehe Task-3-Report für Details): Der
eigentliche Inhalt steckt in `<main id="main-content">` — alle Seiten haben
dieses Element. Wichtig: der Seitenrumpf (Footer-Widget mit "Newsletter",
"Einstellungen", "Kaufen auf Chrono24" usw.) besteht aus `<h2>`-Tags mit
Klassen wie `h4 m-b-3 m-t-0` und liegt AUSSERHALB von `#main-content` — ein
naiver `soup.find_all(["h2", "h3"])` über das ganze Dokument würde diesen
Navigations-Müll als Inhalts-Abschnitte einlesen. Innerhalb von
`#main-content` liegen `<h2>`/`<h3>` und ihr Fließtext als direkte
Geschwister im selben Elternelement (auch bei Tab-Widgets, z. B.
`div.js-tab-panel` auf der Sicherheitshinweise-Seite: jedes Tab-Panel ist
ein eigenständiger Container mit eigenen `<h2>`-Geschwistern). Manche
Abschnitte sind sehr lang (z. B. Datenschutzerklärung, > 8000 Wörter) und
müssen laut `MAX_CHUNK_WORDS` aufgeteilt werden.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

from bs4 import BeautifulSoup, Comment
from bs4.element import Tag

FAQ_CHAPTER_CLASS = "js-faq-chapter"
SUB_CHAPTER_CLASS = "js-sub-chapter"
ACCORDION_ITEM_CLASS = "js-accordion-item"
ACCORDION_TITLE_CLASS = "js-accordion-title"
ACCORDION_BODY_CLASS = "js-accordion-body"

MAIN_CONTENT_ID = "main-content"
HEADING_NAMES = ("h1", "h2", "h3")
CHUNK_HEADING_NAMES = ("h2", "h3")

MIN_DOCS = 30
MAX_CHUNK_WORDS = 600

REQUIRED_FIELDS = {
    "faq": ("id", "question", "answer", "category", "url"),
    "page_chunk": ("id", "title", "heading", "text", "url"),
}


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


def _slug(url: str) -> str:
    return urlparse(url).path.strip("/").replace("/", "-").removesuffix(".htm")


def _text_until_next_heading(tag: Tag) -> str:
    """Sammelt Fließtext ab `tag` bis zum nächsten h1/h2/h3-Geschwister.

    Auf den Info-Seiten liegen Überschrift und zugehöriger Text als direkte
    Geschwister im selben Elternelement (auch innerhalb von Tab-Panels), daher
    reicht ein einfacher Geschwister-Walk ohne Rekursion in Kind-Elemente.
    """
    parts: list[str] = []
    for sibling in tag.next_siblings:
        if isinstance(sibling, Comment):
            continue
        if isinstance(sibling, Tag):
            if sibling.name in HEADING_NAMES:
                break
            text = sibling.get_text("\n", strip=True)
        else:
            text = str(sibling).strip()
        if text:
            parts.append(text)
    return "\n".join(parts)


def _hard_split_words(words: list[str], max_words: int) -> list[str]:
    """Teilt eine Wortliste hart in max_words-Wörter-Stücke.

    Greift, wenn ein einzelner Absatz für sich allein schon über dem Limit
    liegt (z. B. eine 700-Wörter-Zeile ohne Zeilenumbrüche) — sonst würde
    `_split_long_text` diesen Absatz unzerteilt zurückgeben und das
    MAX_CHUNK_WORDS-Limit verletzen.
    """
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


def _split_long_text(text: str, max_words: int = MAX_CHUNK_WORDS) -> list[str]:
    if len(text.split()) <= max_words:
        return [text]
    parts: list[str] = []
    current: list[str] = []
    count = 0
    for paragraph in text.split("\n"):
        words = paragraph.split()
        if len(words) > max_words:
            if current:
                parts.append("\n".join(current))
                current, count = [], 0
            parts.extend(_hard_split_words(words, max_words))
            continue
        if current and count + len(words) > max_words:
            parts.append("\n".join(current))
            current, count = [], 0
        current.append(paragraph)
        count += len(words)
    if current:
        parts.append("\n".join(current))
    return parts


def parse_info_page(html: str, url: str) -> list[dict]:
    """Zerlegt eine Info-Unterseite in `page_chunk`-Dokumente je h2/h3-Abschnitt.

    Sucht Überschriften ausschließlich innerhalb von `#main-content`, damit
    die Footer-Navigation (eigene h2-Tags außerhalb davon) nicht als Inhalt
    landet. Leere Abschnitte (z. B. eine reine Zwischenüberschrift ohne
    eigenen Text vor der nächsten Unterüberschrift) werden übersprungen.
    """
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else url
    main = soup.find(id=MAIN_CONTENT_ID)
    if main is None:
        return []
    docs: list[dict] = []
    for tag in main.find_all(CHUNK_HEADING_NAMES):
        heading = tag.get_text(strip=True)
        text = _text_until_next_heading(tag)
        if not heading or not text:
            continue
        for part in _split_long_text(text):
            docs.append(
                {
                    "id": f"{_slug(url)}-{len(docs) + 1:04d}",
                    "type": "page_chunk",
                    "title": title,
                    "heading": heading,
                    "text": part,
                    "url": url,
                }
            )
    return docs


def build_corpus(raw_dir: Path) -> dict:
    from pipeline.scrape import START_URL, filename_to_url, url_to_filename

    faq_name = url_to_filename(START_URL)
    docs = parse_faq_page((raw_dir / faq_name).read_text(encoding="utf-8"), START_URL)
    for path in sorted(raw_dir.glob("*.html")):
        if path.name == faq_name:
            continue
        docs = docs + parse_info_page(path.read_text(encoding="utf-8"), filename_to_url(path.name))
    scraped_at = datetime.now(tz=UTC).date().isoformat()
    return {"scraped_at": scraped_at, "documents": docs}


def validate_corpus(corpus: dict) -> None:
    docs = corpus.get("documents", [])
    if len(docs) < MIN_DOCS:
        raise CorpusValidationError(f"nur {len(docs)} Dokumente, erwartet mindestens {MIN_DOCS}")
    seen: set[str] = set()
    for doc in docs:
        fields = REQUIRED_FIELDS.get(doc.get("type", ""))
        if fields is None:
            raise CorpusValidationError(f"unbekannter Typ: {doc.get('type')!r} (id={doc.get('id')!r})")
        for field in fields:
            if not doc.get(field):
                raise CorpusValidationError(f"Dokument {doc.get('id')!r}: Feld {field!r} fehlt oder leer")
        if doc["id"] in seen:
            raise CorpusValidationError(f"doppelte id {doc['id']!r}")
        seen.add(doc["id"])


if __name__ == "__main__":
    corpus = build_corpus(Path("data/raw"))
    validate_corpus(corpus)
    out = Path("data/corpus.json")
    out.write_text(json.dumps(corpus, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(corpus['documents'])} Dokumente nach {out} geschrieben")
