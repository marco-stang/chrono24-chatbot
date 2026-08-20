from pathlib import Path

from pipeline.parse import MAX_CHUNK_WORDS, parse_info_page

FIXTURE = Path("tests/fixtures/info_sample.html").read_text(encoding="utf-8")
URL = "https://www.chrono24.de/info/buyer-protection.htm"


def test_parse_info_page_builds_chunks():
    docs = parse_info_page(FIXTURE, URL)
    assert len(docs) >= 2
    for doc in docs:
        assert doc["type"] == "page_chunk"
        assert doc["title"].strip()
        assert doc["heading"].strip()
        assert doc["text"].strip()
        assert doc["url"] == URL


def test_long_sections_are_split():
    docs = parse_info_page(FIXTURE, URL)
    for doc in docs:
        assert len(doc["text"].split()) <= MAX_CHUNK_WORDS
