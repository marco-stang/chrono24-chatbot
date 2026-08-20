from pathlib import Path

from pipeline.parse import parse_faq_page

FIXTURE = Path("tests/fixtures/faq_sample.html").read_text(encoding="utf-8")
URL = "https://www.chrono24.de/info/faqs.htm"


def test_parse_faq_extracts_qa_pairs():
    docs = parse_faq_page(FIXTURE, URL)
    assert len(docs) >= 2
    first = docs[0]
    assert first["type"] == "faq"
    assert first["id"] == "faq-0001"
    assert first["url"] == URL
    assert first["question"].strip()
    assert first["answer"].strip()
    assert first["category"].strip()


def test_parse_faq_ids_are_unique_and_sequential():
    docs = parse_faq_page(FIXTURE, URL)
    ids = [d["id"] for d in docs]
    assert ids == [f"faq-{i:04d}" for i in range(1, len(ids) + 1)]
