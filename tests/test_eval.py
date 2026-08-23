from pathlib import Path

from app.retrieval import RetrievedDoc
from eval.run_eval import QUESTIONS_PATH, _questions_path, abstention_rate, hit_rate_at_k


class StubRetriever:
    def __init__(self, mapping):
        self.mapping = mapping

    def retrieve(self, query, top_k=5):
        return [RetrievedDoc(id=i, type="faq", title=f"Titel {i}", url="u", text="x", score=0.1)
                for i in self.mapping.get(query, [])]


def test_hit_rate_counts_hits_in_top_k():
    retriever = StubRetriever({
        "F1": ["faq-0001", "faq-0002"],
        "F2": ["faq-0009"],
    })
    questions = [{"question": "F1", "expected_doc_id": "faq-0002"},
                 {"question": "F2", "expected_doc_id": "faq-0003"}]
    rate, misses = hit_rate_at_k(retriever, questions, k=5)
    assert rate == 0.5
    assert misses[0]["question"] == "F2"
    assert misses[0]["got"] == ["faq-0009"]


def test_abstention_rate_counts_questions_with_no_hits():
    retriever = StubRetriever({
        "off1": [],
        "off2": ["faq-0005"],
    })
    questions = [{"question": "off1"}, {"question": "off2"}]
    rate, false_hits = abstention_rate(retriever, questions)
    assert rate == 0.5
    assert false_hits[0]["question"] == "off2"
    assert false_hits[0]["got_id"] == "faq-0005"
    assert false_hits[0]["got_title"] == "Titel faq-0005"


def test_abstention_rate_is_one_when_all_questions_abstain():
    retriever = StubRetriever({"off1": [], "off2": []})
    questions = [{"question": "off1"}, {"question": "off2"}]
    rate, false_hits = abstention_rate(retriever, questions)
    assert rate == 1.0
    assert false_hits == []


def test_questions_path_defaults_to_tuning_set():
    assert _questions_path([]) == QUESTIONS_PATH


def test_questions_path_uses_custom_flag():
    assert _questions_path(["--questions", "eval/questions_holdout.json"]) == \
        Path("eval/questions_holdout.json")
