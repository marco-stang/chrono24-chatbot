from app.retrieval import RetrievedDoc
from eval.run_eval import hit_rate_at_k


class StubRetriever:
    def __init__(self, mapping):
        self.mapping = mapping

    def retrieve(self, query, top_k=5):
        return [RetrievedDoc(id=i, type="faq", title="t", url="u", text="x", score=0.1)
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
