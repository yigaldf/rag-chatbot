import ragchat.retrieval.retriever as retr
from ragchat.metrics.models import TokenMeter


class _Hit:
    def __init__(self, id, score, doc_id, text):
        self.id = id
        self.score = score
        self.payload = {"text": text, "title": doc_id, "source_url": "u", "doc_id": doc_id}


class _Pts:
    def __init__(self, pts):
        self.points = pts


def test_retrieve_unions_dedups_and_caps_per_doc(monkeypatch):
    monkeypatch.setattr(retr, "embed", lambda queries, meter=None: [[0.0]] * len(queries))
    hits = [_Hit(f"a{i}", 0.9 - i * 0.01, "A", f"a{i}") for i in range(4)] + [_Hit("b0", 0.5, "B", "b0")]

    class _Q:
        def query_points(self, collection_name, query, limit):
            return _Pts(hits)

    monkeypatch.setattr(retr, "get_qdrant", lambda: _Q())
    monkeypatch.setattr(retr.settings, "score_threshold", 0.3)
    out = retr.retrieve(["q"], TokenMeter(), per_query_k=10, per_doc=3, max_chunks=12)
    doc_ids = [c.doc_id for c in out]
    assert doc_ids.count("A") == 3  # capped
    assert doc_ids.count("B") == 1
    assert out == sorted(out, key=lambda c: c.score, reverse=True)


def test_retrieve_drops_below_threshold(monkeypatch):
    monkeypatch.setattr(retr, "embed", lambda queries, meter=None: [[0.0]])
    hits = [_Hit("x", 0.1, "X", "x")]

    class _Q:
        def query_points(self, **kw):
            return _Pts(hits)

    monkeypatch.setattr(retr, "get_qdrant", lambda: _Q())
    monkeypatch.setattr(retr.settings, "score_threshold", 0.3)
    assert retr.retrieve(["q"], TokenMeter()) == []
