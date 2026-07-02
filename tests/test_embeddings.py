import ragchat.retrieval.embeddings as emb
from ragchat.metrics.models import TokenMeter


class _Resp:
    class _D:
        def __init__(self, v):
            self.embedding = v

    class _U:
        total_tokens = 7

    def __init__(self):
        self.data = [self._D([1.0, 2.0])]
        self.usage = self._U()


def test_embed_records_tokens(monkeypatch):
    class _Client:
        class embeddings:
            @staticmethod
            def create(model, input):
                return _Resp()

    monkeypatch.setattr(emb, "get_openai", lambda: _Client())
    meter = TokenMeter()
    vecs = emb.embed(["hello"], meter)
    assert vecs == [[1.0, 2.0]]
    assert meter.embed == 7
