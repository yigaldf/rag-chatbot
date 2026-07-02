import ragchat.answering.generator as gen
from ragchat.common.models import Chunk


class _U:
    prompt_tokens = 200
    completion_tokens = 40


class _Msg:
    def __init__(self, c):
        self.content = c


class _Choice:
    def __init__(self, c):
        self.message = _Msg(c)


class _Chat:
    def __init__(self, c):
        self.choices = [_Choice(c)]
        self.usage = _U()


def test_answer_grounded(monkeypatch):
    monkeypatch.setattr(gen, "plan_queries", lambda q, meter: ["q1", "q2", "q3"])
    monkeypatch.setattr(gen, "retrieve", lambda queries, meter: [
        Chunk(text="t", title="EV/EBIT", source_url="u", doc_id="02.md", score=0.5)])

    class _Client:
        class chat:
            class completions:
                @staticmethod
                def create(model, messages):
                    return _Chat("grounded answer")

    monkeypatch.setattr(gen, "get_openai", lambda: _Client())
    a = gen.answer("compare things")
    assert a.found is True and a.text == "grounded answer"
    assert a.sources == ["EV/EBIT — u"]
    assert a.metrics.sub_queries == 3 and a.metrics.chunks_used == 1
    assert a.metrics.chat_tokens == 240  # 200 + 40


def test_answer_idk_when_no_chunks(monkeypatch):
    monkeypatch.setattr(gen, "plan_queries", lambda q, meter: ["q1"])
    monkeypatch.setattr(gen, "retrieve", lambda queries, meter: [])
    called = {"chat": False}

    class _Client:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    called["chat"] = True

    monkeypatch.setattr(gen, "get_openai", lambda: _Client())
    a = gen.answer("weather today?")
    assert a.found is False and a.metrics.chunks_used == 0
    assert called["chat"] is False  # gate short-circuits, LLM not called
