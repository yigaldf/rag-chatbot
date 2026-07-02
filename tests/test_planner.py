import ragchat.retrieval.planner as planner
from ragchat.retrieval.planner import QueryPlan
from ragchat.metrics.models import TokenMeter


class _U:
    prompt_tokens = 50
    completion_tokens = 12


class _Msg:
    def __init__(self, plan):
        self.parsed = plan


class _Choice:
    def __init__(self, plan):
        self.message = _Msg(plan)


class _Resp:
    def __init__(self, plan):
        self.choices = [_Choice(plan)]
        self.usage = _U()


def test_plan_queries_returns_parsed_and_records_tokens(monkeypatch):
    class _Client:
        class beta:
            class chat:
                class completions:
                    @staticmethod
                    def parse(model, temperature, response_format, messages):
                        return _Resp(QueryPlan(queries=["q1", "q2"]))

    monkeypatch.setattr(planner, "get_openai", lambda: _Client())
    meter = TokenMeter()
    qs = planner.plan_queries("compare a and b", meter)
    assert qs == ["q1", "q2"]
    assert meter.chat_in == 50 and meter.chat_out == 12


def test_plan_queries_falls_back_on_error(monkeypatch):
    class _Client:
        class beta:
            class chat:
                class completions:
                    @staticmethod
                    def parse(**kw):
                        raise RuntimeError("boom")

    monkeypatch.setattr(planner, "get_openai", lambda: _Client())
    qs = planner.plan_queries("what is x?", TokenMeter())
    assert qs == ["what is x?"]
