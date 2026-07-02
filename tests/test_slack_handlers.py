from ragchat.slack.handlers import handle_mention
from ragchat.common.models import Answer
from ragchat.metrics.models import QuestionMetrics, AggregateStats


class _Say:
    def __init__(self):
        self.calls = []

    def __call__(self, text, thread_ts=None):
        self.calls.append({"text": text, "thread_ts": thread_ts})


class _Registry:
    def __init__(self):
        self.recorded = []

    def record(self, m):
        self.recorded.append(m)

    def stats(self):
        return AggregateStats(num_questions=1, total_sub_queries=2, avg_sub_queries=2.0,
                              total_chunks=3, avg_chunks=3.0, total_embed_tokens=10,
                              total_chat_tokens=100, total_tokens=110, avg_tokens=110.0)


def _answer(text="A", sources=None, found=True):
    m = QuestionMetrics(sub_queries=2, chunks_used=3, embed_tokens=10, chat_tokens=100, total_tokens=110)
    return Answer(text=text, sources=sources or ["S — u"], found=found, metrics=m)


def _event(text, ts="111.1"):
    return {"text": text, "ts": ts, "user": "UUSER"}


def test_question_posts_answer_in_thread_and_records():
    say, reg = _Say(), _Registry()
    handle_mention(_event("<@BOT> what is EV/EBIT?"), say, "BOT",
                   question_fn=lambda q: _answer(), registry=reg)
    # an ack, then the answer — both threaded on the mention ts
    assert len(say.calls) == 2
    assert all(c["thread_ts"] == "111.1" for c in say.calls)
    assert "A" in say.calls[1]["text"]
    assert len(reg.recorded) == 1


def test_empty_question_prompts_and_does_not_call_answer():
    say, reg = _Say(), _Registry()
    called = {"n": 0}

    def qfn(q):
        called["n"] += 1
        return _answer()

    handle_mention(_event("<@BOT>   "), say, "BOT", question_fn=qfn, registry=reg)
    assert called["n"] == 0
    assert len(reg.recorded) == 0
    assert "ask me" in say.calls[-1]["text"].lower()


def test_stats_keyword_posts_aggregate_and_skips_answer():
    say, reg = _Say(), _Registry()
    called = {"n": 0}

    def qfn(q):
        called["n"] += 1
        return _answer()

    handle_mention(_event("<@BOT> stats"), say, "BOT", question_fn=qfn, registry=reg)
    assert called["n"] == 0
    assert "Aggregate metrics" in say.calls[-1]["text"]


def test_answer_error_posts_friendly_message():
    say, reg = _Say(), _Registry()

    def boom(q):
        raise RuntimeError("kaboom")

    handle_mention(_event("<@BOT> break please"), say, "BOT", question_fn=boom, registry=reg)
    assert any("something went wrong" in c["text"].lower() for c in say.calls)
    assert len(reg.recorded) == 0
