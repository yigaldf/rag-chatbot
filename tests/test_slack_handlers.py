from ragchat.metrics.models import AggregateStats
from ragchat.queue.memory import InMemoryJobQueue
from ragchat.slack.handlers import handle_mention


class _Say:
    def __init__(self):
        self.calls = []

    def __call__(self, text, thread_ts=None):
        self.calls.append({"text": text, "thread_ts": thread_ts})


class _Registry:
    def stats(self):
        return AggregateStats(num_questions=1, total_sub_queries=2, avg_sub_queries=2.0,
                              total_chunks=3, avg_chunks=3.0, total_embed_tokens=10,
                              total_chat_tokens=100, total_tokens=110, avg_tokens=110.0)


class _BrokenQueue:
    def send(self, job):
        raise RuntimeError("queue down")


def _event(text, ts="111.1"):
    return {"text": text, "ts": ts, "user": "UUSER", "channel": "C1"}


def _body(event_id="Ev1"):
    return {"event_id": event_id}


def test_question_enqueues_a_job_then_acks():
    say, q = _Say(), InMemoryJobQueue()
    handle_mention(_event("<@BOT> what is EV/EBIT?"), _body(), say, "BOT",
                   queue=q, registry=_Registry())
    rj = q.receive()[0]
    assert rj.job.event_id == "Ev1"
    assert rj.job.channel == "C1"
    assert rj.job.thread_ts == "111.1"
    assert rj.job.text == "what is EV/EBIT?"   # mention stripped
    assert len(say.calls) == 1
    assert "thinking" in say.calls[0]["text"]
    assert say.calls[0]["thread_ts"] == "111.1"


def test_enqueue_failure_posts_error_and_never_acks():
    say = _Say()
    handle_mention(_event("<@BOT> anything"), _body(), say, "BOT",
                   queue=_BrokenQueue(), registry=_Registry())
    assert len(say.calls) == 1
    assert "something went wrong" in say.calls[0]["text"].lower()
    assert "thinking" not in say.calls[0]["text"]


def test_empty_question_prompts_and_does_not_enqueue():
    say, q = _Say(), InMemoryJobQueue()
    handle_mention(_event("<@BOT>   "), _body(), say, "BOT", queue=q, registry=_Registry())
    assert q.receive() == []
    assert "ask me" in say.calls[-1]["text"].lower()


def test_stats_answers_inline_and_does_not_enqueue():
    say, q = _Say(), InMemoryJobQueue()
    handle_mention(_event("<@BOT> stats"), _body(), say, "BOT", queue=q, registry=_Registry())
    assert q.receive() == []
    assert "Aggregate metrics" in say.calls[-1]["text"]
