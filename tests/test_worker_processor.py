from ragchat.common.models import Answer
from ragchat.metrics.models import QuestionMetrics
from ragchat.queue.memory import InMemoryJobQueue
from ragchat.queue.models import AskJob
from ragchat.worker.processor import run_once


class _Poster:
    def __init__(self, boom=False):
        self.posts = []
        self.boom = boom

    def post(self, channel, thread_ts, text):
        if self.boom:
            raise RuntimeError("slack down")
        self.posts.append({"channel": channel, "thread_ts": thread_ts, "text": text})


class _Registry:
    def __init__(self):
        self.recorded = []

    def record(self, m):
        self.recorded.append(m)


def _answer(text="A", found=True):
    m = QuestionMetrics(sub_queries=2, chunks_used=3, embed_tokens=10, chat_tokens=100, total_tokens=110)
    return Answer(text=text, sources=["S — u"], found=found, metrics=m)


def _job(event_id="Ev1"):
    return AskJob(event_id=event_id, channel="C1", thread_ts="111.1", user="U1", text="q?")


def test_success_posts_answer_records_metrics_and_deletes():
    q, poster, reg = InMemoryJobQueue(), _Poster(), _Registry()
    q.send(_job())
    run_once(q, lambda _t: _answer(), poster, reg)
    assert len(poster.posts) == 1
    assert poster.posts[0]["channel"] == "C1"
    assert poster.posts[0]["thread_ts"] == "111.1"
    assert "A" in poster.posts[0]["text"]
    assert len(reg.recorded) == 1
    assert len(q.deleted) == 1


def test_idk_answer_is_a_success_and_is_deleted():
    q, poster, reg = InMemoryJobQueue(), _Poster(), _Registry()
    q.send(_job())
    run_once(q, lambda _t: _answer(text="I don't know.", found=False), poster, reg)
    assert len(poster.posts) == 1
    assert len(q.deleted) == 1


def test_failure_does_not_delete_so_the_message_is_redelivered():
    q, poster, reg = InMemoryJobQueue(), _Poster(), _Registry()
    q.send(_job())

    def boom(_t):
        raise RuntimeError("kaboom")

    run_once(q, boom, poster, reg)
    assert q.deleted == []
    assert reg.recorded == []
    assert poster.posts == []  # not the last attempt: stay silent, let it retry


def test_failure_on_last_attempt_tells_the_user():
    q, poster, reg = InMemoryJobQueue(), _Poster(), _Registry()
    q.send(_job())
    rj = q.receive()[0]

    class _Once:
        def __init__(self, item):
            self.item = item
            self.deleted = []

        def receive(self, wait_seconds=20):
            return [self.item]

        def delete(self, receipt):
            self.deleted.append(receipt)

    rj.receive_count = 3
    q2 = _Once(rj)

    def boom(_t):
        raise RuntimeError("kaboom")

    run_once(q2, boom, poster, reg, max_receive_count=3)
    assert q2.deleted == []  # still goes to the DLQ
    assert "something went wrong" in poster.posts[0]["text"].lower()


def test_error_post_failure_does_not_crash_the_loop():
    q, poster, reg = InMemoryJobQueue(), _Poster(boom=True), _Registry()
    q.send(_job())

    def boom(_t):
        raise RuntimeError("kaboom")

    # max_receive_count=1 forces the last-attempt error post on this first
    # receive; the poster then raises. A dead Slack must not kill the loop.
    run_once(q, boom, poster, reg, max_receive_count=1)  # must not raise
    assert q.deleted == []
