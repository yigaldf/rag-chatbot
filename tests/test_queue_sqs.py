from ragchat.queue.models import AskJob
from ragchat.queue.sqs import SqsJobQueue

URL = "http://queue:9324/000000000000/ragchat-jobs.fifo"


class _FakeSqs:
    def __init__(self, messages=None):
        self.sent = []
        self.deleted = []
        self._messages = messages or []

    def send_message(self, **kw):
        self.sent.append(kw)

    def receive_message(self, **kw):
        self.received_with = kw
        return {"Messages": self._messages} if self._messages else {}

    def delete_message(self, **kw):
        self.deleted.append(kw)


def _job():
    return AskJob(event_id="Ev1", channel="C1", thread_ts="111.1", user="U1", text="q?")


def test_send_sets_group_id_and_dedup_id():
    c = _FakeSqs()
    SqsJobQueue(c, URL).send(_job())
    sent = c.sent[0]
    assert sent["QueueUrl"] == URL
    # one thread is serialized; different threads run in parallel
    assert sent["MessageGroupId"] == "C1:111.1"
    # Slack's own event retries collapse to a single job
    assert sent["MessageDeduplicationId"] == "Ev1"
    assert '"text":"q?"' in sent["MessageBody"].replace(" ", "")


def test_receive_parses_body_and_receive_count():
    body = _job().model_dump_json()
    c = _FakeSqs([{"Body": body, "ReceiptHandle": "rh-1",
                   "Attributes": {"ApproximateReceiveCount": "2"}}])
    got = SqsJobQueue(c, URL).receive(wait_seconds=7)
    assert len(got) == 1
    assert got[0].job.event_id == "Ev1"
    assert got[0].receipt == "rh-1"
    assert got[0].receive_count == 2
    assert c.received_with["WaitTimeSeconds"] == 7


def test_receive_on_empty_queue_returns_empty_list():
    assert SqsJobQueue(_FakeSqs(), URL).receive() == []


def test_delete_passes_the_receipt_handle():
    c = _FakeSqs()
    SqsJobQueue(c, URL).delete("rh-1")
    assert c.deleted == [{"QueueUrl": URL, "ReceiptHandle": "rh-1"}]
