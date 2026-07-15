from ragchat.queue.memory import InMemoryJobQueue
from ragchat.queue.models import AskJob


def _job(event_id="Ev1", text="what is EV/EBIT?"):
    return AskJob(event_id=event_id, channel="C1", thread_ts="111.1", user="U1", text=text)


def test_send_then_receive_returns_the_job():
    q = InMemoryJobQueue()
    q.send(_job())
    got = q.receive()
    assert len(got) == 1
    assert got[0].job.text == "what is EV/EBIT?"
    assert got[0].job.channel == "C1"
    assert got[0].receive_count == 1


def test_receive_on_empty_queue_returns_empty_list():
    assert InMemoryJobQueue().receive() == []


def test_delete_records_the_receipt():
    q = InMemoryJobQueue()
    q.send(_job())
    rj = q.receive()[0]
    q.delete(rj.receipt)
    assert q.deleted == [rj.receipt]


def test_receive_is_fifo():
    q = InMemoryJobQueue()
    q.send(_job("Ev1", "first"))
    q.send(_job("Ev2", "second"))
    assert q.receive()[0].job.text == "first"
    assert q.receive()[0].job.text == "second"
