# SQS Queue + Scalable Workers + Qdrant Server — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Answer Slack mentions concurrently by splitting the single bot process into a producer (Socket Mode listener that enqueues) and N worker replicas (drain the queue, run RAG, post the answer), backed by a shared Qdrant server.

**Architecture:** The `bot` service acks a mention and pushes an `AskJob` onto a FIFO queue, then returns immediately. N `worker` replicas long-poll the queue, run `answer()`, and post the reply straight to Slack via `chat.postMessage`. Embedded (file-locked) Qdrant is replaced by a Qdrant server so multiple processes can share the index; the embedded path survives as a fallback so tests and the CLI need no Docker.

**Tech Stack:** Python 3.12, `uv`, pydantic / pydantic-settings, slack-bolt, qdrant-client, boto3 (SQS), ElasticMQ (local SQS-compatible queue), Docker Compose, pytest.

**Spec:** [docs/superpowers/specs/2026-07-10-sqs-workers-qdrant-server-design.md](../specs/2026-07-10-sqs-workers-qdrant-server-design.md)

---

## File Structure

**Created:**

| File | Responsibility |
|---|---|
| `src/ragchat/queue/__init__.py` | `get_queue()` — pick the queue implementation from config |
| `src/ragchat/queue/models.py` | `AskJob`, `ReceivedJob` — the job contract |
| `src/ragchat/queue/base.py` | `JobQueue` protocol |
| `src/ragchat/queue/memory.py` | `InMemoryJobQueue` — tests + no-Docker local runs |
| `src/ragchat/queue/sqs.py` | `SqsJobQueue` — thin boto3 wrapper |
| `src/ragchat/slack/messages.py` | Shared user-facing strings (ack / empty / error) |
| `src/ragchat/slack/poster.py` | `SlackPoster.post()` — the worker's reply path |
| `src/ragchat/worker/__init__.py` | package marker |
| `src/ragchat/worker/processor.py` | `process_job()` + `run_once()` — the testable core |
| `src/ragchat/worker/main.py` | worker entrypoint: config check + forever loop |
| `elasticmq.conf` | FIFO queue, DLQ redrive, file-backed persistence |

**Modified:**

| File | Change |
|---|---|
| `src/ragchat/common/config.py` | `qdrant_url`, `queue_url`, `queue_endpoint_url`, `aws_region`, `worker_wait_seconds`, `max_receive_count` |
| `src/ragchat/common/clients.py` | `get_qdrant()` uses URL if set; add `wait_for_qdrant()` |
| `src/ragchat/metrics/registry.py` | Drop the in-memory cache; `stats()` re-reads the JSONL |
| `src/ragchat/slack/handlers.py` | Enqueue instead of answering |
| `src/ragchat/slack/app.py` | Pass `body` into the handler |
| `src/ragchat/slack/bot.py` | Drop the `ingest()` call |
| `src/ragchat/cli.py` | `ingest` waits for Qdrant first |
| `pyproject.toml` | `boto3` dep + `ragchat-worker` script |
| `docker-compose.yml` | five services |
| `.env.example` | new settings |
| `README.md` | how to run scaled |

---

## Task 1: Qdrant server + embedded fallback

Removes the directory lock that makes multi-process impossible. After this task the bot still answers inline — no queue yet — but nothing except `ingest` and the answering process opens Qdrant.

**Files:**
- Modify: `src/ragchat/common/config.py`
- Modify: `src/ragchat/common/clients.py`
- Modify: `src/ragchat/slack/bot.py`
- Modify: `src/ragchat/cli.py`
- Test: `tests/test_clients.py`

- [ ] **Step 1: Write the failing tests**

Append the tests below to `tests/test_clients.py`. First make sure these two imports are present at the top of that file (add whichever is missing — do not add a second import block mid-file):

```python
import pytest

from ragchat.common import clients
```

Then append:

```python
def test_get_qdrant_uses_url_when_set(monkeypatch):
    monkeypatch.setattr(clients.settings, "qdrant_url", "http://qdrant:6333")
    clients.reset_clients()
    captured = {}

    class FakeQC:
        def __init__(self, **kw):
            captured.update(kw)

    monkeypatch.setattr(clients, "QdrantClient", FakeQC)
    clients.get_qdrant()
    clients.reset_clients()
    assert captured == {"url": "http://qdrant:6333"}


def test_get_qdrant_uses_path_when_url_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(clients.settings, "qdrant_url", "")
    monkeypatch.setattr(clients.settings, "data_dir", str(tmp_path))
    clients.reset_clients()
    captured = {}

    class FakeQC:
        def __init__(self, **kw):
            captured.update(kw)

    monkeypatch.setattr(clients, "QdrantClient", FakeQC)
    clients.get_qdrant()
    clients.reset_clients()
    assert "path" in captured and "url" not in captured


def test_wait_for_qdrant_is_noop_in_embedded_mode(monkeypatch):
    monkeypatch.setattr(clients.settings, "qdrant_url", "")
    called = {"n": 0}
    monkeypatch.setattr(clients, "get_qdrant", lambda: called.__setitem__("n", 1))
    clients.wait_for_qdrant()
    assert called["n"] == 0


def test_wait_for_qdrant_retries_until_ready(monkeypatch):
    monkeypatch.setattr(clients.settings, "qdrant_url", "http://qdrant:6333")
    calls = {"n": 0}

    class FakeQC:
        def get_collections(self):
            calls["n"] += 1
            if calls["n"] < 3:
                raise ConnectionError("not up yet")
            return []

    monkeypatch.setattr(clients, "get_qdrant", lambda: FakeQC())
    monkeypatch.setattr(clients, "_close_qdrant", lambda: None)
    clients.wait_for_qdrant(attempts=5, delay=0.0, sleep=lambda _s: None)
    assert calls["n"] == 3


def test_wait_for_qdrant_raises_after_exhausting_attempts(monkeypatch):
    monkeypatch.setattr(clients.settings, "qdrant_url", "http://qdrant:6333")

    class FakeQC:
        def get_collections(self):
            raise ConnectionError("never up")

    monkeypatch.setattr(clients, "get_qdrant", lambda: FakeQC())
    monkeypatch.setattr(clients, "_close_qdrant", lambda: None)
    with pytest.raises(RuntimeError, match="not reachable"):
        clients.wait_for_qdrant(attempts=2, delay=0.0, sleep=lambda _s: None)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_clients.py -v`
Expected: FAIL — `AttributeError: 'Settings' object has no attribute 'qdrant_url'` and `module 'ragchat.common.clients' has no attribute 'wait_for_qdrant'`.

- [ ] **Step 3: Add the config field**

In `src/ragchat/common/config.py`, add below `vector_size`:

```python
    # Vector store. Empty => embedded on-disk mode (single process).
    qdrant_url: str = ""
```

- [ ] **Step 4: Implement URL mode + readiness wait**

In `src/ragchat/common/clients.py`, add `import time` at the top, then replace `get_qdrant` and add `wait_for_qdrant`:

```python
def get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        if settings.qdrant_url:
            _qdrant = QdrantClient(url=settings.qdrant_url)
        else:
            settings.qdrant_dir.mkdir(parents=True, exist_ok=True)
            _qdrant = QdrantClient(path=str(settings.qdrant_dir))
    return _qdrant


def wait_for_qdrant(attempts: int = 10, delay: float = 1.0, sleep=time.sleep) -> None:
    """Block until the Qdrant server answers. No-op in embedded mode.

    Compose starts `qdrant` and `ingest` together, so the first connection may
    land before the server is listening. Retrying here keeps the readiness
    check in our code instead of betting on `curl` existing in someone else's image.
    """
    if not settings.qdrant_url:
        return
    last: Exception | None = None
    for _ in range(attempts):
        try:
            get_qdrant().get_collections()
            return
        except Exception as e:  # noqa: BLE001 - any transport error means "not up yet"
            last = e
            _close_qdrant()  # drop the half-open client before retrying
            sleep(delay)
    raise RuntimeError(f"Qdrant not reachable at {settings.qdrant_url} after {attempts} attempts: {last}")
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_clients.py -v`
Expected: PASS (all 5 new tests).

- [ ] **Step 6: Remove `ingest()` from the bot entrypoint**

Replace `src/ragchat/slack/bot.py` entirely:

```python
import logging

from ragchat.slack.app import run_socket_mode


def main() -> None:
    # INFO-level logging so slack_bolt reports the Socket Mode connection and
    # every incoming event (otherwise the bot runs silently and is undebuggable).
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Ingest is a one-shot compose service now: the bot must never open Qdrant,
    # or it would hold the embedded directory lock and block the workers.
    print("starting Slack Socket Mode bot… (Ctrl-C to stop)", flush=True)
    run_socket_mode()


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Make the `ingest` CLI wait for Qdrant**

In `src/ragchat/cli.py`, change the import line and the `ingest` branch:

```python
from ragchat.common.clients import wait_for_qdrant
```

```python
    if args.cmd == "ingest":
        wait_for_qdrant()
        print(ingest(force=args.force))
```

- [ ] **Step 8: Run the full suite**

Run: `uv run pytest`
Expected: PASS — all existing tests still green (they leave `qdrant_url` empty, so they keep using the temp-dir embedded mode).

- [ ] **Step 9: Commit**

```bash
git add src/ragchat/common/config.py src/ragchat/common/clients.py \
        src/ragchat/slack/bot.py src/ragchat/cli.py tests/test_clients.py
git commit -m "feat: support Qdrant server via QDRANT_URL, keep embedded fallback

Removes the directory lock that prevented a second process from opening
the index. ingest() moves out of the bot entrypoint; it becomes a one-shot
step that waits for the server with bounded retry."
```

---

## Task 2: Fix the metrics registry cross-process staleness bug

`MetricsRegistry` loads its items once at construction and serves `stats()` from that list. Once workers own the writes and the bot serves `stats`, the bot would report numbers frozen at boot — forever, and silently. Fix it before the split makes it real.

**Files:**
- Modify: `src/ragchat/metrics/registry.py`
- Test: `tests/test_registry.py`

- [ ] **Step 1: Write the failing regression test**

Append to `tests/test_registry.py`:

```python
def test_stats_reflects_writes_from_another_process(tmp_path):
    """A worker process appends; the bot process must see it.

    The bot holds a long-lived MetricsRegistry and answers `stats` from it.
    Simulate an external append and assert stats() re-reads the file.
    """
    path = tmp_path / "metrics.jsonl"
    reg = MetricsRegistry(path)
    reg.record(_m(1, 1, 10, 90))
    assert reg.stats().num_questions == 1

    # another process appends, bypassing this registry entirely
    with path.open("a", encoding="utf-8") as f:
        f.write(_m(2, 4, 20, 180).model_dump_json() + "\n")

    assert reg.stats().num_questions == 2
    assert reg.stats().total_tokens == 300
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_registry.py::test_stats_reflects_writes_from_another_process -v`
Expected: FAIL — `assert 1 == 2`. The registry served its cached `_items`.

- [ ] **Step 3: Drop the cache**

Replace the `MetricsRegistry` class in `src/ragchat/metrics/registry.py`:

```python
class MetricsRegistry:
    """Append-only JSONL metrics, safe for many processes.

    stats() re-reads the file on every call: the bot process serves `stats`
    while N worker processes do the appending, so any in-memory cache would
    go stale the moment a worker records an answer.
    """

    def __init__(self, path: Path):
        self._path = Path(path)
        self._lock = threading.Lock()

    def _read(self) -> list[QuestionMetrics]:
        items: list[QuestionMetrics] = []
        if self._path.exists():
            for line in self._path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(QuestionMetrics.model_validate_json(line))
                except Exception:
                    continue  # skip corrupt lines
        return items

    def record(self, m: QuestionMetrics) -> None:
        # Short O_APPEND writes are atomic on POSIX, so N workers may append
        # concurrently. The lock only guards threads inside one process.
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as f:
                f.write(m.model_dump_json() + "\n")

    def stats(self) -> AggregateStats:
        return AggregateStats.from_metrics(self._read())
```

- [ ] **Step 4: Run the registry tests**

Run: `uv run pytest tests/test_registry.py -v`
Expected: PASS — all 3 tests, including the pre-existing reload and corrupt-line tests.

- [ ] **Step 5: Commit**

```bash
git add src/ragchat/metrics/registry.py tests/test_registry.py
git commit -m "fix: MetricsRegistry.stats() re-reads the JSONL

The in-memory item list was loaded once at construction. With workers
appending and the bot serving \`stats\`, the bot would report numbers
frozen at process start. Regression test included."
```

---

## Task 3: The queue package — contract, protocol, in-memory implementation

**Files:**
- Create: `src/ragchat/queue/__init__.py`
- Create: `src/ragchat/queue/models.py`
- Create: `src/ragchat/queue/base.py`
- Create: `src/ragchat/queue/memory.py`
- Modify: `src/ragchat/common/config.py`
- Test: `tests/test_queue_memory.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_queue_memory.py`:

```python
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_queue_memory.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ragchat.queue'`.

- [ ] **Step 3: Write the job contract**

Create `src/ragchat/queue/models.py`:

```python
from pydantic import BaseModel


class AskJob(BaseModel):
    """One question to answer. Everything the worker needs, nothing it doesn't."""
    event_id: str    # Slack envelope id — the dedup key
    channel: str     # the worker has no `say` closure; it must know where to post
    thread_ts: str
    user: str
    text: str        # mention already stripped by the producer


class ReceivedJob(BaseModel):
    job: AskJob
    receipt: str
    receive_count: int
```

- [ ] **Step 4: Write the protocol**

Create `src/ragchat/queue/base.py`:

```python
from typing import Protocol

from ragchat.queue.models import AskJob, ReceivedJob


class JobQueue(Protocol):
    def send(self, job: AskJob) -> None: ...
    def receive(self, wait_seconds: int = 20) -> list[ReceivedJob]: ...
    def delete(self, receipt: str) -> None: ...
```

- [ ] **Step 5: Write the in-memory implementation**

Create `src/ragchat/queue/memory.py`:

```python
from collections import deque

from ragchat.queue.models import AskJob, ReceivedJob


class InMemoryJobQueue:
    """Process-local queue for tests and no-Docker local runs.

    `deleted` is exposed so tests can assert delete-on-success semantics.
    """

    def __init__(self) -> None:
        self._q: deque[ReceivedJob] = deque()
        self._seq = 0
        self.deleted: list[str] = []

    def send(self, job: AskJob) -> None:
        self._seq += 1
        self._q.append(ReceivedJob(job=job, receipt=f"r{self._seq}", receive_count=1))

    def receive(self, wait_seconds: int = 20) -> list[ReceivedJob]:
        return [self._q.popleft()] if self._q else []

    def delete(self, receipt: str) -> None:
        self.deleted.append(receipt)
```

- [ ] **Step 6: Add the queue config fields**

In `src/ragchat/common/config.py`, add below `qdrant_url`:

```python
    # Queue. Empty queue_url => in-memory queue (no Docker, tests).
    queue_url: str = ""
    queue_endpoint_url: str = ""   # set => ElasticMQ; empty => real AWS SQS
    aws_region: str = "us-east-1"
    worker_wait_seconds: int = 20
    max_receive_count: int = 3     # must match the queue's DLQ redrive policy
```

- [ ] **Step 7: Write the factory**

Create `src/ragchat/queue/__init__.py`. The `sqs` import is deliberately lazy so `boto3` is only needed when a real queue is configured — that is also why `sqs.py` can arrive in Task 6 without breaking this one.

```python
from ragchat.common.config import settings
from ragchat.queue.models import AskJob, ReceivedJob

_queue = None


def get_queue():
    """Pick the queue implementation from config. Process-wide singleton."""
    global _queue
    if _queue is None:
        if settings.queue_url:
            from ragchat.queue.sqs import build_sqs_queue  # lazy: boto3 only when needed
            _queue = build_sqs_queue()
        else:
            from ragchat.queue.memory import InMemoryJobQueue
            _queue = InMemoryJobQueue()
    return _queue


def reset_queue() -> None:
    """Test helper: drop the cached queue."""
    global _queue
    _queue = None


__all__ = ["AskJob", "ReceivedJob", "get_queue", "reset_queue"]
```

- [ ] **Step 8: Run the tests to verify they pass**

Run: `uv run pytest tests/test_queue_memory.py -v`
Expected: PASS (4 tests).

- [ ] **Step 9: Commit**

```bash
git add src/ragchat/queue tests/test_queue_memory.py src/ragchat/common/config.py
git commit -m "feat: queue package — AskJob contract, JobQueue protocol, in-memory impl

get_queue() lazily imports the SQS backend so boto3 is only required when
QUEUE_URL is set. Tests and no-Docker runs use the in-memory queue."
```

---

## Task 4: Slack poster + worker processor

The worker cannot use Bolt's `say` — that closure only exists inside an event handler. It posts explicitly. `run_once()` holds the delete/retry semantics and is the unit under test; `main.py` (Task 5) is just a loop around it.

**Files:**
- Create: `src/ragchat/slack/messages.py`
- Create: `src/ragchat/slack/poster.py`
- Create: `src/ragchat/worker/__init__.py`
- Create: `src/ragchat/worker/processor.py`
- Test: `tests/test_worker_processor.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_worker_processor.py`:

```python
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
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_worker_processor.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ragchat.worker'`.

- [ ] **Step 3: Extract the shared user-facing strings**

Create `src/ragchat/slack/messages.py` (both the producer and the worker post these — define them once):

```python
ACK = "_thinking…_"
EMPTY = "Ask me a question about the finance corpus. For usage stats, mention me with `stats`."
ERROR = "Sorry, something went wrong answering that."
```

- [ ] **Step 4: Write the poster**

Create `src/ragchat/slack/poster.py`:

```python
from slack_sdk import WebClient

from ragchat.common.config import settings


class SlackPoster:
    """The worker's reply path. Bolt's `say` is a per-event closure, so the
    worker posts explicitly with the channel and thread it was handed."""

    def __init__(self, token: str | None = None):
        self._client = WebClient(token=token or settings.slack_bot_token)

    def post(self, channel: str, thread_ts: str, text: str) -> None:
        self._client.chat_postMessage(channel=channel, thread_ts=thread_ts, text=text)
```

- [ ] **Step 5: Write the processor**

Create `src/ragchat/worker/__init__.py` (empty file), then `src/ragchat/worker/processor.py`:

```python
import logging

from ragchat.common.config import settings
from ragchat.slack import messages
from ragchat.slack.formatting import format_answer

log = logging.getLogger(__name__)


def process_job(job, answer_fn, poster, registry) -> None:
    """Answer one question and post the reply. Raises on any failure."""
    result = answer_fn(job.text)
    registry.record(result.metrics)
    poster.post(job.channel, job.thread_ts, format_answer(result))


def _post_error(poster, job) -> None:
    """Best-effort. A dead Slack must not take the worker loop down with it."""
    try:
        poster.post(job.channel, job.thread_ts, messages.ERROR)
    except Exception:
        log.exception("failed to post the error message for %s", job.event_id)


def run_once(queue, answer_fn, poster, registry, max_receive_count: int | None = None) -> None:
    """Drain one batch. Delete ONLY on success, so failures get redelivered."""
    max_receive_count = settings.max_receive_count if max_receive_count is None else max_receive_count
    for rj in queue.receive(wait_seconds=settings.worker_wait_seconds):
        try:
            process_job(rj.job, answer_fn, poster, registry)
            queue.delete(rj.receipt)
        except Exception:
            log.exception("job failed: %s", rj.job.event_id)
            # Last attempt: tell the user, or they stare at "_thinking…_" forever.
            # Never delete — let the message fall into the DLQ for inspection.
            if rj.receive_count >= max_receive_count:
                _post_error(poster, rj.job)
```

Note: an "I don't know" answer (`found=False`) is a **success** — it posts and deletes. Only exceptions leave a message on the queue.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_worker_processor.py -v`
Expected: PASS (5 tests).

- [ ] **Step 7: Commit**

```bash
git add src/ragchat/slack/messages.py src/ragchat/slack/poster.py \
        src/ragchat/worker tests/test_worker_processor.py
git commit -m "feat: SlackPoster + worker processor with delete-on-success semantics

run_once() deletes only when a job succeeds, so failures are redelivered
and eventually land in the DLQ. On the final attempt it posts an error so
the user isn't left with an unresolved ack."
```

---

## Task 5: Split producer and consumer

**Files:**
- Modify: `src/ragchat/slack/handlers.py`
- Modify: `src/ragchat/slack/app.py`
- Create: `src/ragchat/worker/main.py`
- Modify: `pyproject.toml`
- Test: `tests/test_slack_handlers.py` (rewrite)

- [ ] **Step 1: Rewrite the handler tests**

Replace `tests/test_slack_handlers.py` entirely:

```python
import pytest

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
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_slack_handlers.py -v`
Expected: FAIL — `TypeError: handle_mention() got an unexpected keyword argument 'queue'`.

- [ ] **Step 3: Rewrite the producer**

Replace `src/ragchat/slack/handlers.py` entirely:

```python
import logging

from ragchat.metrics.registry import get_registry as _default_registry
from ragchat.queue import get_queue as _default_queue
from ragchat.queue.models import AskJob
from ragchat.slack import messages
from ragchat.slack.formatting import strip_mention, format_stats

log = logging.getLogger(__name__)


def handle_mention(event, body, say, bot_user_id, queue=None, registry=None):
    """Producer: ack and enqueue. The RAG work happens in a worker.

    - `event`: Slack event dict (uses `text`, `ts`, `channel`, `user`).
    - `body`: Bolt envelope (uses `event_id` — the dedup key).
    - `say(text, thread_ts=...)`: posts a message.
    - `bot_user_id`: this bot's Slack user id (for stripping the mention).
    - `queue`: JobQueue (defaults to get_queue()).
    - `registry`: metrics registry (defaults to get_registry()); only used by `stats`.
    """
    queue = queue or _default_queue()
    registry = registry or _default_registry()
    thread_ts = event.get("ts")
    text = strip_mention(event.get("text", ""), bot_user_id)

    if not text:
        say(text=messages.EMPTY, thread_ts=thread_ts)
        return
    if text.lower() == "stats":
        say(text=format_stats(registry.stats()), thread_ts=thread_ts)
        return

    # Enqueue BEFORE acking: a queue outage must surface as an error, never as
    # a "_thinking…_" that resolves to nothing.
    try:
        queue.send(AskJob(
            event_id=body["event_id"],
            channel=event["channel"],
            thread_ts=thread_ts,
            user=event.get("user", ""),
            text=text,
        ))
    except Exception:
        log.exception("enqueue failed")
        say(text=messages.ERROR, thread_ts=thread_ts)
        return
    say(text=messages.ACK, thread_ts=thread_ts)
```

- [ ] **Step 4: Pass `body` through from Bolt**

In `src/ragchat/slack/app.py`, replace `build_app`:

```python
def build_app() -> App:
    """Construct the Bolt app and register the app_mention handler."""
    app = App(token=settings.slack_bot_token)

    @app.event("app_mention")
    def _on_mention(event, body, say, context):
        # `body["event_id"]` is the dedup key; `say` posts to the event's channel.
        handle_mention(event, body, say, context["bot_user_id"])

    return app
```

- [ ] **Step 5: Run the handler tests to verify they pass**

Run: `uv run pytest tests/test_slack_handlers.py -v`
Expected: PASS (4 tests).

- [ ] **Step 6: Write the worker entrypoint**

Create `src/ragchat/worker/main.py`:

```python
import logging

from ragchat.answering.generator import answer
from ragchat.common.clients import wait_for_qdrant
from ragchat.common.config import settings
from ragchat.metrics.registry import get_registry
from ragchat.queue import get_queue
from ragchat.slack.poster import SlackPoster
from ragchat.worker.processor import run_once

log = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not settings.queue_url:
        raise SystemExit("Missing QUEUE_URL. The worker has no queue to drain.")
    if not settings.slack_bot_token:
        raise SystemExit("Missing SLACK_BOT_TOKEN (xoxb-…). The worker cannot post replies.")

    wait_for_qdrant()
    queue, poster, registry = get_queue(), SlackPoster(), get_registry()
    log.info("worker draining %s", settings.queue_url)
    while True:
        run_once(queue, answer, poster, registry)


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Register the worker console script**

In `pyproject.toml`, under `[project.scripts]`, add:

```toml
ragchat-worker = "ragchat.worker.main:main"
```

- [ ] **Step 8: Run the full suite**

Run: `uv run pytest`
Expected: PASS — every test green.

- [ ] **Step 9: Commit**

```bash
git add src/ragchat/slack/handlers.py src/ragchat/slack/app.py \
        src/ragchat/worker/main.py pyproject.toml tests/test_slack_handlers.py
git commit -m "feat: split producer and consumer

handle_mention() now enqueues an AskJob and acks; ragchat-worker drains the
queue, answers, and posts back. Enqueue happens before the ack so a queue
outage surfaces as an error rather than a hanging '_thinking…_'."
```

---

## Task 6: The SQS backend + ElasticMQ config

**Files:**
- Create: `src/ragchat/queue/sqs.py`
- Create: `elasticmq.conf`
- Modify: `pyproject.toml` (via `uv add`)
- Test: `tests/test_queue_sqs.py`

- [ ] **Step 1: Add the boto3 dependency**

Run: `uv add boto3`
Expected: `pyproject.toml` gains `boto3` under `dependencies`, and `uv.lock` updates.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_queue_sqs.py`. A fake client keeps this offline — no `moto`, no network:

```python
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
```

- [ ] **Step 3: Run them to verify they fail**

Run: `uv run pytest tests/test_queue_sqs.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ragchat.queue.sqs'`.

- [ ] **Step 4: Write the SQS backend**

Create `src/ragchat/queue/sqs.py`:

```python
import boto3

from ragchat.common.config import settings
from ragchat.queue.models import AskJob, ReceivedJob


class SqsJobQueue:
    """Thin boto3 wrapper. Identical against ElasticMQ and real AWS SQS —
    only `endpoint_url` differs."""

    def __init__(self, client, queue_url: str):
        self._client = client
        self._url = queue_url

    def send(self, job: AskJob) -> None:
        self._client.send_message(
            QueueUrl=self._url,
            MessageBody=job.model_dump_json(),
            # Serialize one Slack thread; let different threads run in parallel.
            MessageGroupId=f"{job.channel}:{job.thread_ts}",
            # Slack retries the same event_id; FIFO dedup collapses them.
            MessageDeduplicationId=job.event_id,
        )

    def receive(self, wait_seconds: int = 20) -> list[ReceivedJob]:
        resp = self._client.receive_message(
            QueueUrl=self._url,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=wait_seconds,
            AttributeNames=["ApproximateReceiveCount"],
        )
        return [
            ReceivedJob(
                job=AskJob.model_validate_json(m["Body"]),
                receipt=m["ReceiptHandle"],
                receive_count=int(m["Attributes"]["ApproximateReceiveCount"]),
            )
            for m in resp.get("Messages", [])
        ]

    def delete(self, receipt: str) -> None:
        self._client.delete_message(QueueUrl=self._url, ReceiptHandle=receipt)


def build_sqs_queue() -> SqsJobQueue:
    client = boto3.client(
        "sqs",
        endpoint_url=settings.queue_endpoint_url or None,  # None => real AWS
        region_name=settings.aws_region,
    )
    return SqsJobQueue(client, settings.queue_url)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_queue_sqs.py -v`
Expected: PASS (4 tests).

- [ ] **Step 6: Write the ElasticMQ config**

Create `elasticmq.conf` at the repo root. The `.fifo` suffix is required on FIFO queues and their DLQs. Persistence is on, or a queue restart would silently discard the jobs we added a queue to protect.

```hocon
include classpath("application.conf")

node-address { protocol = http, host = "queue", port = 9324, context-path = "" }
rest-sqs     { enabled = true, bind-port = 9324, bind-hostname = "0.0.0.0" }

messages-storage { enabled = true, uri = "jdbc:h2:/data/elasticmq" }
queues-storage   { enabled = true, path = "/data/queues.conf" }

queues {
  "ragchat-jobs.fifo" {
    fifo = true
    contentBasedDeduplication = false   # we always supply an explicit dedup id
    defaultVisibilityTimeout = 120 seconds
    deadLettersQueue { name = "ragchat-dlq.fifo", maxReceiveCount = 3 }
  }
  "ragchat-dlq.fifo" { }
}
```

- [ ] **Step 7: Run the full suite**

Run: `uv run pytest`
Expected: PASS — every test green.

- [ ] **Step 8: Commit**

```bash
git add src/ragchat/queue/sqs.py tests/test_queue_sqs.py elasticmq.conf pyproject.toml uv.lock
git commit -m "feat: SQS queue backend + ElasticMQ config

FIFO with MessageGroupId=channel:thread_ts (parallel across threads,
ordered within one) and MessageDeduplicationId=event_id (Slack's event
retries collapse). DLQ after 3 receives; message persistence enabled."
```

---

## Task 7: Compose wiring + scaled smoke test

**Files:**
- Modify: `docker-compose.yml`
- Modify: `.env.example`
- Modify: `README.md`

- [ ] **Step 1: Rewrite `docker-compose.yml`**

`bot` is the only service **without** `QDRANT_URL`, and `ingest` the only one without queue settings — the "bot never touches Qdrant, worker never touches the socket" invariant is enforced by configuration, not convention.

```yaml
x-queue-env: &queue-env
  QUEUE_URL: "http://queue:9324/000000000000/ragchat-jobs.fifo"
  QUEUE_ENDPOINT_URL: "http://queue:9324"

services:
  qdrant:
    image: qdrant/qdrant
    volumes:
      - qdrant_data:/qdrant/storage
    restart: unless-stopped

  queue:
    image: softwaremill/elasticmq-native
    volumes:
      - ./elasticmq.conf:/opt/elasticmq.conf:ro
      - queue_data:/data
    restart: unless-stopped

  # One-shot: build/refresh the index, then exit 0. Its success IS the
  # readiness signal that bot and worker gate on.
  ingest:
    build: .
    image: ragchat-bot
    command: ["ragchat", "ingest"]
    env_file: .env
    environment:
      QDRANT_URL: "http://qdrant:6333"
    depends_on:
      qdrant:
        condition: service_started
    restart: "no"

  bot:
    build: .
    image: ragchat-bot
    command: ["ragchat-bot"]
    env_file: .env
    environment: *queue-env
    volumes:
      - ragchat_data:/app/data
    depends_on:
      queue:
        condition: service_started
      ingest:
        condition: service_completed_successfully
    restart: unless-stopped

  worker:
    build: .
    image: ragchat-bot
    command: ["ragchat-worker"]
    env_file: .env
    environment:
      <<: *queue-env
      QDRANT_URL: "http://qdrant:6333"
    volumes:
      - ragchat_data:/app/data
    depends_on:
      queue:
        condition: service_started
      ingest:
        condition: service_completed_successfully
    deploy:
      replicas: 3
    restart: unless-stopped

volumes:
  qdrant_data:
  queue_data:
  ragchat_data:
```

- [ ] **Step 2: Update `.env.example`**

```bash
# Copy to .env and fill in. .env is gitignored.
OPENAI_API_KEY=sk-...
SLACK_BOT_TOKEN=xoxb-...
SLACK_APP_TOKEN=xapp-...

# boto3 requires these to exist; ElasticMQ ignores their values.
AWS_ACCESS_KEY_ID=local
AWS_SECRET_ACCESS_KEY=local
AWS_REGION=us-east-1
```

- [ ] **Step 3: Verify the stack builds and connects**

Run: `docker compose up --build`
Expected, in order:
1. `qdrant` and `queue` start.
2. `ingest` prints an ingest summary dict and exits 0.
3. `bot` logs the Socket Mode WebSocket connection.
4. Three `worker` containers log `worker draining http://queue:9324/...`.

- [ ] **Step 4: Smoke-test concurrency**

In Slack, post three mentions in **three different threads**, quickly:

```
@yourbot what is the acquirer's multiple?
@yourbot explain EV/EBITDA
@yourbot compare EV/EBIT and O'Shaughnessy
```

Run: `docker compose logs -f worker`
Expected: the three jobs are picked up by **different worker containers** and answered in parallel — not serialized. Each answer lands threaded under its own mention.

- [ ] **Step 5: Smoke-test the `stats` cross-process path**

In Slack: `@yourbot stats`
Expected: `num_questions` reflects the three answers above. This is the bug from Task 2 — the bot process never recorded them, the workers did.

- [ ] **Step 6: Smoke-test the failure path**

Run: `docker compose stop qdrant`, then post a mention.
Expected: the worker retries the job three times (visible in `docker compose logs worker`), posts "Sorry, something went wrong answering that." on the final attempt, and the message lands in `ragchat-dlq.fifo`.

Then: `docker compose start qdrant` to recover.

- [ ] **Step 7: Update the README**

Add a "Running with workers" section documenting:

```bash
docker compose up --build                  # 3 workers by default
docker compose up --build --scale worker=5 # more concurrency
docker compose logs -f worker
docker compose run --rm bot ragchat stats  # CLI against the shared volume
```

Note that `uv run ragchat ask "…"` still works with no Docker (embedded Qdrant, in-memory queue) because `QDRANT_URL` and `QUEUE_URL` are unset.

- [ ] **Step 8: Commit**

```bash
git add docker-compose.yml .env.example README.md
git commit -m "feat: compose wiring for qdrant, queue, ingest, bot, worker x3

bot is the only service without QDRANT_URL and ingest the only one without
queue settings, so the process-isolation invariant is enforced by config.
ingest's successful exit is the readiness gate for bot and worker."
```

---

## Verification Checklist

After Task 7, confirm each spec requirement landed:

- [ ] `uv run pytest` — all tests pass
- [ ] Three mentions in three threads are answered by three different workers (Task 7 Step 4)
- [ ] `@bot stats` reflects worker-recorded answers (Task 7 Step 5)
- [ ] A failing job retries 3× and lands in the DLQ with an error posted (Task 7 Step 6)
- [ ] `@bot stats` and empty mentions never enqueue
- [ ] A queue outage produces an error, never a hanging `_thinking…_`
- [ ] `uv run ragchat ask "what is EV/EBIT?"` still works with no Docker running
- [ ] `bot` container has no `QDRANT_URL` in its environment

## Known Limitations (accepted, per the spec)

- **`bot` is a single replica.** One WebSocket, so it is a single point of failure for *accepting* work. Queued jobs survive a bot restart; questions asked while it is down are lost, because Slack does not replay them.
- **Duplicate answers are possible** if a worker dies between `chat.postMessage` and `delete`. Eliminating this would require a distributed transaction across Slack and SQS. We deliberately prefer a rare duplicate answer over a silently lost one.
- **ElasticMQ is a local stand-in.** Production durability means real AWS SQS — a `QUEUE_ENDPOINT_URL` change, no code change.
- **`ingest` is one-shot.** Corpus changes require re-running that service.
