# SQS queue + scalable workers + Qdrant server — Design

**Date:** 2026-07-10
**Goal:** Answer Slack mentions **concurrently**. Split the single bot process into a **producer** (Socket Mode listener that enqueues) and **N worker replicas** (drain the queue, run RAG, post the answer). Replace embedded Qdrant with a **Qdrant server** so multiple processes can share the index.

## Context

Today one process does everything. [`slack/handlers.py`](../../../src/ragchat/slack/handlers.py) calls `answer()` inline in the Bolt event handler, so the whole RAG round-trip (plan → embed → retrieve → chat) runs before the handler returns. Two properties of the current code block concurrency:

1. **`clients.get_qdrant()` uses `QdrantClient(path=...)`** — the embedded, file-locked, single-process mode. No second process may open the same directory.
2. **`slack/bot.py` calls `ingest()` at startup**, so the *bot* process holds that lock.

The [2026-07-02 design](2026-07-02-ragchat-modules-and-docker-design.md) already flagged this: it lists single-process Qdrant as a known limitation and names "Qdrant server / horizontal scaling" as the deferred scaling path. This spec takes that path.

## Decisions (locked with user)

| Decision | Choice |
|---|---|
| Primary goal | **Scale / concurrency** (not merely reliability) |
| Deployment target | `docker compose` on a single host |
| Topology | **Producer / consumer split with a queue** (bot → SQS → worker ×N) |
| Vector store | **Qdrant server** container; embedded path kept as a fallback |
| Queue | **ElasticMQ** (SQS-compatible) locally; `boto3` client works unchanged against real AWS SQS |
| Queue type | **FIFO**, grouped by Slack thread |
| Dedup key | Slack `event_id` as `MessageDeduplicationId` |
| Reply path | Worker posts directly via `chat.postMessage` |
| `stats` command | Answered **inline in the bot**, not queued |
| Ack UX | Unchanged — the existing `_thinking…_` message |
| Scope | Queue + Qdrant server + workers + idempotency + `stats` fix. Nothing else. |

## Out of scope (YAGNI now)

Explicitly rejected for this change, even though they appear in the reference architecture:

- **Redis session context / multi-turn threads.** `answer()` stays stateless.
- **Multiple vector collections** (KB / DevHub / Slack). One collection, `finance_kb`.
- **Emoji-reaction ack.** Keep the `_thinking…_` message; no `reactions:write` scope.
- **Lambda / ECS / any AWS deployment.** Compose only.
- **Changing the chat model.** `gpt-4o-mini` stays.

## Why a queue, honestly

Concurrency alone does **not** require SQS. Two cheaper options were considered and rejected:

- **A — Thread pool in one process.** Bolt already dispatches on a thread pool and the RAG path is I/O-bound. Swapping Qdrant to a server would deliver real concurrency for ~20 lines. Rejected: no durability, single-process ceiling.
- **B — N bot replicas.** Slack permits multiple socket connections per app and load-balances events across them. True horizontal scale, no queue infrastructure. Rejected: still no retry, no DLQ, no backpressure.

**C (chosen)** buys, over B: an accepted question is never lost, a burst queues instead of failing, and answering scales independently of the socket. The cost is ~200 new lines and **mandatory idempotency**.

## Architecture

Five compose services. The invariant that keeps the design clean:
**the `bot` never touches Qdrant or OpenAI; the `worker` never touches the WebSocket.**

| Service | Role | Replicas | State |
|---|---|---|---|
| `qdrant` | Vector store (`qdrant/qdrant`), HTTP :6333 | 1 | own volume |
| `queue` | ElasticMQ, SQS REST on :9324 | 1 | own volume (H2) |
| `ingest` | One-shot: build/refresh index, exit 0 | 1 | — |
| `bot` | Socket Mode listener → enqueue; answers `stats` | **1** (single socket) | metrics volume (read) |
| `worker` | Drain queue → `answer()` → post to Slack | **N** | metrics volume (append) |

**Request path**
```
Slack --app_mention--> bot --send--> queue --receive--> worker --> qdrant + OpenAI
                                                          worker --chat.postMessage--> Slack
```

**Failure path**
```
worker raises -> no delete -> visibility timeout -> redeliver (x3) -> DLQ
```

**Startup ordering**
```
qdrant (started) -> ingest (retries until ready, exits 0) -> bot + worker xN
```

## Package layout (new + changed)

```
src/ragchat/
  queue/
    __init__.py
    models.py      # AskJob
    base.py        # JobQueue protocol, ReceivedJob
    sqs.py         # SqsJobQueue (boto3)
    memory.py      # InMemoryJobQueue (tests, local)
  worker/
    __init__.py
    main.py        # entrypoint: config check + poll loop
    processor.py   # process_job(...) — the testable unit
  slack/
    poster.py      # SlackPoster.post(channel, thread_ts, text)   [new]
    app.py         # [modified] pass body + queue into the handler
    handlers.py    # [modified] enqueue instead of answering
    bot.py         # [modified] drop ingest(); listener only
  common/
    clients.py     # [modified] get_qdrant(): url if set, else path
    config.py      # [modified] new settings
  metrics/
    registry.py    # [modified] stats() re-reads the JSONL
```

## Job contract

```python
class AskJob(BaseModel):
    event_id: str    # Slack envelope id -> dedup key
    channel: str     # worker has no `say` closure; it must know where to post
    thread_ts: str
    user: str
    text: str        # mention already stripped by the producer
```

`channel` and `event_id` are new. `handle_mention` reads only `event["text"]` and `event["ts"]` today, and `event_id` lives on Bolt's `body`, not `event` — so the registered handler gains a `body` parameter.

## Queue abstraction

```python
class ReceivedJob(BaseModel):
    job: AskJob
    receipt: str
    receive_count: int

class JobQueue(Protocol):
    def send(self, job: AskJob) -> None: ...
    def receive(self, wait_seconds: int = 20) -> list[ReceivedJob]: ...
    def delete(self, receipt: str) -> None: ...
```

- **`SqsJobQueue`** — `boto3.client("sqs", endpoint_url=settings.queue_endpoint_url or None, region_name=settings.aws_region)`. Long-polls with `WaitTimeSeconds`, requests the `ApproximateReceiveCount` attribute. `endpoint_url` empty ⇒ real AWS SQS; set ⇒ ElasticMQ. **The client code is identical either way.**
- **`InMemoryJobQueue`** — a `deque`; used by tests and by anyone running without Docker.

Selection happens in one place, `queue/__init__.py::get_queue()`, mirroring `clients.get_openai()`/`get_qdrant()`.

Send parameters:

| Parameter | Value | Why |
|---|---|---|
| `MessageGroupId` | `f"{channel}:{thread_ts}"` | Serializes one Slack thread; different threads run in parallel |
| `MessageDeduplicationId` | `event_id` | Slack's own event retries collapse to one job |

`contentBasedDeduplication = false` — we always supply an explicit dedup id.

## Producer — `slack/handlers.py`

`stats` and the empty-text hint stay inline; they are cheap and need no RAG.

```python
def handle_mention(event, body, say, bot_user_id, queue=None, registry=None):
    queue = queue or get_queue()
    registry = registry or get_registry()
    thread_ts = event.get("ts")
    text = strip_mention(event.get("text", ""), bot_user_id)

    if not text:
        say(text=_EMPTY, thread_ts=thread_ts); return
    if text.lower() == "stats":
        say(text=format_stats(registry.stats()), thread_ts=thread_ts); return

    try:
        queue.send(AskJob(event_id=body["event_id"], channel=event["channel"],
                          thread_ts=thread_ts, user=event.get("user", ""), text=text))
    except Exception:
        say(text=_ERROR, thread_ts=thread_ts); return
    say(text=_ACK, thread_ts=thread_ts)
```

Enqueue **before** acking: a queue outage must surface as an error, never as a `_thinking…_` that resolves to nothing.

The existing `question_fn`/`registry` injection pattern is preserved — `queue` simply replaces `question_fn`.

## Consumer — `worker/`

`processor.py` holds the logic; `main.py` holds the loop.

```python
def process_job(rj, answer_fn, poster, registry):
    result = answer_fn(rj.job.text)
    registry.record(result.metrics)
    poster.post(rj.job.channel, rj.job.thread_ts, format_answer(result))
```

```python
# main.py
while True:
    for rj in queue.receive(wait_seconds=settings.worker_wait_seconds):
        try:
            process_job(rj, answer, poster, registry)
            queue.delete(rj.receipt)          # delete ONLY on success
        except Exception:
            log.exception("job failed", extra={"event_id": rj.job.event_id})
            if rj.receive_count >= settings.max_receive_count:
                _post_error(poster, rj)       # last attempt: tell the user
            # no delete -> redeliver -> DLQ
```

The **"I don't know" path (`found=False`) is a success**: it posts and deletes. Only exceptions leave a message on the queue.

## Slack poster — `slack/poster.py`

The worker has no `say` closure, so it posts explicitly:

```python
class SlackPoster:
    def __init__(self, token: str):
        self._client = WebClient(token=token)
    def post(self, channel: str, thread_ts: str, text: str) -> None:
        self._client.chat_postMessage(channel=channel, thread_ts=thread_ts, text=text)
```

Uses the same `SLACK_BOT_TOKEN` and the existing `chat:write` scope. **No new Slack scopes.**

## Qdrant server + fallback

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
```

Keeping the embedded fallback is deliberate: `uv run ragchat ask "…"` still works with no Docker, and `test_ingest.py` / `test_retriever.py` keep running against a temp directory rather than requiring a live server. Minimal test churn.

## Ingest & startup ordering

`ingest()` is removed from `slack/bot.py`. It becomes a one-shot compose service running the existing `ragchat ingest` CLI command against the Qdrant server. `bot` and `worker` both declare:

```yaml
depends_on:
  ingest:
    condition: service_completed_successfully
```

This is the change that actually unblocks multi-process: nothing but `ingest` and the workers ever open Qdrant, and `ingest` has exited before any worker starts.

## Configuration (new settings)

| Setting | Env | Default | Notes |
|---|---|---|---|
| `qdrant_url` | `QDRANT_URL` | `""` | Empty ⇒ embedded on-disk mode |
| `queue_url` | `QUEUE_URL` | `""` | Full SQS queue URL |
| `queue_endpoint_url` | `QUEUE_ENDPOINT_URL` | `""` | Empty ⇒ real AWS; set ⇒ ElasticMQ |
| `aws_region` | `AWS_REGION` | `us-east-1` | |
| `worker_wait_seconds` | `WORKER_WAIT_SECONDS` | `20` | Long-poll duration |
| `max_receive_count` | `MAX_RECEIVE_COUNT` | `3` | Must match the queue's DLQ redrive policy |

**Visibility timeout is queue-side configuration, not a client setting.** It is declared once in `elasticmq.conf` (`defaultVisibilityTimeout = 120 seconds`), or on the queue itself in real AWS. The client never sends it, so it is deliberately *not* a `Settings` field — a value in `Settings` that no code reads would be a lie.

`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` must be **present** — `boto3` refuses to sign a request without credentials, even against ElasticMQ, which ignores their values. Requiring a developer to invent fake AWS credentials for a local queue is bad ergonomics, so `docker-compose.yml` defaults them (`${AWS_ACCESS_KEY_ID:-local}`); a real `.env` still wins when pointing at real SQS. Without this the workers crash-loop on `NoCredentialsError`.

The worker fails fast at startup if `queue_url` is unset, mirroring the token check in `slack/app.py`.

## Delivery semantics & idempotency

Two distinct sources of duplicates, handled differently:

1. **Slack redelivers the same event** → identical `event_id` → `MessageDeduplicationId` suppresses it within the 5-minute dedup window. No dedup store needed.
2. **Worker crashes after `chat.postMessage` but before `delete`** → the message reappears and the answer posts twice.

Case 2 **cannot be eliminated** — it would require a distributed transaction spanning Slack and SQS. This is a choice of failure mode, and we deliberately choose **duplicate answer** (visible, harmless, rare) over **lost answer** (invisible, and the entire reason the queue exists). Accepted and documented, not mitigated.

**The queue's visibility timeout is a correctness setting, not a tuning knob.** If it is shorter than a slow answer, SQS redelivers while the first worker is still running, producing concurrent duplicate processing. The answer path (planner + embeddings + chat) sits well under 30s; `120 seconds` leaves ample margin.

## Error handling

| Failure | Behavior |
|---|---|
| Producer cannot enqueue | Post `_ERROR` in-thread; **no ack** |
| `plan_queries` fails | Existing fallback to `[question]` — unchanged |
| Worker raises | No delete → redeliver; on the final attempt post `_ERROR`, then let it fall into the DLQ |
| Message exhausts `maxReceiveCount` | Lands in `ragchat-dlq` for inspection |
| `QUEUE_URL` / tokens missing | Fail fast at startup with a clear message |
| Corrupt `metrics.jsonl` line | Skipped — unchanged |

Posting `_ERROR` on the last attempt matters: otherwise the user stares at `_thinking…_` forever with no signal.

## Metrics registry fix (a real bug this change exposes)

`MetricsRegistry.__init__` loads `_items` from disk **once**, then serves `stats()` from that in-memory list. Once *workers* append and the *bot* answers `stats`, the bot's list is frozen at boot — `@bot stats` would report stale numbers forever, and no existing test would catch it.

**Fix:** drop the in-memory cache. `stats()` re-reads the JSONL on every call. The file is small and `stats` is rare. `record()` keeps its `threading.Lock` for intra-process safety; short `O_APPEND` writes are atomic on POSIX, so N workers appending concurrently is safe. `bot` and `worker` share the `ragchat_metrics` volume.

## docker-compose (shape)

Note which service gets which environment: `bot` is the only one **without** `QDRANT_URL`, enforcing the "bot never touches Qdrant" invariant at the configuration level. `ingest` is the only one without queue settings.

```yaml
x-queue-env: &queue-env
  QUEUE_URL: "http://queue:9324/000000000000/ragchat-jobs.fifo"
  QUEUE_ENDPOINT_URL: "http://queue:9324"

services:
  qdrant:
    image: qdrant/qdrant
    volumes: [qdrant_data:/qdrant/storage]

  queue:
    image: softwaremill/elasticmq-native
    volumes:
      - ./elasticmq.conf:/opt/elasticmq.conf:ro
      - queue_data:/data

  ingest:
    build: .
    command: ["ragchat", "ingest"]
    env_file: .env
    environment: {QDRANT_URL: "http://qdrant:6333"}
    depends_on: {qdrant: {condition: service_started}}
    volumes: [ragchat_state:/app/data/qdrant]     # manifest.json
    restart: "no"

  bot:
    build: .
    command: ["ragchat-bot"]
    env_file: .env
    environment: *queue-env
    depends_on:
      queue: {condition: service_started}
      ingest: {condition: service_completed_successfully}
    volumes: [ragchat_metrics:/app/data/metrics]

  worker:
    build: .
    command: ["ragchat-worker"]
    env_file: .env
    environment:
      <<: *queue-env
      QDRANT_URL: "http://qdrant:6333"
    deploy: {replicas: 3}
    depends_on:
      queue: {condition: service_started}
      ingest: {condition: service_completed_successfully}
    volumes: [ragchat_metrics:/app/data/metrics]

volumes: {qdrant_data: {}, queue_data: {}, ragchat_state: {}, ragchat_metrics: {}}
```

**Mount mutable state at narrow paths, never at `/app/data`.** The corpus is baked into the image at `/app/data/corpus`. A named volume mounted over `/app/data` is seeded from the image on first use and then *never updated* — so editing a corpus file and rebuilding would leave the bot serving the stale copy forever. Two narrow volumes avoid this:

| Volume | Mount | Written by | Read by |
|---|---|---|---|
| `ragchat_state` | `/app/data/qdrant` | `ingest` (`manifest.json`) | `ingest` |
| `ragchat_metrics` | `/app/data/metrics` | `worker` (append) | `bot` (`stats`) |

`ingest` needs a volume at all only because `manifest.json` is the sha256 hash-skip ledger. Without it, every restart re-embeds the whole corpus at full OpenAI cost — the points are idempotent (`uuid5` ids), so this wastes money and time rather than corrupting the index.

**Qdrant readiness is handled in code, not by a compose healthcheck.** A healthcheck would depend on `curl`/`wget` existing inside the `qdrant/qdrant` image, which is an assumption about someone else's image. Instead `ingest` retries its first Qdrant call with bounded backoff (~10 attempts, 1s apart) and exits non-zero if the server never comes up. `bot` and `worker` then gate on `service_completed_successfully`, so a successful `ingest` *is* the readiness signal.

`elasticmq.conf` — FIFO queue, DLQ redrive, and **file-backed persistence** so a queue restart does not silently discard jobs:

```hocon
include classpath("application.conf")

node-address { protocol = http, host = "queue", port = 9324, context-path = "" }
rest-sqs     { enabled = true, bind-port = 9324, bind-hostname = "0.0.0.0" }

messages-storage { enabled = true, uri = "jdbc:h2:/data/elasticmq" }
queues-storage   { enabled = true, path = "/data/queues.conf" }

queues {
  "ragchat-jobs.fifo" {
    fifo = true
    contentBasedDeduplication = false
    defaultVisibilityTimeout = 120 seconds
    deadLettersQueue { name = "ragchat-dlq", maxReceiveCount = 3 }
  }
  "ragchat-dlq" { }
}
```

**The DLQ name must NOT end in `.fifo` on ElasticMQ 1.7.1**, despite what ElasticMQ's own documentation shows. It validates `deadLettersQueue.name` against the *standard* queue-name rule and refuses to boot:

```
InvalidParameterValue(ragchat-dlq.fifo, Can only include alphanumeric characters, hyphens, or underscores.)
```

Real AWS SQS is the opposite: a FIFO queue's dead-letter queue must itself be FIFO and end in `.fifo`. So this file is **not** a template for AWS — moving to real SQS means provisioning `ragchat-dlq.fifo` there. Discovered by running the container, not by reading docs.

## Testing

| Test | Asserts |
|---|---|
| `test_slack_handlers.py` (rewritten) | A question enqueues an `AskJob` with the correct `event_id`, `channel`, `thread_ts`, and stripped `text`; then `_ACK` is posted |
| `test_slack_handlers.py` | A failing `queue.send` posts `_ERROR` and **no** ack |
| `test_slack_handlers.py` | `stats` and empty-text paths never enqueue |
| `test_worker_processor.py` (new) | `process_job` posts the formatted answer to `channel`/`thread_ts` and records metrics |
| `test_worker_processor.py` | `answer_fn` raising leaves the message undeleted |
| `test_worker_processor.py` | `found=False` still posts and deletes |
| `test_queue_memory.py` (new) | `InMemoryJobQueue` satisfies the `JobQueue` protocol round-trip |
| `test_registry.py` (extended) | Appending to the JSONL behind the registry's back is reflected by `stats()` — the regression test for the caching bug |

`sqs.py` stays a thin `boto3` wrapper, so there is little logic to test; it is exercised by the compose smoke test rather than unit tests.

**Smoke test:** `docker compose up --build --scale worker=3`, then post three mentions in three different Slack threads and confirm they are answered in parallel by different workers (visible in the logs).

## Accepted limitations

- **`bot` is a single replica** — one WebSocket, so it is a single point of failure for *accepting* work. Queued jobs survive a bot restart; questions asked while it is down are lost (Slack does not replay them).
- **Duplicate answers are possible** if a worker dies between posting and deleting. See *Delivery semantics*.
- **ElasticMQ is a local stand-in.** Persistence is enabled, but production durability would use real AWS SQS — a `QUEUE_ENDPOINT_URL` change, no code change.
- **`ingest` is one-shot.** Corpus changes require re-running that service; there is no hot reload.
- **Metrics are file-based**, so `stats` reads scale linearly with total questions answered. Fine at prototype volume.

## Implementation order

Each step leaves the system working and tested:

1. **Qdrant server + fallback** in `clients.py`; move `ingest()` out of `bot.py` into a one-shot compose service, with bounded connect-retry so it tolerates a cold Qdrant. *(No queue yet — the bot still answers inline. This step alone verifies the directory lock is gone.)*
2. **`MetricsRegistry.stats()` re-reads** the JSONL, with its regression test.
3. **`queue/` package** — `AskJob`, `JobQueue`, `InMemoryJobQueue`, plus tests.
4. **`slack/poster.py`** and **`worker/processor.py`** with tests, driven by the in-memory queue.
5. **Producer/consumer split** — rewrite `handle_mention`, add `worker/main.py` and the `ragchat-worker` entrypoint.
6. **`SqsJobQueue`** + `boto3` dependency + `elasticmq.conf`.
7. **compose wiring** and the `--scale worker=3` smoke test.

Steps 1 and 2 are independently valuable and carry no queue risk; if the work is ever paused, that is a good place to stop.
