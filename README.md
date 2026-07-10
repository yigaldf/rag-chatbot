# rag-chatbot

A Slack bot that answers investment-strategy questions strictly from a vector knowledge base. It refuses to answer from outside knowledge — if the corpus doesn't cover it, it says so.

Mention the bot to ask a question, or `@bot stats` for usage metrics.

## Architecture

The bot accepts questions; workers answer them.

```
Slack --app_mention--> bot --send--> queue --receive--> worker xN --> qdrant + OpenAI
                                                        worker --chat.postMessage--> Slack
```

The `bot` holds a single Socket Mode WebSocket and does nothing but ack and enqueue. N `worker` replicas drain the queue, run retrieval and generation, and post the answer back into the mention's thread. A worker that fails doesn't delete its message, so the job is redelivered and eventually lands in a dead-letter queue.

Design notes live in [docs/superpowers/specs/](docs/superpowers/specs/).

## Running with Docker

```bash
cp .env.example .env        # then fill in your OpenAI and Slack tokens
docker compose up --build   # qdrant, queue, ingest, bot, worker x3
```

To change concurrency, or to watch the workers pick up jobs:

```bash
docker compose up --build --scale worker=5
docker compose logs -f worker
```

The CLI runs against the same shared volume:

```bash
docker compose run --rm bot ragchat stats
```

## Running without Docker

`uv run ragchat ask "what is EV/EBIT?"` works with nothing else running. With `QDRANT_URL` and `QUEUE_URL` unset, the app falls back to an embedded on-disk Qdrant and an in-process queue.

```bash
uv sync
uv run ragchat ingest          # build the index under ./data
uv run ragchat ask "what is the acquirer's multiple?"
uv run ragchat stats
uv run pytest
```

## Slack setup

Socket Mode, so no public URL is needed. See [docs/slack-setup.md](docs/slack-setup.md). You need a bot token (`xoxb-…`, scopes `app_mentions:read` and `chat:write`) and an app-level token (`xapp-…`, scope `connections:write`).
