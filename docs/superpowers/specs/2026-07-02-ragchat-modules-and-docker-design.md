# RAG modules + Docker + Slack bot + metrics — Design

**Date:** 2026-07-02
**Goal:** Extract the RAG pipeline from `notebooks/rag_lab.ipynb` into a package **organized by flow**, run it as a Slack bot (`@ask_finance`) over Socket Mode, package it as a Docker image runnable with `docker compose up`, and add an **aggregate metrics interface** (num questions, total/avg sub-queries, total/avg tokens) viewable from Slack and a CLI.

## Context

All RAG logic currently lives in the notebook (persistent Qdrant, sha256 skip-unchanged ingest, cheap-LLM query planner, grounded synthesis, per-question metrics). The notebook was built "before extracting modules." Slack cannot call notebook cells, so the logic must become importable source with a single entry point.

## Decisions (locked with user)

| Decision | Choice |
|---|---|
| Code shape | Package **organized by flow** (folders per flow) under `src/ragchat/` |
| Models | Pydantic; flow-specific models live with their flow, shared ones in `common/` |
| Config | `pydantic-settings`, env-driven |
| Slack interaction | `@ask_finance <question>` (`app_mention`), reply in thread |
| Slack connection | Socket Mode (outbound WebSocket), local / Docker, no public URL |
| Docker entrypoint | The Slack bot |
| Vector store in Docker | Corpus baked into image; Qdrant index on mounted volume; `ingest()` on startup (hash-skip) |
| Per-question tokens | Per-call `TokenMeter` (no module global) — thread-safe |
| Aggregate metrics | `MetricsRegistry`; persisted as JSONL on the mounted volume |
| Metrics view | Both a Slack `stats` mention and a CLI command |

## Package layout — by flow

```
src/ragchat/
  __init__.py              # public API: answer, ingest, get_registry, Answer, AggregateStats
  common/                  # shared across flows
    config.py              # Settings (pydantic-settings): models, dirs, thresholds, prompts, secrets
    clients.py             # build OpenAI client + persistent QdrantClient(path=...)
    models.py              # cross-cutting Pydantic models: Chunk, Answer
  ingestion/               # ── RAG ingestion flow ──
    chunking.py            # parse_frontmatter, chunk_text, chunk_document
    store.py               # ensure_collection, manifest load/save
    ingest.py              # ingest(force=False): sha256 skip, per-doc replace, uuid5 ids
  retrieval/               # ── query flow ──
    embeddings.py          # embed(texts, meter)
    planner.py             # QueryPlan (pydantic) + plan_queries(question, meter)
    retriever.py           # retrieve(queries, meter): union + dedup + per-doc cap
  answering/               # ── answer / generation flow ──
    generator.py           # answer(question) -> Answer
  metrics/                 # ── metrics flow ──
    models.py              # TokenMeter, QuestionMetrics, AggregateStats
    registry.py            # MetricsRegistry: record(), stats(), JSONL persistence
  slack/
    app.py                 # Bolt Socket Mode bot: app_mention -> answer(); "stats" -> aggregate
  cli.py                   # entrypoints: ingest, ask "<q>", stats
Dockerfile
docker-compose.yml
tests/
```

### Flow responsibilities

- **common/config.py** — `Settings(BaseSettings)`: `embed_model`, `chat_model`, `router_model`, `vector_size`, `top_k`, `score_threshold`, `chunk_words`, `collection`, `corpus_dir`, `qdrant_dir`, `metrics_path`, `idk_message`, `system_prompt`; secrets `openai_api_key`, `slack_bot_token`, `slack_app_token` from env. Singleton `settings`. Path resolution works from repo root or container `WORKDIR`.
- **common/clients.py** — construct the OpenAI client and one shared persistent `QdrantClient(path=settings.qdrant_dir)`.
- **common/models.py** — `Chunk`, `Answer` (Pydantic).
- **ingestion/** — chunking (pure functions), collection + manifest persistence, `ingest()` orchestration (returns a summary).
- **retrieval/** — `embed()`; `plan_queries()` using `client.beta.chat.completions.parse` → `QueryPlan` with fallback `[question]`; `retrieve()` union + per-doc cap. All accept a `TokenMeter`.
- **answering/generator.py** — `answer(question) -> Answer`: create a `TokenMeter`; plan → retrieve → grounding gate → grounded generation; build `QuestionMetrics` from the meter; return `Answer`. **Pure**: no aggregate-metrics side effects (caller records).
- **metrics/** — see below.

### Public API
```python
from ragchat import answer, ingest, get_registry
r = answer("Compare EV/EBIT and O'Shaughnessy")   # Answer (pydantic, incl. per-question metrics)
get_registry().record(r.metrics)                    # aggregate
get_registry().stats()                              # AggregateStats
```

## Per-question token metering (refactor)

The notebook uses a module-global `TOKENS` + snapshot/delta, which corrupts under concurrent Slack mentions. Replacement: a `TokenMeter` created inside each `answer()` call, threaded through `embed`/`plan_queries`/`retrieve`; each OpenAI response's `usage` is added to that meter. Per-question `QuestionMetrics` come from the local meter — correct and thread-safe.

## Metrics interface (new)

- **`QuestionMetrics`** (per question): `sub_queries`, `chunks_used`, `embed_tokens`, `chat_tokens`, `total_tokens`.
- **`AggregateStats`** (Pydantic): `num_questions`, `total_sub_queries`, `avg_sub_queries`, `total_chunks`, `avg_chunks`, `total_embed_tokens`, `total_chat_tokens`, `total_tokens`, `avg_tokens`.
- **`MetricsRegistry`** (`metrics/registry.py`):
  - Thread-safe (a `threading.Lock`).
  - `record(m: QuestionMetrics)` — appends a JSON line to `settings.metrics_path` (`data/metrics/metrics.jsonl`, on the mounted volume) and updates in-memory running totals.
  - `stats() -> AggregateStats` — returns current aggregates (averages computed from totals / count; zero-safe when empty).
  - On construction, loads any existing JSONL so stats survive restarts.
  - Process-wide singleton via `get_registry()`.
- **Recording point:** the Slack handler (and the CLI `ask`) call `registry.record(r.metrics)` after a successful answer. `answer()` itself stays side-effect-free.
- **Viewing — two surfaces:**
  - **Slack:** `@ask_finance stats` → the handler detects the `stats` keyword and replies with a formatted `AggregateStats` summary instead of running RAG.
  - **CLI:** `python -m ragchat.cli stats` (and `docker … stats`) prints the same summary.

## Ingest & startup

Bot startup: build clients → `ingest()` (skips unchanged docs against the mounted volume) → load `MetricsRegistry` → start the Socket Mode listener. Corpus baked at `data/corpus/`; index + metrics persist on the mounted volume.

## Slack bot (`slack/app.py`)

- Bolt `App(token=bot_token)` + `SocketModeHandler(app, app_token)`.
- `app_mention` handler:
  1. Strip the leading `<@BOTID>` mention → question text.
  2. If text is empty → "Ask me a question about the finance corpus."
  3. If text is the `stats` keyword → reply with `get_registry().stats()` formatted.
  4. Otherwise: post an in-thread ack ("_thinking…_"), call `answer(question)`, then post the answer + **Sources** + a compact metrics line as a follow-up **in the mention's thread** (`thread_ts`); `registry.record(r.metrics)`.
  5. Wrap in try/except → post a friendly failure message in-thread; never let the handler raise unhandled.
- Scopes: `app_mentions:read`, `chat:write`. Event: `app_mention`. Socket Mode enabled.

## CLI (`cli.py`)

`python -m ragchat.cli <command>`:
- `ingest [--force]` — build/refresh the index.
- `ask "<question>"` — print a grounded answer + sources + per-question metrics.
- `stats` — print aggregate metrics.

## Docker

- **Dockerfile:** `python:3.12-slim` + `uv`; copy `pyproject.toml` + `uv.lock`; `uv sync --frozen --no-dev`; copy `src/` + `data/corpus/`; `WORKDIR /app`; `CMD` runs the bot (ingest → listen).
- **docker-compose.yml:** service `bot`; `env_file: .env` (OpenAI + Slack tokens); named volume `ragchat_data:/app/data` (holds both `qdrant/` and `metrics/`); `restart: unless-stopped`.
- **Run:** `docker compose up --build`; or `docker run --env-file .env -v ragchat_data:/app/data <image>`; CLI via `docker compose run --rm bot python -m ragchat.cli stats`.
- Secrets injected at runtime via `.env` (gitignored); never baked into the image.

## Error handling

- `plan_queries` failure → fallback `[question]`.
- `answer()` error → Slack handler catches, posts "Sorry, something went wrong answering that." in-thread; logs the exception.
- Missing tokens at startup → fail fast with a clear message (pydantic-settings validation).
- Corrupt/absent `metrics.jsonl` line → skip that line, keep going (never crash on metrics).
- IDK path → standard message, posted as-is (still recorded, with 0 chunks).

## Testing

- **Unit (no network):** `chunking`; `ingest`/`retrieve` with a stubbed embedder + temp Qdrant dir; `plan_queries` JSON/fallback with a mocked OpenAI client; `MetricsRegistry.record`/`stats` with a temp JSONL (totals + averages + reload).
- **Handler:** synthetic `app_mention` (question, empty, and `stats`) with a mocked Slack client + stubbed `answer()`; assert threaded reply, stats path, and error handling.
- **Smoke (optional, real key):** `python -m ragchat.cli ask "What is EV/EBIT?"`.

## Concurrency (known limitation)

Local on-disk Qdrant is single-process; the single bot container holds it — fine for a prototype. `MetricsRegistry` is lock-guarded. Heavy *simultaneous* questions would need a Qdrant **server** container (compose service) with `clients.py` pointed at it — out of scope now; noted as the scaling path.

## Out of scope (YAGNI now)

- HTTP Events API / hosted deployment (Socket Mode local first).
- Qdrant server / horizontal scaling.
- Slash commands or DM interaction (mention only).
- Streaming answers, per-user/per-channel metrics, dashboards, thread memory.
