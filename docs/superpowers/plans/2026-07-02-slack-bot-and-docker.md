# Slack Bot + Docker Implementation Plan (Plan 2 of 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the `ragchat` package as a Slack bot (`@ask_finance`) over Socket Mode, packaged as a Docker image runnable with `docker compose up`.

**Architecture:** A thin Slack layer (`src/ragchat/slack/`) formats `Answer`/`AggregateStats` into Slack messages and wires a Bolt Socket Mode `app_mention` handler that calls `ragchat.answer()` and records metrics. A `bot.py` entrypoint ingests on startup then starts the listener. A Dockerfile installs the package via uv and runs the bot; docker-compose mounts the data volume and injects secrets from `.env`. The RAG core (Plan 1) is unchanged.

**Tech Stack:** slack-bolt (Socket Mode), the `ragchat` package (Plan 1), Docker + docker compose, uv.

**Depends on:** Plan 1 (`ragchat` package) — already implemented.

---

## File Structure

```
src/ragchat/slack/
  __init__.py
  formatting.py     # pure: Answer/AggregateStats -> Slack text (no Slack SDK, no network)
  handlers.py       # handle_mention(event, say, question_fn, registry): the app_mention logic
  app.py            # build_app(): Bolt App + register handler; run_socket_mode()
  bot.py            # entrypoint: ingest() then start Socket Mode  (python -m ragchat.slack.bot)
Dockerfile
.dockerignore
docker-compose.yml
.env.example
tests/
  test_slack_formatting.py
  test_slack_handlers.py
```

Design note: the Slack *logic* (parsing the mention, deciding stats-vs-question, choosing what text to post) lives in `handlers.py` and `formatting.py` as **pure, testable functions** that take injected dependencies (`say`, `question_fn`, `registry`). `app.py`/`bot.py` are thin wiring that we don't unit-test (they need real Slack tokens). This keeps 100% of the interesting behavior testable without tokens or network.

---

### Task 1: Slack message formatting (pure functions)

**Files:**
- Create: `src/ragchat/slack/__init__.py` (empty)
- Create: `src/ragchat/slack/formatting.py`
- Test: `tests/test_slack_formatting.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_slack_formatting.py
from ragchat.slack.formatting import strip_mention, format_answer, format_stats
from ragchat.common.models import Answer
from ragchat.metrics.models import QuestionMetrics, AggregateStats


def _answer(text, sources, found):
    m = QuestionMetrics(sub_queries=2, chunks_used=3, embed_tokens=10, chat_tokens=100, total_tokens=110)
    return Answer(text=text, sources=sources, found=found, metrics=m)


def test_strip_mention_removes_leading_bot_id():
    assert strip_mention("<@U123> what is EV/EBIT?", "U123") == "what is EV/EBIT?"
    assert strip_mention("<@U123>   spaced  ", "U123") == "spaced"
    # mention in the middle is also removed
    assert strip_mention("hey <@U123> hi", "U123") == "hey  hi".strip()


def test_format_answer_found_includes_sources_and_metrics():
    out = format_answer(_answer("The answer.", ["Doc A — u"], True))
    assert "The answer." in out
    assert "Doc A — u" in out
    assert "Sources" in out
    assert "110 tokens" in out


def test_format_answer_idk_has_no_sources_section():
    out = format_answer(_answer("I don't know.", [], False))
    assert "I don't know." in out
    assert "Sources" not in out


def test_format_stats():
    agg = AggregateStats(num_questions=5, total_sub_queries=10, avg_sub_queries=2.0,
                         total_chunks=20, avg_chunks=4.0, total_embed_tokens=100,
                         total_chat_tokens=900, total_tokens=1000, avg_tokens=200.0)
    out = format_stats(agg)
    assert "5" in out and "2.0" in out and "200.0" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_slack_formatting.py -v`
Expected: FAIL (`No module named 'ragchat.slack.formatting'`)

- [ ] **Step 3: Write minimal implementation**

```python
# src/ragchat/slack/__init__.py
# (intentionally empty)
```

```python
# src/ragchat/slack/formatting.py
import re

from ragchat.common.models import Answer
from ragchat.metrics.models import AggregateStats


def strip_mention(text: str, bot_user_id: str) -> str:
    """Remove the <@BOTID> mention token(s) and normalise whitespace."""
    cleaned = re.sub(rf"<@{re.escape(bot_user_id)}>", " ", text)
    return re.sub(r"\s+", " ", cleaned).strip()


def format_answer(a: Answer) -> str:
    lines = [a.text]
    if a.found and a.sources:
        lines.append("\n*Sources:*")
        lines.extend(f"• {s}" for s in a.sources)
    m = a.metrics
    lines.append(
        f"\n_{m.sub_queries} sub-quer{'y' if m.sub_queries == 1 else 'ies'} · "
        f"{m.chunks_used} chunk(s) · {m.total_tokens} tokens "
        f"(embed {m.embed_tokens} + chat {m.chat_tokens})_"
    )
    return "\n".join(lines)


def format_stats(s: AggregateStats) -> str:
    return (
        "*Aggregate metrics*\n"
        f"• questions: {s.num_questions}\n"
        f"• sub-queries: total {s.total_sub_queries}, avg {s.avg_sub_queries:.2f}\n"
        f"• chunks: total {s.total_chunks}, avg {s.avg_chunks:.2f}\n"
        f"• tokens: total {s.total_tokens}, avg {s.avg_tokens:.1f} "
        f"(embed {s.total_embed_tokens} + chat {s.total_chat_tokens})"
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_slack_formatting.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ragchat/slack/__init__.py src/ragchat/slack/formatting.py tests/test_slack_formatting.py
git commit -m "feat: slack message formatting (pure functions)"
```

---

### Task 2: Mention handler (pure logic, injected deps)

**Files:**
- Create: `src/ragchat/slack/handlers.py`
- Test: `tests/test_slack_handlers.py`

The handler is a pure function that takes the Slack `event` dict, a `say` callable (posts a message; accepts `text=` and `thread_ts=`), the bot's user id, a `question_fn` (defaults to `ragchat.answer`), and a `registry` (defaults to `get_registry()`). This lets us test all branches with fakes.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_slack_handlers.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_slack_handlers.py -v`
Expected: FAIL (`No module named 'ragchat.slack.handlers'`)

- [ ] **Step 3: Write minimal implementation**

```python
# src/ragchat/slack/handlers.py
from ragchat.answering.generator import answer as _default_answer
from ragchat.metrics.registry import get_registry as _default_registry
from ragchat.slack.formatting import strip_mention, format_answer, format_stats

_ACK = "_thinking…_"
_EMPTY = "Ask me a question about the finance corpus. For usage stats, mention me with `stats`."
_ERROR = "Sorry, something went wrong answering that."


def handle_mention(event, say, bot_user_id, question_fn=None, registry=None):
    """Core app_mention logic. Pure except for the injected `say`/`registry`.

    - `event`: Slack event dict (uses `text` and `ts`).
    - `say(text, thread_ts=...)`: posts a message.
    - `bot_user_id`: this bot's Slack user id (for stripping the mention).
    - `question_fn`: q -> Answer (defaults to ragchat.answer).
    - `registry`: metrics registry (defaults to get_registry()).
    """
    question_fn = question_fn or _default_answer
    registry = registry or _default_registry()
    thread_ts = event.get("ts")
    text = strip_mention(event.get("text", ""), bot_user_id)

    if not text:
        say(text=_EMPTY, thread_ts=thread_ts)
        return
    if text.lower() == "stats":
        say(text=format_stats(registry.stats()), thread_ts=thread_ts)
        return

    say(text=_ACK, thread_ts=thread_ts)
    try:
        result = question_fn(text)
    except Exception:
        say(text=_ERROR, thread_ts=thread_ts)
        return
    registry.record(result.metrics)
    say(text=format_answer(result), thread_ts=thread_ts)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_slack_handlers.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/ragchat/slack/handlers.py tests/test_slack_handlers.py
git commit -m "feat: slack app_mention handler (question / stats / empty / error)"
```

---

### Task 3: Bolt app wiring + bot entrypoint

**Files:**
- Create: `src/ragchat/slack/app.py`
- Create: `src/ragchat/slack/bot.py`
- Modify: `pyproject.toml` (add a `ragchat-bot` console script)

No unit test — this is thin wiring that requires real tokens. It is exercised by the live smoke test (Task 7). Keep it minimal so there is nothing to test in isolation.

- [ ] **Step 1: Write the Bolt app wiring**

```python
# src/ragchat/slack/app.py
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from ragchat.common.config import settings
from ragchat.slack.handlers import handle_mention


def build_app() -> App:
    """Construct the Bolt app and register the app_mention handler."""
    app = App(token=settings.slack_bot_token)

    @app.event("app_mention")
    def _on_mention(event, say, context):
        # context["bot_user_id"] is provided by Bolt; `say` posts to the event's channel.
        handle_mention(event, say, context["bot_user_id"])

    return app


def run_socket_mode() -> None:
    if not settings.slack_bot_token or not settings.slack_app_token:
        raise SystemExit(
            "Missing Slack tokens. Set SLACK_BOT_TOKEN (xoxb-…) and SLACK_APP_TOKEN (xapp-…) "
            "in .env or the environment."
        )
    app = build_app()
    SocketModeHandler(app, settings.slack_app_token).start()
```

- [ ] **Step 2: Write the bot entrypoint (ingest then listen)**

```python
# src/ragchat/slack/bot.py
from ragchat.ingestion.ingest import ingest
from ragchat.slack.app import run_socket_mode


def main() -> None:
    summary = ingest()  # warm-skip unchanged docs; builds the index on first run
    print(f"ingest on startup: {summary}", flush=True)
    print("starting Slack Socket Mode bot… (Ctrl-C to stop)", flush=True)
    run_socket_mode()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Add the console script**

In `pyproject.toml`, under the existing `[project.scripts]`, add the `ragchat-bot` entry so the block reads:

```toml
[project.scripts]
ragchat = "ragchat.cli:main"
ragchat-bot = "ragchat.slack.bot:main"
```

- [ ] **Step 4: Verify it imports and the entrypoint fails cleanly without tokens**

Run:
```bash
uv sync
uv run python -c "from ragchat.slack.app import run_socket_mode; print('import ok')"
```
Expected: prints `import ok` (module imports without tokens).

Run:
```bash
SLACK_BOT_TOKEN= SLACK_APP_TOKEN= uv run ragchat-bot 2>&1 | head -3 || true
```
Expected: it runs `ingest` then exits with the "Missing Slack tokens" SystemExit message (no traceback beyond the message). If ingest needs the OpenAI key and it is absent, that is fine for this check — the point is the wiring imports and the token guard fires.

- [ ] **Step 5: Commit**

```bash
git add src/ragchat/slack/app.py src/ragchat/slack/bot.py pyproject.toml uv.lock
git commit -m "feat: Bolt Socket Mode app + bot entrypoint (ingest then listen)"
```

---

### Task 4: .dockerignore and .env.example

**Files:**
- Create: `.dockerignore`
- Create: `.env.example`

- [ ] **Step 1: Write .dockerignore**

```
# .dockerignore — keep the build context small and secrets out
.git
.venv
data/qdrant
data/metrics
notebooks
docs
tests
.claude
.env
__pycache__
*.pyc
.ipynb_checkpoints
```

- [ ] **Step 2: Write .env.example**

```
# Copy to .env and fill in. .env is gitignored.
OPENAI_API_KEY=sk-...
SLACK_BOT_TOKEN=xoxb-...
SLACK_APP_TOKEN=xapp-...
```

- [ ] **Step 3: Commit**

```bash
git add .dockerignore .env.example
git commit -m "chore: add .dockerignore and .env.example"
```

---

### Task 5: Dockerfile

**Files:**
- Create: `Dockerfile`

The image installs uv, syncs the project (which installs the `ragchat` package + its console scripts), bakes in the corpus, and runs the bot. Secrets and the Qdrant/metrics data are NOT baked in — they come from `.env` and a mounted volume at runtime.

- [ ] **Step 1: Write the Dockerfile**

```dockerfile
# Dockerfile
FROM python:3.12-slim

# uv for fast, reproducible installs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install dependencies first (better layer caching). Needs the package sources
# because the project itself is installed by `uv sync`.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

# Bake in the corpus (the index + metrics live on a mounted volume at runtime).
COPY data/corpus ./data/corpus

# Data root inside the container; docker-compose mounts a volume here.
ENV DATA_DIR=/app/data

# Run the Slack bot: ingest (warm-skip) then start Socket Mode.
CMD ["uv", "run", "ragchat-bot"]
```

- [ ] **Step 2: Build the image**

Run: `docker build -t ragchat-bot .`
Expected: build succeeds; final line shows the image tagged `ragchat-bot`.

- [ ] **Step 3: Verify the package is installed in the image**

Run: `docker run --rm ragchat-bot uv run ragchat --help`
Expected: prints the CLI usage (subcommands `ingest`, `ask`, `stats`) — proves the package + console scripts are installed.

- [ ] **Step 4: Commit**

```bash
git add Dockerfile
git commit -m "feat: Dockerfile (uv sync + bake corpus + run bot)"
```

---

### Task 6: docker-compose.yml

**Files:**
- Create: `docker-compose.yml`

- [ ] **Step 1: Write docker-compose.yml**

```yaml
# docker-compose.yml
services:
  bot:
    build: .
    image: ragchat-bot
    env_file: .env
    environment:
      DATA_DIR: /app/data
    volumes:
      # Persist the Qdrant index + metrics across restarts (corpus is baked into the image).
      - ragchat_data:/app/data
    restart: unless-stopped

volumes:
  ragchat_data:
```

- [ ] **Step 2: Validate the compose file**

Run: `docker compose config >/dev/null && echo "compose valid"`
Expected: prints `compose valid` (parses without error).

- [ ] **Step 3: Verify the CLI runs through compose (no Slack needed)**

Run:
```bash
docker compose run --rm bot sh -c "uv run ragchat ingest && uv run ragchat ask 'What is EV/EBIT?'"
```
Expected: the ingest summary, then a cited answer — proves the container can ingest into the mounted volume and answer. (`ask` alone would return IDK on a fresh empty volume since it does not ingest; requires `OPENAI_API_KEY` in `.env`.)

- [ ] **Step 4: Commit**

```bash
git add docker-compose.yml
git commit -m "feat: docker-compose (data volume + env + restart)"
```

---

### Task 7: Slack app setup + live smoke test (manual, needs tokens)

**Files:** none (documentation + manual verification). Also create: `docs/slack-setup.md`

- [ ] **Step 1: Write the Slack setup doc**

```markdown
# Slack app setup (Socket Mode)

1. Go to https://api.slack.com/apps → **Create New App** → **From scratch**. Name it (e.g. "ask_finance"), pick your workspace.
2. **Socket Mode** (left nav) → toggle **Enable Socket Mode** on. When prompted, create an **app-level token** with scope `connections:write` → copy it (`xapp-…`) → this is `SLACK_APP_TOKEN`.
3. **OAuth & Permissions** → under **Bot Token Scopes** add: `app_mentions:read`, `chat:write`.
4. **Event Subscriptions** → toggle **Enable Events** on → under **Subscribe to bot events** add `app_mention` → save.
5. **Install App** (left nav) → **Install to Workspace** → authorize → copy the **Bot User OAuth Token** (`xoxb-…`) → this is `SLACK_BOT_TOKEN`.
6. In Slack, invite the bot to a channel: `/invite @ask_finance`.
7. Put both tokens (and `OPENAI_API_KEY`) into `.env` (copy from `.env.example`).
```

- [ ] **Step 2: Fill in .env**

Copy `.env.example` to `.env` and fill `OPENAI_API_KEY`, `SLACK_BOT_TOKEN`, `SLACK_APP_TOKEN`.
(`.env` is gitignored — never commit it.)

- [ ] **Step 3: Run the bot via docker compose**

Run: `docker compose up --build`
Expected: logs show `ingest on startup: {...}` then `starting Slack Socket Mode bot…` and a Bolt "connected" line. No traceback.

- [ ] **Step 4: Ask in Slack**

In the channel where the bot is invited, type: `@ask_finance What is EV/EBIT?`
Expected: the bot replies in a thread — a "_thinking…_" ack, then a cited answer + a metrics line.

- [ ] **Step 5: Check stats in Slack**

Type: `@ask_finance stats`
Expected: the bot replies with the aggregate metrics summary.

- [ ] **Step 6: Commit the setup doc**

```bash
git add docs/slack-setup.md
git commit -m "docs: Slack app setup (Socket Mode) + live smoke steps"
```

---

## Notes for the implementer

- **Tokens are only needed for Tasks 3 (final check is token-optional), 6 step 3, and 7.** Tasks 1–2 (the real logic) are fully tested with fakes — no tokens, no network.
- **`say(text=..., thread_ts=...)`**: Bolt's `say` posts to the event's channel; passing `thread_ts=event["ts"]` threads the reply under the mention. The handler always threads.
- **Concurrency:** the single bot container owns `data/qdrant` (one process), so the notebook/CLI lock friction does not apply inside Docker. Heavy simultaneous questions would need a Qdrant server — out of scope (see spec).
- **Do not** re-run `ingest --force` in the container start command; the entrypoint uses warm `ingest()` so restarts are instant.
- **The notebook and local CLI still work** unchanged; only new files are added.
