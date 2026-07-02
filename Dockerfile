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

# Put the synced venv on PATH so console scripts run directly — no `uv run`
# at runtime (which would re-sync and need network on every start).
ENV PATH="/app/.venv/bin:$PATH"

# Data root inside the container; docker-compose mounts a volume here.
ENV DATA_DIR=/app/data

# Run the Slack bot: ingest (warm-skip) then start Socket Mode.
CMD ["ragchat-bot"]
