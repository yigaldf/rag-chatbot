# Slack app setup (Socket Mode)

1. Go to https://api.slack.com/apps → **Create New App** → **From scratch**. Name it (e.g. "ask_finance"), pick your workspace.
2. **Socket Mode** (left nav) → toggle **Enable Socket Mode** on. When prompted, create an **app-level token** with scope `connections:write` → copy it (`xapp-…`) → this is `SLACK_APP_TOKEN`.
3. **OAuth & Permissions** → under **Bot Token Scopes** add: `app_mentions:read`, `chat:write`.
4. **Event Subscriptions** → toggle **Enable Events** on → under **Subscribe to bot events** add `app_mention` → save.
5. **Install App** (left nav) → **Install to Workspace** → authorize → copy the **Bot User OAuth Token** (`xoxb-…`) → this is `SLACK_BOT_TOKEN`.
6. In Slack, invite the bot to a channel: `/invite @ask_finance`.
7. Put both tokens (and `OPENAI_API_KEY`) into `.env` (copy from `.env.example`).

## Run the bot

```bash
docker compose up --build
```

Expected logs: `ingest on startup: {...}` then `starting Slack Socket Mode bot…` and a Bolt "connected" line.

Then in the channel where the bot is invited:

- `@ask_finance What is EV/EBIT?` → replies in a thread with a cited answer + a metrics line.
- `@ask_finance Compare EV/EBIT and O'Shaughnessy — pros and cons` → multi-strategy cited comparison.
- `@ask_finance stats` → aggregate usage metrics.

## Notes

- Only **one** process may hold `data/qdrant` at a time. Inside Docker the bot container owns its own volume, so there's no conflict with the local notebook/CLI (those use the repo's `data/qdrant`).
- Secrets live only in `.env` (gitignored) and are injected at runtime — never baked into the image.
- To run without Docker: `uv run ragchat-bot` (reads the same `.env`).
