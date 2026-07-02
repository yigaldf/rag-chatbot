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
