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
