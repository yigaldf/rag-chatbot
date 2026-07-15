from slack_sdk import WebClient

from ragchat.common.config import settings


class SlackPoster:
    """The worker's reply path. Bolt's `say` is a per-event closure, so the
    worker posts explicitly with the channel and thread it was handed."""

    def __init__(self, token: str | None = None):
        self._client = WebClient(token=token or settings.slack_bot_token)

    def post(self, channel: str, thread_ts: str, text: str) -> None:
        self._client.chat_postMessage(channel=channel, thread_ts=thread_ts, text=text)
