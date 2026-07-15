from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from ragchat.common.config import settings
from ragchat.slack.handlers import handle_mention


def build_app() -> App:
    """Construct the Bolt app and register the app_mention handler."""
    app = App(token=settings.slack_bot_token)

    @app.event("app_mention")
    def _on_mention(event, body, say, context):
        # `body["event_id"]` is the dedup key; `say` posts to the event's channel.
        handle_mention(event, body, say, context["bot_user_id"])

    return app


def run_socket_mode() -> None:
    if not settings.slack_bot_token or not settings.slack_app_token:
        raise SystemExit(
            "Missing Slack tokens. Set SLACK_BOT_TOKEN (xoxb-…) and SLACK_APP_TOKEN (xapp-…) "
            "in .env or the environment."
        )
    app = build_app()
    SocketModeHandler(app, settings.slack_app_token).start()
