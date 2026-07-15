import logging

from ragchat.answering.generator import answer
from ragchat.common.clients import wait_for_qdrant
from ragchat.common.config import settings
from ragchat.metrics.registry import get_registry
from ragchat.queue import get_queue
from ragchat.slack.poster import SlackPoster
from ragchat.worker.processor import run_once

log = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not settings.queue_url:
        raise SystemExit("Missing QUEUE_URL. The worker has no queue to drain.")
    if not settings.slack_bot_token:
        raise SystemExit("Missing SLACK_BOT_TOKEN (xoxb-…). The worker cannot post replies.")

    wait_for_qdrant()
    queue, poster, registry = get_queue(), SlackPoster(), get_registry()
    log.info("worker draining %s", settings.queue_url)
    while True:
        run_once(queue, answer, poster, registry)


if __name__ == "__main__":
    main()
