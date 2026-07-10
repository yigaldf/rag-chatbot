from ragchat.common.config import settings
from ragchat.queue.models import AskJob, ReceivedJob

_queue = None


def get_queue():
    """Pick the queue implementation from config. Process-wide singleton."""
    global _queue
    if _queue is None:
        if settings.queue_url:
            from ragchat.queue.sqs import build_sqs_queue  # lazy: boto3 only when needed
            _queue = build_sqs_queue()
        else:
            from ragchat.queue.memory import InMemoryJobQueue
            _queue = InMemoryJobQueue()
    return _queue


def reset_queue() -> None:
    """Test helper: drop the cached queue."""
    global _queue
    _queue = None


__all__ = ["AskJob", "ReceivedJob", "get_queue", "reset_queue"]
