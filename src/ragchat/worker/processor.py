import logging

from ragchat.common.config import settings
from ragchat.slack import messages
from ragchat.slack.formatting import format_answer

log = logging.getLogger(__name__)


def process_job(job, answer_fn, poster, registry) -> None:
    """Answer one question and post the reply. Raises on any failure."""
    result = answer_fn(job.text)
    registry.record(result.metrics)
    poster.post(job.channel, job.thread_ts, format_answer(result))


def _post_error(poster, job) -> None:
    """Best-effort. A dead Slack must not take the worker loop down with it."""
    try:
        poster.post(job.channel, job.thread_ts, messages.ERROR)
    except Exception:
        log.exception("failed to post the error message for %s", job.event_id)


def run_once(queue, answer_fn, poster, registry, max_receive_count: int | None = None) -> None:
    """Drain one batch. Delete ONLY on success, so failures get redelivered."""
    max_receive_count = settings.max_receive_count if max_receive_count is None else max_receive_count
    for rj in queue.receive(wait_seconds=settings.worker_wait_seconds):
        try:
            process_job(rj.job, answer_fn, poster, registry)
            queue.delete(rj.receipt)
        except Exception:
            log.exception("job failed: %s", rj.job.event_id)
            # Last attempt: tell the user, or they stare at "_thinking…_" forever.
            # Never delete — let the message fall into the DLQ for inspection.
            if rj.receive_count >= max_receive_count:
                _post_error(poster, rj.job)
