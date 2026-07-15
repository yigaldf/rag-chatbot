from pydantic import BaseModel


class AskJob(BaseModel):
    """One question to answer. Everything the worker needs, nothing it doesn't."""
    event_id: str    # Slack envelope id — the dedup key
    channel: str     # the worker has no `say` closure; it must know where to post
    thread_ts: str
    user: str
    text: str        # mention already stripped by the producer


class ReceivedJob(BaseModel):
    job: AskJob
    receipt: str
    receive_count: int
