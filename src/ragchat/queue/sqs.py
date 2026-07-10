import boto3

from ragchat.common.config import settings
from ragchat.queue.models import AskJob, ReceivedJob


class SqsJobQueue:
    """Thin boto3 wrapper. Identical against ElasticMQ and real AWS SQS —
    only `endpoint_url` differs."""

    def __init__(self, client, queue_url: str):
        self._client = client
        self._url = queue_url

    def send(self, job: AskJob) -> None:
        self._client.send_message(
            QueueUrl=self._url,
            MessageBody=job.model_dump_json(),
            # Serialize one Slack thread; let different threads run in parallel.
            MessageGroupId=f"{job.channel}:{job.thread_ts}",
            # Slack retries the same event_id; FIFO dedup collapses them.
            MessageDeduplicationId=job.event_id,
        )

    def receive(self, wait_seconds: int = 20) -> list[ReceivedJob]:
        resp = self._client.receive_message(
            QueueUrl=self._url,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=wait_seconds,
            AttributeNames=["ApproximateReceiveCount"],
        )
        return [
            ReceivedJob(
                job=AskJob.model_validate_json(m["Body"]),
                receipt=m["ReceiptHandle"],
                receive_count=int(m["Attributes"]["ApproximateReceiveCount"]),
            )
            for m in resp.get("Messages", [])
        ]

    def delete(self, receipt: str) -> None:
        self._client.delete_message(QueueUrl=self._url, ReceiptHandle=receipt)


def build_sqs_queue() -> SqsJobQueue:
    client = boto3.client(
        "sqs",
        endpoint_url=settings.queue_endpoint_url or None,  # None => real AWS
        region_name=settings.aws_region,
    )
    return SqsJobQueue(client, settings.queue_url)
