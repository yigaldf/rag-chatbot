import logging

from ragchat.ingestion.ingest import ingest
from ragchat.slack.app import run_socket_mode


def main() -> None:
    # INFO-level logging so slack_bolt reports the Socket Mode connection and
    # every incoming event (otherwise the bot runs silently and is undebuggable).
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    summary = ingest()  # warm-skip unchanged docs; builds the index on first run
    print(f"ingest on startup: {summary}", flush=True)
    print("starting Slack Socket Mode bot… (Ctrl-C to stop)", flush=True)
    run_socket_mode()


if __name__ == "__main__":
    main()
