import logging

from ragchat.slack.app import run_socket_mode


def main() -> None:
    # INFO-level logging so slack_bolt reports the Socket Mode connection and
    # every incoming event (otherwise the bot runs silently and is undebuggable).
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Ingest is a one-shot compose service now: the bot must never open Qdrant,
    # or it would hold the embedded directory lock and block the workers.
    print("starting Slack Socket Mode bot… (Ctrl-C to stop)", flush=True)
    run_socket_mode()


if __name__ == "__main__":
    main()
