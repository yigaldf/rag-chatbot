import argparse

from ragchat.answering.generator import answer
from ragchat.ingestion.ingest import ingest
from ragchat.metrics.registry import get_registry


def _print_answer(r):
    print(r.text)
    if r.found and r.sources:
        print("\nSources:")
        for s in r.sources:
            print("  -", s)
    m = r.metrics
    print(f"\nMetrics: {m.sub_queries} sub-quer{'y' if m.sub_queries == 1 else 'ies'} · "
          f"{m.chunks_used} chunk(s) · {m.total_tokens} tokens "
          f"(embed {m.embed_tokens} + chat {m.chat_tokens})")


def _print_stats(s):
    print("Aggregate metrics")
    print(f"  questions   : {s.num_questions}")
    print(f"  sub-queries : total {s.total_sub_queries}, avg {s.avg_sub_queries:.2f}")
    print(f"  chunks      : total {s.total_chunks}, avg {s.avg_chunks:.2f}")
    print(f"  tokens      : total {s.total_tokens}, avg {s.avg_tokens:.1f} "
          f"(embed {s.total_embed_tokens} + chat {s.total_chat_tokens})")


def main(argv=None):
    p = argparse.ArgumentParser(prog="ragchat")
    sub = p.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("ingest")
    pi.add_argument("--force", action="store_true")
    pa = sub.add_parser("ask")
    pa.add_argument("question")
    sub.add_parser("stats")
    args = p.parse_args(argv)

    if args.cmd == "ingest":
        print(ingest(force=args.force))
    elif args.cmd == "ask":
        r = answer(args.question)
        get_registry().record(r.metrics)
        _print_answer(r)
    elif args.cmd == "stats":
        _print_stats(get_registry().stats())


if __name__ == "__main__":
    main()
