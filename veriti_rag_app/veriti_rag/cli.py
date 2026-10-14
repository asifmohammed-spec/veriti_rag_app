"""python -m veriti_rag.cli ingest | ask "question" | eval FILE | stats"""
import argparse

from .config import Settings
from .models import make_embeddings
from .store import get_vectorstore


def main(argv=None):
    ap = argparse.ArgumentParser(prog="veriti_rag")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest")
    sub.add_parser("stats")
    a = sub.add_parser("ask"); a.add_argument("question"); a.add_argument("--show-context", action="store_true")
    e = sub.add_parser("eval"); e.add_argument("file"); e.add_argument("--k", type=int, default=5)
    args = ap.parse_args(argv)

    s = Settings()
    vs = get_vectorstore(s, make_embeddings(s))

    if args.cmd == "ingest":
        from .ingest import ingest_all
        for r in ingest_all(s, vs):
            print(r)
    elif args.cmd == "stats":
        print(f"chunks in index: {vs._collection.count()}  embeddings: {s.embeddings}/{s.embed_model}")
    elif args.cmd == "eval":
        from .evaluate import evaluate
        r = evaluate(vs, s.client_id, args.file, args.k)
        print(f"hit@{args.k}={r['hit_rate']}  MRR={r['mrr']}  (n={r['n']})")
        for q in r["misses"]:
            print("  MISS:", q)
    else:
        from .db import load_csvs
        from .graph import build_graph
        from .models import make_llm
        app = build_graph(load_csvs(s.data_dir), vs, make_llm(s), s)
        out = app.invoke({"question": args.question, "client_id": s.client_id})
        if args.show_context:
            for h in out["context"]:
                print(f"[{h.ref}] {h.doc.metadata.get('source')}  score={h.score:.3f}")
        print(f"\nintent={out['intent']}  grounded={out['grounded']}  trace={out['trace']}")
        if out.get("sql"):
            print(f"SQL: {out['sql']}")
        print(f"\n{out['answer']}\n")
        for c in out["citations"]:
            print(f"  [{c['ref']}] {c['source']}")


if __name__ == "__main__":
    main()
