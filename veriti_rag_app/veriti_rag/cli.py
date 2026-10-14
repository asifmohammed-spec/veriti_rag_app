"""python -m veriti_rag.cli ingest | ask "question" | eval FILE | stats"""
import argparse

from .config import Settings
from .models import make_embeddings
from .store import get_vectorstore


def save_chart(result, path="chart.png"):
    rows = result.get("rows") or []
    columns = result.get("columns") or []

    if len(columns) != 2 or not rows:
        raise ValueError("Chart requires a 2-column result set.")

    labels = []
    values = []

    for row in rows:
        if len(row) < 2:
            continue
        labels.append(str(row[0]))
        try:
            values.append(float(row[1]))
        except (TypeError, ValueError) as exc:
            raise ValueError("Chart requires a numeric second column.") from exc

    if not labels:
        raise ValueError("Chart requires at least one data row.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.8), 5))
    ax.bar(labels, values, color="#2563eb")
    ax.set_xlabel(columns[0])
    ax.set_ylabel(columns[1])
    ax.set_title(f"{columns[1]} by {columns[0]}")
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="veriti_rag")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest")
    sub.add_parser("stats")

    a = sub.add_parser("ask")
    a.add_argument("question")
    a.add_argument("--show-context", action="store_true")
    a.add_argument("--chart", nargs="?", const="chart.png", default=None, metavar="PATH")

    e = sub.add_parser("eval")
    e.add_argument("file")
    e.add_argument("--k", type=int, default=5)
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

        if args.chart:
            try:
                chart_path = save_chart(out, args.chart)
                print(f"\nChart saved to {chart_path}")
            except ValueError as exc:
                print(f"\nChart not generated: {exc}")

        for c in out["citations"]:
            print(f"  [{c['ref']}] {c['source']}")


if __name__ == "__main__":
    main()