"""python -m veriti_rag.cli ingest | ask "question" | report | sales-report | eval FILE | stats"""
import argparse
import json
import re
from pathlib import Path
from uuid import uuid4

from .clients import settings_for_client
from .config import Settings
from .db import load_client_csvs
from .models import make_embeddings, make_llm
from .reports import create_dataset_report, create_sales_report
from .sales_report_agent import run_sales_report_agent
from .tracing import thread_trace_project
from render_report import render_report_data
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


def parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"yes", "true", "1", "on"}:
        return True
    if normalized in {"no", "false", "0", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected yes/no, true/false, 1/0, or on/off")


def _versioned_report_paths(args):
    output_dir = Path(args.output_dir)
    pdf_base = Path(args.output) if args.output else output_dir / "sales_report.pdf"
    base_dir = pdf_base.parent if args.output else output_dir
    stem = pdf_base.stem
    html_base = Path(args.html_output) if args.html_output else base_dir / f"{stem}.html"
    data_base = Path(args.data_output) if args.data_output else base_dir / f"{stem}_data.json"
    extract_base = Path(args.extract_output) if args.extract_output else base_dir / f"{stem}_extracted.json"

    base_dir.mkdir(parents=True, exist_ok=True)
    version_pattern = re.compile(rf"^{re.escape(stem)}.*_v(\d+)(?:\.[^.]+)?$")
    existing_versions = [
        int(match.group(1))
        for path in base_dir.iterdir()
        if (match := version_pattern.match(path.name))
    ]
    version = max(existing_versions, default=0) + 1

    def numbered(path):
        path = Path(path)
        return path.with_name(f"{path.stem}_v{version}{path.suffix}")

    paths = {
        "pdf": numbered(pdf_base),
        "html": numbered(html_base),
        "data": numbered(data_base),
        "extract": numbered(extract_base),
    }
    while any(path.exists() for path in paths.values()):
        version += 1
        paths = {
            "pdf": numbered(pdf_base),
            "html": numbered(html_base),
            "data": numbered(data_base),
            "extract": numbered(extract_base),
        }
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(prog="veriti_rag")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ingest_parser = sub.add_parser("ingest")
    ingest_parser.add_argument("--client-id", default=None)
    stats_parser = sub.add_parser("stats")
    stats_parser.add_argument("--client-id", default=None)

    a = sub.add_parser("ask")
    a.add_argument("--client-id", default=None)
    a.add_argument("question")
    a.add_argument("--thread-id", default=None,
                   help="Thread ID for checkpointing (defaults to a new ID per run)")
    a.add_argument("--show-context", action="store_true")
    a.add_argument("--chart", nargs="?", const="chart.png", default=None, metavar="PATH")
    a.add_argument("--report", nargs="?", const="question_report.pdf", default=None, metavar="PATH")

    e = sub.add_parser("eval")
    e.add_argument("--client-id", default=None)
    e.add_argument("file")
    e.add_argument("--k", type=int, default=5)
    r = sub.add_parser("report", help="Create a full-dataset PDF insights report")
    r.add_argument("--client-id", default=None)
    r.add_argument("--output", default="dataset_insights.pdf", metavar="PATH")
    r.add_argument("--clean", nargs="?", const=True, default=False, type=parse_bool, metavar="yes|no",
                   help="Include only business insights, recommendations, and charts")
    sr = sub.add_parser("sales-report", help="Create an agentic sales performance PDF")
    sr.add_argument("--client-id", default=None)
    sr.add_argument("--thread-id", default=None,
                    help="Thread ID for checkpointing (defaults to a new ID per run)")
    sr.add_argument("--output-dir", default="report_output", metavar="DIR",
                    help="Directory for versioned report artifacts (default: report_output)")
    sr.add_argument("--output", default=None, metavar="PATH",
                    help="Optional PDF base path; outputs are still versioned")
    sr.add_argument("--data-output", default=None, metavar="PATH",
                    help="Save the template-compatible structured report JSON")
    sr.add_argument("--extract-output", default=None, metavar="PATH",
                    help="Save the deterministic DuckDB extraction JSON for validation")
    sr.add_argument("--html-output", default=None, metavar="PATH",
                    help="Save the rendered HTML report (defaults beside the PDF)")
    sr.add_argument("--prompt", default="", help="Optional extra focus for the fixed sales report")
    args = ap.parse_args(argv)

    base_settings = Settings()
    client_id = args.client_id or base_settings.client_id
    s = settings_for_client(base_settings, client_id)

    if args.cmd == "ingest":
        from .ingest import ingest_all
        vs = get_vectorstore(s, make_embeddings(s))
        for r in ingest_all(s, vs, client_ids=[s.client_id]):
            print(r)
    elif args.cmd == "stats":
        vs = get_vectorstore(s, make_embeddings(s))
        print(f"chunks in index: {vs._collection.count()}  embeddings: {s.embeddings}/{s.embed_model}")
    elif args.cmd == "eval":
        from .evaluate import evaluate
        vs = get_vectorstore(s, make_embeddings(s))
        r = evaluate(vs, s.client_id, args.file, args.k)
        print(f"hit@{args.k}={r['hit_rate']}  MRR={r['mrr']}  (n={r['n']})")
        for q in r["misses"]:
            print("  MISS:", q)
    elif args.cmd == "report":
        con = load_client_csvs(s.data_dir, s.client_id, s.duckdb_path)
        try:
            output = create_dataset_report(con, args.output, clean=args.clean)
        finally:
            con.close()
        print(f"Dataset insights report saved to {output}")
    elif args.cmd == "sales-report":
        paths = _versioned_report_paths(args)
        thread_id = args.thread_id or str(uuid4())
        con = load_client_csvs(s.data_dir, s.client_id, s.duckdb_path)
        try:
            result = run_sales_report_agent(
                con, make_llm(s), args.prompt, thread_id=thread_id, client_id=s.client_id
            )
        finally:
            con.close()
        output = create_sales_report(result, paths["pdf"])
        paths["data"].write_text(
            json.dumps(result["template_data"], indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        paths["extract"].write_text(
            json.dumps(result["report_data"], indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        render_report_data(result["template_data"], paths["html"])
        print(f"Sales report saved to {output}")
        print(f"HTML report saved to {paths['html']}")
        print(f"Template data saved to {paths['data']}")
        print(f"Extracted analysis saved to {paths['extract']}")
    else:
        from .graph import build_graph

        vs = get_vectorstore(s, make_embeddings(s))
        con = load_client_csvs(s.data_dir, s.client_id, s.duckdb_path)
        app = build_graph(con, vs, make_llm(s), s)
        thread_id = args.thread_id or str(uuid4())
        try:
            with thread_trace_project():
                out = app.invoke(
                    {"question": args.question, "client_id": s.client_id},
                    config={"configurable": {"thread_id": thread_id}},
                )
        finally:
            con.close()
        from langchain_core.tracers.langchain import wait_for_all_tracers

        wait_for_all_tracers()

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

        if args.report:
            from .reports import create_question_report
            report_path = create_question_report(out, args.report)
            print(f"\nQuestion report saved to {report_path}")

        for c in out["citations"]:
            print(f"  [{c['ref']}] {c['source']}")


if __name__ == "__main__":
    main()
