"""LangGraph query pipeline:

  analyze -> retrieve -> knowledge question: answer -> verify -> END
                      -> data/both question: plan_sql <-> execute_sql -> answer -> verify -> END
  (verify sends the answer back for regeneration if it contains unverified numbers)

The LLM (via OpenRouter) writes SQL and prose. DuckDB computes every number.
`verify` checks that every number in the prose appears in the query result.
"""
import functools
import json
import operator
import re
import time
from typing import Annotated, Callable, NamedTuple, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from .checks import unverified_numbers
from .config import Settings
from .db import describe_schema, run_select
from .store import search

ANALYZE_SYSTEM = """You triage questions about a company's CRM data.
Classify the question:
- "data": needs figures computed from the tables
- "knowledge": about definitions, rules, mappings or remediation; no computation
- "both": needs figures AND business context
Also write a keyword-rich search_query for a knowledge-base search (field names, rule ids, business terms).
Respond with JSON only: {"intent": "data"|"knowledge"|"both", "search_query": string}"""

SQL_SYSTEM = """You are a CRM data analyst. Write DuckDB SQL to answer the question.
Rules:
- Use ONLY tables and columns in the schema. Use the context for field meanings and business rules.
- Compute every figure in SQL (counts, sums, percentages, rates). Do not leave arithmetic for later.
- Read-only: a single SELECT (or WITH ... SELECT).
- If no data is needed, set sql to null.
Respond with JSON only: {"sql": string|null, "rationale": string}"""

ANSWER_SYSTEM = """You are a CRM data-quality and sales analyst.
Rules:
- Every number you state MUST appear in the query result. Never compute, estimate or invent figures.
- Cite context with its ref in square brackets, e.g. [S2]. Only cite chunks you actually used.
- If the query failed, returned nothing, or the context lacks the answer, say so plainly.
- Be concise: lead with the answer, then supporting points, then recommendations if relevant."""


class Hit(NamedTuple):
    ref: str
    doc: Document
    score: float


class State(TypedDict, total=False):
    question: str
    client_id: str
    intent: str
    search_query: str
    context: list
    sql: str | None
    rationale: str
    columns: list
    rows: list
    error: str | None
    sql_attempts: int
    answer: str
    answer_attempts: int
    grounded: bool
    unverified: list
    citations: list
    trace: Annotated[list, operator.add]


def parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    return json.loads(text[text.find("{"): text.rfind("}") + 1])


def _ctx(hits: list[Hit]) -> str:
    if not hits:
        return "(none)"
    return "\n\n".join(f"[{h.ref}] {h.doc.metadata.get('source')} | "
                       f"{h.doc.metadata.get('doc_type')}\n{h.doc.page_content}" for h in hits)


def build_graph(con, vs, llm: Callable[[str, str], str], settings: Settings, checkpointer=None):
    schema = describe_schema(con)

    def traced(name):
        def deco(fn):
            @functools.wraps(fn)
            def wrapper(s):
                t = time.perf_counter()
                out = dict(fn(s))
                out["trace"] = [f"{name} {1000 * (time.perf_counter() - t):.0f}ms"]
                return out
            return wrapper
        return deco

    @traced("analyze")
    def analyze(s):
        base = {"sql_attempts": 0, "answer_attempts": 0, "error": None, "unverified": [],
                "sql": None, "rows": [], "columns": []}
        try:
            out = parse_json(llm(ANALYZE_SYSTEM, s["question"]))
            intent = out.get("intent", "both")
            intent = intent if intent in {"data", "knowledge", "both"} else "both"
            q = out.get("search_query") or s["question"]
        except Exception:
            intent, q = "both", s["question"]
        return {**base, "intent": intent, "search_query": q}

    @traced("retrieve")
    def retrieve(s):
        found = search(vs, s["search_query"], s["client_id"], settings.top_k)
        return {"context": [Hit(f"S{i}", d, sc) for i, (d, sc) in enumerate(found, 1)]}

    @traced("plan_sql")
    def plan_sql(s):
        user = f"SCHEMA:\n{schema}\n\nCONTEXT:\n{_ctx(s['context'])}\n\nQUESTION: {s['question']}"
        if s.get("error"):
            user += f"\n\nYour previous attempt failed.\nSQL: {s.get('sql')}\nError: {s['error']}\nFix it."
        n = s["sql_attempts"] + 1
        try:
            out = parse_json(llm(SQL_SYSTEM, user))
            return {"sql": out.get("sql"), "rationale": out.get("rationale", ""),
                    "sql_attempts": n, "error": None}
        except Exception as e:
            return {"sql": None, "error": f"Could not parse model output: {e}", "sql_attempts": n}

    @traced("execute_sql")
    def execute_sql(s):
        try:
            cols, rows = run_select(con, s["sql"])
            return {"columns": cols, "rows": rows, "error": None}
        except Exception as e:
            return {"error": str(e), "columns": [], "rows": []}

    def _result_ok(s):
        return bool(s.get("sql")) and not s.get("error")

    @traced("answer")
    def answer(s):
        if _result_ok(s):
            data = f"SQL: {s['sql']}\nColumns: {s['columns']}\nRows (max 50): {s['rows'][:50]}"
        elif s.get("error"):
            data = f"The query could not be completed. Last error: {s['error']}"
        else:
            data = "No query was run."
        user = f"QUESTION: {s['question']}\n\nCONTEXT:\n{_ctx(s['context'])}\n\nQUERY RESULT:\n{data}"
        if s.get("unverified"):
            user += (f"\n\nYour previous answer contained numbers that are not in the query result: "
                     f"{s['unverified']}. Rewrite it using only numbers from the result.")
        return {"answer": llm(ANSWER_SYSTEM, user), "answer_attempts": s["answer_attempts"] + 1}

    @traced("verify")
    def verify(s):
        rows = s["rows"] if _result_ok(s) else []
        allowed = s["question"] + " " + " ".join(h.doc.page_content for h in s["context"])
        bad = unverified_numbers(s["answer"], rows, allowed)
        text = s["answer"]
        if bad and s["answer_attempts"] >= settings.max_answer_attempts:
            text += f"\n\nWarning: these figures could not be verified against the data: {', '.join(bad)}"
        used = set(re.findall(r"\[(S\d+)\]", text))
        cites = [{"ref": h.ref, "chunk_id": h.doc.metadata.get("chunk_id"),
                  "source": h.doc.metadata.get("source")} for h in s["context"] if h.ref in used]
        return {"answer": text, "unverified": bad, "grounded": not bad, "citations": cites}

    def after_retrieve(s):
        return "answer" if s["intent"] == "knowledge" else "plan_sql"

    def after_plan(s):
        if s.get("error"):
            return "answer" if s["sql_attempts"] >= settings.max_sql_attempts else "plan_sql"
        return "execute_sql" if s.get("sql") else "answer"

    def after_execute(s):
        if s.get("error") and s["sql_attempts"] < settings.max_sql_attempts:
            return "plan_sql"
        return "answer"

    def after_verify(s):
        if not s["grounded"] and s["answer_attempts"] < settings.max_answer_attempts:
            return "answer"
        return END

    g = StateGraph(State)
    for name, fn in [("analyze", analyze), ("retrieve", retrieve), ("plan_sql", plan_sql),
                     ("execute_sql", execute_sql), ("answer", answer), ("verify", verify)]:
        g.add_node(name, fn)
    g.add_edge(START, "analyze")
    g.add_edge("analyze", "retrieve")
    g.add_conditional_edges("retrieve", after_retrieve, ["plan_sql", "answer"])
    g.add_conditional_edges("plan_sql", after_plan, ["plan_sql", "execute_sql", "answer"])
    g.add_conditional_edges("execute_sql", after_execute, ["plan_sql", "answer"])
    g.add_edge("answer", "verify")
    g.add_conditional_edges("verify", after_verify, ["answer", END])
    return g.compile(checkpointer=checkpointer)
