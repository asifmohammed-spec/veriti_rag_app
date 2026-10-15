import json
from veriti_rag import graph as G
from veriti_rag.db import load_csvs, run_select
from veriti_rag.ingest import ingest_all


def make_llm(intent="data", first_sql_bad=False, bad_answer_first=False, expected_count=1):
    log = {"sql": 0, "answer": 0}

    def llm(system, user):
        if system == G.ANALYZE_SYSTEM:
            return json.dumps({"intent": intent, "search_query": "won deals missing close value DRC-COMP-002"})
        if system == G.SQL_SYSTEM:
            log["sql"] += 1
            if first_sql_bad and log["sql"] == 1:
                return json.dumps({"sql": "SELECT COUNT(*) FROM nope", "rationale": "x"})
            return json.dumps({"sql": "SELECT COUNT(*) AS n FROM sales_pipeline "
                                      "WHERE LOWER(TRIM(deal_stage))='won' AND close_value IS NULL", "rationale": "r"})
        if system == G.ANSWER_SYSTEM:
            log["answer"] += 1
            if intent == "knowledge":
                return "Win rate is won deals over closed deals [S1]."
            if bad_answer_first and "not in the query result" not in user:
                return "There are 7 won deals without a close value [S1]."
            return f"There are {expected_count} won deals without a close value [S1]."
        raise AssertionError("unexpected prompt")
    return llm, log


def run(env, llm, q="How many won deals have no close value?"):
    s, vs = env
    ingest_all(s, vs)
    app = G.build_graph(load_csvs("data/raw"), vs, llm, s)
    return app.invoke({"question": q, "client_id": "maven_demo"})


def test_sql_guard_blocks_writes():
    con = load_csvs("data/raw")
    for bad in ["DROP TABLE accounts", "SELECT 1; DELETE FROM accounts", "SET threads=1"]:
        try:
            run_select(con, bad)
            raise AssertionError("not blocked")
        except ValueError:
            pass


def test_data_question_with_sql_retry(env):
    data = load_csvs("data/raw")
    expected_count = data.execute(
        "SELECT COUNT(*) FROM sales_pipeline WHERE LOWER(TRIM(deal_stage))='won' AND close_value IS NULL"
    ).fetchone()[0]
    data.close()
    llm, log = make_llm(first_sql_bad=True, expected_count=expected_count)
    out = run(env, llm)
    assert log["sql"] == 2 and out["rows"] == [[expected_count]] and out["error"] is None
    assert out["grounded"] and out["citations"][0]["ref"] == "S1"
    assert [t.split()[0] for t in out["trace"]][:2] == ["analyze", "retrieve"]


def test_invented_number_triggers_regeneration(env):
    data = load_csvs("data/raw")
    expected_count = data.execute(
        "SELECT COUNT(*) FROM sales_pipeline WHERE LOWER(TRIM(deal_stage))='won' AND close_value IS NULL"
    ).fetchone()[0]
    data.close()
    llm, log = make_llm(bad_answer_first=True, expected_count=expected_count)
    out = run(env, llm)
    assert log["answer"] == 2 and out["grounded"] and "7" not in out["answer"]


def test_knowledge_question_skips_sql(env):
    llm, log = make_llm(intent="knowledge")
    out = run(env, llm, "What is the win rate definition?")
    assert log["sql"] == 0 and out["sql"] is None and out["grounded"]
