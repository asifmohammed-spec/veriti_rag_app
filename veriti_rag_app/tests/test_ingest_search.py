from pathlib import Path
from veriti_rag.evaluate import evaluate
from veriti_rag.ingest import ingest_all
from veriti_rag.store import search


def run(env):
    s, vs = env
    return ingest_all(s, vs)[0]


def test_ingest_is_incremental(env):
    s, vs = env
    first = run(env)
    assert first["added"] == first["chunks"] == 33 and first["deleted"] == 0
    again = run(env)
    assert again["added"] == 0 and again["unchanged"] == 33
    p = next(Path(s.kb_dir).rglob("drc_rules.yml"))
    p.write_text(p.read_text().replace("Won deals must have a close value",
                                       "Won deals must carry a close value"))
    third = run(env)
    assert third["added"] == 1 and third["deleted"] == 1   # edited chunk replaced, rest untouched


def test_deleted_file_is_purged(env):
    s, vs = env
    run(env)
    (Path(s.kb_dir) / "maven_demo" / "bsd_definitions.md").unlink()
    r = run(env)
    assert r["deleted"] == 3 and r["chunks"] == 30


def test_client_isolation(env):
    s, vs = env
    run(env)
    assert search(vs, "close value", "maven_demo")
    assert search(vs, "close value", "another_client") == []


def test_search_finds_rule_chunk(env):
    s, vs = env
    run(env)
    docs = [d for d, _ in search(vs, "DRC-COMP-002 won deals missing close value", "maven_demo", 3)]
    assert any(d.metadata.get("rule_id") == "DRC-COMP-002" for d in docs)


def test_eval_runner(env):
    s, vs = env
    run(env)
    r = evaluate(vs, "maven_demo", "evaluation/retrieval_eval.jsonl", k=5)
    assert r["n"] == 7 and r["hit_rate"] >= 0.5, r   # floor for the offline word-overlap embedder
