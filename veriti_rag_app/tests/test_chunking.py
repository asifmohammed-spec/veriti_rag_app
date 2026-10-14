from pathlib import Path
from veriti_rag.chunking import load_documents, split_documents

KB = Path("knowledge_base/maven_demo")


def chunks(source=None, size=800, overlap=100):
    docs = [d for d in load_documents(KB, "maven_demo") if source in (None, d.metadata["source"])]
    return split_documents(docs, size, overlap)


def test_one_chunk_per_unit_with_metadata():
    assert len(chunks("drc_rules.yml")) == 4        # one per rule
    assert len(chunks("schema.yml")) == 4           # one per table
    assert len(chunks("data_dictionary.csv")) == 10 # one per row
    assert len(chunks("remediation_guide.md")) == 4 # one per section; heading-only fragment dropped
    c = next(c for c in chunks("drc_rules.yml") if c.metadata["rule_id"] == "DRC-COMP-002")
    assert c.metadata["doc_type"] == "drc_rule" and "close_value" in c.page_content


def test_markdown_section_gets_rule_id_from_heading():
    c = next(c for c in chunks("remediation_guide.md") if c.metadata.get("rule_id") == "DRC-INTG-001")
    assert c.metadata["doc_type"] == "remediation" and "Orphan" in c.page_content


def test_schema_chunk_keeps_table_name_and_foreign_keys():
    c = next(c for c in chunks("schema.yml") if c.metadata["table"] == "sales_pipeline")
    assert "FK account -> accounts.account" in c.page_content


def test_long_text_respects_chunk_size():
    out = chunks("business_rules.md", size=120, overlap=20)
    assert len(out) > 3 and all(len(c.page_content) <= 120 for c in out)


def test_ids_stable_and_unique():
    a, b = chunks(), chunks()
    ids = [c.metadata["chunk_id"] for c in a]
    assert ids == [c.metadata["chunk_id"] for c in b] and len(set(ids)) == len(ids) == 33
