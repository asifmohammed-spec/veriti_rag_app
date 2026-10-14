"""Simple chunking built on LangChain splitters.

  .md/.txt   split by headings (MarkdownHeaderTextSplitter), then size-limited
  .yml       one document per rule (rules file) or per table (schema file)
  .csv       one document per row
Then one RecursiveCharacterTextSplitter pass enforces chunk_size/overlap.
Chunk id = hash(client, source, version, text): unchanged text keeps its id,
which is what makes re-ingestion incremental.
"""
import csv
import hashlib
import io
import re
from pathlib import Path

import yaml
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

SUPPORTED = {".md", ".txt", ".yml", ".yaml", ".csv"}
RULE_RE = re.compile(r"\b[A-Z]{2,5}-[A-Z]{2,5}-\d{1,4}\b")
MIN_CHARS = 25  # drops heading-only fragments

DOC_TYPES = [("business_profile", "business_profile"), ("schema", "schema"),
             ("dictionary", "data_dictionary"), ("mapping", "field_mapping"),
             ("remediation", "remediation"), ("drc", "drc_rule"), ("rule", "business_rule"),
             ("bsd", "bsd_definition"), ("technical", "technical_profile"),
             ("handover", "handover_note")]


def doc_type_for(source: str) -> str:
    name = Path(source).stem.lower()
    return next((t for kw, t in DOC_TYPES if kw in name), "document")


def _markdown(text: str, base: dict) -> list[Document]:
    splitter = MarkdownHeaderTextSplitter(
        [("#", "h1"), ("##", "h2"), ("###", "h3")], strip_headers=False)
    out = []
    for d in splitter.split_text(text):
        meta = dict(base)
        headers = " ".join(v for k, v in d.metadata.items() if k.startswith("h"))
        meta.update({k: v for k, v in d.metadata.items()})
        if m := RULE_RE.search(headers):
            meta["rule_id"] = m.group(0)
        out.append(Document(page_content=d.page_content, metadata=meta))
    return out


def _yaml(text: str, base: dict) -> list[Document]:
    data = yaml.safe_load(text) or {}
    items: list[tuple[str, dict]] = []
    if isinstance(data, dict) and isinstance(data.get("rules"), list):
        for r in data["rules"]:
            items.append((yaml.safe_dump(r, sort_keys=False), {"rule_id": str(r.get("id", ""))}))
    elif isinstance(data, dict) and isinstance(data.get("tables"), dict):
        for t, spec in data["tables"].items():
            lines = [f"Table {t} (primary key: {spec.get('primary_key', '-')})"]
            lines += [f"FK {fk['column']} -> {fk['references']}" for fk in spec.get("foreign_keys", [])]
            for c, cs in (spec.get("columns") or {}).items():
                lines.append(f"- {c}: " + (", ".join(f"{k}={v}" for k, v in cs.items())
                                            if isinstance(cs, dict) else str(cs)))
            items.append(("\n".join(lines), {"table": t}))
    else:
        items.append((text, {}))
    return [Document(page_content=t, metadata={**base, **m}) for t, m in items]


def _csv(text: str, base: dict) -> list[Document]:
    docs = []
    for i, row in enumerate(csv.DictReader(io.StringIO(text)), 1):
        body = "\n".join(f"{k.strip()}: {v.strip()}" for k, v in row.items() if k and v and v.strip())
        if body:
            docs.append(Document(page_content=body, metadata={**base, "row": i}))
    return docs


def load_documents(client_dir: Path, client_id: str, version: str = "v1") -> list[Document]:
    docs = []
    for f in sorted(client_dir.rglob("*")):
        if f.suffix.lower() not in SUPPORTED or f.name.startswith("_"):
            continue
        source = f.relative_to(client_dir).as_posix()
        base = {"client_id": client_id, "source": source,
                "doc_type": doc_type_for(source), "version": version}
        text = f.read_text(encoding="utf-8")
        loader = {".yml": _yaml, ".yaml": _yaml, ".csv": _csv}.get(f.suffix.lower(), _markdown)
        docs += loader(text, base)
    return docs


def split_documents(docs: list[Document], size: int = 800, overlap: int = 100) -> list[Document]:
    chunks = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap).split_documents(docs)
    unique: dict[str, Document] = {}
    for c in chunks:
        if len(c.page_content.strip()) < MIN_CHARS:
            continue
        m = c.metadata
        key = f"{m['client_id']}|{m['source']}|{m['version']}|{c.page_content}"
        cid = hashlib.sha1(key.encode()).hexdigest()[:16]
        m["chunk_id"] = cid
        unique.setdefault(cid, c)
    return list(unique.values())
