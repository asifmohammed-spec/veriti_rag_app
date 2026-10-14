"""Ingestion as a LangGraph graph:  load -> split -> index

`index` compares chunk ids with what is already stored: it embeds and adds
only new chunks, and deletes stale ones (edited text, removed sections or
files). Re-running on unchanged files embeds nothing."""
from pathlib import Path
from typing import TypedDict

import yaml
from langgraph.graph import END, START, StateGraph

from .chunking import load_documents, split_documents
from .config import Settings


class IngestState(TypedDict, total=False):
    client_id: str
    docs: list
    chunks: list
    report: dict


def _version(client_dir: Path) -> str:
    meta = client_dir / "_meta.yml"
    return str((yaml.safe_load(meta.read_text()) or {}).get("version", "v1")) if meta.exists() else "v1"


def build_ingest_graph(s: Settings, vs, batch: int = 128):
    def load(st):
        d = Path(s.kb_dir) / st["client_id"]
        if not d.is_dir():
            raise FileNotFoundError(f"No knowledge base folder: {d}")
        return {"docs": load_documents(d, st["client_id"], _version(d))}

    def split(st):
        return {"chunks": split_documents(st["docs"], s.chunk_size, s.chunk_overlap)}

    def index(st):
        cid = st["client_id"]
        existing = set(vs.get(where={"client_id": cid}, include=[])["ids"])
        want = {c.metadata["chunk_id"]: c for c in st["chunks"]}
        stale = sorted(existing - want.keys())
        new = [c for i, c in want.items() if i not in existing]
        if stale:
            vs.delete(ids=stale)
        for i in range(0, len(new), batch):  # embedding happens inside add_documents
            part = new[i:i + batch]
            vs.add_documents(part, ids=[c.metadata["chunk_id"] for c in part])
        return {"report": {"client": cid, "chunks": len(want), "added": len(new),
                           "deleted": len(stale), "unchanged": len(want) - len(new)}}

    g = StateGraph(IngestState)
    g.add_node("load", load)
    g.add_node("split", split)
    g.add_node("index", index)
    g.add_edge(START, "load")
    g.add_edge("load", "split")
    g.add_edge("split", "index")
    g.add_edge("index", END)
    return g.compile()


def ingest_all(s: Settings, vs, client_ids=None) -> list[dict]:
    graph = build_ingest_graph(s, vs)
    clients = client_ids or sorted(p.name for p in Path(s.kb_dir).iterdir() if p.is_dir())
    return [graph.invoke({"client_id": c})["report"] for c in clients]
