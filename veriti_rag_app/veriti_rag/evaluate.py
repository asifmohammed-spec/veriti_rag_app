"""Retrieval evaluation. Each line of the JSONL file:
  {"question": "...", "expected": [{"source": "drc_rules.yml", "contains": "DRC-COMP-002"}]}
A retrieved chunk matches when its source equals `source` and its text contains
`contains` (case-insensitive). Reports hit-rate@k and MRR."""
import json
from pathlib import Path

from .store import search


def evaluate(vs, client_id: str, path: str, k: int = 5) -> dict:
    cases = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    hits, rr, misses = 0, 0.0, []
    for c in cases:
        res = search(vs, c["question"], client_id, k)
        rank = next((i for i, (d, _) in enumerate(res, 1)
                     if any(d.metadata.get("source") == e["source"]
                            and e.get("contains", "").lower() in d.page_content.lower()
                            for e in c["expected"])), None)
        hits += rank is not None
        rr += 1.0 / rank if rank else 0.0
        if rank is None:
            misses.append(c["question"])
    n = max(len(cases), 1)
    return {"hit_rate": round(hits / n, 3), "mrr": round(rr / n, 3), "n": len(cases), "misses": misses}
