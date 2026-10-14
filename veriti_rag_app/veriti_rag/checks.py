"""Deterministic grounding check: every number in the answer must be traceable
to the query result, the question, or the retrieved context. The LLM is never
trusted to do arithmetic; if it states a number we can't find, we flag it.

Known limits: abbreviations like '$1.2M' are not normalised, and a number that
coincidentally appears elsewhere in the context will pass."""
import re
from decimal import Decimal

CITE = re.compile(r"\[[^\]]*\]")
IDS = re.compile(r"\b[A-Z]{2,5}-[A-Z]{2,5}-\d{1,4}\b")
DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
LIST_MARK = re.compile(r"(?m)^\s*\d+[.)]\s")
CODE = re.compile(r"```.*?```|`[^`]*`", re.S)
NUM = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)(\s?%)?")


def _clean(text: str) -> str:
    for rx in (CODE, CITE, IDS, DATE, LIST_MARK):
        text = rx.sub(" ", text)
    return text


def _numbers_in(text: str) -> list[float]:
    return [float(m.group(1).replace(",", "")) for m in NUM.finditer(text)]


def _allowed(rows, allowed_text: str) -> list[float]:
    vals = [float(len(rows))]
    for row in rows:
        for cell in row:
            if isinstance(cell, bool) or cell is None:
                continue
            if isinstance(cell, (int, float, Decimal)):
                vals.append(abs(float(cell)))
            else:
                vals += _numbers_in(str(cell))
    vals += _numbers_in(allowed_text)
    return vals


def unverified_numbers(answer: str, rows: list[list], allowed_text: str = "") -> list[str]:
    allowed = _allowed(rows, allowed_text)
    bad = []
    for m in NUM.finditer(_clean(answer)):
        raw, pct = m.group(1), bool(m.group(2))
        x = float(raw.replace(",", ""))
        dec = len(raw.split(".")[1]) if "." in raw else 0
        tol = 0.5 * 10 ** (-dec) + 1e-9
        cands = lambda v: (v, v * 100) if pct else (v,)
        if not any(abs(c - x) <= tol for v in allowed for c in cands(v)):
            bad.append(raw + ("%" if pct else ""))
    return list(dict.fromkeys(bad))
