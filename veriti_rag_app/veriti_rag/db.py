"""DuckDB helpers: load CSVs, describe the schema, run guarded read-only SQL."""
from pathlib import Path
import re
import duckdb

FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|copy|pragma|install|load|export|set)\b", re.I)


def load_csvs(folder: str) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    files = sorted(Path(folder).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files in {folder}")
    for f in files:
        con.execute(f'CREATE TABLE "{f.stem}" AS SELECT * FROM '
                    f"read_csv_auto('{f.as_posix()}', header=true)")
    try:  # block file access from model-written SQL
        con.execute("SET enable_external_access=false")
    except Exception:
        pass
    return con


def describe_schema(con, sample_values: int = 3) -> str:
    lines = []
    tables = [r[0] for r in con.execute(
        "SELECT table_name FROM information_schema.tables ORDER BY 1").fetchall()]
    for t in tables:
        n = con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
        lines.append(f"TABLE {t} ({n} rows)")
        cols = con.execute("SELECT column_name, data_type FROM information_schema.columns "
                           "WHERE table_name = ? ORDER BY ordinal_position", [t]).fetchall()
        for c, dtype in cols:
            vals = con.execute(f'SELECT DISTINCT "{c}"::VARCHAR FROM "{t}" '
                               f'WHERE "{c}" IS NOT NULL LIMIT {sample_values}').fetchall()
            lines.append(f"  - {c} {dtype}  e.g. {[v[0] for v in vals]}")
    return "\n".join(lines)


def run_select(con, sql: str, max_rows: int = 200):
    s = sql.strip().rstrip(";")
    if ";" in s:
        raise ValueError("Only a single statement is allowed.")
    if not re.match(r"(?is)^\s*(select|with)\b", s) or FORBIDDEN.search(s):
        raise ValueError("Only read-only SELECT queries are allowed.")
    cur = con.execute(s)
    cols = [d[0] for d in cur.description]
    return cols, [list(r) for r in cur.fetchmany(max_rows)]
