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


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _table_identifier(client_id: str, source_name: str) -> str:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", client_id):
        raise ValueError("Invalid client ID.")
    safe_stem = re.sub(r"[^a-zA-Z0-9_]+", "_", source_name).strip("_").lower()
    if not safe_stem:
        raise ValueError(f"CSV filename has no safe table name: {source_name}")
    return f"{client_id}__{safe_stem}"


def ingest_csvs_to_shared(folder: str, client_id: str, database_path: str) -> list[dict]:
    """Upsert this client's CSVs into the shared DuckDB file and prune its stale tables."""
    source_dir = Path(folder)
    files = sorted(source_dir.glob("*.csv"))
    db_path = Path(database_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        con.execute("""CREATE TABLE IF NOT EXISTS _veriti_client_tables (
            client_id VARCHAR NOT NULL, source_name VARCHAR NOT NULL,
            table_name VARCHAR NOT NULL, PRIMARY KEY (client_id, source_name)
        )""")
        current: dict[str, tuple[str, Path]] = {}
        for file in files:
            table_name = _table_identifier(client_id, file.stem)
            if table_name in (entry[0] for entry in current.values()):
                raise ValueError(f"CSV filenames in {source_dir} map to the same table: {file.stem}")
            current[file.stem] = (table_name, file)

        con.execute("BEGIN TRANSACTION")
        existing = dict(con.execute(
            "SELECT source_name, table_name FROM _veriti_client_tables WHERE client_id = ?",
            [client_id],
        ).fetchall())
        for source_name, table_name in existing.items():
            if source_name not in current:
                con.execute(f"DROP TABLE IF EXISTS {_quote_identifier(table_name)}")
                con.execute(
                    "DELETE FROM _veriti_client_tables WHERE client_id = ? AND source_name = ?",
                    [client_id, source_name],
                )

        report = []
        for source_name, (table_name, file) in current.items():
            csv_path = file.resolve().as_posix().replace("'", "''")
            con.execute(
                f"CREATE OR REPLACE TABLE {_quote_identifier(table_name)} "
                f"AS SELECT * FROM read_csv_auto('{csv_path}', header=true)"
            )
            con.execute(
                "INSERT OR REPLACE INTO _veriti_client_tables "
                "(client_id, source_name, table_name) VALUES (?, ?, ?)",
                [client_id, source_name, table_name],
            )
            report.append({"client_id": client_id, "file": file.name, "table": table_name,
                           "rows": con.execute(
                               f"SELECT COUNT(*) FROM {_quote_identifier(table_name)}"
                           ).fetchone()[0]})
        con.execute("COMMIT")
        return report
    except Exception:
        try:
            con.execute("ROLLBACK")
        except Exception:
            pass
        raise
    finally:
        con.close()


def load_client_csvs(folder: str, client_id: str, database_path: str) -> duckdb.DuckDBPyConnection:
    """Load one client's shared-database tables into an isolated query connection."""
    ingest_csvs_to_shared(folder, client_id, database_path)
    shared = duckdb.connect(str(database_path), read_only=True)
    local = duckdb.connect(":memory:")
    try:
        mappings = shared.execute(
            "SELECT source_name, table_name FROM _veriti_client_tables "
            "WHERE client_id = ? ORDER BY source_name", [client_id]
        ).fetchall()
        if not mappings:
            raise FileNotFoundError(f"No CSV files in {folder}")
        for source_name, table_name in mappings:
            table = _quote_identifier(table_name)
            columns = shared.execute(f"DESCRIBE SELECT * FROM {table}").fetchall()
            schema = ", ".join(
                f"{_quote_identifier(column[0])} {column[1]}" for column in columns
            )
            local.execute(f"CREATE TABLE {_quote_identifier(source_name)} ({schema})")
            placeholders = ", ".join("?" for _ in columns)
            insert = (f"INSERT INTO {_quote_identifier(source_name)} VALUES ({placeholders})")
            cursor = shared.execute(f"SELECT * FROM {table}")
            while rows := cursor.fetchmany(10_000):
                local.executemany(insert, rows)
        try:
            local.execute("SET enable_external_access=false")
        except Exception:
            pass
        return local
    except Exception:
        local.close()
        raise
    finally:
        shared.close()


def client_data_status(folder: str, client_id: str, database_path: str) -> list[dict]:
    """Compare selected-client CSV files with the shared DuckDB ingestion registry."""
    source_dir = Path(folder)
    indexed = {}
    if Path(database_path).exists():
        con = duckdb.connect(database_path, read_only=True)
        try:
            exists = con.execute(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_name = '_veriti_client_tables'"
            ).fetchone()[0]
            if exists:
                indexed = dict(con.execute(
                    "SELECT source_name, table_name FROM _veriti_client_tables WHERE client_id = ?",
                    [client_id],
                ).fetchall())
        finally:
            con.close()
    return [
        {"file": path.name, "table": indexed.get(path.stem),
         "indexed": path.stem in indexed}
        for path in sorted(source_dir.glob("*.csv"))
    ]


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
