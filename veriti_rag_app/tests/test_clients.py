import json
import re

import pytest
from pathlib import Path
from dataclasses import replace

from veriti_rag.clients import create_client, list_clients, settings_for_client
from veriti_rag.config import Settings
from veriti_rag.db import client_data_status, ingest_csvs_to_shared, load_client_csvs
from veriti_rag.ingest import ingest_all
from veriti_rag.store import search
from veriti_rag.tracing import thread_trace_project
from langsmith import get_tracing_context


def make_settings(tmp_path):
    return Settings(
        kb_dir=str(tmp_path / "knowledge_base"),
        data_dir=str(tmp_path / "data" / "raw"),
        index_dir=str(tmp_path / "vector_index"),
        duckdb_path=str(tmp_path / "data" / "veriti.duckdb"),
        embeddings="hash",
    )


def test_create_client_creates_workspace_and_persists_registry(tmp_path):
    settings = make_settings(tmp_path)

    created = create_client(settings, "North Star Health")
    selected = settings_for_client(settings, created["client_id"])
    registered = {client["client_id"] for client in list_clients(settings)}

    assert created["client_id"] == "north_star_health"
    assert re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", created["client_id"])
    assert (tmp_path / "knowledge_base" / created["client_id"] / "_meta.yml").is_file()
    assert (tmp_path / "data" / "raw" / created["client_id"]).is_dir()
    assert (tmp_path / "vector_index" / created["client_id"]).is_dir()
    assert created["client_id"] in registered
    assert selected.client_id == created["client_id"]

    persisted = json.loads((tmp_path / "data" / "clients.json").read_text(encoding="utf-8"))
    assert persisted["clients"][created["client_id"]]["name"] == "North Star Health"


def test_duplicate_and_invalid_client_names_are_safe(tmp_path):
    settings = make_settings(tmp_path)
    first = create_client(settings, "North Star")
    second = create_client(settings, "North-Star")

    assert first["client_id"] == "north_star"
    assert second["client_id"] == "north_star_2"
    with pytest.raises(ValueError, match="already registered"):
        create_client(settings, " north star ")
    with pytest.raises(ValueError, match="English letter or number"):
        create_client(settings, "---")


def test_new_clients_are_visible_and_maven_default_remains(tmp_path):
    settings = make_settings(tmp_path)
    created = create_client(settings, "Client One")

    clients = {item["client_id"]: item["name"] for item in list_clients(settings)}
    assert clients["maven_demo"] == "Maven Demo"
    assert clients[created["client_id"]] == "Client One"


def test_client_switch_uses_separate_data_and_index_paths(tmp_path):
    settings = make_settings(tmp_path)
    created = create_client(settings, "Path Check")
    selected = settings_for_client(settings, created["client_id"])

    assert selected.data_dir == str(tmp_path / "data" / "raw" / created["client_id"])
    assert selected.index_dir == str(tmp_path / "vector_index" / created["client_id"])
    assert settings_for_client(settings, "maven_demo").index_dir == settings.index_dir


def test_kb_ingestion_and_search_are_scoped_to_selected_client(env, tmp_path):
    settings, vs = env
    isolated = replace(
        settings,
        data_dir=str(tmp_path / "data" / "raw"),
        duckdb_path=str(tmp_path / "data" / "veriti.duckdb"),
    )
    client = create_client(isolated, "Quasar Customer")
    kb_file = Path(settings.kb_dir) / client["client_id"] / "business_rules.md"
    kb_file.write_text(
        "Quasar reconciliation requires matching account identifiers before settlement.",
        encoding="utf-8",
    )

    before = ingest_all(settings, vs, client_ids=["maven_demo"])[0]
    beta = ingest_all(settings, vs, client_ids=[client["client_id"]])[0]

    assert beta["client"] == client["client_id"]
    beta_hits = search(vs, "Quasar reconciliation account identifiers", client["client_id"])
    maven_hits = search(vs, "Quasar reconciliation account identifiers", "maven_demo")
    assert any("Quasar" in doc.page_content for doc, _ in beta_hits)
    assert all(doc.metadata["client_id"] == "maven_demo" for doc, _ in maven_hits)
    assert all("Quasar" not in doc.page_content for doc, _ in maven_hits)
    assert ingest_all(settings, vs, client_ids=["maven_demo"])[0]["chunks"] == before["chunks"]


def test_shared_duckdb_tables_and_query_connection_are_client_isolated(tmp_path):
    settings = make_settings(tmp_path)
    alpha = create_client(settings, "Alpha")
    beta = create_client(settings, "Beta")
    alpha_dir = tmp_path / "data" / "raw" / alpha["client_id"]
    beta_dir = tmp_path / "data" / "raw" / beta["client_id"]
    alpha_dir.joinpath("orders.csv").write_text("account,amount\nAlpha Corp,11\n", encoding="utf-8")
    beta_dir.joinpath("orders.csv").write_text("account,amount\nBeta Corp,99\n", encoding="utf-8")

    ingest_csvs_to_shared(str(alpha_dir), alpha["client_id"], settings.duckdb_path)
    ingest_csvs_to_shared(str(beta_dir), beta["client_id"], settings.duckdb_path)
    alpha_con = load_client_csvs(str(alpha_dir), alpha["client_id"], settings.duckdb_path)
    beta_con = load_client_csvs(str(beta_dir), beta["client_id"], settings.duckdb_path)
    try:
        assert alpha_con.execute("SELECT SUM(amount) FROM orders").fetchone()[0] == 11
        assert beta_con.execute("SELECT SUM(amount) FROM orders").fetchone()[0] == 99
        with pytest.raises(Exception):
            alpha_con.execute('SELECT * FROM "beta__orders"').fetchall()
        with pytest.raises(Exception):
            beta_con.execute('SELECT * FROM "alpha__orders"').fetchall()
    finally:
        alpha_con.close()
        beta_con.close()


def test_csv_deletion_sync_removes_only_selected_clients_table(tmp_path):
    settings = make_settings(tmp_path)
    alpha = create_client(settings, "Alpha")
    beta = create_client(settings, "Beta")
    alpha_dir = tmp_path / "data" / "raw" / alpha["client_id"]
    beta_dir = tmp_path / "data" / "raw" / beta["client_id"]
    alpha_dir.joinpath("orders.csv").write_text("amount\n11\n", encoding="utf-8")
    beta_dir.joinpath("orders.csv").write_text("amount\n99\n", encoding="utf-8")
    ingest_csvs_to_shared(str(alpha_dir), alpha["client_id"], settings.duckdb_path)
    ingest_csvs_to_shared(str(beta_dir), beta["client_id"], settings.duckdb_path)

    (alpha_dir / "orders.csv").unlink()
    assert ingest_csvs_to_shared(str(alpha_dir), alpha["client_id"], settings.duckdb_path) == []
    status = client_data_status(str(beta_dir), beta["client_id"], settings.duckdb_path)
    assert status == [{"file": "orders.csv", "table": "beta__orders", "indexed": True}]
    with pytest.raises(FileNotFoundError):
        load_client_csvs(str(alpha_dir), alpha["client_id"], settings.duckdb_path)


def test_thread_tracing_selects_project_without_agent_address(monkeypatch):
    monkeypatch.setenv("LANGSMITH_AGENT_ID", "example-agent")
    monkeypatch.setenv("LANGSMITH_AGENT_ENVIRONMENT", "test")
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)

    with thread_trace_project("veriti-rag"):
        context = get_tracing_context()
        assert context["project_name"] == "veriti-rag"
        assert context["address"] is None
