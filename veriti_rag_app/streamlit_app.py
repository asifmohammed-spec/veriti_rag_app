# Run with: python -m streamlit run streamlit_app.py
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import streamlit as st

from veriti_rag.clients import (
    DEFAULT_CLIENT_ID,
    client_data_dir,
    create_client,
    list_clients,
    settings_for_client,
)
from veriti_rag.config import Settings
from veriti_rag.db import client_data_status, ingest_csvs_to_shared, load_client_csvs
from veriti_rag.graph import build_graph
from veriti_rag.ingest import ingest_all
from veriti_rag.models import make_embeddings, make_llm
from veriti_rag.reports import create_sales_report
from veriti_rag.sales_report_agent import run_sales_report_agent
from veriti_rag.store import get_vectorstore, search
from veriti_rag.tracing import thread_trace_project
from render_report import render_report_data


KB_EXTENSIONS = {".md", ".txt", ".yml", ".yaml", ".csv"}


def _safe_upload_name(upload) -> str:
    name = Path(upload.name).name
    if name != upload.name or name in {"", ".", ".."}:
        raise ValueError(f"Unsafe filename: {upload.name}")
    return name


def _indexed_kb_sources(vs, client_id: str) -> set[str]:
    result = vs.get(where={"client_id": client_id}, include=["metadatas"])
    return {meta.get("source") for meta in result.get("metadatas", []) if meta}


def _save_uploads(uploads, destination: Path, extensions: set[str], overwrite: bool) -> tuple[int, list[str]]:
    destination.mkdir(parents=True, exist_ok=True)
    saved = 0
    messages = []
    for upload in uploads or []:
        try:
            name = _safe_upload_name(upload)
            if Path(name).suffix.lower() not in extensions:
                messages.append(f"Skipped {name}: unsupported file type.")
                continue
            target = destination / name
            if target.exists() and not overwrite:
                messages.append(f"Skipped {name}: already exists. Enable replace to overwrite it.")
                continue
            target.write_bytes(upload.getvalue())
            saved += 1
            messages.append(f"Saved {name}.")
        except (OSError, ValueError) as exc:
            messages.append(f"Could not save {upload.name}: {exc}")
    return saved, messages


def _delete_kb_file(path: Path, vs, client_id: str, kb_root: Path) -> None:
    source = path.relative_to(kb_root / client_id).as_posix()
    result = vs.get(where={"client_id": client_id}, include=["metadatas"])
    ids = [
        chunk_id for chunk_id, metadata in zip(result.get("ids", []), result.get("metadatas", []))
        if metadata and metadata.get("source") == source
    ]
    if ids:
        vs.delete(ids=ids)
    path.unlink()


def setup_state():
    st.session_state.setdefault("thread_id", str(uuid4()))
    st.session_state.setdefault("embedding_mode", "hash")
    st.session_state.setdefault("client_selector", DEFAULT_CLIENT_ID)


st.set_page_config(page_title="Veriti RAG", layout="wide")
setup_state()

base_settings = Settings(embeddings=st.session_state.embedding_mode)
try:
    clients = list_clients(base_settings)
except Exception as exc:
    st.error(f"Could not load the client registry: {exc}")
    st.stop()

if pending_client := st.session_state.pop("pending_client_selection", None):
    st.session_state["client_selector"] = pending_client
if created_message := st.session_state.pop("client_created_message", None):
    st.success(created_message)

with st.sidebar:
    st.subheader("Client")
    client_names = {item["client_id"]: item["name"] for item in clients}
    if st.session_state.client_selector not in client_names:
        st.session_state.client_selector = DEFAULT_CLIENT_ID
    client_id = st.selectbox(
        "Select a client",
        options=list(client_names),
        format_func=lambda value: client_names[value],
        key="client_selector",
    )

    st.subheader("Settings")
    st.selectbox("Embedding mode", ["hash", "openrouter"], key="embedding_mode")
    base_settings = Settings(embeddings=st.session_state.embedding_mode)
    settings = settings_for_client(base_settings, client_id)

    if (st.session_state.get("active_client_id") != client_id
            or st.session_state.get("active_embedding_mode") != settings.embeddings):
        st.session_state.thread_id = str(uuid4())
        st.session_state.active_client_id = client_id
        st.session_state.active_embedding_mode = settings.embeddings
        st.session_state.vs = get_vectorstore(settings, make_embeddings(settings))
        st.session_state.last_kb_ingest = None
    vs = st.session_state.vs

    with st.form("create_client_form", clear_on_submit=True):
        st.markdown("**Create New Client**")
        client_name = st.text_input("Client name", placeholder="Acme Health")
        create_clicked = st.form_submit_button("Create client")
    if create_clicked:
        try:
            created = create_client(base_settings, client_name)
            st.session_state.pending_client_selection = created["client_id"]
            st.session_state.client_created_message = (
                f"Created {created['name']} ({created['client_id']})."
            )
            st.rerun()
        except (OSError, ValueError) as exc:
            st.error(str(exc))

    st.divider()
    if st.button("Re-index selected knowledge base", width="stretch"):
        try:
            with st.spinner(f"Indexing {client_names[client_id]} knowledge files..."):
                status = ingest_all(settings, vs, client_ids=[client_id])[0]
            st.session_state.last_kb_ingest = status
            st.success(
                f"Indexed {status['chunks']} chunks: {status['added']} added, "
                f"{status['deleted']} removed, {status['unchanged']} unchanged."
            )
        except Exception as exc:
            st.error(f"Knowledge-base ingestion failed: {exc}")

st.title("VERITI RAG")
st.caption(f"Client: **{client_names[client_id]}** · ID: `{client_id}`")

knowledge_tab, data_tab, query_tab, report_tab = st.tabs(
    ["Knowledge Base", "Client Data", "Ask VERITI", "Sales Report"]
)

with knowledge_tab:
    st.subheader("Client-specific knowledge base")
    kb_root = Path(settings.kb_dir)
    client_kb = kb_root / client_id
    client_kb.mkdir(parents=True, exist_ok=True)
    indexed_sources = _indexed_kb_sources(vs, client_id)
    knowledge_files = sorted(
        path for path in client_kb.rglob("*")
        if path.is_file() and path.suffix.lower() in KB_EXTENSIONS and not path.name.startswith("_")
    )
    if knowledge_files:
        st.dataframe(
            [{"File": path.relative_to(client_kb).as_posix(),
              "Status": "Indexed" if path.relative_to(client_kb).as_posix() in indexed_sources else "Not indexed",
              "Size": f"{path.stat().st_size:,} bytes"} for path in knowledge_files],
            hide_index=True,
            width="stretch",
        )
    else:
        st.info("No knowledge files yet. Upload business rules, schema, data dictionaries, mappings, or remediation guidance.")

    with st.form("knowledge_upload_form"):
        kb_uploads = st.file_uploader(
            "Upload knowledge files",
            type=["md", "txt", "yml", "yaml", "csv"],
            accept_multiple_files=True,
            key=f"kb_upload_{client_id}",
        )
        replace_kb = st.checkbox("Replace files with matching names", key=f"replace_kb_{client_id}")
        save_kb = st.form_submit_button("Save knowledge files")
    if save_kb:
        saved, messages = _save_uploads(kb_uploads, client_kb, KB_EXTENSIONS, replace_kb)
        for message in messages:
            st.success(message) if message.startswith("Saved") else st.warning(message)
        if saved:
            st.info("Re-index this client's knowledge base to include the uploaded files.")

    if knowledge_files:
        selected_kb_file = st.selectbox(
            "Knowledge file to delete",
            options=[path.relative_to(client_kb).as_posix() for path in knowledge_files],
            key=f"delete_kb_select_{client_id}",
        )
        if st.button("Delete selected knowledge file", key=f"delete_kb_{client_id}"):
            try:
                _delete_kb_file(client_kb / selected_kb_file, vs, client_id, kb_root)
                st.success(f"Deleted {selected_kb_file} and its indexed chunks for this client.")
                st.rerun()
            except OSError as exc:
                st.error(f"Could not delete file: {exc}")

    if st.session_state.get("last_kb_ingest"):
        st.caption(f"Last selected-client ingestion: {st.session_state.last_kb_ingest}")

with data_tab:
    st.subheader("Client-specific CSV datasets")
    data_dir = client_data_dir(settings, client_id)
    with st.form("csv_upload_form"):
        csv_uploads = st.file_uploader(
            "Upload CSV datasets",
            type=["csv"],
            accept_multiple_files=True,
            key=f"csv_upload_{client_id}",
        )
        replace_csv = st.checkbox("Replace files with matching names", key=f"replace_csv_{client_id}")
        save_csvs = st.form_submit_button("Save CSV files")
    if save_csvs:
        saved, messages = _save_uploads(csv_uploads, data_dir, {".csv"}, replace_csv)
        for message in messages:
            st.success(message) if message.startswith("Saved") else st.warning(message)

    try:
        csv_status = client_data_status(str(data_dir), client_id, settings.duckdb_path)
        if csv_status:
            st.dataframe(
                [{"File": item["file"], "DuckDB table": item["table"] or "—",
                  "Status": "Indexed in shared DuckDB" if item["indexed"] else "Not indexed"}
                 for item in csv_status],
                hide_index=True,
                width="stretch",
            )
        else:
            st.info("No CSV datasets uploaded for this client.")
    except Exception as exc:
        st.warning(f"Could not read dataset ingestion status: {exc}")

    col_ingest, col_delete = st.columns(2)
    with col_ingest:
        if st.button("Ingest selected client's CSVs", key=f"ingest_csv_{client_id}"):
            try:
                result = ingest_csvs_to_shared(str(data_dir), client_id, settings.duckdb_path)
                st.success(f"Loaded {len(result)} CSV tables into the shared DuckDB database for {client_id}.")
                st.json(result)
                st.rerun()
            except Exception as exc:
                st.error(f"CSV ingestion failed: {exc}")
    with col_delete:
        csv_files = sorted(data_dir.glob("*.csv"))
        if csv_files:
            file_to_delete = st.selectbox(
                "CSV file to delete",
                options=[path.name for path in csv_files],
                key=f"delete_csv_select_{client_id}",
            )
            if st.button("Delete CSV and sync DuckDB", key=f"delete_csv_{client_id}"):
                try:
                    (data_dir / file_to_delete).unlink()
                    ingest_csvs_to_shared(str(data_dir), client_id, settings.duckdb_path)
                    st.success(f"Deleted {file_to_delete}; only {client_id}'s shared tables were updated.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not delete or sync dataset: {exc}")

with query_tab:
    st.subheader("Ask a question about this client's data and business rules")
    question = st.text_area(
        "Question",
        value="Which rule checks that won deals have revenue recorded?",
        key=f"question_{client_id}",
    )
    if st.button("Run RAG query", key=f"run_query_{client_id}"):
        try:
            with st.spinner("Loading this client's datasets and running the RAG workflow..."):
                con = load_client_csvs(str(data_dir), client_id, settings.duckdb_path)
                try:
                    hits = search(vs, question, client_id, k=settings.top_k)
                    app = build_graph(con, vs, make_llm(settings), settings)
                    with thread_trace_project():
                        result = app.invoke(
                            {"question": question, "client_id": client_id},
                            config={"configurable": {"thread_id": st.session_state.thread_id}},
                        )
                finally:
                    con.close()
            st.subheader("Answer")
            st.write(result.get("answer", ""))
            if result.get("sql"):
                st.subheader("Generated SQL")
                st.code(result["sql"], language="sql")
            st.caption(f"Grounded: {result.get('grounded')} · Client: {client_id}")
            if result.get("citations"):
                st.subheader("Knowledge citations")
                st.json(result["citations"])
            with st.expander(f"Knowledge matches ({len(hits)})"):
                for idx, (doc, score) in enumerate(hits, start=1):
                    st.markdown(f"**{idx}. {doc.metadata.get('source')}** · score={score:.3f}")
                    st.write(doc.page_content[:800])
        except Exception as exc:
            st.error(f"RAG query failed for {client_names[client_id]}: {exc}")

with report_tab:
    st.subheader("Sales performance report")
    report_prompt = st.text_input(
        "Optional report focus",
        placeholder="Emphasize product conversion",
        key=f"report_prompt_{client_id}",
    )
    if st.button("Generate sales report", key=f"run_report_{client_id}"):
        try:
            with st.spinner(f"Generating sales report for {client_names[client_id]}..."):
                con = load_client_csvs(str(data_dir), client_id, settings.duckdb_path)
                try:
                    result = run_sales_report_agent(
                        con, make_llm(settings), report_prompt,
                        thread_id=st.session_state.thread_id, client_id=client_id,
                    )
                finally:
                    con.close()
                report_dir = Path("report_output") / client_id
                report_dir.mkdir(parents=True, exist_ok=True)
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                pdf_path = report_dir / f"sales_report_{stamp}.pdf"
                html_path = report_dir / f"sales_report_{stamp}.html"
                create_sales_report(result, pdf_path)
                render_report_data(result["template_data"], html_path)
            st.success(f"Sales report created for {client_names[client_id]}.")
            st.download_button(
                "Download PDF report", data=pdf_path.read_bytes(),
                file_name=pdf_path.name, mime="application/pdf",
            )
            st.download_button(
                "Download HTML report", data=html_path.read_bytes(),
                file_name=html_path.name, mime="text/html",
            )
            st.json(result["template_data"])
        except Exception as exc:
            st.error(f"Sales report failed for {client_names[client_id]}: {exc}")
