# python -m streamlit run streamlit_app.py

from uuid import uuid4

import streamlit as st

from veriti_rag.config import Settings
from veriti_rag.db import load_csvs
from veriti_rag.graph import build_graph
from veriti_rag.ingest import ingest_all
from veriti_rag.models import make_embeddings, make_llm
from veriti_rag.store import get_vectorstore, search
from veriti_rag.tracing import thread_trace_project


def setup_state():
    st.session_state.setdefault("thread_id", str(uuid4()))
    if "settings" not in st.session_state:
        st.session_state.settings = Settings(
            client_id="maven_demo",
            embeddings="hash",
        )
        st.session_state.vs = get_vectorstore(
            st.session_state.settings,
            make_embeddings(st.session_state.settings),
        )


def refresh_store():
    s = Settings(
        client_id=st.session_state.client_id,
        embeddings=st.session_state.embedding_mode,
    )
    st.session_state.settings = s
    st.session_state.vs = get_vectorstore(s, make_embeddings(s))
    st.success("Vector store refreshed.")


st.set_page_config(page_title="Veriti RAG Debugger", layout="wide")

setup_state()

st.title("Veriti RAG Streamlit Debugger")

with st.sidebar:
    st.subheader("Settings")
    st.session_state.client_id = st.text_input("Client ID", value=st.session_state.settings.client_id)
    st.session_state.embedding_mode = st.selectbox(
        "Embedding mode",
        ["hash", "openrouter"],
        index=0 if st.session_state.settings.embeddings == "hash" else 1,
    )

    if st.button("Refresh store"):
        refresh_store()

    if st.button("Ingest knowledge base"):
        s = st.session_state.settings
        vs = st.session_state.vs
        report = ingest_all(s, vs)
        st.json(report)

question = st.text_area("Ask a question", value="Which rule checks that won deals have revenue recorded?")

if st.button("Run query"):
    s = st.session_state.settings
    vs = st.session_state.vs

    with st.spinner("Running retrieval..."):
        hits = search(vs, question, s.client_id, k=5)

    st.subheader("Retrieval matches")
    if not hits:
        st.info("No retrieval hits found.")
    else:
        for idx, (doc, score) in enumerate(hits, start=1):
            with st.expander(f"#{idx} {doc.metadata.get('source')}  |  score={score:.3f}"):
                st.write("Metadata:", doc.metadata)
                st.write(doc.page_content[:800])

    try:
        llm = make_llm(s)
        app = build_graph(load_csvs(s.data_dir), vs, llm, s)

        with st.spinner("Running full RAG graph..."):
            with thread_trace_project():
                result = app.invoke(
                    {"question": question, "client_id": s.client_id},
                    config={"configurable": {"thread_id": st.session_state.thread_id}},
                )

        st.subheader("Answer")
        st.write(result.get("answer", ""))

        if result.get("sql"):
            st.subheader("Generated SQL")
            st.code(result["sql"])

        if result.get("grounded") is not None:
            st.caption(f"Grounded: {result.get('grounded')}")

        if result.get("citations"):
            st.subheader("Citations")
            st.json(result["citations"])

        if st.checkbox("Show raw graph output"):
            st.json(result)
    except Exception as e:
        st.warning(f"LLM pipeline unavailable: {e}")
        st.info("Retrieval results above are still useful for debugging without OpenRouter.")
