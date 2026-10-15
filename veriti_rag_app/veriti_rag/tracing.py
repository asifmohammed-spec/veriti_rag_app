"""Helpers for thread-based LangGraph runs with LangSmith tracing."""
import os

from langsmith import tracing_context


def thread_trace_project(default_project="veriti-rag"):
    """Prefer project/session tracing during threaded runs.

    LangSmith cannot address the same run by both an agent address and a
    session ID. LangGraph's ``thread_id`` is exported as that session ID, so
    remove agent-address routing and explicitly route to a project. Keep that
    routing for the process lifetime because LangSmith flushes run patches in
    a background batch after the graph invocation returns.
    """
    os.environ.pop("LANGSMITH_AGENT_ID", None)
    os.environ.pop("LANGSMITH_AGENT_ENVIRONMENT", None)
    project = os.environ.setdefault("LANGSMITH_PROJECT", default_project)
    return tracing_context(project_name=project)
