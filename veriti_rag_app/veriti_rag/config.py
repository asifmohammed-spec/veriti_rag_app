"""All settings in one place. Override with environment variables.
Required for real runs: OPENROUTER_API_KEY."""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

def _e(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    # paths
    kb_dir: str = _e("VERITI_KB_DIR", "knowledge_base")
    data_dir: str = _e("VERITI_DATA_DIR", "data/raw")
    index_dir: str = _e("VERITI_INDEX_DIR", "vector_index")
    client_id: str = _e("VERITI_CLIENT_ID", "maven_demo")
    # OpenRouter (OpenAI-compatible) for both chat and embeddings
    base_url: str = _e("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    llm_model: str = _e("VERITI_MODEL", "openai/gpt-oss-120b:nitro")
    embed_model: str = _e("VERITI_EMBED_MODEL", "openai/text-embedding-3-small")
    embeddings: str = _e("VERITI_EMBEDDINGS", "openrouter")  # or "hash" (offline, tests only)
    # chunking / retrieval
    chunk_size: int = int(_e("VERITI_CHUNK_SIZE", "800"))
    chunk_overlap: int = int(_e("VERITI_CHUNK_OVERLAP", "100"))
    top_k: int = int(_e("VERITI_TOP_K", "5"))
    # graph limits
    max_sql_attempts: int = 3
    max_answer_attempts: int = 2

    # LangSmith tracing
    langsmith_tracing: bool = (
        _e("LANGSMITH_TRACING", "false").lower()
        in ("true", "1", "yes", "on")
    )
    langsmith_endpoint: str = _e(
        "LANGSMITH_ENDPOINT",
        "https://api.smith.langchain.com",
    )
    langsmith_api_key: str = _e("LANGSMITH_API_KEY", "")
    langsmith_project: str = _e(
        "LANGSMITH_PROJECT",
        "veriti-rag",
    )


