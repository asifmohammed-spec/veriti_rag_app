"""Chat + embedding models, both through OpenRouter's OpenAI-compatible API."""
import hashlib
import math
import os
import re

from langchain_core.embeddings import Embeddings

from .config import Settings

from dotenv import load_dotenv

load_dotenv()

HEADERS = {"X-Title": "VERITI RAG"}  # optional OpenRouter attribution


def _key() -> str:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("Set OPENROUTER_API_KEY (https://openrouter.ai/keys).")
    return key


class HashEmbeddings(Embeddings):
    """Offline, deterministic word-overlap embeddings. For tests/dev only:
    it has no semantic understanding."""
    def __init__(self, dim: int = 384):
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        toks = re.findall(r"[a-z0-9]+", text.lower())
        v = [0.0] * self.dim
        for f in toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]:
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "big")
            v[h % self.dim] += 1.0 if (h >> 40) & 1 else -1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def embedding_tag(s: Settings) -> str:
    """Used in the collection name so changing the model never mixes vector spaces."""
    tag = "hash" if s.embeddings == "hash" else s.embed_model
    return re.sub(r"[^a-zA-Z0-9]+", "_", tag).strip("_")[:50]


def make_embeddings(s: Settings) -> Embeddings:
    if s.embeddings == "hash":
        return HashEmbeddings()
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(
        model=s.embed_model, api_key=_key(), base_url=s.base_url,
        check_embedding_ctx_length=False,  # send plain strings, not OpenAI token ids
        chunk_size=100, default_headers=HEADERS)


def make_llm(s: Settings):
    """Returns llm(system, user) -> str. The graph only needs this callable,
    so tests can pass a fake."""
    from langchain_openai import ChatOpenAI
    chat = ChatOpenAI(model=s.llm_model, api_key=_key(), base_url=s.base_url,
                      temperature=0, max_retries=5, timeout=120, default_headers=HEADERS)

    def call(system: str, user: str) -> str:
        content = chat.invoke([("system", system), ("human", user)]).content
        if isinstance(content, list):
            content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
        return content
    return call
