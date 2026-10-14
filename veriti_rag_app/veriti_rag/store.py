"""Chroma vector store through LangChain, plus a client-filtered search."""
from langchain_chroma import Chroma

from .config import Settings
from .models import embedding_tag


def get_vectorstore(s: Settings, embeddings) -> Chroma:
    return Chroma(collection_name=f"kb_{embedding_tag(s)}", embedding_function=embeddings,
                  persist_directory=s.index_dir, collection_metadata={"hnsw:space": "cosine"})


def search(vs: Chroma, query: str, client_id: str, k: int = 5):
    """Returns [(Document, similarity)]; filtered by client so rules never leak across clients."""
    res = vs.similarity_search_with_score(query, k=k, filter={"client_id": client_id})
    return [(d, 1.0 - dist) for d, dist in res]
