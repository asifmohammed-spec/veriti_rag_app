import shutil
import pytest
from veriti_rag.config import Settings
from veriti_rag.models import HashEmbeddings
from veriti_rag.store import get_vectorstore


@pytest.fixture
def env(tmp_path):
    kb = tmp_path / "kb"
    shutil.copytree("knowledge_base", kb)
    s = Settings(kb_dir=str(kb), index_dir=str(tmp_path / "idx"), embeddings="hash")
    return s, get_vectorstore(s, HashEmbeddings())
