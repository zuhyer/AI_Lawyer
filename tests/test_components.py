import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "../src"))

from AI_Lawyer.components.local_embedding import (
    LocalSentenceTransformerEmbeddings,
    validate_index_dimension,
)
from AI_Lawyer.components.query_component import QueryComponent
from AI_Lawyer.entity.config_entity import EmbeddingConfig, LLMConfig
from langchain_core.documents import Document


def test_dummy_embedding_init():
    # model might not be installed, just exercise constructor failure mode
    try:
        LocalSentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    except ImportError:
        pass


def test_query_component_return_type(monkeypatch):
    # create a fake faiss db object with similarity_search method
    class DummyDB:
        def similarity_search(self, query, k):
            return []

    llm_cfg = LLMConfig(provider="none", model="m", api_key="k")
    qc = QueryComponent(llm_config=llm_cfg, faiss_db=DummyDB())
    # monkeypatch the llm so we don't need actual service
    class DummyLLM:
        def __init__(self):
            pass
        def invoke(self, args):
            class R:
                content = "ok"
            return R()
    qc.llm = DummyLLM()
    res = qc.query_with_user_files("q", [], None)
    assert isinstance(res.get("sources"), list)
