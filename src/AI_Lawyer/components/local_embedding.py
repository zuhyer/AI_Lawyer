
from pathlib import Path
from AI_Lawyer.entity.config_entity import EmbeddingConfig
from AI_Lawyer.utils.logging_setup import logger
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings

# Local Sentence-Transformer based embeddings (all-MiniLM-L6-v2)
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
except Exception:
    SentenceTransformer = None


class LocalSentenceTransformerEmbeddings(Embeddings):
    """Embeddings wrapper using sentence-transformers for local inference.

    Implements LangChain's Embeddings interface with `embed_documents`
    and `embed_query` using SentenceTransformer.encode.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is not installed. Install it with `pip install sentence-transformers`"
            )
        self.model_name = model_name
        logger.info(f"Initializing local SentenceTransformer model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        # sentence-transformers returns ndarray; convert to list[list[float]]
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            # Ensure 2D
            if embeddings.ndim == 1:
                embeddings = np.expand_dims(embeddings, 0)
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            logger.error(f"Error in embed_documents: {e}")
            raise

    def embed_query(self, text):
        try:
            emb = self.model.encode(text, convert_to_numpy=True)
            return emb.tolist()
        except Exception as e:
            logger.error(f"Error in embed_query: {e}")
            raise



class EmbeddingCreator:

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.model_name = config.model or "all-MiniLM-L6-v2"
        self.db_path = Path(config.vector_store_path)

    def get_embedding_model(self):
        """Return a local sentence-transformers based embeddings instance."""
        try:
            logger.info(f"Initializing local embedding model: {self.model_name}")
            return LocalSentenceTransformerEmbeddings(self.model_name)
        except Exception as e:
            logger.error(f"Failed to initialize local embedding model: {e}")
            raise

    def create_vector_store(self, text_chunks):
        try:
            logger.info("Creating FAISS vector store using local embeddings...")
            embedding_model = self.get_embedding_model()

            faiss_db = FAISS.from_documents(
                text_chunks,
                embedding_model
            )

            # Ensure directory exists
            self.db_path.mkdir(parents=True, exist_ok=True)

            faiss_db.save_local(str(self.db_path))

            logger.info(f"FAISS database saved successfully at: {self.db_path}")
            return faiss_db

        except Exception as e:
            logger.error(f"Error during FAISS vector store creation: {e}")
            raise

    def main(self, text_chunks):
        return self.create_vector_store(text_chunks)
