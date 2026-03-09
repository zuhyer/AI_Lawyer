
from pathlib import Path
from AI_Lawyer.entity.config_entity import EmbeddingConfig
from AI_Lawyer.utils.logging_setup import logger
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings

# sentence-transformers + numpy availability
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _ST_AVAILABLE = True
except ImportError:
    SentenceTransformer = None  # type: ignore
    np = None  # type: ignore
    _ST_AVAILABLE = False


def validate_index_dimension(
    faiss_index_path: Path,
    embedding_model,
    config_dimension: int,
) -> None:
    """Ensure stored FAISS index dimension matches model output & config.

    Raises ValueError on mismatch; non-fatal when index does not exist yet.
    """
    index_faiss = faiss_index_path / "index.faiss"
    if not index_faiss.exists():
        return
    try:
        import faiss as _faiss  # type: ignore
        idx = _faiss.read_index(str(index_faiss))
        stored_dim = idx.d
        model_dim = embedding_model.dimension
        if stored_dim != model_dim:
            raise ValueError(
                f"FAISS index dimension mismatch!\n"
                f"  Stored index: {stored_dim} dimensions\n"
                f"  Current model: {model_dim} dimensions ({embedding_model.model_name})\n"
                f"  Config says: {config_dimension} dimensions\n"
                f"Delete the index and re-run ingestion to rebuild."
            )
        if model_dim != config_dimension:
            logger.warning(
                f"Config dimension ({config_dimension}) does not match model output ({model_dim})."
            )
        logger.info(f"✅ Dimension check passed: {stored_dim}d for {faiss_index_path.name}")
    except ValueError:
        raise
    except Exception as e:
        logger.warning(f"Could not validate FAISS dimension: {e}")



class LocalSentenceTransformerEmbeddings(Embeddings):
    """Embeddings wrapper using sentence-transformers for local inference.

    Implements LangChain's Embeddings interface with `embed_documents`
    and `embed_query` using SentenceTransformer.encode.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not _ST_AVAILABLE or SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is not installed. Install it with `pip install sentence-transformers`"
            )
        self.model_name = model_name
        logger.info(f"Initializing local SentenceTransformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        # probe a dummy embedding to capture output dimension
        probe = self.model.encode(["test"], convert_to_numpy=True)
        self.dimension: int = int(probe.shape[1])
        logger.info(f"Embedding model ready — dimension={self.dimension}")

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

    def __init__(self, config: EmbeddingConfig, domain: str = None, config_manager = None):
        self.config = config
        self.model_name = config.model or "all-MiniLM-L6-v2"
        self.domain = domain
        self.config_manager = config_manager
        
        # Set path based on domain or use default
        if domain and config_manager:
            self.db_path = config_manager.get_domain_vector_db_path(domain)
        else:
            self.db_path = Path(config.vector_store_path)

    def get_embedding_model(self):
        """Return a local sentence-transformers based embeddings instance.

        Also perform dimension validation against existing FAISS index and
        configuration.  This will raise if dimensions mismatch.
        """
        try:
            logger.info(f"Initializing local embedding model: {self.model_name}")
            model = LocalSentenceTransformerEmbeddings(self.model_name)
            # if the config specifies a dimension, validate index consistency
            validate_index_dimension(self.db_path, model, self.config.dimension)
            return model
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
