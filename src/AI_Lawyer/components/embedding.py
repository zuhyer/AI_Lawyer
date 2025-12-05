from pathlib import Path
from AI_Lawyer.entity.config_entity import EmbeddingConfig
from AI_Lawyer.utils.logging_setup import logger
from AI_Lawyer.utils.secret_loader import resolve_secret

import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings


# ===========================================================
# Gemini Embedding Wrapper Class
# ===========================================================

class GeminiEmbedding(Embeddings):

    def __init__(self, model_name: str, api_key: str):
        """
        Wrapper to use Gemini embeddings with LangChain.
        """
        # Validate API key before configuring
        if not api_key or len(api_key.strip()) == 0:
            raise ValueError("❌ API key is empty or None")
        
        if not api_key.startswith("AIza"):
            logger.warning(f"⚠️  API key format may be invalid. Expected to start with 'AIza', got: {api_key[:4]}...")
        
        try:
            genai.configure(api_key=api_key)
            logger.info("✅ Gemini API configured successfully")
        except Exception as e:
            logger.error(f"❌ Failed to configure Gemini API: {e}")
            raise ValueError(f"API configuration failed: {e}")

        self.model_name = model_name

    def embed_documents(self, texts):
        """
        Embeds a list of documents using Gemini.
        """
        try:
            response = genai.embed_content(
                model=self.model_name,
                content=texts
            )
            return response["embedding"]
        except Exception as e:
            logger.error(f"❌ Error in embed_documents: {e}")
            if "API_KEY_INVALID" in str(e):
                raise ValueError(f"Invalid API key. Please check your secret.yaml file. Error: {e}")
            raise e

    def embed_query(self, text):
        """
        Embeds a single query string.
        """
        try:
            response = genai.embed_content(
                model=self.model_name,
                content=text
            )
            return response["embedding"]
        except Exception as e:
            logger.error(f"❌ Error in embed_query: {e}")
            if "API_KEY_INVALID" in str(e):
                raise ValueError(f"Invalid API key. Please check your secret.yaml file. Error: {e}")
            raise e


# ===========================================================
# Embedding Component Class
# ===========================================================

class EmbeddingCreator:

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.model_name = config.model
        self.db_path = Path(config.vector_store_path)

        # Resolve API key safely from config.yaml → secret.yaml
        try:
            self.api_key = resolve_secret(config.api_key)
            
            # Validate the resolved key
            if not self.api_key or len(self.api_key.strip()) == 0:
                raise ValueError("API key resolved to empty or None value")
                
            logger.info(f"✅ API key resolved successfully (length: {len(self.api_key)})")
            
        except Exception as e:
            logger.error(f"❌ Failed to resolve API key from '{config.api_key}': {e}")
            raise ValueError(f"API key resolution failed. Ensure secret.yaml has the correct key entry. Error: {e}")

    def get_embedding_model(self):
        """
        Initializes the Gemini embedding model.
        """
        try:
            logger.info(f"Initializing Gemini embedding model: {self.model_name}")
            return GeminiEmbedding(self.model_name, self.api_key)
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise e

    def create_vector_store(self, text_chunks):
        """
        Creates a FAISS vector store from the document chunks.
        """
        try:
            logger.info("Creating FAISS vector store...")

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
            raise e

    def main(self, text_chunks):
        """
        Pipeline-compatible entry point.
        """
        return self.create_vector_store(text_chunks)
    

    