from AI_Lawyer.config.configuration import ConfigurationManager
from AI_Lawyer.components.local_embedding import EmbeddingCreator
from AI_Lawyer.utils.logging_setup import logger
from langchain_community.vectorstores import FAISS
from pathlib import Path


STAGE_NAME = "Embedding Stage"


def start_embedding_pipeline(text_chunks, domain: str = "constitution"):
    """
    Runs the embedding creation process:
      - Loads embedding config
      - Uses domain-specific vector store path
      - Creates FAISS vector store
      - Saves it to domain-specific location
      
    Args:
        text_chunks: List of document chunks to embed
        domain: Domain name for domain-specific vector store path
    """
    try:
        logger.info(f"===== Starting Embedding Pipeline for domain: {domain} =====")

        # Load embedding config from configuration manager
        config_manager = ConfigurationManager()
        embedding_config = config_manager.get_embeddings_config()
        
        # Get domain-specific vector DB path
        domain_vector_db_path = config_manager.get_domain_vector_db_path(domain)

        # Initialize embedding component with domain-specific path
        embedding_creator = EmbeddingCreator(
            config=embedding_config,
            domain=domain,
            vector_store_path=domain_vector_db_path
        )

        # Create the FAISS vector store
        faiss_db = embedding_creator.main(text_chunks)

        logger.info(f"✅ Embedding Pipeline completed for domain '{domain}'")
        return faiss_db

    except Exception as e:
        logger.exception(f"Embedding Pipeline failed due to: {e}")
        raise e



def load_existing_vector_store(domain: str = "constitution"):
    """
    Load FAISS DB from disk for a specific domain.
    
    Args:
        domain: Domain name to load vector store for
    """
    try:
        logger.info(f"===== Loading Existing FAISS Database for domain: {domain} =====")

        config_manager = ConfigurationManager()
        embedding_config = config_manager.get_embeddings_config()
        domain_vector_db_path = config_manager.get_domain_vector_db_path(domain)

        embedding_creator = EmbeddingCreator(
            config=embedding_config,
            domain=domain,
            vector_store_path=domain_vector_db_path
        )

        # Load from domain-specific path
        db = FAISS.load_local(
            str(domain_vector_db_path),
            embedding_creator.get_embedding_model(),
            allow_dangerous_deserialization=True
        )

        logger.info(f"✅ FAISS Database loaded for domain '{domain}' from: {domain_vector_db_path}")
        return db

    except Exception as e:
        logger.exception(f"Failed to load FAISS database for domain '{domain}': {e}")
        raise e



if __name__ == "__main__":
    try:
        logger.info(f">>>> Stage {STAGE_NAME} started <<<<")

        logger.warning("This file cannot run directly. It requires text_chunks input.")
        logger.warning("Run via main.py after data loader + text chunking pipeline.")

        logger.info(f">>>> Stage {STAGE_NAME} completed <<<<")

    except Exception as e:
        logger.exception(e)

