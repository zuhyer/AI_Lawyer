from AI_Lawyer.config.configuration import ConfigurationManager
from AI_Lawyer.components.local_embedding import EmbeddingCreator
from AI_Lawyer.utils.logging_setup import logger
from langchain_community.vectorstores import FAISS


STAGE_NAME = "Embedding Stage"


def start_embedding_pipeline(text_chunks):
    """
    Runs the embedding creation process:
      - Loads embedding config
      - Creates FAISS vector store
      - Saves it to disk
    """
    try:
        logger.info("===== Starting Embedding Pipeline =====")

        # Load embedding config from configuration manager
        config_manager = ConfigurationManager()
        embedding_config = config_manager.get_embeddings_config()

        # Initialize embedding component
        embedding_creator = EmbeddingCreator(config=embedding_config)

        # Create the FAISS vector store
        faiss_db = embedding_creator.main(text_chunks)

        logger.info("Embedding Pipeline completed successfully.")
        return faiss_db

    except Exception as e:
        logger.exception(f"Embedding Pipeline failed due to: {e}")
        raise e



def load_existing_vector_store():
    """
    Optional method:
    Load FAISS DB from disk when needed.
    """
    try:
        logger.info("===== Loading Existing FAISS Database =====")

        config_manager = ConfigurationManager()
        embedding_config = config_manager.get_embeddings_config()

        embedding_creator = EmbeddingCreator(config=embedding_config)

        # Load from path
        db = FAISS.load_local(
            embedding_creator.db_path,
            embedding_creator.get_embedding_model(),
            allow_dangerous_deserialization=True
        )

        logger.info("Existing FAISS Database loaded successfully.")
        return db

    except Exception as e:
        logger.exception(f"Failed to load FAISS database: {e}")
        raise e



if __name__ == "__main__":
    try:
        logger.info(f">>>> Stage {STAGE_NAME} started <<<<")

        logger.warning("This file cannot run directly. It requires text_chunks input.")
        logger.warning("Run via main.py after data loader + text chunking pipeline.")

        logger.info(f">>>> Stage {STAGE_NAME} completed <<<<")

    except Exception as e:
        logger.exception(e)

