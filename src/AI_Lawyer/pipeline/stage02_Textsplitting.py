from AI_Lawyer.config.configuration import ConfigurationManager
from AI_Lawyer.components.chunking_component import Data_Loader,Chunking_text
from AI_Lawyer.utils.logging_setup import logger

STAGE_NAME = "Text_Chunking"

def start_data_loader_pipeline():
    try:
        logger.info(f"===== Starting Data Loading Pipeline =====")

        config_manager = ConfigurationManager()
        data_config = config_manager.get_data_ingestion_config()

        loader = Data_Loader(config=data_config)

        # Load all PDF documents
        documents = loader.main()

        logger.info(f"Documents Loaded: {len(documents)}")

        return documents

    except Exception as e:
        logger.exception(f"Data Loading Pipeline failed due to: {e}")
        raise e
    

def start_chunking_pipeline(documents):
    try:
        logger.info(f"===== Starting Text Chunking Pipeline =====")

        config_manager = ConfigurationManager()
        chunk_config = config_manager.get_chunking_config()

        chunker = Chunking_text(config=chunk_config)

        # Pass documents to chunker
        text_chunks = chunker.main(documents)

        logger.info(f"Total Chunks Created: {len(text_chunks)}")

        return text_chunks

    except Exception as e:
        logger.exception(f"Text Chunking Pipeline failed due to: {e}")
        raise e


if __name__ == '__main__':
    try:
        logger.info(">>>> Stage Text_Chunking started <<<<")

        documents = start_data_loader_pipeline()
        text_chunks = start_chunking_pipeline(documents)

        logger.info(">>>> Stage Text_Chunking completed <<<<")

    except Exception as e:
        logger.exception(e)