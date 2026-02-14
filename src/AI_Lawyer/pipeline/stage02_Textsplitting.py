from AI_Lawyer.config.configuration import ConfigurationManager
from AI_Lawyer.components.chunking_component import Data_Loader,Chunking_text
from AI_Lawyer.utils.logging_setup import logger
from AI_Lawyer.utils.common import deduplicate_documents, add_document_metadata

STAGE_NAME = "Text_Chunking"

def start_data_loader_pipeline(domain: str = "constitution"):
    try:
        logger.info(f"===== Starting Data Loading Pipeline for domain: {domain} =====")

        config_manager = ConfigurationManager()
        data_config = config_manager.get_data_ingestion_config()
        domain_config = config_manager.get_domain_chunking_config(domain)

        loader = Data_Loader(config=data_config, domain=domain, domain_config=domain_config)

        # Load documents (single file or all PDFs based on domain config)
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

def start_chunking_pipeline(documents, domain: str = "constitution"):
    try:
        logger.info(f"===== Starting Text Chunking Pipeline for domain: {domain} =====")

        # Apply deduplication before chunking
        documents = deduplicate_documents(documents)
        
        # Add domain metadata
        config_manager = ConfigurationManager()
        domain_config = config_manager.get_domain_chunking_config(domain)
        documents = add_document_metadata(
            documents, 
            domain=domain,
            source_file=domain_config.data_source
        )
        
        # Skip chunking if preserve_full_document is True
        if domain_config.preserve_full_document:
            logger.info(f"Domain '{domain}' preserves full documents — skipping chunking")
            # Return documents as-is (each document = one chunk)
            return documents

        chunker = Chunking_text(config=domain_config)

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

        # Default to constitution domain; can be changed to any domain
        domain = "constitution"
        documents = start_data_loader_pipeline(domain=domain)
        text_chunks = start_chunking_pipeline(documents, domain=domain)

        logger.info(">>>> Stage Text_Chunking completed <<<<")

    except Exception as e:
        logger.exception(e)