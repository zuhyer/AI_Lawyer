from AI_Lawyer.config.configuration import ConfigurationManager
from AI_Lawyer.components.data_ingestion import DataIngestion  # assuming this is saved as a component
from AI_Lawyer.utils.logging_setup import logger


STAGE_NAME = "Data Ingestion Pipeline"

def start_data_ingestion():
    try:
        logger.info("Starting Data Ingestion Pipeline")

        config_manager = ConfigurationManager()
        data_config = config_manager.get_data_ingestion_config()

        ingestion = DataIngestion(config=data_config)
        ingestion.download_pdfs()

        logger.info(" Data Ingestion Pipeline completed successfully!")

        return ingestion

    except Exception as e:
        logger.exception(f" Data Ingestion Pipeline failed: {e}")


if __name__ == '__main__':
    try:
        logger.info(f">>>> stage {STAGE_NAME} started <<<<")
        obj = start_data_ingestion()
        obj.main()
        logger.info(f">>>> stage {STAGE_NAME} Completed <<<<")

    except Exception as e:
        logger.exception(e)
        raise e   
    

