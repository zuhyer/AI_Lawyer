from AI_Lawyer.config.configuration import ConfigurationMannager
from AI_Lawyer.components.chunking_component import Data_Loader
from AI_Lawyer.utils.logging_setup import logger

STAGE_NAME = "data_loading"

def start_data_loader_pipeline():

    try:
        logger.info(f"===== Starting {STAGE_NAME} Pipeline =====")

        # Load configuration
        config_manager = ConfigurationMannager()
        data_config = config_manager.get_data_ingestion_config()   # same config object

        # Initialize Data Loader Component
        loader = Data_Loader(config=data_config)

        # Load all PDFs inside directory
        documents = loader.main()

        logger.info(f"===== {STAGE_NAME} Pipeline Completed Successfully! =====")
        return documents

    except Exception as e:
        logger.exception(f"{STAGE_NAME} Pipeline failed due to: {e}")


if __name__ == '__main__':
    try:
        logger.info(f">>>> Stage {STAGE_NAME} started <<<<")
        start_data_loader_pipeline()
        logger.info(f">>>> Stage {STAGE_NAME} completed <<<<")
    except Exception as e:
        logger.exception(e)
