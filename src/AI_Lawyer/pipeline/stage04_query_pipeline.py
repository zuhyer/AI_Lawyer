from AI_Lawyer.config.configuration import ConfigurationManager
from AI_Lawyer.components.query_component import QueryComponent
from AI_Lawyer.utils.logging_setup import logger
from langchain_community.vectorstores import FAISS

STAGE_NAME = "Query Stage"
def start_query_pipeline(question, faiss_db):
    """
    Runs the query process:
      - Loads LLM config
      - Initializes QueryComponent
      - Executes query against FAISS vector store
    """
    try:
        logger.info("===== Starting Query Pipeline =====")

        # Load LLM config from configuration manager
        config_manager = ConfigurationManager()
        llm_config = config_manager.get_llm_config()

        # Initialize QueryComponent
        query_component = QueryComponent(llm_config=llm_config, faiss_db=faiss_db)

        # Execute the query
        response = query_component.execute_query(question)

        logger.info("Query Pipeline completed successfully.")
        return response

    except Exception as e:
        logger.exception(f"Query Pipeline failed due to: {e}")
        raise e