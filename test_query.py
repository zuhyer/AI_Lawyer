#!/usr/bin/env python3
"""
Test script to verify the query engine works with local embeddings.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from AI_Lawyer.config.configuration import ConfigurationManager
from AI_Lawyer.components.query_component import QueryComponent
from AI_Lawyer.pipeline.stage03_embedding_creation import load_existing_vector_store
from AI_Lawyer.utils.logging_setup import logger


def main():
    try:
        logger.info("===== Testing Query Engine =====")

        # Load configuration
        config_manager = ConfigurationManager()
        llm_cfg = config_manager.get_llm_config()

        # Load existing FAISS database
        logger.info("Loading FAISS database...")
        faiss_db = load_existing_vector_store()

        if faiss_db is None:
            logger.error("FAISS database not found. Please run main.py first.")
            return

        logger.info(f"✓ FAISS database loaded successfully")

        # Initialize QueryComponent
        logger.info("Initializing QueryComponent...")
        query_engine = QueryComponent(llm_config=llm_cfg, faiss_db=faiss_db)
        logger.info("✓ QueryComponent initialized")

        # Test query
        test_question = "What are the basic human rights?"
        logger.info(f"Executing test query: '{test_question}'")

        response = query_engine.execute_query(test_question)

        logger.info("=" * 80)
        logger.info("QUERY RESPONSE:")
        logger.info("=" * 80)
        print(response)
        logger.info("=" * 80)
        logger.info("✓ Query executed successfully!")

    except Exception as e:
        logger.exception(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    main()
