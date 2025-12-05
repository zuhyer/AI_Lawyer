"""
main.py
Orchestrates the full AI_Lawyer pipeline:
  - Stage 01: Data Ingestion
  - Stage 02: Text Splitting / Chunking
  - Stage 03: Embedding Creation (FAISS)
  - Stage 04: Query Pipeline (RAG)

Notes:
- Config path: /workspaces/AI_Lawyer/config/config.yaml
- Vectorstore path: /workspaces/AI_Lawyer/vectorstore
- FAISS loading uses: FAISS.load_local(path, embedding_model)
"""

import sys
import logging
from pathlib import Path

# Project imports (expected to exist in your repo)
from AI_Lawyer.config.configuration import ConfigurationManager
from AI_Lawyer.utils.logging_setup import logger as project_logger

# Stage functions (your pipeline files)
from AI_Lawyer.pipeline.stage01_data_ingestion import start_data_ingestion
from AI_Lawyer.pipeline.stage02_Textsplitting import (
    start_data_loader_pipeline,
    start_chunking_pipeline,
)
from AI_Lawyer.pipeline.stage03_embedding_creation import (
    start_embedding_pipeline,
    load_existing_vector_store,
)
# stage04 may expose a start function; we attempt to import it.
try:
    from AI_Lawyer.pipeline.stage04_query_pipeline import start_query_pipeline
    _HAS_STAGE04_START = True
except Exception:
    _HAS_STAGE04_START = False

# Fallback imports (if stage04 not present)
try:
    from AI_Lawyer.components.query_component import QueryComponent
    from AI_Lawyer.entity.config_entity import LLMConfig
    _HAS_QUERY_COMPONENT = True
except Exception:
    _HAS_QUERY_COMPONENT = False

# FAISS
from langchain_community.vectorstores import FAISS  # for load_local if needed

# Setup top-level logging (merge with your project's logger)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("AI_Lawyer.main")
# ensure project logger also prints
project_logger.setLevel(logging.INFO)


# CONFIGURED PATHS (as provided)
CONFIG_PATH = Path("/workspaces/AI_Lawyer/config/config.yaml")
VECTORSTORE_PATH = Path("/workspaces/AI_Lawyer/vectorstore")


# def run_stage_01():
#     """
#     Run Stage 01 - Data ingestion.
#     (You said earlier you may skip it normally; this runs it because you asked for all stages.)
#     """
#     try:
#         logger.info("===== Stage 01: Data Ingestion (start) =====")
#         start_data_ingestion()
#         logger.info("===== Stage 01: Data Ingestion (completed) =====")
#     except Exception as e:
#         logger.exception(f"Stage 01 failed: {e}")
#         raise e


def run_stage_02():
    """
    Run Stage 02 - Text loading + chunking.
    Uses your start_data_loader_pipeline() and start_chunking_pipeline()
    """
    try:
        logger.info("===== Stage 02: Text Loading & Chunking (start) =====")

        documents = start_data_loader_pipeline()
        if not documents:
            logger.warning("No documents loaded in Stage 02. Aborting pipeline.")
            raise RuntimeError("Stage 02 produced no documents.")

        text_chunks = start_chunking_pipeline(documents)
        if not text_chunks:
            logger.warning("No text chunks created in Stage 02. Aborting pipeline.")
            raise RuntimeError("Stage 02 produced no text chunks.")

        logger.info(f"===== Stage 02 completed: {len(text_chunks)} chunks created =====")
        return text_chunks

    except Exception as e:
        logger.exception(f"Stage 02 failed: {e}")
        raise e


def run_stage_03(text_chunks):
    """
    Run Stage 03 - Embedding creation & FAISS saving.
    Uses start_embedding_pipeline(text_chunks) which returns faiss_db.
    """
    try:
        logger.info("===== Stage 03: Embedding Creation (start) =====")

        faiss_db = start_embedding_pipeline(text_chunks)
        if not faiss_db:
            logger.warning("Stage 03 did not return a FAISS DB object. Attempting to load from disk.")
            # as fallback attempt to load from disk using saved path and embedding model
            # NOTE: This requires embedding_model instance; try best-effort:
            config_manager = ConfigurationManager()
            embed_cfg = config_manager.get_embeddings_config()
            # create embedding model to pass to load_local if your component provides it
            # We call load_existing_vector_store() if available (imported above)
            try:
                faiss_db = load_existing_vector_store()
            except Exception as e:
                logger.exception("Failed to load existing FAISS as fallback: %s", e)
                raise e

        logger.info("===== Stage 03 completed: FAISS DB ready =====")
        return faiss_db

    except Exception as e:
        logger.exception(f"Stage 03 failed: {e}")
        raise e


def run_stage_04(faiss_db):
    """
    Run Stage 04 - Query pipeline.
    Note: start_query_pipeline(question, faiss_db) requires user input, so we use QueryComponent directly.
    """
    try:
        logger.info("===== Stage 04: Query Pipeline (start) =====")

        config_manager = ConfigurationManager()
        llm_cfg = config_manager.get_llm_config()

        # Use QueryComponent directly (start_query_pipeline requires user question which orchestration doesn't have)
        if _HAS_QUERY_COMPONENT:
            logger.info("Initializing QueryComponent directly.")
            query_engine = QueryComponent(llm_config=llm_cfg, faiss_db=faiss_db)
            logger.info("QueryComponent initialized. Ready for interactive queries.")
            return query_engine

        # Fallback to pipeline function if QueryComponent not available
        if _HAS_STAGE04_START:
            logger.warning("QueryComponent not found, but start_query_pipeline is available.")
            logger.warning("Note: start_query_pipeline requires a user question parameter.")
            raise RuntimeError("Cannot run stage04 without user question in non-interactive context.")

        # Nothing available
        logger.error("No query pipeline or QueryComponent found in project.")
        raise RuntimeError("Query pipeline not found.")

    except Exception as e:
        logger.exception(f"Stage 04 failed: {e}")
        raise e


def main():
    logger.info("========== AI_Lawyer: Full Pipeline Orchestration START ==========")
    try:
        # confirm config exists
        if not CONFIG_PATH.exists():
            logger.error("Config file not found at: %s", CONFIG_PATH)
            raise FileNotFoundError(f"Config file missing: {CONFIG_PATH}")

        # Stage 1 - SKIPPED (data ingestion assumed to be already done)
        logger.info("===== Stage 01: Data Ingestion (SKIPPED - assuming PDFs already downloaded) =====")

        # Stage 2
        text_chunks = run_stage_02()

        # Stage 3
        faiss_db = run_stage_03(text_chunks)

        # Stage 4
        query_obj = run_stage_04(faiss_db)

        logger.info("========== AI_Lawyer: Full Pipeline Orchestration FINISHED ==========")

        # return objects for programmatic use if imported as module
        return {
            "documents_count": len(text_chunks) if text_chunks else 0,
            "faiss_db": faiss_db,
            "query_engine": query_obj,
        }

    except Exception as e:
        logger.exception(f"Pipeline execution failed: {e}")
        raise e


if __name__ == "__main__":
    main()

