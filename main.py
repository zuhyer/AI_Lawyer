import sys
import os
import logging
from pathlib import Path

# ============================================================
# Project Imports
# ============================================================

from AI_Lawyer.config.configuration import ConfigurationManager

from AI_Lawyer.utils.logging_setup import logger as project_logger

from AI_Lawyer.pipeline.stage02_Textsplitting import (
    start_data_loader_pipeline,
    start_chunking_pipeline,
)

# 🔥 IMPORTANT: Import classes directly (no wrapper functions)
from AI_Lawyer.pipeline.stage03_embedding_creation import EmbeddingCreator

# ============================================================
# Logging Setup
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("AI_Lawyer.main")
project_logger.setLevel(logging.INFO)

CONFIG_PATH = Path("/workspaces/AI_Lawyer/config/config.yaml")


# ============================================================
# Stage 02 - Load + Chunk
# ============================================================

def run_stage_02(domain: str):

    logger.info(f"📂 Stage 02: Loading + Chunking for '{domain}'")

    documents = start_data_loader_pipeline(domain=domain)
    if not documents:
        logger.warning(f"No documents found for domain '{domain}'")
        return []

    text_chunks = start_chunking_pipeline(documents, domain=domain)
    if not text_chunks:
        logger.warning(f"No chunks created for domain '{domain}'")
        return []

    logger.info(f"✅ Stage 02 complete for '{domain}' → {len(text_chunks)} chunks")
    return text_chunks


# ============================================================
# Stage 03 - Embedding + FAISS (DIRECT CLASS USAGE)
# ============================================================

def run_stage_03(text_chunks, domain: str):

    logger.info(f"🔗 Stage 03: Embedding + FAISS for '{domain}'")

    config_manager = ConfigurationManager()
    embedding_config = config_manager.get_embeddings_config()

    embedding_creator = EmbeddingCreator(
        config=embedding_config,
        domain=domain,
        config_manager=config_manager,
    )

    faiss_db = embedding_creator.main(text_chunks)

    logger.info(f"✅ FAISS created for '{domain}'")

    return faiss_db


# ============================================================
# MAIN ORCHESTRATION
# ============================================================

def main():

    logger.info("=" * 80)
    logger.info("🚀 AI_Lawyer Multi-Domain Pipeline START")
    logger.info("=" * 80)

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")

    config_manager = ConfigurationManager()
    llm_cfg = config_manager.get_llm_config()
    # show prompt template if provided
    if llm_cfg.prompt_template:
        logger.info("LLM prompt template loaded from config")
    vector_db_config = config_manager.get_vector_db_config()

    domains = vector_db_config.domains

    logger.info(f"📊 Domains Found: {len(domains)}")
    logger.info(f"📌 Domain List: {', '.join(domains)}")

    domain_results = {}

    for domain in domains:

        logger.info("\n" + "=" * 80)
        logger.info(f"🔄 PROCESSING DOMAIN: {domain}")
        logger.info("=" * 80)

        try:
            # Stage 02
            text_chunks = run_stage_02(domain)

            if not text_chunks:
                domain_results[domain] = {
                    "status": "skipped",
                    "reason": "no_chunks"
                }
                continue

            # Stage 03
            faiss_db = run_stage_03(text_chunks, domain)

            domain_results[domain] = {
                "status": "success",
                "chunks": len(text_chunks),
            }

            logger.info(f"✅ Domain '{domain}' completed successfully")

        except Exception as e:
            logger.exception(f"❌ Domain '{domain}' failed")
            domain_results[domain] = {
                "status": "failed",
                "error": str(e),
            }

    # ============================================================
    # SUMMARY
    # ============================================================

    logger.info("\n" + "=" * 80)
    logger.info("📈 PIPELINE SUMMARY")
    logger.info("=" * 80)

    success = sum(1 for r in domain_results.values() if r["status"] == "success")
    failed = sum(1 for r in domain_results.values() if r["status"] == "failed")
    skipped = sum(1 for r in domain_results.values() if r["status"] == "skipped")

    logger.info(f"✅ Successful Domains: {success}")
    logger.info(f"❌ Failed Domains: {failed}")
    logger.info(f"⚠️ Skipped Domains: {skipped}")

    for domain, result in domain_results.items():
        icon = "✅" if result["status"] == "success" else "❌" if result["status"] == "failed" else "⚠️"
        logger.info(f"{icon} {domain} → {result['status']}")

    logger.info("\n🎯 Domain-separated vector stores ready.")
    logger.info("=" * 80)

    return domain_results


if __name__ == "__main__":
    main()

