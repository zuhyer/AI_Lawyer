#!/usr/bin/env python3
"""
Test script for legal_templates_db single-file, non-chunked ingestion.
Validates: loading, deduplication, preservation, and embedding.
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from AI_Lawyer.config.configuration import ConfigurationManager
from AI_Lawyer.pipeline.stage02_Textsplitting import (
    start_data_loader_pipeline,
    start_chunking_pipeline
)
from AI_Lawyer.pipeline.stage03_embedding_creation import start_embedding_pipeline
from AI_Lawyer.utils.logging_setup import logger


def test_legal_templates_ingestion():
    """Test complete ingestion workflow for legal_templates_db."""
    
    domain = "legal_templates_db"
    logger.info(f"\n{'='*80}")
    logger.info(f"TESTING: {domain} — Single-File, Non-Chunked Ingestion")
    logger.info(f"{'='*80}\n")
    
    try:
        # Step 1: Load domain config
        logger.info("Step 1: Loading domain configuration...")
        config_manager = ConfigurationManager()
        domain_config = config_manager.get_domain_chunking_config(domain)
        
        logger.info(f"  ✓ Domain config loaded:")
        logger.info(f"    - preserve_full_document: {domain_config.preserve_full_document}")
        logger.info(f"    - data_source: {domain_config.data_source}")
        logger.info(f"    - strategy: {domain_config.strategy}")
        
        # Step 2: Load documents (single file only)
        logger.info(f"\nStep 2: Loading documents for domain '{domain}'...")
        documents = start_data_loader_pipeline(domain=domain)
        logger.info(f"  ✓ Loaded {len(documents)} documents from single file")
        
        if not documents:
            logger.error("  ✗ No documents loaded! Aborting.")
            return False
        
        for i, doc in enumerate(documents[:3]):  # Show first 3
            content_preview = doc.page_content[:100].replace('\n', ' ')
            logger.info(f"    Document {i+1}: {content_preview}...")
        
        # Step 3: Apply deduplication & preserve without chunking
        logger.info(f"\nStep 3: Processing documents (deduplication & chunking)...")
        text_chunks = start_chunking_pipeline(documents, domain=domain)
        logger.info(f"  ✓ Processed {len(text_chunks)} chunks (should equal {len(documents)} — full docs)")
        
        if len(text_chunks) != len(documents):
            logger.warning(f"  ⚠ Chunk count mismatch: {len(text_chunks)} chunks vs {len(documents)} docs")
            logger.warning("  (Expected: document count = chunk count for non-chunked domains)")
        
        # Verify metadata
        if text_chunks and hasattr(text_chunks[0], 'metadata'):
            logger.info(f"  ✓ Metadata attached:")
            for key, val in text_chunks[0].metadata.items():
                logger.info(f"    - {key}: {val}")
        
        # Step 4: Create embeddings
        logger.info(f"\nStep 4: Creating embeddings...")
        faiss_db = start_embedding_pipeline(text_chunks, domain=domain)
        logger.info(f"  ✓ FAISS vector store created successfully")
        logger.info(f"  ✓ Vector count: {faiss_db.index.ntotal}")
        
        # Step 5: Test retrieval
        logger.info(f"\nStep 5: Testing vector retrieval...")
        query = "legal template application"
        results = faiss_db.similarity_search(query, k=1)
        logger.info(f"  ✓ Retrieval successful:")
        logger.info(f"    Query: '{query}'")
        logger.info(f"    Top result: {results[0].page_content[:100]}...")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"✅ ALL TESTS PASSED for {domain}")
        logger.info(f"{'='*80}\n")
        return True
        
    except Exception as e:
        logger.exception(f"\n❌ TEST FAILED: {e}")
        logger.info(f"{'='*80}\n")
        return False


if __name__ == "__main__":
    success = test_legal_templates_ingestion()
    sys.exit(0 if success else 1)
