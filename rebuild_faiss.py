#!/usr/bin/env python
"""
Rebuild FAISS vector store due to Pydantic version compatibility issues.
This script regenerates the embeddings and FAISS index from scratch.
"""

import sys
import traceback
from AI_Lawyer.utils.logging_setup import logger

def main():
    try:
        logger.info("🔄 Starting FAISS Rebuild Process")
        logger.info("=" * 60)
        
        # Stage 2: Load and chunk documents
        logger.info("📄 Stage 2: Loading and chunking documents...")
        from AI_Lawyer.pipeline.stage02_Textsplitting import (
            start_data_loader_pipeline,
            start_chunking_pipeline,
        )
        
        documents = start_data_loader_pipeline()
        logger.info(f"✅ Loaded {len(documents)} documents")
        
        text_chunks = start_chunking_pipeline(documents)
        logger.info(f"✅ Created {len(text_chunks)} text chunks")
        
        # Stage 3: Create embeddings and FAISS index
        logger.info("\n🔗 Stage 3: Creating embeddings and FAISS index...")
        from AI_Lawyer.pipeline.stage03_embedding_creation import start_embedding_pipeline
        
        faiss_db = start_embedding_pipeline(text_chunks)
        logger.info("✅ FAISS vector store created and saved successfully!")
        
        logger.info("=" * 60)
        logger.info("✨ FAISS rebuild completed successfully!")
        logger.info("You can now run: python main.py")
        
        return 0
        
    except Exception as e:
        logger.exception(f"❌ FAISS rebuild failed: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
