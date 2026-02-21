#!/usr/bin/env python3
"""
Ingest all domains - Create FAISS indices from PDFs
Run this once to populate all domain FAISS indices with actual data
"""

import sys
from pathlib import Path
sys.path.insert(0, 'src')

from AI_Lawyer.pipeline.stage02_Textsplitting import start_data_loader_pipeline, start_chunking_pipeline
from AI_Lawyer.pipeline.stage03_embedding_creation import start_embedding_pipeline
from AI_Lawyer.utils.logging_setup import logger

def ingest_all_domains():
    """Ingest data for all legal domains"""
    
    domains = [
        'constitution',
        'bns_criminal_law',
        'bnss_procedure',
        'sakshya_evidence',
        'case_law_sc_recent',
        'procedure_guides_db',
        'legal_templates_db'
    ]
    
    print("\n" + "="*70)
    print("🚀 INGESTING ALL LEGAL DOMAINS")
    print("="*70 + "\n")
    
    successful = []
    failed = []
    
    for domain in domains:
        try:
            print(f"\n{'='*70}")
            print(f"📚 Processing Domain: {domain.upper()}")
            print(f"{'='*70}")
            
            # Step 1: Load documents
            print(f"\n✏️  Step 1: Loading documents...")
            documents = start_data_loader_pipeline(domain=domain)
            print(f"   ✅ Loaded {len(documents)} documents")
            
            # Step 2: Chunk documents
            print(f"\n✏️  Step 2: Chunking documents...")
            chunks = start_chunking_pipeline(documents, domain=domain)
            print(f"   ✅ Created {len(chunks)} chunks")
            
            # Step 3: Create embeddings and FAISS index
            print(f"\n✏️  Step 3: Creating embeddings and FAISS index...")
            start_embedding_pipeline(chunks, domain=domain)
            print(f"   ✅ FAISS index created for {domain}")
            
            successful.append(domain)
            print(f"\n✅ SUCCESS: {domain} domain ingested!")
            
        except Exception as e:
            failed.append((domain, str(e)))
            print(f"\n❌ FAILED: {domain} - {str(e)[:100]}")
            logger.exception(f"Failed to ingest {domain}: {e}")
    
    # Summary
    print(f"\n\n{'='*70}")
    print("📊 INGESTION SUMMARY")
    print(f"{'='*70}")
    
    print(f"\n✅ Successfully ingested ({len(successful)}/{len(domains)}):")
    for domain in successful:
        print(f"   • {domain}")
    
    if failed:
        print(f"\n❌ Failed ({len(failed)}/{len(domains)}):")
        for domain, error in failed:
            print(f"   • {domain}: {error[:50]}...")
    
    print(f"\n{'='*70}\n")
    
    if len(failed) == 0:
        print("🎉 All domains successfully ingested!")
        return 0
    else:
        print(f"⚠️  {len(failed)} domain(s) failed")
        return 1

if __name__ == "__main__":
    try:
        exit_code = ingest_all_domains()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(2)
