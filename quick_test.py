#!/usr/bin/env python3
"""
Quick test to check which domain has data for "Section 304A"
"""
import sys
from pathlib import Path
sys.path.insert(0, 'src')

from AI_Lawyer.config.configuration import ConfigurationManager
from AI_Lawyer.components.local_embedding import LocalSentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

config = ConfigurationManager()
embedding_model = LocalSentenceTransformerEmbeddings(config.get_embeddings_config().model)

# Check each domain
domains = ['constitution', 'bns_criminal_law', 'bnss_procedure', 'sakshya_evidence', 
           'case_law_sc_recent', 'procedure_guides_db', 'legal_templates_db']

test_query = "Section 304A IPC criminal punishment"

print("\nSearching for 'Section 304A' in each domain:\n")
print("="*70)

for domain in domains:
    path = Path(f'vector_db/{domain}')
    faiss_file = path / 'index.faiss'
    
    if faiss_file.exists():
        try:
            db = FAISS.load_local(str(path), embedding_model, allow_dangerous_deserialization=True)
            results = db.similarity_search(test_query, k=1)
            
            if results:
                score = "High" if len(results) > 0 else "None"
                print(f"✅ {domain:25} | Found {len(results)} result(s)")
                # Print snippet
                preview = results[0].page_content[:80].replace('\n', ' ')
                print(f"   📄 {preview}...")
            else:
                print(f"⚠️  {domain:25} | No results")
        except Exception as e:
            print(f"❌ {domain:25} | Error: {str(e)[:50]}")
    else:
        print(f"❌ {domain:25} | Index not found")

print("="*70)
