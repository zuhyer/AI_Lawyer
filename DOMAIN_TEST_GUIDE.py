#!/usr/bin/env python3
"""
Query Examples for Testing All 7 Legal Domains
Shows Python code and curl commands for testing each domain
"""

# ============================================================================
# METHOD 1: Python Script - Direct Domain Testing
# ============================================================================
"""
Run with: python -c "exec(open('test_domains.py').read())"

import sys
from pathlib import Path
sys.path.insert(0, 'src')

from AI_Lawyer.config.configuration import ConfigurationManager
from AI_Lawyer.components.local_embedding import LocalSentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

config_manager = ConfigurationManager()
embedding_config = config_manager.get_embeddings_config()
embedding_model = LocalSentenceTransformerEmbeddings(embedding_config.model)

# Test queries for each domain
test_queries = {
    'constitution': 'What are fundamental rights in India?',
    'bns_criminal_law': 'What is the punishment for theft?',
    'bnss_procedure': 'What is the procedure for filing an FIR?',
    'sakshya_evidence': 'What types of evidence are admissible in court?',
    'case_law_sc_recent': 'What are important Supreme Court judgments?',
    'procedure_guides_db': 'What are the steps to file a petition?',
    'legal_templates_db': 'What is a power of attorney template?'
}

domains_status = {}
for domain, query in test_queries.items():
    try:
        faiss_db = FAISS.load_local(
            f'vector_db/{domain}',
            embedding_model,
            allow_dangerous_deserialization=True
        )
        results = faiss_db.similarity_search(query, k=2)
        domains_status[domain] = f"✅ OK - Found {len(results)} results"
        print(f"✅ {domain}: {len(results)} documents retrieved")
    except Exception as e:
        domains_status[domain] = f"❌ Failed - {str(e)}"
        print(f"❌ {domain}: {str(e)}")

# Summary
print("\\n" + "="*60)
print("SUMMARY")
print("="*60)
for domain, status in domains_status.items():
    print(f"{domain:25} | {status}")
"""

# ============================================================================
# METHOD 2: Curl Commands - API Testing
# ============================================================================
"""
First, start the API server:
python api_server.py

Then run these curl commands in a separate terminal:
"""

# QUERY EACH DOMAIN
"""
# 1. CONSTITUTION
curl -X POST http://localhost:8000/query/domain/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are fundamental rights in India?",
    "domain": "constitution"
  }'

# 2. BNS CRIMINAL LAW
curl -X POST http://localhost:8000/query/domain/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the punishment for theft?",
    "domain": "bns_criminal_law"
  }'

# 3. BNSS PROCEDURE
curl -X POST http://localhost:8000/query/domain/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the procedure for filing an FIR?",
    "domain": "bnss_procedure"
  }'

# 4. SAKSHYA EVIDENCE
curl -X POST http://localhost:8000/query/domain/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What types of evidence are admissible in court?",
    "domain": "sakshya_evidence"
  }'

# 5. CASE LAW (SUPREME COURT)
curl -X POST http://localhost:8000/query/domain/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are important Supreme Court judgments?",
    "domain": "case_law_sc_recent"
  }'

# 6. PROCEDURE GUIDES
curl -X POST http://localhost:8000/query/domain/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the steps to file a petition?",
    "domain": "procedure_guides_db"
  }'

# 7. LEGAL TEMPLATES
curl -X POST http://localhost:8000/query/domain/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is a power of attorney?",
    "domain": "legal_templates_db"
  }'

# CHECK HEALTH
curl -X GET http://localhost:8000/health
"""

# ============================================================================
# METHOD 3: Python - Comprehensive Test Script
# ============================================================================
"""
Save as: test_comprehensive.py

python test_comprehensive.py
"""

COMPREHENSIVE_TEST_CODE = '''
#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, 'src')

from AI_Lawyer.config.configuration import ConfigurationManager
from AI_Lawyer.components.local_embedding import LocalSentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
import time

# Configuration
config_manager = ConfigurationManager()
embedding_config = config_manager.get_embeddings_config()
embedding_model = LocalSentenceTransformerEmbeddings(embedding_config.model)

# Domain test queries
DOMAIN_TESTS = {
    'constitution': [
        'What are fundamental rights?',
        'What is Article 15?',
        'Explain the Preamble'
    ],
    'bns_criminal_law': [
        'What is the punishment for theft?',
        'Define criminal offence',
        'What is Section 392?'
    ],
    'bnss_procedure': [
        'What is FIR?',
        'Explain arrest procedure',
        'What is bail?'
    ],
    'sakshya_evidence': [
        'What is evidence?',
        'Types of testimony',
        'What is hearsay evidence?'
    ],
    'case_law_sc_recent': [
        'landmark Supreme Court decisions',
        'Important case laws',
        'What is precedent?'
    ],
    'procedure_guides_db': [
        'How to file a petition?',
        'Filing requirements',
        'Steps in civil procedure'
    ],
    'legal_templates_db': [
        'Power of attorney template',
        'Contract templates',
        'Legal document samples'
    ]
}

def test_domain(domain: str):
    """Test a single domain"""
    print(f"\\n{'='*70}")
    print(f"Testing: {domain.upper().replace('_', ' ')}")
    print(f"{'='*70}")
    
    try:
        # Load FAISS index
        faiss_db = FAISS.load_local(
            f'vector_db/{domain}',
            embedding_model,
            allow_dangerous_deserialization=True
        )
        print(f"✅ Index loaded successfully")
        
        # Test multiple queries
        queries = DOMAIN_TESTS[domain]
        for i, query in enumerate(queries, 1):
            print(f"\\n  Query {i}: {query}")
            results = faiss_db.similarity_search(query, k=2)
            print(f"  ✅ Found {len(results)} results")
            if results:
                print(f"  Top result: {results[0].page_content[:80]}...")
        
        return True
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

# Run tests
print("\\n" + "="*70)
print("COMPREHENSIVE DOMAIN TEST SUITE".center(70))
print("="*70)

results = {}
start_time = time.time()

for domain in DOMAIN_TESTS.keys():
    results[domain] = test_domain(domain)

# Summary
elapsed = time.time() - start_time
working = sum(results.values())

print(f"\\n\\n{'='*70}")
print("TEST SUMMARY".center(70))
print(f"{'='*70}")
print(f"Total Domains: {len(results)}")
print(f"Working: {working}")
print(f"Failed: {len(results) - working}")
print(f"Time: {elapsed:.2f}s")
print(f"{'='*70}")

for domain, status in results.items():
    status_str = "✅ PASS" if status else "❌ FAIL"
    print(f"{domain:25} | {status_str}")

if working == len(results):
    print(f"\\n✅ SUCCESS: All {len(results)} domains working!")
else:
    print(f"\\n⚠️  PARTIAL: {working}/{len(results)} domains working")
'''

# ============================================================================
# METHOD 4: SQ-Like Query Monitor
# ============================================================================
"""
SQL-Like Query Examples:

SELECT * FROM constitution 
WHERE query = 'What are fundamental rights?'
LIMIT 3;

SELECT * FROM bns_criminal_law 
WHERE query = 'punishment for theft'
LIMIT 3;

SELECT * FROM bnss_procedure 
WHERE query = 'FIR filing procedure'
TOP_K = 5;

SELECT * FROM sakshya_evidence 
WHERE query = 'types of evidence'
SIMILARITY_THRESHOLD = 0.7;

SELECT * FROM case_law_sc_recent 
WHERE query = 'landmark decisions'
LIMIT 2;

SELECT * FROM procedure_guides_db 
WHERE query = 'petition filing steps'
LIMIT 3;

SELECT * FROM legal_templates_db 
WHERE query = 'power of attorney'
LIMIT 2;
"""

print(__doc__)

if __name__ == "__main__":
    print("\n" + "="*70)
    print("DOMAIN TEST METHODS".center(70))
    print("="*70)
    print("""
✅ METHOD 1: Direct Python FAISS Testing
   - Test each domain by loading FAISS index
   - Verify documents can be retrieved
   - Shows embeddings are working

✅ METHOD 2: API REST Testing
   - Use /query/domain/ask endpoint
   - Test with different queries
   - Verify API integration

✅ METHOD 3: Comprehensive Test Suite
   - Multi-query testing per domain
   - Performance metrics
   - Detailed reporting

✅ METHOD 4: SQL-Like Monitoring
   - Simulate SQL queries
   - Set similarity thresholds
   - Monitor search quality
    """)
    print("="*70)
