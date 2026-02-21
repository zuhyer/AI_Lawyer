#!/usr/bin/env python3
"""
Simple Domain Test - Verify all 7 domains are accessible
Usage: python verify_domains.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Domain information
DOMAINS = {
    'constitution': 'Indian Constitution articles and amendments',
    'bns_criminal_law': 'Bharatiya Nyaya Sanhita (Criminal Offences)',
    'bnss_procedure': 'Bharatiya Nyaya Sanhita (Criminal Procedure)',
    'sakshya_evidence': 'Indian Evidence Act & Rules',
    'case_law_sc_recent': 'Supreme Court Recent Cases & Precedents',
    'procedure_guides_db': 'Legal Procedures & Guidelines',
    'legal_templates_db': 'Legal Documents & Templates'
}

def check_domain_files(domain: str) -> tuple[bool, str]:
    """Check if domain FAISS files exist"""
    domain_path = Path('vector_db') / domain
    faiss_file = domain_path / 'index.faiss'
    pkl_file = domain_path / 'index.pkl'
    
    if not domain_path.exists():
        return False, f"Directory not found: {domain_path}"
    
    if not faiss_file.exists():
        return False, f"FAISS index missing: {faiss_file}"
    
    if not pkl_file.exists():
        return False, f"Metadata missing: {pkl_file}"
    
    return True, "OK"

def main():
    print("\n" + "="*70)
    print("📊 DOMAIN FILES VERIFICATION")
    print("="*70 + "\n")
    
    working = 0
    failed = 0
    
    for domain, description in DOMAINS.items():
        exists, message = check_domain_files(domain)
        
        domain_display = domain.upper().replace('_', ' ')
        
        if exists:
            size_mb = sum(f.stat().st_size for f in Path('vector_db/'+domain).glob('*')) / (1024*1024)
            print(f"✅ {domain_display:25} | {description:40} | {size_mb:.1f}MB")
            working += 1
        else:
            print(f"❌ {domain_display:25} | {message}")
            failed += 1
    
    print("\n" + "="*70)
    print("📈 SUMMARY")
    print("="*70)
    print(f"✅ Available: {working}/{len(DOMAINS)}")
    print(f"❌ Missing:  {failed}/{len(DOMAINS)}")
    
    if failed == 0:
        print(f"\n✅ All {len(DOMAINS)} domains are ready for use!\n")
        return 0
    else:
        print(f"\n⚠️  {failed} domain(s) need attention\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
