"""
Query Router - Routes user queries to appropriate domains.
Classifies queries and loads the correct FAISS index for retrieval.
"""

from typing import Tuple, Dict, List
from pathlib import Path
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer

from AI_Lawyer.config.configuration import ConfigurationManager
from AI_Lawyer.utils.logging_setup import logger


class QueryRouter:
    """
    Routes queries to the appropriate legal domain.
    Classifies queries and loads domain-specific FAISS indices.
    """
    
    # Domain keywords for classification
    DOMAIN_KEYWORDS = {
        'constitution': [
            'article', 'constitution', 'fundamental rights', 'directive principles',
            'appendix', 'schedule', 'preamble', 'amendment', 'rajya sabha', 'lok sabha',
            'governor', 'president', 'voting rights', 'citizenship'
        ],
        'bns_criminal_law': [
            'section', 'bns', 'offence', 'crime', 'criminal', 'punishment', 'bail',
            'arrest', 'theft', 'rape', 'murder', 'theft', 'fraud', 'embezzlement',
            'guilty', 'conviction', 'sentence', 'imprisonment'
        ],
        'bnss_procedure': [
            'procedure', 'bnss', 'code of criminal procedure', 'investigation',
            'fir', 'charge sheet', 'trial', 'evidence', 'witness', 'cross examination',
            'appeal', 'revision', 'jurisdiction'
        ],
        'sakshya_evidence': [
            'evidence', 'sakshya', 'indian evidence act', 'witness', 'documentary',
            'testimony', 'admission', 'confession', 'hearsay', 'expert opinion',
            'burden of proof', 'relevancy', 'admissibility'
        ],
        'case_law_sc_recent': [
            'judgment', 'ruling', 'case law', 'supreme court', 'precedent',
            'landmark decision', 'writ', 'petition', 'verdict', 'order',
            'bench', 'ratio decidendi', 'obiter dictum'
        ],
        'procedure_guides_db': [
            'procedure', 'process', 'filing', 'petition', 'application', 'form',
            'fee', 'deadline', 'documentation', 'compliance', 'requirement',
            'step', 'guideline', 'instruction'
        ],
        'legal_templates_db': [
            'template', 'format', 'draft', 'sample', 'example', 'document',
            'deed', 'agreement', 'contract', 'will', 'power of attorney',
            'affidavit', 'memorandum', 'clause'
        ]
    }
    
    # Domain to classification query mapping
    DOMAIN_QUERIES = {
        'constitutional': 'constitution',
        'criminal_offence': 'bns_criminal_law',
        'criminal_procedure': 'bnss_procedure',
        'evidence': 'sakshya_evidence',
        'case_law': 'case_law_sc_recent',
        'procedure': 'procedure_guides_db',
        'template_generation': 'legal_templates_db'
    }
    
    def __init__(self, config_manager: ConfigurationManager = None, 
                 embedding_model: SentenceTransformer = None):
        """
        Initialize QueryRouter.
        
        Args:
            config_manager: ConfigurationManager instance
            embedding_model: Pre-loaded SentenceTransformer for embeddings
        """
        self.config_manager = config_manager or ConfigurationManager()
        self.embedding_model = embedding_model
        
        # Load embedding model if not provided
        if not self.embedding_model:
            embedding_config = self.config_manager.get_embeddings_config()
            self.embedding_model = SentenceTransformer(embedding_config.model)
        
        # Load vector DB config
        self.vdb_config = self.config_manager.get_vector_db_config()
        self.faiss_indices: Dict[str, FAISS] = {}
        
        logger.info("✅ QueryRouter initialized")
    
    def classify_query(self, query: str) -> Tuple[str, float]:
        """
        Classify query into appropriate domain using keyword matching and semantic similarity.
        
        Args:
            query: User query text
            
        Returns:
            Tuple of (domain_name, confidence_score)
        """
        query_lower = query.lower()
        
        # Step 1: Keyword-based scoring
        keyword_scores = {}
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            keyword_scores[domain] = score
        
        # If clear keyword match, use it
        max_keyword_score = max(keyword_scores.values()) if keyword_scores else 0
        if max_keyword_score >= 2:  # At least 2 keywords match
            top_domain = max(keyword_scores, key=keyword_scores.get)
            confidence = min(0.95, 0.6 + (max_keyword_score * 0.1))
            logger.info(f"🎯 Query classified to '{top_domain}' (keyword-based, confidence={confidence:.2f})")
            return top_domain, confidence
        
        # Step 2: Semantic similarity fallback
        try:
            query_embedding = self.embedding_model.encode(query)
            
            # Get representative terms for each domain
            domain_terms = {
                domain: ' '.join(keywords[:5]) 
                for domain, keywords in self.DOMAIN_KEYWORDS.items()
            }
            
            # Compute similarity
            similarities = {}
            for domain, terms in domain_terms.items():
                terms_embedding = self.embedding_model.encode(terms)
                # Simple cosine similarity
                from sklearn.metrics.pairwise import cosine_similarity
                sim = cosine_similarity([query_embedding], [terms_embedding])[0][0]
                similarities[domain] = sim
            
            top_domain = max(similarities, key=similarities.get)
            confidence = max(0.5, min(0.9, similarities[top_domain]))
            
            logger.info(f"🎯 Query classified to '{top_domain}' (semantic-based, confidence={confidence:.2f})")
            return top_domain, confidence
            
        except Exception as e:
            logger.warning(f"Semantic classification failed: {e}. Using default domain.")
            return 'bns_criminal_law', 0.5  # Default to criminal law
    
    def load_domain_index(self, domain: str, force_reload: bool = False) -> FAISS:
        """
        Load FAISS index for specific domain.
        
        Args:
            domain: Domain name
            force_reload: Force reload from disk
            
        Returns:
            FAISS index for the domain
            
        Raises:
            FileNotFoundError: If domain index doesn't exist
        """
        # Check cache
        if domain in self.faiss_indices and not force_reload:
            logger.info(f"📚 Using cached FAISS index for domain: {domain}")
            return self.faiss_indices[domain]
        
        try:
            domain_path = self.config_manager.get_domain_vector_db_path(domain)
            index_path = domain_path / "index.faiss"
            
            if not index_path.exists():
                raise FileNotFoundError(
                    f"FAISS index not found for domain '{domain}' at {domain_path}\n"
                    f"Please run ingestion for this domain first."
                )
            
            # Load FAISS index
            faiss_db = FAISS.load_local(
                str(domain_path),
                self.embedding_model,
                allow_dangerous_deserialization=True
            )
            
            # Cache the index
            self.faiss_indices[domain] = faiss_db
            
            logger.info(f"✅ Loaded FAISS index for domain: {domain}")
            return faiss_db
            
        except FileNotFoundError as e:
            logger.error(f"❌ {str(e)}")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to load FAISS index for domain '{domain}': {e}")
            raise
    
    def get_domain_config(self, domain: str) -> Dict:
        """
        Get chunking and retrieval config for domain.
        
        Args:
            domain: Domain name
            
        Returns:
            Dictionary with domain configuration
        """
        try:
            domain_chunking = self.config_manager.get_domain_chunking_config(domain)
            
            return {
                'domain': domain,
                'chunk_size': domain_chunking.chunk_size,
                'chunk_overlap': domain_chunking.chunk_overlap,
                'strategy': domain_chunking.strategy,
                'description': domain_chunking.description
            }
        except Exception as e:
            logger.warning(f"Failed to get domain config for '{domain}': {e}")
            return {
                'domain': domain,
                'chunk_size': 1000,
                'chunk_overlap': 200,
                'strategy': 'default',
                'description': f'Default configuration for {domain}'
            }
    
    def route_query(self, query: str) -> Tuple[str, FAISS, Dict]:
        """
        Route query to appropriate domain and return loaded index.
        
        Args:
            query: User query text
            
        Returns:
            Tuple of (domain_name, faiss_index, domain_config)
        """
        # Classify query
        domain, confidence = self.classify_query(query)
        
        # Load domain index
        try:
            faiss_index = self.load_domain_index(domain)
        except FileNotFoundError:
            logger.warning(f"Index not available for '{domain}'. Using default instead.")
            domain = 'bns_criminal_law'
            faiss_index = self.load_domain_index(domain)
        
        # Get domain configuration
        domain_config = self.get_domain_config(domain)
        domain_config['classification_confidence'] = confidence
        
        logger.info(f"✅ Query routed to domain: {domain} (confidence={confidence:.2f})")
        
        return domain, faiss_index, domain_config
    
    def get_all_available_domains(self) -> List[str]:
        """
        Get list of all available domains with loaded indices.
        
        Returns:
            List of domains with available indices
        """
        available_domains = []
        base_path = Path(self.vdb_config.base_path)
        
        for domain in self.vdb_config.domains:
            domain_path = base_path / domain
            if (domain_path / "index.faiss").exists():
                available_domains.append(domain)
        
        logger.info(f"📊 Available domains: {available_domains}")
        return available_domains
    
    def clear_cache(self):
        """Clear cached FAISS indices."""
        self.faiss_indices.clear()
        logger.info("🧹 Cleared FAISS index cache")
