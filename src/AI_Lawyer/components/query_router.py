"""
Query Router - Routes user queries to appropriate domains.
Classifies queries and loads the correct FAISS index for retrieval.
"""

from typing import Tuple, Dict, List
from pathlib import Path
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from AI_Lawyer.config.configuration import ConfigurationManager
from AI_Lawyer.utils.logging_setup import logger


class QueryRouter:
    """
    Routes queries to the appropriate legal domain.
    Classifies queries and loads domain-specific FAISS indices.
    """
    
    # Domain keywords for classification (see weights in classify_query)
    DOMAIN_KEYWORDS = {
        'constitution': [
            'article', 'constitution', 'fundamental rights', 'directive principles',
            'appendix', 'schedule', 'preamble', 'amendment', 'rajya sabha', 'lok sabha',
            'governor', 'president', 'voting rights', 'citizenship', 'constitutional',
            'basic structure', 'constitutional amendments', 'part', 'chapter', 'supreme court',
            'judicial review', 'constitutional law', 'fundamental duty', 'directive principle',
            'right to equality', 'right to freedom', 'right to life', 'right to property',
            'right to constitutional remedies', 'freedom of speech', 'freedom of religion',
            'freedom of movement', 'right against exploitation', 'right to education',
            'cultural and educational rights', 'right to work', 'social security',
            'state policy', 'reservation', 'sc st obc', 'affirmative action',
            'constitutional bodies', 'union territory', 'centre state relations',
            'emergency provisions', 'constitutional emergency', 'constitutional validity'
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
    
    def preprocess_query(self, query: str) -> str:
        """Clean and normalize user query before classification.

        - Lowercases text
        - Removes punctuation
        - Expands common abbreviations
        - Strips extra whitespace
        """
        import re

        q = query.lower()
        # remove punctuation
        q = re.sub(r"[^\w\s]", " ", q)

        # expand some known abbreviations into full forms to aid keyword matching
        expansions = {
            'fir': 'first information report',
            'bns': 'bharatiya nyaya sanha',
            'bnss': 'bharatiya nagarik suraksha sanhita',
            'sc': 'supreme court',
            'st': 'scheduled tribe',
            'obc': 'other backward class',
        }
        for abbr, full in expansions.items():
            q = re.sub(rf"\b{abbr}\b", full, q)

        # collapse whitespace
        q = re.sub(r"\s+", " ", q).strip()
        return q

    def apply_domain_boosters(self, scores: Dict[str, float], query: str) -> Dict[str, float]:
        """Lightweight rule-based boosting for certain domain signals."""
        query_lower = query.lower()
        boosted = scores.copy()

        # constitution booster: presence of 'article' or 'amendment'
        if 'article' in query_lower or 'amendment' in query_lower:
            boosted['constitution'] = boosted.get('constitution', 0) * 1.3

        # criminal law booster: explicit 'section' with crime terms
        if 'section' in query_lower and any(w in query_lower for w in ['offence', 'crime', 'punishment']):
            boosted['bns_criminal_law'] = boosted.get('bns_criminal_law', 0) * 1.2

        # procedure booster: words like 'process' or 'steps'
        if any(w in query_lower for w in ['process', 'step', 'procedure', 'filing']):
            boosted['procedure_guides_db'] = boosted.get('procedure_guides_db', 0) * 1.2

        # template booster: presence of 'template' or 'format'
        if any(w in query_lower for w in ['template', 'format', 'draft', 'sample']):
            boosted['legal_templates_db'] = boosted.get('legal_templates_db', 0) * 1.4

        return boosted

    def classify_query(self, query: str) -> Tuple[str, float]:
        """
        Classify query into appropriate domain using keyword matching and semantic similarity.
        
        Args:
            query: User query text
            
        Returns:
            Tuple of (domain_name, confidence_score)
        """
        prepped = self.preprocess_query(query)
        query_lower = prepped
        
        # Step 1: Keyword-based scoring
        keyword_scores = {}
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            keyword_scores[domain] = score

        # apply rule-based boosters
        keyword_scores = self.apply_domain_boosters(keyword_scores, query_lower)
        
        # If clear keyword match, use it
        max_keyword_score = max(keyword_scores.values()) if keyword_scores else 0
        if max_keyword_score >= 2:  # At least 2 keywords match
            top_domain = max(keyword_scores, key=keyword_scores.get)
            confidence = min(0.95, 0.6 + (max_keyword_score * 0.1))
            logger.info(f"🎯 Query classified to '{top_domain}' (keyword-based, confidence={confidence:.2f})")
            return top_domain, confidence
        
        # Step 2: Semantic similarity fallback
        try:
            query_embedding = self.embedding_model.encode(prepped)
            
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
                sim = cosine_similarity([query_embedding], [terms_embedding])[0][0]
                similarities[domain] = sim
            
            top_domain = max(similarities, key=similarities.get)
            confidence = max(0.5, min(0.9, similarities[top_domain]))
            
            logger.info(f"🎯 Query classified to '{top_domain}' (semantic-based, confidence={confidence:.2f})")
            return top_domain, confidence
            
        except Exception as e:
            logger.error(f"❌ Semantic classification failed: {e}", exc_info=True)
            logger.warning(f"Falling back to default domain: 'bns_criminal_law'")
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
