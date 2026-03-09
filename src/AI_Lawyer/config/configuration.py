import os 
from pathlib import Path
from AI_Lawyer.utils.common import read_yaml, create_directories
from AI_Lawyer.utils.logging_setup import *
from AI_Lawyer.entity.config_entity import (
    DataConfig,
    ChunkingConfig,
    EmbeddingConfig,
    LLMConfig,
    FileExtractorConfig,
    UserUploadProcessorConfig,
    DomainChunkingConfig,
    VectorDBConfig,
    VerificationConfig,
    # additional v2 dataclasses
    RetrievalConfig,
    BM25Config,
    RerankerConfig,
    CachingConfig,
    RateLimitConfig,
    APIKeyConfig,
    CORSConfig,
    InputValidationConfig,
    SecurityConfig,
    LoggingConfig,
)
from AI_Lawyer.constants import *


class ConfigurationManager:
    """Configuration manager for AI Lawyer application."""

    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        """
        Initialize configuration manager.
        
        Args:
            config_filepath: Path to config.yaml
            params_filepath: Path to params.yaml
        """
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config['data']['root_dir']])

    def get_data_ingestion_config(self) -> DataConfig:
        """Get data ingestion configuration."""
        config = self.config['data']
        create_directories([config['pdf_directory']])
        
        data_config = DataConfig(   
            root_dir=Path(config['root_dir']),
            pdf_directory=Path(config['pdf_directory']),
            source_url=config['source_url']
        )
        return data_config
    
    def get_domain_chunking_config(self, domain: str) -> DomainChunkingConfig:
        """Get domain-specific chunking configuration.
        
        Args:
            domain: Domain name (e.g., 'constitution', 'bns_criminal_law')
            
        Returns:
            DomainChunkingConfig instance for the specified domain
        """
        try:
            chunking_config = self.config.get('chunking', {})
            domain_config = chunking_config.get(domain)
            
            if not domain_config:
                logger.warning(f"Domain '{domain}' not found in config. Using default chunking.")
                # Fall back to default params
                default_config = self.params.get('chunkingparams', {})
                domain_config = {
                    'chunk_size': default_config.get('chunk_size', 1000),
                    'chunk_overlap': default_config.get('chunk_overlap', 200),
                    'strategy': 'default',
                    'description': 'Default chunking strategy'
                }
            
            domain_chunking = DomainChunkingConfig(
                domain=domain,
                chunk_size=domain_config.get('chunk_size', 1000),
                chunk_overlap=domain_config.get('chunk_overlap', 200),
                strategy=domain_config.get('strategy', 'default'),
                description=domain_config.get('description', f'Configuration for {domain}'),
                data_source=domain_config.get('data_source'),
                preserve_full_document=domain_config.get('preserve_full_document', False),
                add_start_index=domain_config.get('add_start_index', True)
            )
            
            logger.info(f"✅ Domain '{domain}' config loaded: "
                       f"chunk_size={domain_chunking.chunk_size}, "
                       f"overlap={domain_chunking.chunk_overlap}, "
                       f"strategy={domain_chunking.strategy}, "
                       f"preserve_full_document={domain_chunking.preserve_full_document}, "
                       f"data_source={domain_chunking.data_source}")
            
            return domain_chunking
            
        except Exception as e:
            logger.error(f"Error loading domain config for '{domain}': {e}")
            raise
    
    def get_vector_db_config(self) -> VectorDBConfig:
        """Get vector database configuration for domain-separated indices.
        
        Returns:
            VectorDBConfig instance with domain paths and settings
        """
        try:
            vdb_config = self.config.get('vector_db', {})
            
            vector_db = VectorDBConfig(
                base_path=vdb_config.get('base_path', 'vector_db'),
                default_top_k=vdb_config.get('default_top_k', 5),
                domains=vdb_config.get('domains', [
                    'constitution',
                    'bns_criminal_law',
                    'bnss_procedure',
                    'sakshya_evidence',
                    'case_law_sc_recent',
                    'procedure_guides_db',
                    'legal_templates_db'
                ])
            )
            
            # Create domain directories
            base_path = Path(vector_db.base_path)
            for domain in vector_db.domains:
                domain_path = base_path / domain
                domain_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"✅ Vector DB config loaded: base_path={vector_db.base_path}, "
                       f"domains={len(vector_db.domains)}")
            
            return vector_db
            
        except Exception as e:
            logger.error(f"Error loading vector DB config: {e}")
            raise
    
    def get_verification_config(self) -> VerificationConfig:
        """Get verification pipeline configuration.
        
        Returns:
            VerificationConfig instance with verification thresholds
        """
        try:
            verify_config = self.config.get('verification', {})
            
            verification = VerificationConfig(
                min_confidence_threshold=verify_config.get('min_confidence_threshold', 0.65),
                enable_citation_validation=verify_config.get('enable_citation_validation', True),
                citation_types=verify_config.get('citation_types', [
                    'Section', 'Article', 'Rule', 'Schedule', 'Case Law'
                ])
            )
            
            logger.info(f"✅ Verification config loaded: "
                       f"min_threshold={verification.min_confidence_threshold}, "
                       f"citation_validation={verification.enable_citation_validation}")
            
            return verification
            
        except Exception as e:
            logger.error(f"Error loading verification config: {e}")
            raise
    
    def get_domain_vector_db_path(self, domain: str) -> Path:
        """Get FAISS index path for a specific domain.
        
        Args:
            domain: Domain name
            
        Returns:
            Path to domain-specific FAISS index directory
        """
        vdb_config = self.get_vector_db_config()
        domain_path = Path(vdb_config.base_path) / domain
        domain_path.mkdir(parents=True, exist_ok=True)
        return domain_path

    def get_embeddings_config(self) -> EmbeddingConfig:
        """Get embeddings configuration."""
        config = self.config['embeddings']
        embedding_config = EmbeddingConfig(
            model=config.get('model', ''),
            vector_store=config.get('vector_store', ''),
            vector_store_path=config.get('vector_store_path', ''),
            api_key=config.get('api_key', ''),
            device=config.get('device', 'cpu'),
            batch_size=int(config.get('batch_size', 32)),
            dimension=int(config.get('dimension', 384)),
            normalize=bool(config.get('normalize', True)),
        )
        return embedding_config

    def get_chunking_config(self) -> ChunkingConfig:
        """Get text chunking configuration."""
        config = self.params['chunkingparams']

        chunking_config = ChunkingConfig(
            chunk_size=config['chunk_size'],
            chunk_overlap=config['chunk_overlap'],
            add_start_index=config['add_start_index']
        )
        return chunking_config

    def get_llm_config(self) -> LLMConfig:
        """Get LLM configuration, including optional prompt template."""
        config = self.config['llm']
        llm_config = LLMConfig(
            provider=config.get('provider', ''),
            model=config.get('model', ''),
            api_key=config.get('api_key', ''),
            prompt_template=config.get('prompt_template', ''),
        )
        return llm_config

    def get_file_extractor_config(self) -> FileExtractorConfig:
        """Get file extraction configuration."""
        config = self.config['file_extraction']
        file_extractor_config = FileExtractorConfig(
            supported_formats=config['supported_formats'],
            ocr_enabled=config['ocr_enabled'],
            tesseract_path=config['tesseract_path'],
            ocr_language=config['ocr_language'],
            log_extraction_details=config['log_extraction_details'],
            batch_processing=config['batch_processing']
        )
        return file_extractor_config

    def get_user_upload_processor_config(self) -> UserUploadProcessorConfig:
        """Get user upload processor configuration."""
        config = self.config.get('user_upload_processor', {})
        params = self.params.get('user_upload_params', {})
        
        user_upload_config = UserUploadProcessorConfig(
            chunk_size=params.get('chunk_size', 1000),
            chunk_overlap=params.get('chunk_overlap', 200),
            add_start_index=params.get('add_start_index', True),
            max_upload_size_mb=config.get('max_upload_size_mb', 50),
            temp_index_ttl_seconds=config.get('temp_index_ttl_seconds', 3600)
        )
        return user_upload_config

    # --- additional getters migrated from v2 upgrade ---

    def get_retrieval_config(self) -> RetrievalConfig:
        r = self.config.get("retrieval", {})
        bm25_raw = r.get("bm25", {})
        reranker_raw = r.get("reranker", {})
        return RetrievalConfig(
            top_k=int(r.get("top_k", 8)),
            vector_top_k_multiplier=int(r.get("vector_top_k_multiplier", 3)),
            hybrid_mode=bool(r.get("hybrid_mode", False)),
            vector_weight=float(r.get("vector_weight", 0.65)),
            bm25_weight=float(r.get("bm25_weight", 0.35)),
            rrf_k=int(r.get("rrf_k", 60)),
            score_threshold=float(r.get("score_threshold", 0.0)),
            multi_query_expansion=bool(r.get("multi_query_expansion", False)),
            num_expanded_queries=int(r.get("num_expanded_queries", 2)),
            query_rewriting=bool(r.get("query_rewriting", False)),
            bm25=BM25Config(
                k1=float(bm25_raw.get("k1", 1.5)),
                b=float(bm25_raw.get("b", 0.75)),
                index_path=bm25_raw.get("index_path", "bm25_indices"),
            ),
            reranker=RerankerConfig(
                enabled=bool(reranker_raw.get("enabled", False)),
                model=reranker_raw.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
                device=reranker_raw.get("device", "cpu"),
                batch_size=int(reranker_raw.get("batch_size", 16)),
                score_threshold=float(reranker_raw.get("score_threshold", 0.0)),
            ),
        )

    def get_caching_config(self) -> CachingConfig:
        c = self.config.get("caching", {})
        return CachingConfig(
            enabled=bool(c.get("enabled", False)),
            backend=c.get("backend", "memory"),
            redis_host=os.environ.get("REDIS_HOST", c.get("redis_host", "localhost")),
            redis_port=int(os.environ.get("REDIS_PORT", c.get("redis_port", 6379))),
            redis_db=int(c.get("redis_db", 0)),
            query_cache_ttl=int(c.get("query_cache_ttl", 3600)),
            retrieval_cache_ttl=int(c.get("retrieval_cache_ttl", 1800)),
            key_prefix=c.get("key_prefix", "ai_lawyer:"),
            max_memory_items=int(c.get("max_memory_items", 500)),
        )

    def get_security_config(self) -> SecurityConfig:
        s = self.config.get("security", {})
        rl = s.get("rate_limiting", {})
        ak = s.get("api_key", {})
        cors = s.get("cors", {})
        iv = s.get("input_validation", {})
        pi = s.get("prompt_injection", {})
        return SecurityConfig(
            rate_limiting=RateLimitConfig(
                enabled=bool(rl.get("enabled", False)),
                requests_per_minute=int(rl.get("requests_per_minute", 60)),
                requests_per_hour=int(rl.get("requests_per_hour", 1000)),
                burst_limit=int(rl.get("burst_limit", 10)),
            ),
            api_key=APIKeyConfig(
                enabled=bool(ak.get("enabled", False)),
                header_name=ak.get("header_name", "X-API-Key"),
                keys_env_var=ak.get("keys_env_var", "AILAWYER_API_KEYS"),
                admin_key_env_var=ak.get("admin_key_env_var", "ADMIN_API_KEY"),
            ),
            cors=CORSConfig(
                allow_origins=cors.get("allow_origins", ["*"]),
                allow_methods=cors.get("allow_methods", ["*"]),
                allow_headers=cors.get("allow_headers", ["*"]),
                allow_credentials=bool(cors.get("allow_credentials", False)),
            ),
            input_validation=InputValidationConfig(
                max_query_length=int(iv.get("max_query_length", 2000)),
                max_file_size_mb=int(iv.get("max_file_size_mb", 50)),
                injection_patterns_enabled=bool(iv.get("injection_patterns_enabled", True)),
            ),
            prompt_injection_enabled=bool(pi.get("enabled", True)),
        )

    def get_logging_config(self) -> LoggingConfig:
        lg = self.config.get("logging", {})
        return LoggingConfig(
            level=lg.get("level", "INFO"),
            format=lg.get("format", "text"),
            file_path=lg.get("file_path", None),
            max_bytes=int(lg.get("max_bytes", 10485760)),
            backup_count=int(lg.get("backup_count", 5)),
            include_request_id=bool(lg.get("include_request_id", True)),
            mlflow_tracking_uri=lg.get("mlflow_tracking_uri", "mlruns"),
            mlflow_experiment=lg.get("mlflow_experiment", "ai_lawyer"),
        )
