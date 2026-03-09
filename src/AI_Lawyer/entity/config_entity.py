from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Any

@dataclass(frozen=True)
class DataConfig:
    root_dir: Path
    source_url: List[str]
    pdf_directory: Path


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str
    vector_store: str
    vector_store_path: str
    api_key: str
    # additional fields added in upgraded version
    device: str = "cpu"
    batch_size: int = 32
    dimension: int = 384  # typical for MiniLM, adjust for other models
    normalize: bool = True


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    api_key: str
    # new field controlled via configuration; if blank uses default prompt
    prompt_template: str = ""


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int
    chunk_overlap: int
    add_start_index: bool


@dataclass(frozen=True)
class FileExtractorConfig:
    supported_formats: List[str]
    ocr_enabled: bool
    tesseract_path: str
    ocr_language: str
    log_extraction_details: bool
    batch_processing: bool


@dataclass(frozen=True)
class UserUploadProcessorConfig:
    """Configuration for processing user-uploaded documents."""
    chunk_size: int
    chunk_overlap: int
    add_start_index: bool
    max_upload_size_mb: int
    temp_index_ttl_seconds: int


@dataclass(frozen=True)
class DomainChunkingConfig:
    """Domain-specific chunking configuration."""
    domain: str
    chunk_size: int
    chunk_overlap: int
    strategy: str
    description: str
    data_source: Optional[str] = None
    preserve_full_document: bool = False
    add_start_index: bool = True


@dataclass(frozen=True)
class VectorDBConfig:
    """Vector database configuration with domain support."""
    base_path: str
    default_top_k: int
    domains: List[str]


@dataclass(frozen=True)
class VerificationConfig:
    """Verification pipeline configuration."""
    min_confidence_threshold: float
    enable_citation_validation: bool
    citation_types: List[str]


# ─── New dataclasses introduced in v2 upgrade ───────────────────────────────

@dataclass(frozen=True)
class BM25Config:
    k1: float
    b: float
    index_path: str


@dataclass(frozen=True)
class RerankerConfig:
    enabled: bool
    model: str
    device: str
    batch_size: int
    score_threshold: float


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int
    vector_top_k_multiplier: int
    hybrid_mode: bool
    vector_weight: float
    bm25_weight: float
    rrf_k: int
    score_threshold: float
    multi_query_expansion: bool
    num_expanded_queries: int
    query_rewriting: bool
    bm25: BM25Config
    reranker: RerankerConfig


@dataclass(frozen=True)
class CachingConfig:
    enabled: bool
    backend: str
    redis_host: str
    redis_port: int
    redis_db: int
    query_cache_ttl: int
    retrieval_cache_ttl: int
    key_prefix: str
    max_memory_items: int


@dataclass(frozen=True)
class RateLimitConfig:
    enabled: bool
    requests_per_minute: int
    requests_per_hour: int
    burst_limit: int


@dataclass(frozen=True)
class APIKeyConfig:
    enabled: bool
    header_name: str
    keys_env_var: str
    admin_key_env_var: str


@dataclass(frozen=True)
class CORSConfig:
    allow_origins: List[str]
    allow_methods: List[str]
    allow_headers: List[str]
    allow_credentials: bool


@dataclass(frozen=True)
class InputValidationConfig:
    max_query_length: int
    max_file_size_mb: int
    injection_patterns_enabled: bool


@dataclass(frozen=True)
class SecurityConfig:
    rate_limiting: RateLimitConfig
    api_key: APIKeyConfig
    cors: CORSConfig
    input_validation: InputValidationConfig
    prompt_injection_enabled: bool


@dataclass(frozen=True)
class LoggingConfig:
    level: str
    format: str
    file_path: Optional[str]
    max_bytes: int
    backup_count: int
    include_request_id: bool
    mlflow_tracking_uri: str


@dataclass
class config:
    data: DataConfig
    embeddings: EmbeddingConfig
    LLM: LLMConfig
    chunks: ChunkingConfig

