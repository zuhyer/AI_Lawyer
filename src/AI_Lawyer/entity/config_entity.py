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


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    api_key: str


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int
    chunk_overlap: int
    add_start_index: bool


@dataclass(frozen=True)
class DomainChunkingConfig:
    """Domain-specific chunking configuration."""
    domain: str
    chunk_size: int
    chunk_overlap: int
    strategy: str
    description: str
    # Optional path to the original data source for this domain
    data_source: Optional[str] = None
    # If True, the domain's documents should be preserved as full documents
    # and not passed through the chunking pipeline (e.g., legal templates)
    preserve_full_document: bool = False


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
class VectorDBConfig:
    """Vector database configuration for domain-separated indices."""
    base_path: str
    default_top_k: int
    domains: List[str]


@dataclass(frozen=True)
class VerificationConfig:
    """Verification pipeline configuration."""
    min_confidence_threshold: float
    enable_citation_validation: bool
    citation_types: List[str]


@dataclass(frozen=True)
class FinalResponse:
    """Final structured response with domain, citations, and confidence."""
    answer: str
    domain_used: str
    citations: List[str]
    verified: bool
    confidence_score: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class config:
    data: DataConfig
    embeddings: EmbeddingConfig
    LLM: LLMConfig
    chunks: ChunkingConfig
    vector_db: VectorDBConfig
    verification: VerificationConfig


