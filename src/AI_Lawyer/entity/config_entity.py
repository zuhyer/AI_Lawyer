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


@dataclass
class config:
    data: DataConfig
    embeddings: EmbeddingConfig
    LLM: LLMConfig
    chunks: ChunkingConfig


