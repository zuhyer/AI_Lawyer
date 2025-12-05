from pathlib import Path
from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class DataConfig:
    root_dir: Path
    source_url: List[str]
    pdf_directory: Path



@dataclass(frozen= True)
class EmbeddingConfig:
    model: str
    vector_store: str
    vector_store_path: str
    api_key: str

@dataclass(frozen= True)
class LLMConfig:
    provider: str
    model: str
    api_key: str

@dataclass(frozen= True)
class ChunkingConfig:
    chunk_size: int
    chunk_overlap: int
    add_start_index: bool

@dataclass
class config:
    data : DataConfig
    embeddings : EmbeddingConfig
    LLM : LLMConfig
    chunks : ChunkingConfig


