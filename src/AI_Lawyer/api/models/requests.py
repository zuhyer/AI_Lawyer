"""Pydantic request models for API endpoints - Production Grade."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator, ConfigDict
from enum import Enum
from datetime import datetime


# ===== ENUMS =====

class DocumentTypeEnum(str, Enum):
    """Supported document types."""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    IMAGE = "image"
    ALL = "all"


class ProcessingPriorityEnum(str, Enum):
    """Processing priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class QueryModeEnum(str, Enum):
    """Query processing modes."""
    STANDARD = "standard"
    HYBRID = "hybrid"
    SEMANTIC = "semantic"


# ===== BASE MODELS =====

class BaseRequest(BaseModel):
    """Base request model with common fields."""
    
    model_config = ConfigDict(validate_assignment=True)
    
    request_id: Optional[str] = Field(
        None,
        description="Unique request identifier for tracking"
    )
    timeout: int = Field(
        30,
        ge=5,
        le=300,
        description="Request timeout in seconds"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata for request"
    )


# ===== FILE EXTRACTION MODELS =====

class ExtractionRequest(BaseRequest):
    """Request model for file extraction endpoint."""
    
    file_path: Optional[str] = Field(
        None, 
        description="Path to single file for extraction"
    )
    file_paths: Optional[List[str]] = Field(
        None,
        description="List of file paths for batch extraction"
    )
    directory_path: Optional[str] = Field(
        None,
        description="Directory path to extract all supported files from"
    )
    document_type: DocumentTypeEnum = Field(
        DocumentTypeEnum.ALL,
        description="Filter by document type"
    )
    extract_images: bool = Field(
        False,
        description="Whether to extract images from documents"
    )
    preserve_formatting: bool = Field(
        True,
        description="Preserve original formatting in extracted text"
    )
    priority: ProcessingPriorityEnum = Field(
        ProcessingPriorityEnum.MEDIUM,
        description="Processing priority"
    )

    @validator("file_paths")
    def validate_file_paths(cls, v):
        if v is not None and len(v) > 100:
            raise ValueError("Maximum 100 files per request")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "file_path": "/path/to/document.pdf",
                "extract_images": False,
                "preserve_formatting": True
            }
        }


class BatchExtractionRequest(BaseRequest):
    """Request model for batch file extraction."""
    
    extraction_requests: List[ExtractionRequest] = Field(
        ...,
        min_items=1,
        max_items=10,
        description="List of extraction requests"
    )
    parallel: bool = Field(
        True,
        description="Process extractions in parallel"
    )


# ===== DATA INGESTION MODELS =====

class DataIngestionRequest(BaseRequest):
    """Request model for data ingestion/indexing."""
    
    documents: List[str] = Field(
        ...,
        min_items=1,
        description="List of document texts to ingest"
    )
    collection_name: str = Field(
        "default",
        description="Collection/category name for the documents"
    )
    chunk_size: int = Field(
        512,
        ge=100,
        le=4096,
        description="Size of text chunks for embedding"
    )
    chunk_overlap: int = Field(
        128,
        ge=0,
        le=512,
        description="Overlap between chunks"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata for documents"
    )
    reindex: bool = Field(
        False,
        description="Whether to reindex existing documents"
    )


class BulkIngestionRequest(BaseRequest):
    """Request model for bulk data ingestion."""
    
    ingestion_requests: List[DataIngestionRequest] = Field(
        ...,
        min_items=1,
        max_items=5,
        description="List of ingestion requests"
    )


# ===== QUERY/RAG MODELS =====

class QueryRequest(BaseRequest):
    """Request model for query/RAG endpoint."""
    
    query: str = Field(
        ..., 
        min_length=1, 
        max_length=5000,
        description="User query/question for the RAG system"
    )
    mode: QueryModeEnum = Field(
        QueryModeEnum.STANDARD,
        description="Query processing mode"
    )
    top_k: int = Field(
        5,
        ge=1,
        le=50,
        description="Number of top results to return"
    )
    score_threshold: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score threshold"
    )
    use_reranker: bool = Field(
        False,
        description="Whether to use reranking on results"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are the legal obligations under Section 304A of IPC?",
                "mode": "standard",
                "top_k": 5,
                "score_threshold": 0.3,
                "use_reranker": False
            }
        }


class HybridQueryRequest(BaseRequest):
    """Request model for hybrid query endpoint (with user uploads)."""
    
    query: str = Field(
        ..., 
        min_length=1, 
        max_length=5000,
        description="User query/question for the RAG system"
    )
    top_k: int = Field(
        5,
        ge=1,
        le=50,
        description="Number of top results to return per search"
    )
    use_permanent_db: bool = Field(
        True,
        description="Include permanent legal database in search"
    )
    use_user_uploads: bool = Field(
        True,
        description="Include user-uploaded documents in search"
    )
    use_reranker: bool = Field(
        False,
        description="Whether to use reranking on results"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are the constitutional rights in India?",
                "top_k": 5,
                "use_permanent_db": True,
                "use_user_uploads": True,
                "use_reranker": False
            }
        }


# ===== FEEDBACK & LOGGING MODELS =====

class FeedbackRequest(BaseRequest):
    """Request model for feedback on query results."""
    
    query_id: str = Field(
        ...,
        description="ID of the original query"
    )
    rating: int = Field(
        ...,
        ge=1,
        le=5,
        description="Rating from 1-5"
    )
    feedback_text: Optional[str] = Field(
        None,
        max_length=1000,
        description="Feedback text"
    )
    was_helpful: bool = Field(
        ...,
        description="Whether the response was helpful"
    )
    tags: Optional[List[str]] = Field(
        None,
        description="Tags for categorizing feedback"
    )
