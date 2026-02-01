"""Pydantic response models for API endpoints - Production Grade."""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from enum import Enum


# ===== ENUMS =====

class StatusEnum(str, Enum):
    """Status values for responses."""
    SUCCESS = "success"
    PROCESSING = "processing"
    FAILED = "failed"
    PARTIAL = "partial"


class ComponentStatusEnum(str, Enum):
    """Component status values."""
    OK = "ok"
    DEGRADED = "degraded"
    ERROR = "error"
    UNKNOWN = "unknown"


# ===== BASE RESPONSE =====

class BaseResponse(BaseModel):
    """Base response model with common fields."""
    
    model_config = ConfigDict(validate_assignment=True)
    
    success: bool = Field(..., description="Whether request was successful")
    message: str = Field(..., description="Status message")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response timestamp"
    )
    request_id: Optional[str] = Field(
        None,
        description="Request ID for tracking"
    )


# ===== EXTRACTION RESPONSES =====

class ExtractionResponse(BaseResponse):
    """Response model for file extraction endpoint."""
    
    data: Dict[str, str] = Field(
        default_factory=dict,
        description="Dictionary mapping filename -> extracted text"
    )
    errors: Dict[str, str] = Field(
        default_factory=dict,
        description="Dictionary mapping filename -> error message (if any)"
    )
    file_count: int = Field(..., description="Number of files processed")
    successful_count: int = Field(
        0,
        description="Number of successfully extracted files"
    )
    failed_count: int = Field(
        0,
        description="Number of failed extractions"
    )
    processing_time_seconds: float = Field(
        0.0,
        description="Total processing time"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Successfully extracted from 2 files",
                "data": {
                    "document.pdf": "This is the extracted text from PDF...",
                    "report.docx": "This is text from Word document..."
                },
                "errors": {},
                "file_count": 2,
                "successful_count": 2,
                "failed_count": 0,
                "processing_time_seconds": 1.234,
                "timestamp": "2024-12-24T10:30:00"
            }
        }


class BatchExtractionResponse(BaseResponse):
    """Response for batch extraction requests."""
    
    extractions: List[ExtractionResponse] = Field(
        default_factory=list,
        description="List of extraction responses"
    )
    total_files: int = Field(
        0,
        description="Total files processed across all requests"
    )
    total_successful: int = Field(
        0,
        description="Total successful extractions"
    )
    total_failed: int = Field(
        0,
        description="Total failed extractions"
    )


# ===== INGESTION RESPONSES =====

class IngestionResponse(BaseResponse):
    """Response for data ingestion."""
    
    document_count: int = Field(
        ...,
        description="Number of documents ingested"
    )
    chunk_count: int = Field(
        ...,
        description="Number of text chunks created"
    )
    collection_name: str = Field(
        ...,
        description="Collection where documents were ingested"
    )
    index_size: int = Field(
        0,
        description="Size of the updated index"
    )
    processing_time_seconds: float = Field(
        0.0,
        description="Processing time"
    )


# ===== QUERY RESPONSES =====

class QueryResult(BaseResponse):
    """Single query result item."""
    
    model_config = ConfigDict(validate_assignment=True, use_attribute_docstrings=True)
    
    text: str = Field(..., description="Extracted text snippet")
    source: str = Field(..., description="Source file or document")
    score: float = Field(..., description="Similarity/relevance score (0-1)")
    rank: int = Field(
        0,
        description="Ranking position in results"
    )
    source_type: str = Field(
        default="legal_db",
        description="Type of source (legal_db, user_upload, or cached)"
    )
    chunk_index: Optional[int] = Field(
        None,
        description="Index of chunk in source document"
    )
    page_number: Optional[int] = Field(
        None,
        description="Page number in source document"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )


class QueryResponse(BaseResponse):
    """Response model for standard query endpoint."""
    
    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer from LLM")
    results: List[QueryResult] = Field(
        default_factory=list,
        description="Retrieved context chunks"
    )
    result_count: int = Field(..., description="Number of results returned")
    processing_time_seconds: float = Field(
        ...,
        description="Processing time in seconds"
    )
    embedding_time_seconds: float = Field(
        0.0,
        description="Time spent on embedding"
    )
    retrieval_time_seconds: float = Field(
        0.0,
        description="Time spent on retrieval"
    )
    generation_time_seconds: float = Field(
        0.0,
        description="Time spent on LLM generation"
    )
    confidence_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Confidence score of the answer (0-1)"
    )
    mode: str = Field(
        "standard",
        description="Query processing mode used"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "query": "What are fundamental rights?",
                "answer": "The fundamental rights in the Indian Constitution include...",
                "results": [
                    {
                        "success": True,
                        "message": "",
                        "text": "Article 14 grants equality before law...",
                        "source": "COI_2024.pdf",
                        "score": 0.92,
                        "rank": 1,
                        "source_type": "legal_db",
                        "page_number": 5,
                        "metadata": {"section": "Part 3"}
                    }
                ],
                "result_count": 1,
                "processing_time_seconds": 0.234,
                "confidence_score": 0.87,
                "mode": "standard"
            }
        }


class HybridQueryResponse(BaseResponse):
    """Response model for hybrid query endpoint (with user uploads)."""
    
    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer from LLM")
    results: List[QueryResult] = Field(
        default_factory=list,
        description="Retrieved context chunks from both sources"
    )
    result_count: int = Field(..., description="Total number of results")
    permanent_db_results: int = Field(
        0,
        description="Number of results from permanent legal database"
    )
    user_upload_results: int = Field(
        0,
        description="Number of results from user uploads"
    )
    processing_time_seconds: float = Field(
        ...,
        description="Total processing time in seconds"
    )
    confidence_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Confidence score of the answer"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "query": "What are my obligations under section 304A?",
                "answer": "Section 304A of IPC deals with...",
                "results": [
                    {
                        "text": "Section 304A - Causing death by negligence...",
                        "source": "IPC_2024.pdf",
                        "score": 0.95,
                        "rank": 1,
                        "source_type": "legal_db"
                    }
                ],
                "result_count": 1,
                "permanent_db_results": 1,
                "user_upload_results": 0,
                "processing_time_seconds": 0.456,
                "confidence_score": 0.89
            }
        }


# ===== HEALTH & STATUS RESPONSES =====

class ComponentHealth(BaseModel):
    """Health status of a single component."""
    
    name: str = Field(..., description="Component name")
    status: ComponentStatusEnum = Field(
        ...,
        description="Component status"
    )
    message: Optional[str] = Field(
        None,
        description="Additional status message"
    )
    response_time_ms: Optional[float] = Field(
        None,
        description="Response time in milliseconds"
    )


class HealthResponse(BaseResponse):
    """Response model for health check endpoint."""
    
    status: str = Field(..., description="Overall health status")
    uptime_seconds: float = Field(
        ...,
        description="Server uptime in seconds"
    )
    version: str = Field(
        "1.0.0",
        description="API version"
    )
    components: List[ComponentHealth] = Field(
        default_factory=list,
        description="Status of individual components"
    )
    database_connected: bool = Field(
        False,
        description="Whether database is connected"
    )
    vector_store_available: bool = Field(
        False,
        description="Whether vector store is available"
    )
    llm_available: bool = Field(
        False,
        description="Whether LLM is available"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "status": "healthy",
                "message": "All systems operational",
                "uptime_seconds": 3600.5,
                "version": "1.0.0",
                "components": [
                    {
                        "name": "database",
                        "status": "ok",
                        "response_time_ms": 12.5
                    }
                ],
                "database_connected": True,
                "vector_store_available": True,
                "llm_available": True
            }
        }


# ===== ERROR RESPONSES =====

class ValidationError(BaseModel):
    """Validation error details."""
    
    field: str = Field(..., description="Field name")
    error: str = Field(..., description="Error message")
    value: Optional[Any] = Field(None, description="Invalid value")


class ErrorResponse(BaseResponse):
    """Response model for error responses."""
    
    error_code: str = Field(..., description="Error code")
    error_type: str = Field(..., description="Error type")
    validation_errors: Optional[List[ValidationError]] = Field(
        None,
        description="Validation errors if applicable"
    )
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional error details"
    )
    stack_trace: Optional[str] = Field(
        None,
        description="Stack trace (only in development)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "message": "Failed to extract from file",
                "error_code": "EXTRACTION_ERROR",
                "error_type": "IOError",
                "details": {"file": "document.pdf", "reason": "Invalid PDF format"},
                "timestamp": "2024-12-24T10:30:00"
            }
        }


# ===== FEEDBACK RESPONSES =====

class FeedbackResponse(BaseResponse):
    """Response for feedback submission."""
    
    feedback_id: str = Field(
        ...,
        description="Unique feedback ID"
    )
    saved: bool = Field(
        ...,
        description="Whether feedback was saved"
    )


# ===== STATS & ANALYTICS RESPONSES =====

class QueryStats(BaseModel):
    """Statistics for a query."""
    
    query_id: str = Field(..., description="Query ID")
    query_text: str = Field(..., description="Query text")
    processing_time_ms: float = Field(..., description="Processing time in ms")
    results_count: int = Field(..., description="Number of results")
    answer_generated: bool = Field(..., description="Whether answer was generated")
    timestamp: datetime = Field(..., description="Query timestamp")


class StatsResponse(BaseResponse):
    """Response for statistics endpoint."""
    
    total_queries: int = Field(..., description="Total number of queries")
    total_documents: int = Field(..., description="Total indexed documents")
    avg_query_time_ms: float = Field(
        ...,
        description="Average query processing time"
    )
    recent_queries: List[QueryStats] = Field(
        default_factory=list,
        description="Recent query statistics"
    )
