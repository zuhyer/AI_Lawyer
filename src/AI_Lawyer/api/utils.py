"""
Utility functions and helpers for FastAPI endpoints.
Includes validation, formatting, and common operations.
"""

import os
import uuid
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from AI_Lawyer.utils.logging_setup import logger


class APIUtils:
    """Utility functions for API operations."""
    
    @staticmethod
    def generate_request_id() -> str:
        """Generate a unique request ID."""
        return f"req-{uuid.uuid4().hex[:12]}"
    
    @staticmethod
    def generate_query_id() -> str:
        """Generate a unique query ID."""
        return f"q-{uuid.uuid4().hex[:12]}"
    
    @staticmethod
    def generate_task_id() -> str:
        """Generate a unique task ID for background jobs."""
        return f"task-{uuid.uuid4().hex[:12]}"
    
    @staticmethod
    def get_timestamp() -> str:
        """Get current timestamp in ISO format."""
        return datetime.utcnow().isoformat() + "Z"
    
    @staticmethod
    def get_timestamp_ms() -> int:
        """Get current timestamp in milliseconds."""
        return int(time.time() * 1000)
    
    @staticmethod
    def format_processing_time(start_time: float) -> float:
        """Format processing time from start time to now."""
        return round(time.time() - start_time, 3)
    
    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """Format bytes to human-readable size."""
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.2f} TB"
    
    @staticmethod
    def truncate_text(text: str, max_length: int = 500) -> str:
        """Truncate text to maximum length with ellipsis."""
        if len(text) <= max_length:
            return text
        return text[:max_length - 3] + "..."
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename to prevent directory traversal."""
        # Remove directory components
        filename = os.path.basename(filename)
        # Remove potentially dangerous characters
        dangerous_chars = ["<", ">", ":", "\"", "/", "\\", "|", "?", "*", "\0"]
        for char in dangerous_chars:
            filename = filename.replace(char, "_")
        return filename
    
    @staticmethod
    def validate_file_extension(filename: str, allowed_extensions: List[str]) -> bool:
        """Validate file extension."""
        if not filename:
            return False
        ext = os.path.splitext(filename)[1].lstrip(".").lower()
        return ext in allowed_extensions
    
    @staticmethod
    def get_file_extension(filename: str) -> str:
        """Get file extension without dot."""
        return os.path.splitext(filename)[1].lstrip(".").lower()
    
    @staticmethod
    def batch_list(items: List[Any], batch_size: int) -> List[List[Any]]:
        """Split list into batches."""
        batches = []
        for i in range(0, len(items), batch_size):
            batches.append(items[i:i + batch_size])
        return batches
    
    @staticmethod
    def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = "_") -> Dict[str, Any]:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(APIUtils.flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    @staticmethod
    def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple dictionaries."""
        result = {}
        for d in dicts:
            if d:
                result.update(d)
        return result


class ValidationUtils:
    """Validation utilities for request data."""
    
    @staticmethod
    def validate_query_text(query: str, min_length: int = 1, max_length: int = 5000) -> tuple[bool, str]:
        """Validate query text."""
        if not query or not query.strip():
            return False, "Query cannot be empty"
        
        if len(query) < min_length:
            return False, f"Query must be at least {min_length} characters"
        
        if len(query) > max_length:
            return False, f"Query must be at most {max_length} characters"
        
        return True, "Valid"
    
    @staticmethod
    def validate_document_text(text: str, min_length: int = 1) -> tuple[bool, str]:
        """Validate document text."""
        if not text or not text.strip():
            return False, "Document text cannot be empty"
        
        if len(text) < min_length:
            return False, f"Document must be at least {min_length} characters"
        
        return True, "Valid"
    
    @staticmethod
    def validate_chunk_size(chunk_size: int, min_size: int = 100, max_size: int = 4096) -> tuple[bool, str]:
        """Validate chunk size."""
        if chunk_size < min_size:
            return False, f"Chunk size must be at least {min_size}"
        
        if chunk_size > max_size:
            return False, f"Chunk size must be at most {max_size}"
        
        return True, "Valid"
    
    @staticmethod
    def validate_top_k(top_k: int, min_k: int = 1, max_k: int = 50) -> tuple[bool, str]:
        """Validate top_k parameter."""
        if top_k < min_k:
            return False, f"top_k must be at least {min_k}"
        
        if top_k > max_k:
            return False, f"top_k must be at most {max_k}"
        
        return True, "Valid"
    
    @staticmethod
    def validate_score_threshold(threshold: float) -> tuple[bool, str]:
        """Validate score threshold."""
        if not isinstance(threshold, (int, float)):
            return False, "Threshold must be a number"
        
        if threshold < 0.0 or threshold > 1.0:
            return False, "Threshold must be between 0.0 and 1.0"
        
        return True, "Valid"


class ResponseFormatting:
    """Formatting utilities for API responses."""
    
    @staticmethod
    def format_query_result(
        text: str,
        source: str,
        score: float,
        rank: int = 1,
        source_type: str = "legal_db",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Format a single query result."""
        return {
            "text": APIUtils.truncate_text(text),
            "source": source,
            "score": round(score, 4),
            "rank": rank,
            "source_type": source_type,
            "metadata": metadata or {},
        }
    
    @staticmethod
    def format_extraction_result(
        filename: str,
        text: str,
        success: bool = True,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Format extraction result."""
        return {
            "filename": filename,
            "success": success,
            "text": text if success else None,
            "error": error,
            "character_count": len(text) if success else 0,
        }
    
    @staticmethod
    def format_timing_info(
        embedding_time: float = 0.0,
        retrieval_time: float = 0.0,
        generation_time: float = 0.0,
    ) -> Dict[str, float]:
        """Format timing information."""
        return {
            "embedding_seconds": round(embedding_time, 3),
            "retrieval_seconds": round(retrieval_time, 3),
            "generation_seconds": round(generation_time, 3),
            "total_seconds": round(embedding_time + retrieval_time + generation_time, 3),
        }


class PaginationUtils:
    """Pagination utilities for list responses."""
    
    @staticmethod
    def paginate(
        items: List[Any],
        page: int = 1,
        page_size: int = 10,
    ) -> Dict[str, Any]:
        """Paginate a list of items."""
        if page < 1:
            page = 1
        
        if page_size < 1:
            page_size = 10
        
        if page_size > 100:
            page_size = 100
        
        total = len(items)
        total_pages = (total + page_size - 1) // page_size
        
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        paginated_items = items[start_idx:end_idx]
        
        return {
            "items": paginated_items,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total": total,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_previous": page > 1,
            },
        }


class CacheUtils:
    """Utilities for caching responses."""
    
    @staticmethod
    def should_cache(cache_ttl: int) -> bool:
        """Determine if response should be cached."""
        return cache_ttl > 0
    
    @staticmethod
    def get_cache_key(
        prefix: str,
        *args,
        **kwargs,
    ) -> str:
        """Generate cache key from parameters."""
        import hashlib
        key_parts = [prefix] + list(args) + [str(v) for v in kwargs.values()]
        key_str = "|".join(str(p) for p in key_parts)
        key_hash = hashlib.md5(key_str.encode()).hexdigest()
        return f"{prefix}:{key_hash}"
    
    @staticmethod
    def is_cache_expired(cached_time: float, ttl: int) -> bool:
        """Check if cache has expired."""
        return time.time() - cached_time > ttl


class LoggingUtils:
    """Logging utilities for API operations."""
    
    @staticmethod
    def log_request(request_id: str, method: str, path: str, details: Optional[Dict] = None):
        """Log request."""
        logger.info(
            f"[{request_id}] {method} {path}",
            extra={"details": details} if details else {}
        )
    
    @staticmethod
    def log_response(request_id: str, status_code: int, processing_time: float):
        """Log response."""
        logger.info(
            f"[{request_id}] Response: {status_code} ({processing_time:.3f}s)"
        )
    
    @staticmethod
    def log_error(request_id: str, error: Exception, details: Optional[Dict] = None):
        """Log error."""
        logger.error(
            f"[{request_id}] Error: {str(error)}",
            extra={"details": details} if details else {},
            exc_info=True
        )
