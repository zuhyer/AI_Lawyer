"""
Custom exceptions and error handling for FastAPI.
Production-grade error handling with proper HTTP status codes.
"""

from fastapi import HTTPException, status
from typing import Optional, Dict, Any
from enum import Enum


# ===== ERROR CODES ENUM =====

class ErrorCode(str, Enum):
    """Standard error codes for API responses."""
    
    # Validation errors (400)
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INVALID_INPUT = "INVALID_INPUT"
    MISSING_REQUIRED_FIELD = "MISSING_REQUIRED_FIELD"
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    INVALID_FILE_FORMAT = "INVALID_FILE_FORMAT"
    
    # Authentication errors (401)
    AUTHENTICATION_FAILED = "AUTHENTICATION_FAILED"
    INVALID_TOKEN = "INVALID_TOKEN"
    TOKEN_EXPIRED = "TOKEN_EXPIRED"
    
    # Authorization errors (403)
    PERMISSION_DENIED = "PERMISSION_DENIED"
    INSUFFICIENT_PERMISSIONS = "INSUFFICIENT_PERMISSIONS"
    RESOURCE_FORBIDDEN = "RESOURCE_FORBIDDEN"
    
    # Not found errors (404)
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
    DOCUMENT_NOT_FOUND = "DOCUMENT_NOT_FOUND"
    QUERY_NOT_FOUND = "QUERY_NOT_FOUND"
    
    # Conflict errors (409)
    RESOURCE_ALREADY_EXISTS = "RESOURCE_ALREADY_EXISTS"
    DUPLICATE_ENTRY = "DUPLICATE_ENTRY"
    
    # Rate limiting (429)
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"
    
    # Processing errors (422)
    PROCESSING_FAILED = "PROCESSING_FAILED"
    EXTRACTION_ERROR = "EXTRACTION_ERROR"
    QUERY_ERROR = "QUERY_ERROR"
    INGESTION_ERROR = "INGESTION_ERROR"
    EMBEDDING_ERROR = "EMBEDDING_ERROR"
    
    # Service errors (503)
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    DATABASE_ERROR = "DATABASE_ERROR"
    VECTOR_STORE_ERROR = "VECTOR_STORE_ERROR"
    LLM_SERVICE_ERROR = "LLM_SERVICE_ERROR"
    
    # Timeout errors (504)
    REQUEST_TIMEOUT = "REQUEST_TIMEOUT"
    PROCESSING_TIMEOUT = "PROCESSING_TIMEOUT"
    
    # Generic errors (500)
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


# ===== CUSTOM EXCEPTIONS =====

class APIException(HTTPException):
    """Base custom exception for API errors."""
    
    def __init__(
        self,
        status_code: int,
        error_code: ErrorCode,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize API exception.
        
        Args:
            status_code: HTTP status code
            error_code: Error code enum
            message: Error message
            details: Additional error details
            headers: HTTP headers to include
        """
        self.error_code = error_code
        self.details = details or {}
        
        content = {
            "success": False,
            "error_code": error_code.value,
            "message": message,
            "details": self.details if self.details else None,
        }
        
        # Remove None details
        if content["details"] is None:
            del content["details"]
        
        super().__init__(
            status_code=status_code,
            detail=content,
            headers=headers,
        )


class ValidationError(APIException):
    """Raised when request validation fails."""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize validation error."""
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            error_code=ErrorCode.VALIDATION_ERROR,
            message=message,
            details=details,
        )


class InvalidInputError(APIException):
    """Raised when input is invalid."""
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
    ):
        """Initialize invalid input error."""
        details = {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = str(value)
        
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            error_code=ErrorCode.INVALID_INPUT,
            message=message,
            details=details if details else None,
        )


class FileNotFoundError(APIException):
    """Raised when file is not found."""
    
    def __init__(self, file_path: str):
        """Initialize file not found error."""
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            error_code=ErrorCode.FILE_NOT_FOUND,
            message=f"File not found: {file_path}",
            details={"file_path": file_path},
        )


class InvalidFileFormatError(APIException):
    """Raised when file format is not supported."""
    
    def __init__(
        self,
        file_name: str,
        supported_formats: Optional[list] = None,
    ):
        """Initialize invalid file format error."""
        details = {"file_name": file_name}
        if supported_formats:
            details["supported_formats"] = supported_formats
        
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_code=ErrorCode.INVALID_FILE_FORMAT,
            message=f"Invalid file format: {file_name}",
            details=details,
        )


class AuthenticationError(APIException):
    """Raised when authentication fails."""
    
    def __init__(self, message: str = "Authentication failed"):
        """Initialize authentication error."""
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            error_code=ErrorCode.AUTHENTICATION_FAILED,
            message=message,
            headers={"WWW-Authenticate": "Bearer"},
        )


class PermissionDeniedError(APIException):
    """Raised when user lacks permissions."""
    
    def __init__(self, message: str = "Permission denied"):
        """Initialize permission denied error."""
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            error_code=ErrorCode.PERMISSION_DENIED,
            message=message,
        )


class ResourceNotFoundError(APIException):
    """Raised when resource is not found."""
    
    def __init__(
        self,
        resource_type: str,
        resource_id: Optional[str] = None,
    ):
        """Initialize resource not found error."""
        if resource_id:
            message = f"{resource_type} not found: {resource_id}"
            details = {
                "resource_type": resource_type,
                "resource_id": resource_id,
            }
        else:
            message = f"{resource_type} not found"
            details = {"resource_type": resource_type}
        
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            error_code=ErrorCode.RESOURCE_NOT_FOUND,
            message=message,
            details=details,
        )


class DocumentNotFoundError(ResourceNotFoundError):
    """Raised when document is not found."""
    
    def __init__(self, document_id: str):
        """Initialize document not found error."""
        super().__init__(
            resource_type="Document",
            resource_id=document_id,
        )


class ExtractionError(APIException):
    """Raised when file extraction fails."""
    
    def __init__(
        self,
        file_name: str,
        reason: str,
    ):
        """Initialize extraction error."""
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_code=ErrorCode.EXTRACTION_ERROR,
            message=f"Failed to extract from {file_name}: {reason}",
            details={
                "file_name": file_name,
                "reason": reason,
            },
        )


class QueryError(APIException):
    """Raised when query processing fails."""
    
    def __init__(self, message: str, reason: Optional[str] = None):
        """Initialize query error."""
        details = {}
        if reason:
            details["reason"] = reason
        
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_code=ErrorCode.QUERY_ERROR,
            message=message,
            details=details if details else None,
        )


class IngestionError(APIException):
    """Raised when data ingestion fails."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize ingestion error."""
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_code=ErrorCode.INGESTION_ERROR,
            message=message,
            details=details,
        )


class EmbeddingError(APIException):
    """Raised when embedding generation fails."""
    
    def __init__(self, message: str, reason: Optional[str] = None):
        """Initialize embedding error."""
        details = {}
        if reason:
            details["reason"] = reason
        
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_code=ErrorCode.EMBEDDING_ERROR,
            message=message,
            details=details if details else None,
        )


class VectorStoreError(APIException):
    """Raised when vector store operations fail."""
    
    def __init__(self, message: str, reason: Optional[str] = None):
        """Initialize vector store error."""
        details = {}
        if reason:
            details["reason"] = reason
        
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code=ErrorCode.VECTOR_STORE_ERROR,
            message=message,
            details=details if details else None,
        )


class LLMServiceError(APIException):
    """Raised when LLM service is unavailable."""
    
    def __init__(self, message: str = "LLM service unavailable"):
        """Initialize LLM service error."""
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code=ErrorCode.LLM_SERVICE_ERROR,
            message=message,
        )


class RateLimitError(APIException):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, message: str = "Rate limit exceeded"):
        """Initialize rate limit error."""
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            error_code=ErrorCode.RATE_LIMIT_EXCEEDED,
            message=message,
        )


class TimeoutError(APIException):
    """Raised when request times out."""
    
    def __init__(self, message: str = "Request timeout"):
        """Initialize timeout error."""
        super().__init__(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            error_code=ErrorCode.REQUEST_TIMEOUT,
            message=message,
        )


class InternalServerError(APIException):
    """Raised when internal server error occurs."""
    
    def __init__(self, message: str = "Internal server error"):
        """Initialize internal server error."""
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code=ErrorCode.INTERNAL_SERVER_ERROR,
            message=message,
        )
