"""
Production-grade FastAPI application with comprehensive middleware,
error handling, security, and observability.
"""

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZIPMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import time
import logging
import os
import uuid
from datetime import datetime

from AI_Lawyer.api.routes import health, extraction, query, ingestion
from AI_Lawyer.api.exceptions import APIException, ErrorCode
from AI_Lawyer.api.dependencies import lifespan_manager
from AI_Lawyer.utils.logging_setup import logger


# ===== MIDDLEWARE & REQUEST CONTEXT =====

class RequestIDMiddleware:
    """Middleware to add request IDs for tracking."""
    
    def __init__(self, app):
        """Initialize middleware."""
        self.app = app
    
    async def __call__(self, request: Request, call_next):
        """Process request with ID."""
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        request.state.start_time = time.time()
        
        response = await call_next(request)
        
        response.headers["X-Request-ID"] = request_id
        
        return response


class LoggingMiddleware:
    """Middleware for request/response logging."""
    
    def __init__(self, app):
        """Initialize middleware."""
        self.app = app
    
    async def __call__(self, request: Request, call_next):
        """Log request and response."""
        start_time = time.time()
        request_id = getattr(request.state, "request_id", "unknown")
        
        # Log request
        logger.info(
            f"[{request_id}] {request.method} {request.url.path} "
            f"client={request.client.host if request.client else 'unknown'}"
        )
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Log response
            logger.info(
                f"[{request_id}] {request.method} {request.url.path} "
                f"status={response.status_code} time={process_time:.3f}s"
            )
            
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"[{request_id}] {request.method} {request.url.path} "
                f"error={str(e)[:100]} time={process_time:.3f}s",
                exc_info=True
            )
            raise


def create_app():
    """
    Create and configure production-grade FastAPI application.
    
    Features:
    - CORS protection
    - Request/response logging
    - Error handling
    - Security headers
    - Gzip compression
    - Request ID tracking
    - Health checks
    - Comprehensive API documentation
    """
    
    app = FastAPI(
        title="AI Lawyer API",
        description="Production-grade legal document analysis and Q&A system",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan_manager,
    )
    
    # ===== MIDDLEWARE STACK =====
    
    # Request ID middleware (must be first)
    app.add_middleware(RequestIDMiddleware)
    
    # Logging middleware
    app.add_middleware(LoggingMiddleware)
    
    # CORS configuration
    allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
    allowed_origins = [origin.strip() for origin in allowed_origins if origin.strip()]
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[
            "X-Request-ID",
            "X-Process-Time",
            "X-Total-Count",
            "X-Page-Count",
        ],
        max_age=3600,
    )
    
    # Trusted hosts middleware
    trusted_hosts = os.getenv("TRUSTED_HOSTS", "*").split(",")
    trusted_hosts = [host.strip() for host in trusted_hosts if host.strip()]
    
    if trusted_hosts != ["*"]:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=trusted_hosts
        )
    
    # GZIP compression
    app.add_middleware(GZIPMiddleware, minimum_size=1000)
    
    # HTTPS redirect (disable in development)
    if os.getenv("REQUIRE_HTTPS", "false").lower() == "true":
        app.add_middleware(HTTPSRedirectMiddleware)
    
    # ===== EXCEPTION HANDLERS =====
    
    @app.exception_handler(APIException)
    async def api_exception_handler(request: Request, exc: APIException):
        """Handle custom API exceptions."""
        request_id = getattr(request.state, "request_id", "unknown")
        logger.warning(f"[{request_id}] API Exception: {exc.detail}")
        
        response = exc.detail.copy() if isinstance(exc.detail, dict) else {"error": str(exc.detail)}
        response["success"] = False
        response["request_id"] = request_id
        response["timestamp"] = datetime.utcnow().isoformat()
        
        return JSONResponse(
            status_code=exc.status_code,
            content=response,
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle validation errors."""
        request_id = getattr(request.state, "request_id", "unknown")
        logger.warning(f"[{request_id}] Validation Error: {exc.error_count()} error(s)")
        
        errors = []
        for error in exc.errors():
            errors.append({
                "field": ".".join(str(loc) for loc in error["loc"][1:]),
                "message": error["msg"],
                "type": error["type"],
            })
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "success": False,
                "error_code": ErrorCode.VALIDATION_ERROR.value,
                "message": "Request validation failed",
                "errors": errors,
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle unhandled exceptions."""
        request_id = getattr(request.state, "request_id", "unknown")
        logger.exception(f"[{request_id}] Unhandled Exception: {exc}")
        
        # Don't expose internal details in production
        show_details = os.getenv("DEBUG", "false").lower() == "true"
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error_code": ErrorCode.INTERNAL_SERVER_ERROR.value,
                "message": "An unexpected error occurred",
                "details": str(exc) if show_details else None,
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )
    
    # ===== ROUTES =====
    
    # Health check routes
    app.include_router(health.router)
    
    # File extraction routes
    app.include_router(extraction.router)
    
    # Data ingestion routes
    app.include_router(ingestion.router)
    
    # Query/RAG routes
    app.include_router(query.router)
    
    # ===== ROOT ENDPOINTS =====
    
    @app.get("/", tags=["root"])
    async def root():
        """API root endpoint with information."""
        return {
            "name": "AI Lawyer API",
            "version": "1.0.0",
            "description": "Legal document analysis and Q&A system",
            "documentation": {
                "swagger": "/docs",
                "redoc": "/redoc",
                "openapi": "/openapi.json",
            },
            "endpoints": {
                "health": "/health",
                "extraction": "/extraction",
                "ingestion": "/ingestion",
                "query": "/query",
            },
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    # ===== STATIC FILES (Optional) =====
    
    templates_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "templates"
    )
    if os.path.exists(templates_dir):
        try:
            app.mount("/static", StaticFiles(directory=templates_dir), name="static")
            logger.info(f"✓ Static files mounted from: {templates_dir}")
        except Exception as e:
            logger.warning(f"Could not mount static files: {e}")
    
    return app


# ===== APPLICATION INSTANCE =====

app = create_app()


# ===== DEVELOPMENT SERVER =====

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    log_level = os.getenv("LOG_LEVEL", "info")
    
    logger.info(f"Starting FastAPI server at {host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
    )
