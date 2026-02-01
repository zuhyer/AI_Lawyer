"""
Health check and system status endpoints.
Production-grade health monitoring with component status checks.
"""

from fastapi import APIRouter, HTTPException, status
from datetime import datetime
import time
import logging

from AI_Lawyer.api.models.responses import (
    HealthResponse, ComponentHealth, ComponentStatusEnum
)
from AI_Lawyer.api.dependencies import ServiceManager
from AI_Lawyer.utils.logging_setup import logger

router = APIRouter(prefix="/health", tags=["health"])

# Track server start time for uptime calculation
SERVER_START_TIME = time.time()


@router.get("/", response_model=HealthResponse)
async def health_check():
    """
    Comprehensive health check endpoint.
    
    Checks:
    - API availability
    - Configuration system
    - Vector store connectivity
    - Embedding model availability
    - LLM service connectivity
    
    Returns:
    - Overall health status
    - Individual component statuses
    - Server uptime
    """
    try:
        uptime = time.time() - SERVER_START_TIME
        service_manager = ServiceManager.get_instance()
        
        # Perform comprehensive health checks
        components = []
        
        # Check API
        components.append(ComponentHealth(
            name="api",
            status=ComponentStatusEnum.OK,
            message="API is responding",
            response_time_ms=0
        ))
        
        # Check configuration
        try:
            config = service_manager.initialize_config()
            components.append(ComponentHealth(
                name="configuration",
                status=ComponentStatusEnum.OK,
                message="Configuration loaded"
            ))
        except Exception as e:
            components.append(ComponentHealth(
                name="configuration",
                status=ComponentStatusEnum.ERROR,
                message=f"Configuration error: {str(e)[:50]}"
            ))
        
        # Check embedding model
        try:
            start = time.time()
            service_manager.initialize_embedding_model()
            response_time = (time.time() - start) * 1000
            components.append(ComponentHealth(
                name="embedding_model",
                status=ComponentStatusEnum.OK,
                message="Embedding model ready",
                response_time_ms=response_time
            ))
        except Exception as e:
            components.append(ComponentHealth(
                name="embedding_model",
                status=ComponentStatusEnum.ERROR,
                message=f"Embedding error: {str(e)[:50]}"
            ))
        
        # Check vector store
        try:
            start = time.time()
            service_manager.initialize_vector_store()
            response_time = (time.time() - start) * 1000
            components.append(ComponentHealth(
                name="vector_store",
                status=ComponentStatusEnum.OK,
                message="Vector store connected",
                response_time_ms=response_time
            ))
        except Exception as e:
            components.append(ComponentHealth(
                name="vector_store",
                status=ComponentStatusEnum.ERROR,
                message=f"Vector store error: {str(e)[:50]}"
            ))
        
        # Check query component
        try:
            start = time.time()
            service_manager.initialize_query_component()
            response_time = (time.time() - start) * 1000
            components.append(ComponentHealth(
                name="query_engine",
                status=ComponentStatusEnum.OK,
                message="Query engine ready",
                response_time_ms=response_time
            ))
        except Exception as e:
            components.append(ComponentHealth(
                name="query_engine",
                status=ComponentStatusEnum.DEGRADED,
                message=f"Query engine issue: {str(e)[:50]}"
            ))
        
        # Check extraction component
        try:
            start = time.time()
            service_manager.initialize_extraction_component()
            response_time = (time.time() - start) * 1000
            components.append(ComponentHealth(
                name="extraction_engine",
                status=ComponentStatusEnum.OK,
                message="Extraction engine ready",
                response_time_ms=response_time
            ))
        except Exception as e:
            components.append(ComponentHealth(
                name="extraction_engine",
                status=ComponentStatusEnum.DEGRADED,
                message=f"Extraction issue: {str(e)[:50]}"
            ))
        
        # Determine overall status
        error_count = sum(1 for c in components if c.status == ComponentStatusEnum.ERROR)
        degraded_count = sum(1 for c in components if c.status == ComponentStatusEnum.DEGRADED)
        
        if error_count > 0:
            overall_status = "unhealthy"
        elif degraded_count > 0:
            overall_status = "degraded"
        else:
            overall_status = "healthy"
        
        # Check component availability
        db_connected = any(
            c.name == "vector_store" and c.status == ComponentStatusEnum.OK
            for c in components
        )
        vector_store_available = any(
            c.name == "vector_store" and c.status in [ComponentStatusEnum.OK, ComponentStatusEnum.DEGRADED]
            for c in components
        )
        llm_available = any(
            c.name == "query_engine" and c.status in [ComponentStatusEnum.OK, ComponentStatusEnum.DEGRADED]
            for c in components
        )
        
        return HealthResponse(
            success=overall_status in ["healthy", "degraded"],
            message=f"System is {overall_status}",
            status=overall_status,
            uptime_seconds=uptime,
            version="1.0.0",
            components=components,
            database_connected=db_connected,
            vector_store_available=vector_store_available,
            llm_available=llm_available,
        )
    
    except Exception as e:
        logger.exception("Health check failed")
        return HealthResponse(
            success=False,
            message=f"Health check failed: {str(e)[:100]}",
            status="unhealthy",
            uptime_seconds=time.time() - SERVER_START_TIME,
            components=[],
            database_connected=False,
            vector_store_available=False,
            llm_available=False,
        )


@router.get("/ready", response_model=dict)
async def readiness_check():
    """
    Kubernetes-style readiness check.
    
    Returns 200 if service is ready to accept traffic,
    503 if it's still initializing or degraded.
    """
    try:
        service_manager = ServiceManager.get_instance()
        
        # Check critical components
        try:
            service_manager.initialize_config()
            service_manager.initialize_vector_store()
            
            return {
                "ready": True,
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.warning(f"Readiness check failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"ready": False, "reason": str(e)[:100]},
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Readiness check error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"ready": False, "reason": "Unknown error"},
        )


@router.get("/live", response_model=dict)
async def liveness_check():
    """
    Kubernetes-style liveness check.
    
    Returns 200 if service is running,
    503 if the process should be restarted.
    """
    return {
        "alive": True,
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_seconds": time.time() - SERVER_START_TIME,
    }


@router.get("/startup", response_model=dict)
async def startup_check():
    """
    Startup verification endpoint.
    
    Performs initialization checks and returns system readiness info.
    """
    try:
        service_manager = ServiceManager.get_instance()
        health = await service_manager.health_check()
        
        return {
            "success": True,
            "message": "System startup complete",
            "health_status": health,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"Startup check failed: {e}")
        return {
            "success": False,
            "message": f"Startup incomplete: {str(e)[:100]}",
            "timestamp": datetime.utcnow().isoformat(),
        }
