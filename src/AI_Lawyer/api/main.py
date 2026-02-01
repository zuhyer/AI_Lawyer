"""
Entry point for running the AI Lawyer FastAPI server.
Includes comprehensive logging and configuration management.

Usage:
    python -m AI_Lawyer.api.main
    python api_server.py
    uvicorn AI_Lawyer.api.app:app
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from AI_Lawyer.utils.logging_setup import logger


def load_environment():
    """Load environment variables from .env file."""
    env_file = Path(__file__).parent.parent.parent.parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        logger.info(f"✓ Environment loaded from: {env_file}")
    else:
        logger.warning(f"⚠ .env file not found at: {env_file}")


def main():
    """Run the FastAPI server with production-grade configuration."""
    import uvicorn
    
    # Load environment
    load_environment()
    
    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    log_level = os.getenv("LOG_LEVEL", "info")
    environment = os.getenv("ENVIRONMENT", "development")
    
    # Import app here to allow environment to be loaded first
    from AI_Lawyer.api.app import app
    
    logger.info("=" * 70)
    logger.info("🚀 AI LAWYER API - STARTING UP")
    logger.info("=" * 70)
    logger.info(f"Environment: {environment}")
    logger.info(f"Host: {host}")
    logger.info(f"Port: {port}")
    logger.info(f"Auto-reload: {reload}")
    logger.info(f"Log level: {log_level}")
    logger.info("")
    logger.info("📖 API Documentation:")
    logger.info(f"   Swagger UI: http://{host}:{port}/docs")
    logger.info(f"   ReDoc: http://{host}:{port}/redoc")
    logger.info(f"   OpenAPI JSON: http://{host}:{port}/openapi.json")
    logger.info("")
    logger.info("🏥 Health Checks:")
    logger.info(f"   Full Health: http://{host}:{port}/health")
    logger.info(f"   Readiness: http://{host}:{port}/health/ready")
    logger.info(f"   Liveness: http://{host}:{port}/health/live")
    logger.info("=" * 70)
    
    # Run the server
    try:
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=reload,
            log_level=log_level,
            access_log=log_level in ["debug", "info"],
            use_colors=not environment == "production",
        )
    except KeyboardInterrupt:
        logger.info("🛑 Server interrupted by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
