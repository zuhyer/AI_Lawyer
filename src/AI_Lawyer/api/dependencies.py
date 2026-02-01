"""
Dependency injection and service initialization for FastAPI.
Handles singleton initialization and lazy loading of expensive components.
"""

from typing import Optional
import logging
from contextlib import asynccontextmanager

from AI_Lawyer.config.configuration import ConfigurationManager
from AI_Lawyer.utils.logging_setup import logger

# ===== SINGLETON INSTANCES =====

class ServiceManager:
    """Manages all service instances and dependencies."""
    
    _instance: Optional['ServiceManager'] = None
    _initialized: bool = False
    
    def __init__(self):
        """Initialize service manager (singleton pattern)."""
        if ServiceManager._instance is not None:
            raise RuntimeError("ServiceManager is a singleton, use get_instance()")
        
        self.config_manager: Optional[ConfigurationManager] = None
        self.query_component = None
        self.embedding_model = None
        self.faiss_db = None
        self.extraction_component = None
        self.llm = None
        self._lock = None
        
        logger.info("✓ ServiceManager initialized")
    
    @classmethod
    def get_instance(cls) -> 'ServiceManager':
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def initialize_config(self) -> ConfigurationManager:
        """Initialize configuration manager."""
        if self.config_manager is None:
            try:
                self.config_manager = ConfigurationManager()
                logger.info("✓ ConfigurationManager loaded")
            except Exception as e:
                logger.error(f"✗ Failed to initialize ConfigurationManager: {e}")
                raise
        return self.config_manager
    
    def initialize_embedding_model(self):
        """Lazy initialize embedding model."""
        if self.embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                
                config = self.initialize_config()
                embedding_config = config.get_embeddings_config()
                
                self.embedding_model = SentenceTransformer(
                    embedding_config.model,
                    cache_folder=embedding_config.cache_folder if hasattr(embedding_config, 'cache_folder') else None
                )
                logger.info(f"✓ Embedding model loaded: {embedding_config.model}")
            except Exception as e:
                logger.error(f"✗ Failed to initialize embedding model: {e}")
                raise
        return self.embedding_model
    
    def initialize_vector_store(self):
        """Lazy initialize FAISS vector store."""
        if self.faiss_db is None:
            try:
                from langchain_community.vectorstores import FAISS
                
                config = self.initialize_config()
                embedding_config = config.get_embeddings_config()
                embedding_model = self.initialize_embedding_model()
                
                vector_store_path = embedding_config.vector_store_path
                self.faiss_db = FAISS.load_local(
                    vector_store_path,
                    embedding_model,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"✓ Vector store loaded from: {vector_store_path}")
            except Exception as e:
                logger.error(f"✗ Failed to initialize vector store: {e}")
                raise
        return self.faiss_db
    
    def initialize_query_component(self):
        """Lazy initialize query component."""
        if self.query_component is None:
            try:
                from AI_Lawyer.components.query_component import QueryComponent
                
                config = self.initialize_config()
                llm_config = config.get_llm_config()
                faiss_db = self.initialize_vector_store()
                
                self.query_component = QueryComponent(llm_config, faiss_db)
                logger.info("✓ QueryComponent initialized")
            except Exception as e:
                logger.error(f"✗ Failed to initialize QueryComponent: {e}")
                raise
        return self.query_component
    
    def initialize_extraction_component(self):
        """Lazy initialize file extraction component."""
        if self.extraction_component is None:
            try:
                from AI_Lawyer.components.extraction_component import FileExtractionComponent
                
                self.extraction_component = FileExtractionComponent()
                logger.info("✓ FileExtractionComponent initialized")
            except Exception as e:
                logger.error(f"✗ Failed to initialize FileExtractionComponent: {e}")
                raise
        return self.extraction_component
    
    async def health_check(self) -> dict:
        """Perform health check on all components."""
        health_status = {
            "config": "ok",
            "embedding_model": "error",
            "vector_store": "error",
            "query_component": "error",
            "extraction_component": "error"
        }
        
        try:
            self.initialize_config()
            health_status["config"] = "ok"
        except Exception as e:
            health_status["config"] = f"error: {str(e)[:50]}"
        
        try:
            self.initialize_embedding_model()
            health_status["embedding_model"] = "ok"
        except Exception as e:
            health_status["embedding_model"] = f"error: {str(e)[:50]}"
        
        try:
            self.initialize_vector_store()
            health_status["vector_store"] = "ok"
        except Exception as e:
            health_status["vector_store"] = f"error: {str(e)[:50]}"
        
        try:
            self.initialize_query_component()
            health_status["query_component"] = "ok"
        except Exception as e:
            health_status["query_component"] = f"error: {str(e)[:50]}"
        
        try:
            self.initialize_extraction_component()
            health_status["extraction_component"] = "ok"
        except Exception as e:
            health_status["extraction_component"] = f"error: {str(e)[:50]}"
        
        return health_status
    
    async def shutdown(self):
        """Cleanup resources on shutdown."""
        logger.info("🛑 Cleaning up resources...")
        
        # Close vector store if needed
        if self.faiss_db is not None:
            try:
                # FAISS doesn't need explicit closing, but we can log it
                logger.info("✓ Vector store cleanup complete")
            except Exception as e:
                logger.error(f"Error closing vector store: {e}")
        
        # Clear references
        self.faiss_db = None
        self.query_component = None
        self.embedding_model = None
        self.extraction_component = None
        self.config_manager = None
        
        logger.info("✓ All resources cleaned up")


# ===== DEPENDENCY INJECTION FUNCTIONS =====

async def get_config_manager() -> ConfigurationManager:
    """FastAPI dependency for ConfigurationManager."""
    service_manager = ServiceManager.get_instance()
    return service_manager.initialize_config()


async def get_query_component():
    """FastAPI dependency for QueryComponent."""
    service_manager = ServiceManager.get_instance()
    return service_manager.initialize_query_component()


async def get_extraction_component():
    """FastAPI dependency for FileExtractionComponent."""
    service_manager = ServiceManager.get_instance()
    return service_manager.initialize_extraction_component()


async def get_vector_store():
    """FastAPI dependency for FAISS vector store."""
    service_manager = ServiceManager.get_instance()
    return service_manager.initialize_vector_store()


async def get_embedding_model():
    """FastAPI dependency for embedding model."""
    service_manager = ServiceManager.get_instance()
    return service_manager.initialize_embedding_model()


# ===== LIFESPAN MANAGEMENT =====

@asynccontextmanager
async def lifespan_manager(app):
    """
    Manage application lifespan events.
    Initializes services on startup and cleans up on shutdown.
    """
    logger.info("=" * 60)
    logger.info("🚀 AI Lawyer API - Starting up...")
    logger.info("=" * 60)
    
    service_manager = ServiceManager.get_instance()
    
    try:
        # Initialize critical services
        service_manager.initialize_config()
        logger.info("✓ Configuration loaded")
        
        # Health check
        health = await service_manager.health_check()
        logger.info(f"✓ Health status: {health}")
        
        logger.info("=" * 60)
        logger.info("✅ API Ready for requests")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield
    
    logger.info("=" * 60)
    logger.info("🛑 AI Lawyer API - Shutting down...")
    logger.info("=" * 60)
    
    await service_manager.shutdown()
    
    logger.info("=" * 60)
    logger.info("✅ Shutdown complete")
    logger.info("=" * 60)
