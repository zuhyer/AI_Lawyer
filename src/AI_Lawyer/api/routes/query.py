"""
Query/RAG endpoint - supports both standard and hybrid queries.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import Optional, List
import time
import io

from AI_Lawyer.api.models.requests import QueryRequest, HybridQueryRequest
from AI_Lawyer.api.models.responses import QueryResponse, HybridQueryResponse, QueryResult
from AI_Lawyer.components.file_extractor import FileExtractor
from AI_Lawyer.components.user_upload_processor import UserUploadProcessor
from AI_Lawyer.components.query_component import QueryComponent
from AI_Lawyer.config.configuration import ConfigurationManager
from AI_Lawyer.utils.logging_setup import logger

router = APIRouter(prefix="/query", tags=["query"])

# Initialize components (singleton for efficiency)
_config_manager = None
_query_component = None
_embedding_model = None

def get_components():
    """Lazy initialize components."""
    global _config_manager, _query_component, _embedding_model
    
    if _config_manager is None:
        try:
            _config_manager = ConfigurationManager()
            logger.info("✅ ConfigurationManager initialized")
        except Exception as e:
            logger.error(f"✗ Failed to initialize ConfigurationManager: {e}")
            raise

    if _query_component is None:
        try:
            from langchain_community.vectorstores import FAISS
            from sentence_transformers import SentenceTransformer
            
            # Load FAISS and query component
            embedding_config = _config_manager.get_embeddings_config()
            vector_store_path = embedding_config.vector_store_path
            
            # Initialize embedding model
            _embedding_model = SentenceTransformer(embedding_config.model)
            
            # Load FAISS
            faiss_db = FAISS.load_local(
                vector_store_path,
                _embedding_model,
                allow_dangerous_deserialization=True
            )
            
            # Create query component
            llm_config = _config_manager.get_llm_config()
            _query_component = QueryComponent(llm_config, faiss_db)
            _query_component.embedding_model = _embedding_model
            
            logger.info("✅ QueryComponent and FAISS initialized")
        except Exception as e:
            logger.error(f"✗ Failed to initialize QueryComponent: {e}")
            raise
    
    return _config_manager, _query_component, _embedding_model


@router.post("/ask", response_model=QueryResponse)
async def ask_query(request: QueryRequest):
    """
    Submit a standard query to the RAG system (permanent legal database only).
    
    Parameters:
    - query: User question
    - top_k: Number of top results to retrieve (1-50)
    - score_threshold: Minimum similarity score (0.0-1.0)
    
    Returns:
    - answer: Generated answer from LLM
    - results: Retrieved context chunks with scores
    - processing_time: Time taken to process
    """
    start_time = time.time()
    
    try:
        logger.info(f"📝 Standard query received: {request.query[:100]}")
        
        _, query_component, _ = get_components()
        
        # Get answer
        answer = query_component.answer_query(request.query)
        
        # Retrieve source documents
        docs = query_component.retrieve_docs(request.query, k=request.top_k)
        
        # Convert to response format
        results = []
        for doc in docs:
            result = QueryResult(
                text=doc.page_content[:500],
                source=doc.metadata.get("source", "Unknown"),
                score=0.0,  # Would need similarity_search_with_scores for actual scores
                source_type="legal_db",
                metadata=doc.metadata
            )
            results.append(result)
        
        processing_time = time.time() - start_time
        
        logger.info(f"✅ Query processed in {processing_time:.3f}s - {len(results)} results")
        
        return QueryResponse(
            success=True,
            query=request.query,
            answer=answer,
            results=results,
            result_count=len(results),
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.exception(f"✗ Query processing failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {str(e)}"
        )


@router.post("/hybrid")
async def hybrid_query(
    query: str = Form(...),
    files: List[UploadFile] = File(default=[]),
    top_k: int = Form(5)
):
    """
    Submit a hybrid query with optional file uploads.
    Searches both permanent legal database AND user-uploaded documents.
    
    Parameters:
    - query: User question
    - files: Optional list of files to upload and search
    - top_k: Number of top results per search
    
    Returns:
    - answer: Generated answer combining both sources
    - results: Combined and ranked results
    - permanent_db_results: Count from legal DB
    - user_upload_results: Count from user uploads
    """
    start_time = time.time()
    
    try:
        logger.info(f"🔄 Hybrid query received: {query[:100]}, files: {len(files)}")
        
        config_manager, query_component, embedding_model = get_components()
        
        user_documents = []
        
        # Process uploaded files if provided
        if files and len(files) > 0:
            try:
                # Extract text from uploaded files
                file_extractor = FileExtractor(config_manager.get_file_extractor_config())
                files_dict = {}
                
                for uploaded_file in files:
                    logger.info(f"📄 Processing upload: {uploaded_file.filename}")
                    
                    # Read file content
                    content = await uploaded_file.read()
                    file_size_mb = len(content) / (1024 * 1024)
                    
                    # Save to temporary location
                    import tempfile
                    with tempfile.NamedTemporaryFile(
                        suffix=uploaded_file.filename[-10:], 
                        delete=False
                    ) as temp_file:
                        temp_file.write(content)
                        temp_path = temp_file.name
                    
                    # Extract text
                    try:
                        extracted_text = file_extractor.extract_from_file(temp_path)
                        files_dict[uploaded_file.filename] = extracted_text
                        logger.info(f"✓ Extracted from {uploaded_file.filename}")
                    except Exception as e:
                        logger.error(f"✗ Extraction failed for {uploaded_file.filename}: {e}")
                        continue
                    finally:
                        # Cleanup temp file
                        import os
                        try:
                            os.remove(temp_path)
                        except:
                            pass
                
                # Process extracted text into chunks
                if files_dict:
                    upload_processor = UserUploadProcessor(
                        config_manager.get_user_upload_processor_config()
                    )
                    user_documents = upload_processor.process_uploaded_files(files_dict)
                    logger.info(f"✅ Created {len(user_documents)} chunks from uploads")
                    
            except Exception as e:
                logger.error(f"✗ Failed to process uploads: {e}")
                # Continue without user documents
                user_documents = []
        
        # Execute hybrid query
        result = query_component.query_with_user_files(
            question=query,
            user_documents=user_documents,
            top_k=top_k,
            embedding_model=embedding_model
        )
        
        if not result["success"]:
            raise Exception(result.get("error", "Hybrid query failed"))
        
        # Convert to response format
        results = []
        for source in result.get("sources", []):
            result_item = QueryResult(
                text=source["text"],
                source=source["source_name"],
                score=source["score"],
                source_type=source["source_type"],
                metadata={"chunk_index": source.get("chunk_index", 0)}
            )
            results.append(result_item)
        
        processing_time = time.time() - start_time
        
        logger.info(
            f"✅ Hybrid query completed in {processing_time:.3f}s - "
            f"{result['permanent_db_results']} from legal DB, "
            f"{result['user_upload_results']} from uploads"
        )
        
        return HybridQueryResponse(
            success=True,
            query=query,
            answer=result["answer"],
            results=results,
            result_count=result["source_count"],
            permanent_db_results=result["permanent_db_results"],
            user_upload_results=result["user_upload_results"],
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.exception(f"✗ Hybrid query failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Hybrid query failed: {str(e)}"
        )


@router.get("/status", response_model=dict)
async def query_status():
    """Check if query system is operational."""
    try:
        config_manager, query_component, embedding_model = get_components()
        
        status = {
            "operational": True,
            "permanent_db": "ready",
            "hybrid_support": "enabled",
            "embedding_model": config_manager.get_embeddings_config().model,
            "llm_model": config_manager.get_llm_config().model
        }
        
        logger.info("✅ Query status: operational")
        return status
        
    except Exception as e:
        logger.exception("Failed to get query status")
        raise HTTPException(
            status_code=500,
            detail=f"Query system error: {str(e)}"
        )

