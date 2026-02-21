"""
Query/RAG endpoint - supports both standard and hybrid queries.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List
import time

from AI_Lawyer.api.models.requests import QueryRequest
from AI_Lawyer.api.models.responses import QueryResponse, HybridQueryResponse, QueryResult
from AI_Lawyer.components.file_extractor import FileExtractor
from AI_Lawyer.components.user_upload_processor import UserUploadProcessor
from AI_Lawyer.components.query_component import QueryComponent
from AI_Lawyer.config.configuration import ConfigurationManager
from AI_Lawyer.utils.logging_setup import logger

router = APIRouter(prefix="/query", tags=["query"])

# ============================================================
# GLOBAL SINGLETONS
# ============================================================

_config_manager = None
_query_component = None
_embedding_model = None
_query_router = None


# ============================================================
# COMPONENT INITIALIZATION
# ============================================================

def get_components():
    """Lazy initialize shared components."""
    global _config_manager, _query_component, _embedding_model, _query_router

    # Config
    if _config_manager is None:
        _config_manager = ConfigurationManager()
        logger.info("✅ ConfigurationManager initialized")

    # Embedding model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        embedding_config = _config_manager.get_embeddings_config()
        _embedding_model = SentenceTransformer(embedding_config.model)
        logger.info("✅ Embedding model loaded")

    # Query Router
    if _query_router is None:
        from AI_Lawyer.components.query_router import QueryRouter
        _query_router = QueryRouter(_config_manager, _embedding_model)
        logger.info("✅ QueryRouter initialized")

    # Query Component (NO FAISS HERE)
    if _query_component is None:
        llm_config = _config_manager.get_llm_config()
        _query_component = QueryComponent(llm_config, None)
        _query_component.embedding_model = _embedding_model
        logger.info("✅ QueryComponent initialized")

    return _config_manager, _query_component, _embedding_model, _query_router


# ============================================================
# STANDARD QUERY
# ============================================================

@router.post("/ask", response_model=QueryResponse)
async def ask_query(request: QueryRequest):

    start_time = time.time()

    try:
        logger.info(f"📝 Query: {request.query[:100]}")

        config_manager, query_component, embedding_model, query_router = get_components()

        # Detect domain
        domain, faiss_index, domain_config = query_router.route_query(request.query)

        logger.info(f"🎯 Routed to domain: {domain}")

        # Attach correct FAISS
        query_component.faiss_db = faiss_index

        if query_component.faiss_db is None:
            raise Exception(f"No vector DB found for domain {domain}")

        # Generate answer
        answer = query_component.answer_query(request.query)

        # Retrieve docs
        docs = query_component.retrieve_docs(request.query, top_k=request.top_k)

        results = []

        for doc in docs:
            results.append(
                QueryResult(
                    text=doc.page_content[:500],
                    source=doc.metadata.get("source", "Unknown"),
                    score=0.0,
                    source_type="legal_db",
                    metadata=doc.metadata
                )
            )

        processing_time = time.time() - start_time

        logger.info(
            f"✅ Completed in {processing_time:.2f}s with {len(results)} results"
        )

        return QueryResponse(
            success=True,
            message="Query processed successfully",
            query=request.query,
            answer=answer,
            results=results,
            result_count=len(results),
            processing_time_seconds=processing_time
        )

    except Exception as e:
        logger.exception("Query failed")

        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {str(e)}"
        )


# ============================================================
# HYBRID QUERY
# ============================================================

@router.post("/hybrid", response_model=HybridQueryResponse)
async def hybrid_query(
    query: str = Form(...),
    files: List[UploadFile] = File(default=[]),
    top_k: int = Form(5)
):

    start_time = time.time()

    try:

        config_manager, query_component, embedding_model, query_router = get_components()

        domain, faiss_index, domain_config = query_router.route_query(query)

        query_component.faiss_db = faiss_index

        user_documents = []

        # Process uploads
        if files:

            file_extractor = FileExtractor(
                config_manager.get_file_extractor_config()
            )

            files_dict = {}

            import tempfile
            import os

            for uploaded_file in files:

                content = await uploaded_file.read()

                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(content)
                    temp_path = temp_file.name

                try:
                    text = file_extractor.extract_from_file(temp_path)
                    files_dict[uploaded_file.filename] = text
                finally:
                    os.remove(temp_path)

            if files_dict:
                upload_processor = UserUploadProcessor(
                    config_manager.get_user_upload_processor_config()
                )

                user_documents = upload_processor.process_uploaded_files(files_dict)

        result = query_component.query_with_user_files(
            question=query,
            user_documents=user_documents,
            top_k=top_k,
            embedding_model=embedding_model
        )

        results = []

        for source in result.get("sources", []):
            results.append(
                QueryResult(
                    text=source["text"],
                    source=source["source_name"],
                    score=source["score"],
                    source_type=source["source_type"],
                    metadata={"chunk_index": source.get("chunk_index", 0)}
                )
            )

        processing_time = time.time() - start_time

        return HybridQueryResponse(
            success=True,
            message="Hybrid query processed successfully",
            query=query,
            answer=result["answer"],
            results=results,
            result_count=result["source_count"],
            permanent_db_results=result["permanent_db_results"],
            user_upload_results=result["user_upload_results"],
            processing_time_seconds=processing_time
        )

    except Exception as e:

        logger.exception("Hybrid query failed")

        raise HTTPException(
            status_code=500,
            detail=f"Hybrid query failed: {str(e)}"
        )


# ============================================================
# STATUS ENDPOINT
# ============================================================

@router.get("/status")
async def query_status():

    try:
        config_manager, query_component, embedding_model, query_router = get_components()

        return {
            "operational": True,
            "embedding_model": config_manager.get_embeddings_config().model,
            "llm_model": config_manager.get_llm_config().model,
            "router": "active",
            "vector_db": "domain_based"
        }

    except Exception as e:

        logger.exception("Status check failed")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
