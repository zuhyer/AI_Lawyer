"""
Domain-Aware Query Endpoint - Routes queries to appropriate domains with verification.
"""

from fastapi import APIRouter, HTTPException
import time
from typing import List

from AI_Lawyer.api.models.requests import QueryRequest
from AI_Lawyer.api.models.responses import DomainAwareQueryResponse, QueryResult
from AI_Lawyer.components.query_router import QueryRouter
from AI_Lawyer.components.verification_pipeline import VerificationPipeline
from AI_Lawyer.components.query_component import QueryComponent
from AI_Lawyer.config.configuration import ConfigurationManager
from AI_Lawyer.utils.logging_setup import logger
from sentence_transformers import SentenceTransformer

router = APIRouter(prefix="/query/domain", tags=["domain-aware-query"])

# Global components (singleton pattern)
_config_manager = None
_query_router = None
_verification_pipeline = None
_embedding_model = None
_query_component = None


def get_domain_components():
    """Lazy initialize domain-aware components."""
    global _config_manager, _query_router, _verification_pipeline, _embedding_model, _query_component
    
    if _config_manager is None:
        try:
            _config_manager = ConfigurationManager()
            logger.info("✅ ConfigurationManager initialized")
        except Exception as e:
            logger.error(f"✗ Failed to initialize ConfigurationManager: {e}")
            raise
    
    if _embedding_model is None:
        try:
            embedding_config = _config_manager.get_embeddings_config()
            _embedding_model = SentenceTransformer(embedding_config.model)
            logger.info("✅ Embedding model loaded")
        except Exception as e:
            logger.error(f"✗ Failed to load embedding model: {e}")
            raise
    
    if _query_router is None:
        try:
            _query_router = QueryRouter(_config_manager, _embedding_model)
            logger.info("✅ QueryRouter initialized")
        except Exception as e:
            logger.error(f"✗ Failed to initialize QueryRouter: {e}")
            raise
    
    if _verification_pipeline is None:
        try:
            _verification_pipeline = VerificationPipeline(_config_manager)
            logger.info("✅ VerificationPipeline initialized")
        except Exception as e:
            logger.error(f"✗ Failed to initialize VerificationPipeline: {e}")
            raise
    
    if _query_component is None:
        try:
            llm_config = _config_manager.get_llm_config()
            # FAISS will be loaded per-domain, so we pass None for now
            _query_component = QueryComponent(llm_config, None)
            _query_component.embedding_model = _embedding_model
            logger.info("✅ QueryComponent initialized")
        except Exception as e:
            logger.error(f"✗ Failed to initialize QueryComponent: {e}")
            raise
    
    return _config_manager, _query_router, _verification_pipeline, _embedding_model, _query_component


@router.post("/ask", response_model=DomainAwareQueryResponse)
async def ask_domain_aware_query(request: QueryRequest):
    """
    Submit a domain-aware query with automatic domain classification and verification.
    
    This endpoint:
    1. Classifies the query into appropriate legal domain
    2. Loads domain-specific FAISS index
    3. Retrieves relevant documents
    4. Generates answer using LLM
    5. Verifies answer against retrieved chunks
    6. Returns comprehensive response with citations and confidence
    
    Parameters:
    - query: User question
    - top_k: Number of top results to retrieve (default: 5)
    
    Returns:
    - answer: Generated answer from LLM
    - domain_used: Classified domain
    - citations: Extracted and validated citations
    - verified: Whether answer passed verification
    - confidence_score: Overall confidence score
    
    Example:
    ```
    POST /query/domain/ask
    {
        "query": "What is the punishment for theft under BNS?",
        "top_k": 5
    }
    ```
    """
    start_time = time.time()
    
    try:
        logger.info(f"📝 Domain-aware query received: {request.query[:100]}")
        
        # Get components
        config_manager, query_router, verification_pipeline, embedding_model, query_component = get_domain_components()
        
        # Step 1: Classify query and route to domain
        domain, faiss_index, domain_config = query_router.route_query(request.query)
        classification_confidence = domain_config.get('classification_confidence', 0.5)
        
        logger.info(f"🎯 Query routed to domain: {domain} (confidence={classification_confidence:.2f})")
        
        # Step 2: Set domain-specific FAISS index for this query
        query_component.faiss_db = faiss_index
        
        # Step 3: Retrieve documents using domain-specific index
        retrieval_start = time.time()
        docs = query_component.retrieve_docs(request.query, top_k=request.top_k)
        retrieval_time = time.time() - retrieval_start
        
        if not docs:
            logger.warning(f"⚠️  No documents retrieved for domain '{domain}'")
            return DomainAwareQueryResponse(
                success=False,
                message=f"No relevant documents found in {domain} database",
                query=request.query,
                answer="Unable to find relevant information in the legal database. Please consult a licensed advocate.",
                domain_used=domain,
                domain_confidence=classification_confidence,
                citations=[],
                verified=False,
                confidence_score=0.0,
                confidence_category="Very Low",
                results=[],
                result_count=0,
                processing_time_seconds=time.time() - start_time,
                fallback_message="No relevant documents found for your query."
            )
        
        logger.info(f"✅ Retrieved {len(docs)} documents for domain '{domain}'")
        
        # Step 4: Generate answer using LLM
        generation_start = time.time()
        try:
            answer = query_component.answer_query(request.query)
        except Exception as e:
            logger.error(f"❌ LLM generation failed: {e}")
            answer = "Unable to generate answer at this time. Please try again later."
        generation_time = time.time() - generation_start
        
        logger.info(f"✅ Answer generated in {generation_time:.3f}s")
        
        # Step 5: Verify answer against retrieved documents
        verification_result = verification_pipeline.verify_answer(answer, docs, domain)
        
        logger.info(f"✅ Answer verification complete: verified={verification_result.is_verified}, "
                   f"confidence={verification_result.confidence_score:.2f}")
        
        # Step 6: Prepare results for response
        results = []
        for idx, doc in enumerate(docs):
            result = QueryResult(
                success=True,
                message="",
                text=doc.page_content[:500],
                source=doc.metadata.get("source_file", doc.metadata.get("source", "Unknown")),
                score=0.85,  # Would need similarity_search_with_scores for actual scores
                rank=idx + 1,
                source_type="legal_db",
                chunk_index=doc.metadata.get("chunk_index"),
                page_number=doc.metadata.get("page_number"),
                metadata={
                    "domain": doc.metadata.get("domain", domain),
                    "chunk_total": doc.metadata.get("chunk_total"),
                    "source_file": doc.metadata.get("source_file")
                }
            )
            results.append(result)
        
        # Step 7: Determine final answer (use fallback if low confidence)
        final_answer = answer
        if not verification_result.is_verified and verification_result.fallback_message:
            final_answer = verification_result.fallback_message
        
        # Step 8: Get confidence category
        confidence_category = verification_pipeline.get_confidence_category(
            verification_result.confidence_score
        )
        
        # Step 9: Build response
        processing_time = time.time() - start_time
        
        response = DomainAwareQueryResponse(
            success=verification_result.is_verified,
            message="Query processed successfully" if verification_result.is_verified else "Low confidence answer - caution advised",
            query=request.query,
            answer=final_answer,
            domain_used=domain,
            domain_confidence=classification_confidence,
            citations=verification_result.cited_sections,
            verified=verification_result.is_verified,
            confidence_score=verification_result.confidence_score,
            confidence_category=confidence_category,
            results=results,
            result_count=len(results),
            processing_time_seconds=processing_time,
            fallback_message=verification_result.fallback_message,
            verification_details=verification_result.details
        )
        
        logger.info(f"✅ Domain-aware query processed in {processing_time:.3f}s - "
                   f"domain={domain}, verified={verification_result.is_verified}, "
                   f"confidence={verification_result.confidence_score:.2f}")
        
        return response
        
    except Exception as e:
        logger.exception(f"✗ Domain-aware query processing failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {str(e)}"
        )


@router.get("/domains")
async def get_available_domains():
    """
    Get list of all available domains with their indices.
    
    Returns:
    list: Array of domain information with status
    """
    try:
        logger.info("📊 Fetching available domains")
        
        config_manager, query_router, _, _, _ = get_domain_components()
        
        available_domains = query_router.get_all_available_domains()
        
        domains_info = []
        for domain in available_domains:
            stats = config_manager.get_vector_db_config()
            domain_path = config_manager.get_domain_vector_db_path(domain)
            
            domains_info.append({
                "domain": domain,
                "status": "available",
                "path": str(domain_path),
                "config": query_router.get_domain_config(domain)
            })
        
        logger.info(f"✅ Found {len(domains_info)} available domains")
        
        return {
            "success": True,
            "domains": domains_info,
            "total_domains": len(domains_info),
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get domains: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get domains: {str(e)}"
        )


@router.post("/classify")
async def classify_query(request: QueryRequest):
    """
    Classify a query into appropriate domain without generating answer.
    
    Returns:
    - domain: Classified domain name
    - confidence: Classification confidence (0-1)
    - domain_keywords: Relevant keywords that matched
    """
    try:
        logger.info(f"🔍 Classifying query: {request.query[:100]}")
        
        _, query_router, _, _, _ = get_domain_components()
        
        domain, confidence = query_router.classify_query(request.query)
        domain_config = query_router.get_domain_config(domain)
        
        logger.info(f"✅ Query classified to '{domain}' with confidence {confidence:.2f}")
        
        return {
            "success": True,
            "query": request.query,
            "domain": domain,
            "confidence": confidence,
            "domain_config": domain_config,
            "message": f"Query classified to {domain} domain"
        }
        
    except Exception as e:
        logger.error(f"❌ Classification failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Classification failed: {str(e)}"
        )
