"""
Query Component - RAG and Hybrid Query Handler
Supports both permanent FAISS index and temporary user-uploaded documents.
"""

from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_community.vectorstores import FAISS

from AI_Lawyer.entity.config_entity import LLMConfig
from AI_Lawyer.utils.logging_setup import logger
from AI_Lawyer.utils.secret_loader import resolve_secret


class QueryComponent:
    """
    Query component for hybrid RAG system.
    Supports both permanent FAISS index and temporary user-uploaded documents.
    """

    def __init__(self, llm_config: LLMConfig, faiss_db):
        """
        Initialize QueryComponent with LLM and FAISS vector store.
        
        Args:
            llm_config: LLMConfig instance with provider/model/api_key
            faiss_db: Loaded FAISS vector store instance (permanent legal database)
        """
        self.llm_config = llm_config
        self.faiss_db = faiss_db
        self.embedding_model = None  # Will be set if available

        logger.info("🔧 Initializing QueryComponent...")

        # Resolve encrypted API key
        api_key = resolve_secret(self.llm_config.api_key)

        if not api_key:
            raise ValueError("❌ ERROR: LLM API key could not be resolved. Check your secret config.")

        # Mask key for logging
        masked_key = (api_key[:6] + "...") if isinstance(api_key, str) and len(api_key) > 6 else "<empty>"
        logger.info(f"LLM API key resolved (masked): {masked_key}")

        # Initialize Groq LLM with error handling
        try:
            self.llm = ChatGroq(
                model=self.llm_config.model,
                groq_api_key=api_key
            )
            logger.info(
                f"✅ LLM initialized - Provider: {self.llm_config.provider}, "
                f"Model: {self.llm_config.model}"
            )

        except Exception as e:
            err_str = str(e)
            if "invalid api key" in err_str.lower() or "401" in err_str:
                guidance = (
                    "❌ Groq authentication failed: Invalid API Key (401).\n"
                    "Action: Ensure you placed a valid Groq API key in `config/secret.yaml`.\n"
                    f"Current resolved key (masked): {masked_key}"
                )
                logger.error(guidance)
                raise RuntimeError(guidance) from e
            logger.exception("Failed to initialize Groq LLM:")
            raise

        # Load prompt template
        self.prompt_template = ChatPromptTemplate.from_template(self._get_prompt())

    # =====================================================================
    # HELPER METHODS
    # =====================================================================
    def _extract_source_name(self, source_path: str) -> str:
        """
        Extract a clean source name from the document metadata.
        Handles both file paths and direct filenames.
        
        Args:
            source_path: Source path or filename from metadata
            
        Returns:
            Clean filename or document name
        """
        if not source_path:
            return "Unknown Source"
        
        # If it's a file path, extract just the filename
        try:
            source_name = Path(source_path).name
            # Clean up URL-encoded characters
            source_name = source_name.replace("%20", " ")
            return source_name if source_name else "Unknown Source"
        except Exception:
            return source_path if source_path else "Unknown Source"

    # =====================================================================
    # PROMPT TEMPLATE (Legal-Grade)
    # =====================================================================
    def _get_prompt(self) -> str:
        """Get the legal-safe prompt template."""
        return """
You are an AI Legal Research Assistant. Your job is to provide factual, context-based legal information with clarity and professionalism.

RULES:
1. Context First: Use the provided documents as the primary source.
2. If context insufficient: Reply: "The provided documents do not contain enough information to answer this."
3. No Legal Advice: Provide information, not recommendations.
4. Adaptive Length: Match response depth to question complexity.
5. Tone: Clear, simple, structured for legal systems.
6. Citations: Cite context minimally without disrupting readability.

------------------------------------
QUESTION:
{question}

CONTEXT:
{context}

------------------------------------
ANSWER:
"""

    # =====================================================================
    # DOCUMENT RETRIEVAL (Permanent FAISS)
    # =====================================================================
    def retrieve_docs(self, query: str, top_k: int = 5) -> List[Document]:
        """
        Retrieve documents from permanent FAISS index.
        
        Args:
            query: User question
            top_k: Number of top results to return
            
        Returns:
            List of relevant Document objects
        """
        logger.info(f"🔍 Retrieving {top_k} documents from permanent FAISS for: {query[:80]}")
        
        try:
            results = self.faiss_db.similarity_search(query, k=top_k)
            logger.info(f"✓ Retrieved {len(results)} documents from permanent database")
            return results
        except Exception as e:
            logger.error(f"✗ Error retrieving from FAISS: {e}")
            return []

    def retrieve_docs_with_scores(
        self, 
        query: str, 
        top_k: int = 5
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve documents with similarity scores from permanent FAISS.
        
        Args:
            query: User question
            top_k: Number of top results
            
        Returns:
            List of (Document, similarity_score) tuples
        """
        logger.info(f"🔍 Retrieving scored documents from permanent FAISS")
        
        try:
            results = self.faiss_db.similarity_search_with_relevance_scores(query, k=top_k)
            logger.info(f"✓ Retrieved {len(results)} scored documents")
            return results
        except Exception as e:
            logger.error(f"✗ Error retrieving scored documents: {e}")
            return []

    def get_context(self, documents: List[Document]) -> str:
        """
        Convert Document objects into context string.
        
        Args:
            documents: List of Document objects
            
        Returns:
            Concatenated context string
        """
        logger.info(f"📄 Preparing context from {len(documents)} documents")
        
        if not documents:
            return "[No relevant context found]"
        
        context_parts = []
        for idx, doc in enumerate(documents, 1):
            raw_source = doc.metadata.get("source", "Unknown")
            clean_source = self._extract_source_name(raw_source)
            content = doc.page_content[:500]  # Limit per chunk
            context_parts.append(f"[Source {idx}: {clean_source}]\n{content}")
        
        return "\n\n---\n\n".join(context_parts)

    # =====================================================================
    # HYBRID QUERY (Permanent + User Uploads)
    # =====================================================================
    def query_with_user_files(
        self,
        question: str,
        user_documents: List[Document],
        top_k: int = 5,
        embedding_model = None
    ) -> Dict[str, Any]:
        """
        Execute hybrid query: search both permanent FAISS and user uploads.
        
        Args:
            question: User question
            user_documents: List of Document objects from user uploads
            top_k: Number of results per search
            embedding_model: Embedding model for creating temporary FAISS
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        logger.info(
            f"🔄 HYBRID QUERY - Question: {question[:80]}, "
            f"User docs: {len(user_documents)}, top_k: {top_k}"
        )
        
        try:
            # Create temporary FAISS from user documents
            if user_documents and embedding_model:
                logger.info(f"📦 Creating temporary FAISS from {len(user_documents)} user documents...")
                temp_faiss = FAISS.from_documents(user_documents, embedding_model)
                logger.info("✅ Temporary FAISS created")
            else:
                temp_faiss = None
                logger.warning("⚠️  No embedding model or user documents for temporary FAISS")

            # Search temporary FAISS (user documents) FIRST - prioritize user uploads
            temp_results = []
            if temp_faiss:
                temp_results = temp_faiss.similarity_search_with_relevance_scores(question, k=top_k)
                logger.info(f"✓ Temporary FAISS (User Docs): {len(temp_results)} results")
            
            # Track results from permanent database
            permanent_results = []
            
            # If user documents provide good results, use them primarily
            if temp_results:
                # Use user documents as primary source
                combined_sorted = temp_results[:top_k]
                logger.info(f"✅ Using user-uploaded documents as primary source: {len(combined_sorted)} results")
            else:
                # Fallback to permanent FAISS if no user documents or no good matches
                logger.warning("⚠️  No relevant results from user documents, falling back to legal database...")
                permanent_results = self.retrieve_docs_with_scores(question, top_k=top_k)
                combined_sorted = permanent_results[:top_k]
                logger.info(f"✓ Using permanent legal database: {len(combined_sorted)} results")

            # Extract documents for context
            combined_docs = [doc for doc, _ in combined_sorted]
            context = self.get_context(combined_docs)

            # Generate LLM response
            logger.info("🤖 Generating LLM response...")
            chain = self.prompt_template | self.llm
            response = chain.invoke({"question": question, "context": context})

            # Extract answer text
            answer = response.content if hasattr(response, 'content') else str(response)

            # Build sources list
            sources = []
            for doc, score in combined_sorted:
                # Extract proper source name from metadata
                raw_source = doc.metadata.get("source", "Unknown")
                clean_source_name = self._extract_source_name(raw_source)
                
                source_info = {
                    "text": doc.page_content[:300],
                    "score": float(score),
                    "source_type": "user_upload" if doc.metadata.get("file_type") == "user_upload" else "legal_db",
                    "source_name": clean_source_name,
                    "chunk_index": doc.metadata.get("chunk_index", 0)
                }
                sources.append(source_info)

            logger.info(f"✅ Hybrid query completed - Answer generated with {len(sources)} sources")

            return {
                "success": True,
                "answer": answer,
                "sources": sources,
                "source_count": len(sources),
                "permanent_db_results": len(permanent_results),
                "user_upload_results": len(temp_results)
            }

        except Exception as e:
            logger.error(f"✗ Hybrid query failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "answer": "Error processing query"
            }

    # =====================================================================
    # STANDARD RAG QUERY (Permanent FAISS Only)
    # =====================================================================
    def answer_query(self, query: str) -> str:
        """
        Standard RAG query using permanent FAISS only.
        
        Args:
            query: User question
            
        Returns:
            LLM-generated answer
        """
        logger.info(f"🎯 Processing standard query: {query[:80]}")

        documents = self.retrieve_docs(query)

        if not documents:
            logger.warning("⚠️  No relevant documents found")
            return "❌ No relevant legal information found in the indexed documents."

        context = self.get_context(documents)

        chain = self.prompt_template | self.llm
        response = chain.invoke({"question": query, "context": context})

        answer = response.content if hasattr(response, 'content') else str(response)
        logger.info("✅ Answer generated")
        
        return answer

    def execute_query(self, question: str) -> str:
        """Compatibility wrapper for legacy code."""
        return self.answer_query(question)
