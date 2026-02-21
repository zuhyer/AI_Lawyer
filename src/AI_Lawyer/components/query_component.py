from typing import List, Dict, Any
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from AI_Lawyer.entity.config_entity import LLMConfig
from AI_Lawyer.utils.logging_setup import logger
from AI_Lawyer.utils.secret_loader import resolve_secret


class QueryComponent:

    def __init__(self, llm_config: LLMConfig, faiss_db):

        self.llm_config = llm_config
        self.faiss_db = faiss_db

        logger.info("🔧 Initializing QueryComponent...")

        api_key = resolve_secret(self.llm_config.api_key)

        if not api_key:
            raise ValueError("LLM API key missing")

        self.llm = ChatGroq(
            model=self.llm_config.model,
            groq_api_key=api_key
        )

        self.prompt_template = ChatPromptTemplate.from_template(
            self._get_prompt()
        )

    def _get_prompt(self) -> str:
        return """
You are an AI Legal Research Assistant.

Use provided legal context first.
If insufficient say:
"The provided documents do not contain enough information."

QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
"""

    # SAFE RETRIEVAL
    def retrieve_docs(
        self,
        query: str,
        top_k: int = 8,
        domain: str = None
    ) -> List[Document]:

        logger.info(
            f"🔍 Retrieval | Query='{query[:60]}' | Domain={domain} | k={top_k}"
        )

        try:

            results = self.faiss_db.similarity_search(
                query,
                k=max(top_k * 2, 10)
            )

            # Manual domain filtering
            if domain:
                filtered = []

                for doc in results:
                    metadata = doc.metadata if isinstance(doc.metadata, dict) else {}

                    if metadata.get("domain") == domain:
                        filtered.append(doc)

                results = filtered

            results = results[:top_k]

            logger.info(f"✓ Retrieved {len(results)} docs")
            return results

        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return []

    # CONTEXT
    def get_context(self, documents: List[Document]) -> str:

        if not documents:
            return "[No relevant context found]"

        parts = []

        for i, doc in enumerate(documents, 1):

            metadata = doc.metadata if isinstance(doc.metadata, dict) else {}

            source = metadata.get("source_file", "Unknown")
            domain = metadata.get("domain", "Unknown")

            text = doc.page_content[:500]

            parts.append(
                f"[Source {i} | Domain: {domain} | File: {source}]\n{text}"
            )

        return "\n\n---\n\n".join(parts)

    # STANDARD QUERY
    def answer_query(
        self,
        query: str,
        domain: str = None
    ) -> str:

        docs = self.retrieve_docs(
            query=query,
            domain=domain
        )

        if not docs:
            return "No relevant legal information found."

        context = self.get_context(docs)

        chain = self.prompt_template | self.llm

        response = chain.invoke(
            {"question": query, "context": context}
        )

        return response.content

    # HYBRID QUERY
    def query_with_user_files(
        self,
        question: str,
        user_documents: List[Document],
        embedding_model,
        top_k: int = 8
    ) -> Dict[str, Any]:

        logger.info("🔄 Hybrid query started")

        try:

            temp_results = []

            if user_documents:

                temp_db = FAISS.from_documents(
                    user_documents,
                    embedding_model
                )

                temp_results = temp_db.similarity_search(
                    question,
                    k=top_k
                )

            perm_results = self.retrieve_docs(
                question,
                top_k=top_k
            )

            combined_docs = (temp_results + perm_results)[:top_k]

            context = self.get_context(combined_docs)

            chain = self.prompt_template | self.llm

            response = chain.invoke(
                {"question": question, "context": context}
            )

            return {
                "success": True,
                "answer": response.content,
                "sources": len(combined_docs)
            }

        except Exception as e:

            logger.error(f"Hybrid error: {e}")

            return {
                "success": False,
                "answer": "Query failed",
                "error": str(e)
            }

    def execute_query(self, question: str) -> str:
        return self.answer_query(question)