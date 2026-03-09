from typing import List, Dict, Any
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from AI_Lawyer.entity.config_entity import LLMConfig
from AI_Lawyer.utils.logging_setup import logger
from AI_Lawyer.utils.secret_loader import resolve_secret
# (secret_loader now handles environment and file resolution automatically)


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

        # prompt may be supplied via config; otherwise use default template
        template_str = (
            self.llm_config.prompt_template
            if getattr(self.llm_config, 'prompt_template', '')
            else self._get_prompt()
        )
        self.prompt_template = ChatPromptTemplate.from_template(template_str)

    def _get_prompt(self) -> str:
        # default hard‑coded prompt; overridden by config if supplied
        # includes instructions for template generation when relevant
        return """\
You are an AI Legal Research Assistant.\n\nUse provided legal context first.\nIf insufficient say:\n\"The provided documents do not contain enough information.\"\n\nWhen the query relates to generating a legal template or format,
provide a clear, structured document with placeholders for the user
(e.g. <PartyA>, <Date>, <Signature>). Use sections/headings as
appropriate and include any explanatory comments.\n
QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\nANSWER:\n"""

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

        # if templates domain, use stronger prompt instructions
        if domain == 'legal_templates_db':
            prompt_text = (
                "You are an expert legal template generator. "
                "Produce the full template with placeholders and logic, "
                "based on question and context.\n\nQUESTION:\n{question}\n\nCONTEXT:\n{context}\n\nANSWER:\n"
            )
            template = ChatPromptTemplate.from_template(prompt_text)
            chain = template | self.llm
        else:
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

            # return detailed source list rather than just a count (bug fix)
            source_list = []
            for i, doc in enumerate(combined_docs):
                meta = doc.metadata if isinstance(doc.metadata, dict) else {}
                source_list.append({
                    "text": doc.page_content,
                    "source_name": meta.get("source_file"),
                    "score": meta.get("score"),
                    "source_type": meta.get("source_type"),
                    "chunk_index": meta.get("chunk_index"),
                })
            return {
                "success": True,
                "answer": response.content,
                "sources": source_list,
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