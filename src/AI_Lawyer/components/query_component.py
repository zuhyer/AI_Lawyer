
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from AI_Lawyer.entity.config_entity import LLMConfig
from AI_Lawyer.utils.logging_setup import logger
from AI_Lawyer.utils.secret_loader import resolve_secret


class QueryComponent:
    def __init__(self, llm_config: LLMConfig, faiss_db):
        """
        llm_config : LLMConfig instance
        faiss_db   : Loaded FAISS vector store instance
        """
        self.llm_config = llm_config
        self.faiss_db = faiss_db

        logger.info("Initializing QueryComponent...")

        # Resolve encrypted API key from config.yaml → secret loader
        api_key = resolve_secret(self.llm_config.api_key)

        if not api_key:
            raise ValueError("❌ ERROR: LLM API key could not be resolved. Check your secret config.")

        # Basic masking for logs - don't print full key
        masked_key = (api_key[:6] + "...") if isinstance(api_key, str) and len(api_key) > 6 else "<empty>"

        logger.info(f"LLM API key resolved (masked): {masked_key}")

        # Initialize Groq LLM with helpful error handling for auth failures
        try:
            self.llm = ChatGroq(
                model=self.llm_config.model,
                groq_api_key=api_key
            )

            logger.info(f"Using LLM provider: {self.llm_config.provider}, model: {self.llm_config.model}")

        except Exception as e:
            err_str = str(e)
            # Common Groq authentication failure indicators
            if "invalid api key" in err_str.lower() or "invalid_api_key" in err_str.lower() or "401" in err_str:
                guidance = (
                    "Groq authentication failed: Invalid API Key (401).\n"
                    "Action: Ensure you placed a valid Groq API key in `config/secret.yaml` under the key referenced by `config/config.yaml` (e.g. 'response_model_API_Key'),\n"
                    "or set an appropriate environment variable if your setup uses one.\n"
                    "Current resolved key (masked): {}\n"
                    "If you previously stored a Hugging Face token (prefix 'hf_') or a different provider token, replace it with your Groq API key.\n"
                    "After updating, re-run the pipeline. Original error: {}"
                ).format(masked_key, err_str)

                logger.error(guidance)
                raise RuntimeError(guidance) from e

            # For other errors, re-raise
            logger.exception("Failed to initialize Groq LLM:")
            raise

        # Load the enhanced legal prompt template
        self.prompt_template = ChatPromptTemplate.from_template(self._get_prompt())

    # --------------------------------------------------------------------
    # PROMPT TEMPLATE (Legal-Grade)
    # --------------------------------------------------------------------
    def _get_prompt(self):
        return """
You are an AI Legal Research Assistant. Provide factual, context-based legal information ONLY.

⚖️ STRICT RULES:
1. Use ONLY the information provided in the context. No assumptions.
2. If the answer is not present in the context, reply:
   "The provided documents do not contain enough information to answer this."
3. Do NOT create legal advice or interpretations.
4. Cite context lines concisely when possible.
5. Keep answers clear, structured, and professional.

------------------------------------
QUESTION:
{question}

CONTEXT:
{context}

------------------------------------
ANSWER:
"""

    # --------------------------------------------------------------------
    # DOCUMENT RETRIEVAL
    # --------------------------------------------------------------------
    def retrieve_docs(self, query):
        """Retrieve the top semantic matches from FAISS."""
        logger.info(f"Retrieving documents for query: {query}")
        return self.faiss_db.similarity_search(query)

    def get_context(self, documents):
        """Convert a list of Document objects into a single concatenated context string."""
        logger.info(f"Preparing context from {len(documents)} retrieved chunks.")
        return "\n\n".join([doc.page_content for doc in documents])

    # --------------------------------------------------------------------
    # RAG QUERY HANDLER
    # --------------------------------------------------------------------
    def answer_query(self, query):
        """
        Full legal reasoning pipeline:
        1. Retrieve documents
        2. Build context
        3. Pass through LLM with legal-safe prompt
        """
        logger.info(f"Processing query: {query}")

        documents = self.retrieve_docs(query)

        if not documents:
            logger.warning("No relevant documents found in FAISS.")
            return "No relevant legal information found in the indexed documents."

        context = self.get_context(documents)

        chain = self.prompt_template | self.llm

        logger.info("Generating final LLM response...")

        response = chain.invoke({"question": query, "context": context})

        logger.info("LLM response generated successfully.")

        # Extract text content if response is a BaseMessage object
        if hasattr(response, 'content'):
            return response.content
        
        return str(response)

    # Backwards-compatible wrapper used by other modules
    def execute_query(self, question):
        """Compatibility shim: previous code called execute_query(question)."""
        return self.answer_query(question)

