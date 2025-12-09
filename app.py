#!/usr/bin/env python3
"""
AI Lawyer - Streamlit Web Interface
A legal document Q&A system powered by local embeddings and Groq LLM.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st
from AI_Lawyer.config.configuration import ConfigurationManager
from AI_Lawyer.components.query_component import QueryComponent
from AI_Lawyer.pipeline.stage03_embedding_creation import load_existing_vector_store
from AI_Lawyer.utils.logging_setup import logger

# Page configuration
st.set_page_config(
    page_title="AI Lawyer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3em;
        color: #1f77b4;
        margin-bottom: 0.5em;
    }
    .subtitle {
        font-size: 1.2em;
        color: #555;
        margin-bottom: 1em;
    }
    .response-box {
        background-color: #f0f2f6;
        padding: 1.5em;
        border-radius: 0.5em;
        border-left: 4px solid #1f77b4;
        margin-top: 1em;
    }
    .error-box {
        background-color: #ffcccc;
        padding: 1.5em;
        border-radius: 0.5em;
        border-left: 4px solid #d9534f;
        margin-top: 1em;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_query_engine():
    """Load FAISS DB and QueryComponent once (cached for performance)."""
    try:
        with st.spinner("Loading legal document database..."):
            logger.info("Loading FAISS database...")
            faiss_db = load_existing_vector_store()

            if faiss_db is None:
                st.error(
                    "❌ FAISS database not found. Please run `python main.py` first to build the embeddings."
                )
                st.stop()

            logger.info("Loading LLM configuration...")
            config_manager = ConfigurationManager()
            llm_cfg = config_manager.get_llm_config()

            logger.info("Initializing QueryComponent...")
            query_engine = QueryComponent(llm_config=llm_cfg, faiss_db=faiss_db)

            return query_engine

    except RuntimeError as e:
        # Groq authentication error or similar
        st.error(f"⚠️ Configuration Error:\n\n{str(e)}")
        st.stop()

    except Exception as e:
        st.error(f"❌ Failed to initialize query engine: {str(e)}")
        logger.exception("Query engine initialization failed")
        st.stop()


def main():
    """Main Streamlit app."""
    # Header
    st.markdown(
        '<div class="main-header">⚖️ AI Lawyer</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="subtitle">Legal Document Q&A System</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Ask questions about Indian legal documents including the Constitution, "
        "Criminal Procedure Code, Civil Procedure Code, IPC, and more."
    )

    # Sidebar info
    with st.sidebar:
        st.header("ℹ️ About")
        st.write(
            "This system uses:\n"
            "- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)\n"
            "- **Vector Store**: FAISS\n"
            "- **LLM**: Groq (deepseek-r1-distill-llama-70b)\n"
            "- **RAG**: Retrieval-Augmented Generation"
        )

        st.header("🚀 Quick Tips")
        st.write(
            "- Ask specific legal questions (e.g., 'What is bail?')\n"
            "- Queries are answered from indexed documents only\n"
            "- Responses cite relevant legal text when available"
        )

        st.header("📚 Indexed Documents")
        st.write(
            "- Constitution of India (2024)\n"
            "- Indian Penal Code (IPC)\n"
            "- Criminal Procedure Code (CrPC), 1973\n"
            "- Civil Procedure Code (CPC), 1908\n"
            "- Indian Evidence Act, 1872\n"
            "- Registration Act, 1908\n"
            "- Supreme Court Landmark Judgments\n"
            "- And more..."
        )

    # Load the query engine
    query_engine = load_query_engine()

    # User input section
    st.header("❓ Ask a Question")

    col1, col2 = st.columns([5, 1])

    with col1:
        user_question = st.text_input(
            "Enter your legal question:",
            placeholder="e.g., What are the provisions related to bail under the CrPC?",
            label_visibility="collapsed"
        )

    with col2:
        search_button = st.button("🔍 Search", use_container_width=True)

    # Process query
    if search_button and user_question.strip():
        try:
            with st.spinner("Searching documents and generating response..."):
                logger.info(f"Processing query: {user_question}")
                response = query_engine.execute_query(user_question)

            st.markdown(
                '<div class="response-box">'
                '<strong>📋 Response:</strong><br><br>'
                + response.replace("\n", "<br>")
                + '</div>',
                unsafe_allow_html=True
            )

        except RuntimeError as e:
            # Config/auth errors from QueryComponent
            st.markdown(
                '<div class="error-box">'
                '<strong>⚠️ Configuration Error:</strong><br><br>'
                + str(e).replace("\n", "<br>")
                + '</div>',
                unsafe_allow_html=True
            )
            logger.exception("Query execution failed (config error)")

        except Exception as e:
            st.markdown(
                '<div class="error-box">'
                '<strong>❌ Error:</strong><br><br>'
                + str(e).replace("\n", "<br>")
                + '</div>',
                unsafe_allow_html=True
            )
            logger.exception("Query execution failed")

    elif search_button and not user_question.strip():
        st.warning("⚠️ Please enter a question before searching.")

    # Footer
    st.divider()
    st.caption(
        "⚖️ AI Lawyer | Powered by Local Embeddings, FAISS, and Groq | "
        "Educational Purpose Only"
    )


if __name__ == "__main__":
    main()
