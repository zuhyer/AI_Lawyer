import streamlit as st
import time
from pathlib import Path

from AI_Lawyer.config.configuration import ConfigurationManager
from AI_Lawyer.components.query_component import QueryComponent
from AI_Lawyer.components.query_router import QueryRouter
from AI_Lawyer.utils.logging_setup import logger

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Lawyer - Intelligent Legal Assistant",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ AI Lawyer")
st.markdown("### 🤖 Intelligent Multi-Domain Legal Research Assistant")

# ============================================================
# SESSION INITIALIZATION
# ============================================================

if "config_manager" not in st.session_state:
    st.session_state.config_manager = ConfigurationManager()

if "embedding_model" not in st.session_state:
    with st.spinner("Loading embedding model..."):
        embedding_config = (
            st.session_state.config_manager.get_embeddings_config()
        )
        st.session_state.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_config.model
        )
        logger.info("✅ HuggingFace embedding model loaded")

if "query_router" not in st.session_state:
    st.session_state.query_router = QueryRouter(
        config_manager=st.session_state.config_manager,
        embedding_model=st.session_state.embedding_model
    )

if "llm_config" not in st.session_state:
    st.session_state.llm_config = (
        st.session_state.config_manager.get_llm_config()
    )

# ============================================================
# QUERY INPUT
# ============================================================

query = st.text_area(
    "Ask your legal question",
    placeholder="e.g., What is Article 21 of the Constitution of India?",
    height=120
)

search_button = st.button("🚀 Search")

# ============================================================
# QUERY EXECUTION
# ============================================================

if search_button and query.strip():

    with st.spinner("Analyzing query and retrieving legal context..."):

        try:
            start_time = time.time()

            # 🔥 STEP 1: AUTO DOMAIN ROUTING
            domain, faiss_index, domain_config = (
                st.session_state.query_router.route_query(query)
            )

            confidence = domain_config.get("classification_confidence", 0)

            # 🔥 STEP 2: CREATE QUERY COMPONENT WITH CORRECT FAISS
            query_component = QueryComponent(
                llm_config=st.session_state.llm_config,
                faiss_db=faiss_index
            )

            # 🔥 STEP 3: GENERATE ANSWER
            answer = query_component.answer_query(query)

            processing_time = time.time() - start_time

            # ============================================================
            # DISPLAY RESULTS
            # ============================================================

            st.success("✅ Query processed successfully")

            st.markdown(
                f"**🎯 Detected Domain:** `{domain}` "
                f"(Confidence: {confidence:.2f})"
            )

            st.markdown("### 💡 Legal Opinion")

            st.markdown(
                f"""
                <div style='background-color:#f0f8ff;
                            padding:1.5rem;
                            border-radius:0.5rem;
                            border-left:4px solid #0066cc;'>
                {answer}
                </div>
                """,
                unsafe_allow_html=True
            )

            col1, col2 = st.columns(2)

            with col1:
                st.metric("⏱️ Processing Time", f"{processing_time:.2f}s")

            with col2:
                st.metric("📚 Domain Used", domain)

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            logger.exception(e)