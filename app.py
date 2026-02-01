"""
Streamlit UI for AI Lawyer with Hybrid Query Support
Allows users to query legal documents + upload their own documents
"""

import streamlit as st
import pandas as pd
import time
from pathlib import Path
from typing import List, Dict, Any

# Project imports
from AI_Lawyer.config.configuration import ConfigurationManager
from AI_Lawyer.components.file_extractor import FileExtractor
from AI_Lawyer.components.user_upload_processor import UserUploadProcessor
from AI_Lawyer.components.query_component import QueryComponent
from AI_Lawyer.components.local_embedding import LocalSentenceTransformerEmbeddings
from AI_Lawyer.utils.logging_setup import logger

from langchain_community.vectorstores import FAISS


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================
def extract_clean_source_name(source_path: str) -> str:
    """
    Extract a clean source name from file path.
    
    Args:
        source_path: Full file path or name
        
    Returns:
        Clean filename without path and with URL decoding
    """
    if not source_path:
        return "Unknown Source"
    
    try:
        # Extract just the filename from path
        filename = Path(source_path).name
        # Decode URL-encoded characters
        filename = filename.replace("%20", " ")
        return filename if filename else "Unknown Source"
    except Exception:
        return source_path if source_path else "Unknown Source"


# =====================================================================
# PAGE CONFIGURATION
# =====================================================================
st.set_page_config(
    page_title="AI Lawyer - Hybrid Legal Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚖️ AI Lawyer - Hybrid Legal Assistant")
st.markdown("""
*A powerful legal research assistant powered by RAG with support for user documents*
""")

# =====================================================================
# SESSION STATE & INITIALIZATION (MUST BE FIRST!)
# =====================================================================
if "config_manager" not in st.session_state:
    st.session_state.config_manager = ConfigurationManager()
    logger.info("✅ ConfigurationManager initialized in session")

if "embedding_model" not in st.session_state:
    with st.spinner("📦 Loading embedding model..."):
        embedding_config = st.session_state.config_manager.get_embeddings_config()
        st.session_state.embedding_model = LocalSentenceTransformerEmbeddings(embedding_config.model)
        logger.info("✅ Embedding model loaded")

if "faiss_db" not in st.session_state:
    with st.spinner("📚 Loading legal database..."):
        embedding_config = st.session_state.config_manager.get_embeddings_config()
        st.session_state.faiss_db = FAISS.load_local(
            embedding_config.vector_store_path,
            st.session_state.embedding_model,
            allow_dangerous_deserialization=True
        )
        logger.info("✅ FAISS legal database loaded")

if "query_component" not in st.session_state:
    llm_config = st.session_state.config_manager.get_llm_config()
    st.session_state.query_component = QueryComponent(
        llm_config, 
        st.session_state.faiss_db
    )
    st.session_state.query_component.embedding_model = st.session_state.embedding_model
    logger.info("✅ QueryComponent initialized")

if "user_documents" not in st.session_state:
    st.session_state.user_documents = []

if "upload_processor" not in st.session_state:
    upload_config = st.session_state.config_manager.get_user_upload_processor_config()
    st.session_state.upload_processor = UserUploadProcessor(upload_config)
    logger.info("✅ UserUploadProcessor initialized")

if "file_extractor" not in st.session_state:
    file_config = st.session_state.config_manager.get_file_extractor_config()
    st.session_state.file_extractor = FileExtractor(file_config)
    logger.info("✅ FileExtractor initialized")




# =====================================================================
# SIDEBAR - UPLOAD & SETTINGS
# =====================================================================
with st.sidebar:
    st.title("⚙️ Settings & Upload")
    
    # ===== UPLOAD SECTION =====
    st.header("📤 Upload Documents")
    st.markdown("*Upload files to search alongside the legal database*")
    
    uploaded_files = st.file_uploader(
        "Select files to upload:",
        type=["pdf", "docx", "doc", "txt", "png", "jpg", "jpeg", "bmp", "tiff"],
        accept_multiple_files=True,
        help="Upload multiple documents at once for hybrid search",
        key="file_uploader_sidebar"
    )
    
    # Auto-process uploaded files when files selected
    if uploaded_files and len(uploaded_files) > 0:
        with st.spinner("⏳ Extracting and processing documents..."):
            try:
                files_dict = {}
                temp_dir = Path("/tmp/ai_lawyer_uploads")
                temp_dir.mkdir(exist_ok=True)
                
                progress_bar = st.progress(0)
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    # Save to temporary file
                    temp_path = temp_dir / uploaded_file.name
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Extract text
                    try:
                        extracted_text = st.session_state.file_extractor.extract_from_file(str(temp_path))
                        files_dict[uploaded_file.name] = extracted_text
                        st.success(f"✅ Extracted: {uploaded_file.name}", icon="✅")
                    except Exception as e:
                        st.error(f"❌ Failed to extract {uploaded_file.name}: {e}")
                        logger.error(f"Extraction failed: {e}")
                        continue
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                # Process into chunks
                if files_dict:
                    st.session_state.user_documents = st.session_state.upload_processor.process_uploaded_files(
                        files_dict
                    )
                    
                    # Show statistics
                    try:
                        stats = st.session_state.upload_processor.get_chunk_statistics(
                            st.session_state.user_documents
                        )
                        
                        st.metric("📊 Total Chunks", len(st.session_state.user_documents))
                        
                    except Exception as stats_err:
                        st.success(f"✅ Processed {len(st.session_state.user_documents)} chunks from {len(files_dict)} files")
                    
                    logger.info(f"✅ Uploaded {len(st.session_state.user_documents)} chunks")
                    
            except Exception as e:
                st.error(f"⚠️ Error processing uploads: {e}")
                logger.exception(f"File processing failed: {e}")
    else:
        st.info("📌 No files uploaded yet.")
    
    # Display status if files are already uploaded
    if len(st.session_state.user_documents) > 0:
        st.success(f"✅ {len(st.session_state.user_documents)} chunks ready for Hybrid search!")
    
    st.divider()
    
    # ===== SEARCH SETTINGS SECTION =====
    st.header("🔍 Search Settings")
    
    use_permanent_db = st.checkbox(
        "Search Legal Database",
        value=True,
        help="Include the permanent legal database in search"
    )
    
    use_user_uploads = st.checkbox(
        "Search Uploaded Documents",
        value=len(st.session_state.user_documents) > 0,
        disabled=len(st.session_state.user_documents) == 0,
        help="Include uploaded documents in search"
    )
    
    top_k = st.slider(
        "Number of Results",
        min_value=1,
        max_value=20,
        value=5,
        help="How many results to retrieve and combine"
    )
    
    st.divider()
    
    # ===== SYSTEM STATUS SECTION =====
    st.header("ℹ️ System Status")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Legal DB", "✅ Ready")
    with col2:
        user_docs_count = len(st.session_state.user_documents)
        st.metric("Uploads", f"{'✅' if user_docs_count > 0 else '⚠️'} {user_docs_count} chunks")


if "query_mode" not in st.session_state:
    st.session_state.query_mode = "Standard"

# Auto-switch to Hybrid when documents are uploaded
if len(st.session_state.user_documents) > 0:
    st.session_state.query_mode = "Hybrid"


# =====================================================================
# MAIN CONTENT - QUERY INTERFACE
# =====================================================================
st.header("🔍 Legal Query Assistant")

# Query input
query = st.text_area(
    "Ask a legal question:",
    placeholder="e.g., What are the penalties under Section 304A of the Indian Penal Code?",
    height=100,
    help="Enter your legal question. The assistant will search both the legal database and your uploaded documents."
)

# Query mode selection and button row
col1, col2, col3, col4 = st.columns([1.2, 1, 1.2, 1.5])

with col1:
    # Display mode info - auto-switches to Hybrid when documents uploaded
    if len(st.session_state.user_documents) > 0:
        st.info(f"📤 **Mode: Hybrid** (Auto-enabled with {len(st.session_state.user_documents)} chunks)")
        hybrid_mode = "Hybrid"
    else:
        st.info("📚 **Mode: Standard** (Upload docs to enable Hybrid)")
        hybrid_mode = "Standard"

with col2:
    if st.button("🚀 Search", type="primary", use_container_width=True):
        if not query.strip():
            st.error("❌ Please enter a query")
        else:
            with st.spinner("⏳ Processing query..."):
                try:
                    start_time = time.time()
                    
                    if hybrid_mode == "Hybrid" and len(st.session_state.user_documents) > 0:
                        # Hybrid query
                        result = st.session_state.query_component.query_with_user_files(
                            question=query,
                            user_documents=st.session_state.user_documents,
                            top_k=top_k,
                            embedding_model=st.session_state.embedding_model
                        )
                        
                        if result["success"]:
                            processing_time = time.time() - start_time
                            
                            # Display answer
                            st.success("✅ Query processed successfully")
                            
                            st.subheader("💡 Answer")
                            st.write(result["answer"])
                            
                            # Display metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Processing Time", f"{processing_time:.2f}s")
                            with col2:
                                st.metric("Legal DB Results", result["permanent_db_results"])
                            with col3:
                                st.metric("User Doc Results", result["user_upload_results"])
                            
                            # Display sources
                            st.subheader("📚 Sources")
                            
                            sources_data = []
                            for source in result.get("sources", []):
                                sources_data.append({
                                    "Source": source["source_name"],
                                    "Type": source["source_type"],
                                    "Score": f"{source['score']:.3f}",
                                    "Preview": source["text"][:100] + "..."
                                })
                            
                            if sources_data:
                                df = pd.DataFrame(sources_data)
                                st.dataframe(df, use_container_width=True)
                            
                            # Display full sources
                            with st.expander("📖 Full Source Texts"):
                                for i, source in enumerate(result.get("sources", []), 1):
                                    st.markdown(f"**Source {i}: {source['source_name']} (Score: {source['score']:.3f})**")
                                    st.text(source["text"])
                                    st.divider()
                        else:
                            st.error(f"Query failed: {result.get('error', 'Unknown error')}")
                    
                    else:
                        if hybrid_mode == "Hybrid" and len(st.session_state.user_documents) == 0:
                            st.warning("⚠️ No uploaded documents for Hybrid mode. Using Standard mode instead.")
                        
                        # Standard query (legal DB only)
                        answer = st.session_state.query_component.answer_query(query)
                        docs = st.session_state.query_component.retrieve_docs(query, top_k=top_k)
                        processing_time = time.time() - start_time
                        
                        st.success("✅ Query processed successfully")
                        
                        # Display answer
                        st.subheader("💡 Answer")
                        st.write(answer)
                        
                        # Display metrics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Processing Time", f"{processing_time:.2f}s")
                        with col2:
                            st.metric("Sources Found", len(docs))
                        
                        # Display sources
                        st.subheader("📚 Sources")
                        
                        sources_data = []
                        for doc in docs:
                            clean_source = extract_clean_source_name(doc.metadata.get("source", "Unknown"))
                            sources_data.append({
                                "Source": clean_source,
                                "Preview": doc.page_content[:100] + "..."
                            })
                        
                        if sources_data:
                            df = pd.DataFrame(sources_data)
                            st.dataframe(df, use_container_width=True)
                        
                        # Display full sources
                        with st.expander("📖 Full Source Texts"):
                            for i, doc in enumerate(docs, 1):
                                clean_source = extract_clean_source_name(doc.metadata.get('source', 'Unknown'))
                                st.markdown(f"**Source {i}: {clean_source}**")
                                st.text(doc.page_content)
                                st.divider()
                    
                except Exception as e:
                    st.error(f"❌ Error processing query: {str(e)}")
                    logger.exception(f"Query failed: {e}")

with col3:
    st.markdown("**Help**")
    if st.button("❓ How to use", use_container_width=True):
        st.info("See the How to use section below for guidance")

with col4:
    st.markdown("**Status**")
    upload_count = len(st.session_state.user_documents)
    if upload_count > 0:
        st.success(f"✅ {upload_count} chunks", icon="✅")
    else:
        st.info("⬆️ Upload in sidebar")


st.divider()

# Help section
with st.expander("❓ How to use"):
    st.markdown("""
    ### Standard Mode
    - Searches only the legal database of Indian legal documents
    - Fast and reliable for general legal queries
    
    ### Hybrid Mode
    - Searches both the legal database AND your uploaded documents
    - Combines results for comprehensive coverage
    - Requires at least one uploaded document
    
    ### Tips for Best Results
    1. Upload relevant documents first (Hybrid mode)
    2. Ask specific, clear legal questions
    3. Reference specific sections or acts when known
    4. Review the sources to verify information
    
    ### Supported File Types
    - **Documents**: PDF, DOCX, TXT
    - **Images**: PNG, JPG, JPEG, BMP, TIFF (with OCR)
    """)

with st.expander("⚖️ About"):
    st.markdown("""
    **AI Lawyer** is a hybrid legal research assistant powered by:
    - **RAG (Retrieval Augmented Generation)**: Combines retrieval + LLM generation
    - **FAISS**: Vector similarity search
    - **Groq LLM**: Fast legal reasoning
    - **LangChain**: Orchestration framework
    
    The system maintains two search indices:
    1. **Permanent**: Pre-indexed Indian legal documents
    2. **Temporary**: Your uploaded documents (session-based)
    """)

