# Hybrid Implementation - Complete Guide

## Overview

This document describes the **hybrid RAG implementation** that allows users to search both:
1. **Permanent Legal Database**: Pre-indexed Indian legal documents (Constitution, IPC, etc.)
2. **User Uploads**: Temporary documents uploaded by users (session-based)

---

## Architecture

### Components Added

#### 1. **UserUploadProcessorConfig** (Entity)
Location: `src/AI_Lawyer/entity/config_entity.py`

Configuration dataclass for upload processing:
- `chunk_size`: Size of text chunks (default 1000)
- `chunk_overlap`: Overlap between chunks (default 200)
- `max_upload_size_mb`: Max file size allowed (default 50MB)
- `temp_index_ttl_seconds`: Temporary index lifetime (default 3600s)

#### 2. **UserUploadProcessor** (Component)
Location: `src/AI_Lawyer/components/user_upload_processor.py`

Processes uploaded documents:
- Extracts text using `FileExtractor`
- Chunks text using `RecursiveCharacterTextSplitter`
- Creates LangChain `Document` objects with metadata
- Validates file sizes
- Provides chunk statistics

**Key Methods:**
- `process_single_file()`: Process one file → chunks
- `process_uploaded_files()`: Batch process multiple files
- `validate_file_size()`: Check file size limits
- `get_chunk_statistics()`: Return chunk info

#### 3. **Enhanced QueryComponent** (Component)
Location: `src/AI_Lawyer/components/query_component.py`

Extended with hybrid query support:
- `answer_query()`: Standard query (legal DB only)
- `retrieve_docs()`: Get docs from permanent FAISS
- `query_with_user_files()`: **Hybrid query** (legal DB + uploads)

**Hybrid Query Flow:**
```
User Question
    ↓
├─ Search Permanent FAISS (legal DB) → Results A
├─ Create Temporary FAISS from user uploads
├─ Search Temporary FAISS → Results B
├─ Merge & Rank (Results A + B)
├─ Generate Context
├─ LLM Generation
└─ Return Answer + Sources
```

#### 4. **Enhanced API Routes**
Location: `src/AI_Lawyer/api/routes/query.py`

New endpoints:
- `POST /query/ask`: Standard query (legal DB)
- `POST /query/hybrid`: Hybrid query with file uploads
- `GET /query/status`: System status

**Hybrid Endpoint Request:**
```json
{
  "query": "What are my obligations?",
  "files": [file1.pdf, file2.docx],
  "top_k": 5
}
```

**Response:**
```json
{
  "success": true,
  "answer": "...",
  "results": [
    {
      "text": "...",
      "source": "my_file.pdf",
      "score": 0.95,
      "source_type": "user_upload"
    }
  ],
  "permanent_db_results": 3,
  "user_upload_results": 2,
  "processing_time": 0.456
}
```

#### 5. **Streamlit UI**
Location: `streamlit_app.py`

Enhanced UI with:
- **File Upload Panel**: Upload documents to sidebar
- **Query Mode Toggle**: Standard vs Hybrid
- **Search Settings**: Adjust top_k, enable/disable sources
- **Results Display**: 
  - Answer in natural language
  - Source metrics (processing time, result counts)
  - Source previews with type indication (legal_db vs user_upload)
  - Expandable full source texts

---

## Configuration

### config.yaml
```yaml
user_upload_processor:
  max_upload_size_mb: 50
  temp_index_ttl_seconds: 3600
```

### params.yaml
```yaml
user_upload_params:
  chunk_size: 1000
  chunk_overlap: 200
  add_start_index: true
```

### ConfigurationManager
New method:
```python
def get_user_upload_processor_config(self) -> UserUploadProcessorConfig:
    """Get user upload processor configuration."""
```

---

## Usage

### Streamlit App

**Run the app:**
```bash
streamlit run streamlit_app.py
```

**Workflow:**
1. Open sidebar → Upload documents
2. Files processed automatically
3. Select **Hybrid** mode
4. Enter query
5. Click "Search"
6. View results with source breakdown

### API Server

**Run FastAPI:**
```bash
python -m uvicorn src.AI_Lawyer.api.app:app --host 0.0.0.0 --port 8000
```

**Standard Query (cURL):**
```bash
curl -X POST http://localhost:8000/query/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is Article 14?",
    "top_k": 5
  }'
```

**Hybrid Query (cURL):**
```bash
curl -X POST http://localhost:8000/query/hybrid \
  -F "query=What are my obligations?" \
  -F "files=@my_document.pdf" \
  -F "files=@case_file.docx" \
  -F "top_k=5"
```

### Python Code

**Standard Query:**
```python
from AI_Lawyer.config.configuration import ConfigurationManager
from AI_Lawyer.components.query_component import QueryComponent
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer

config = ConfigurationManager()
embedding_config = config.get_embeddings_config()

embedding_model = SentenceTransformer(embedding_config.model)
faiss_db = FAISS.load_local(
    embedding_config.vector_store_path,
    embedding_model
)

llm_config = config.get_llm_config()
query_comp = QueryComponent(llm_config, faiss_db)

answer = query_comp.answer_query("What is Article 14?")
print(answer)
```

**Hybrid Query:**
```python
from AI_Lawyer.components.user_upload_processor import UserUploadProcessor
from AI_Lawyer.components.file_extractor import FileExtractor

# Extract and process user files
file_extractor = FileExtractor(config.get_file_extractor_config())
files_dict = {
    "my_doc.pdf": file_extractor.extract_from_file("my_doc.pdf"),
    "case.docx": file_extractor.extract_from_file("case.docx")
}

# Create chunks
upload_processor = UserUploadProcessor(config.get_user_upload_processor_config())
user_documents = upload_processor.process_uploaded_files(files_dict)

# Hybrid query
query_comp.embedding_model = embedding_model
result = query_comp.query_with_user_files(
    question="What are my obligations?",
    user_documents=user_documents,
    top_k=5,
    embedding_model=embedding_model
)

print(f"Answer: {result['answer']}")
print(f"From Legal DB: {result['permanent_db_results']}")
print(f"From Uploads: {result['user_upload_results']}")
```

---

## Data Flow

### Standard Query Flow
```
User Question
    ↓
QueryComponent.answer_query()
    ├─ retrieve_docs(question, k=5)
    │   └─ FAISS.similarity_search()
    ├─ get_context(docs)
    ├─ LLM.invoke(prompt + context)
    └─ Return answer
```

### Hybrid Query Flow
```
User Question + Upload Files
    ↓
FileExtractor.extract_from_file()
    └─ Extract text from PDF/DOCX/IMG
    ↓
UserUploadProcessor.process_uploaded_files()
    └─ Chunk text, create Documents with metadata
    ↓
QueryComponent.query_with_user_files()
    ├─ retrieve_docs_with_scores(question)
    │   └─ Search Permanent FAISS
    ├─ FAISS.from_documents(user_documents, embedding_model)
    │   └─ Create Temporary FAISS
    ├─ Temp_FAISS.similarity_search_with_scores(question)
    │   └─ Search User Uploads
    ├─ Merge Results (Legal DB + Uploads)
    ├─ Sort by Score
    ├─ get_context(top_k merged results)
    ├─ LLM.invoke(prompt + context)
    └─ Return {answer, sources, counts}
```

---

## Metadata Structure

### Legal DB Documents
```python
Document(
    page_content="...",
    metadata={
        "source": "COI_2024.pdf",
        "chunk_index": 0,
        "file_type": "legal_db"
    }
)
```

### User Upload Documents
```python
Document(
    page_content="...",
    metadata={
        "source": "my_case.pdf",
        "chunk_index": 5,
        "chunk_size": 1000,
        "file_type": "user_upload",
        "total_chunks": 25
    }
)
```

---

## Error Handling

### File Extraction Errors
- Empty files → Logged, continue processing
- Unsupported formats → Handled gracefully
- OCR failures → Fallback to empty string

### Chunking Errors
- Empty text → Returns empty list with warning
- Processing errors → Logged, continues with next file

### Query Errors
- No FAISS → HTTPException 500
- Missing embedding model → Graceful hybrid query skip
- LLM API errors → Detailed error response

---

## Performance Considerations

### Temporary FAISS Creation
- Creates in-memory index from user documents
- Size = number_of_chunks × embedding_dim
- For 1000 chunks × 384 dims = ~1.5MB RAM

### Chunking Performance
- RecursiveCharacterTextSplitter: O(n) where n = text length
- Typical: 50MB file → ~200-300 chunks in <1s

### Embedding Performance
- Sentence-Transformers: ~100 chunks/s on CPU
- FAISS search: <100ms for K=5

### Total Hybrid Query Time
- Extract + chunk: 1-5 seconds
- FAISS searches: <500ms
- LLM generation: 2-10 seconds
- **Total: 3-15 seconds** (depending on file size + LLM)

---

## Testing

### Unit Test Example
```python
def test_user_upload_processor():
    config = ConfigurationManager()
    processor = UserUploadProcessor(config.get_user_upload_processor_config())
    
    # Process test file
    files_dict = {"test.txt": "This is a test document."}
    docs = processor.process_uploaded_files(files_dict)
    
    assert len(docs) > 0
    assert docs[0].metadata["source"] == "test.txt"
    assert docs[0].metadata["file_type"] == "user_upload"
```

### Integration Test Example
```python
def test_hybrid_query():
    config = ConfigurationManager()
    
    # Setup
    embedding_model = SentenceTransformer(...)
    faiss_db = FAISS.load_local(...)
    query_comp = QueryComponent(config.get_llm_config(), faiss_db)
    
    # Create mock user documents
    from langchain.schema import Document
    user_docs = [
        Document(page_content="My case file content", 
                metadata={"source": "case.pdf", "file_type": "user_upload"})
    ]
    
    # Query
    result = query_comp.query_with_user_files(
        "What happened?",
        user_docs,
        5,
        embedding_model
    )
    
    assert result["success"]
    assert result["permanent_db_results"] >= 0
    assert result["user_upload_results"] >= 0
```

---

## Troubleshooting

### Issue: "Temporary FAISS creation failed"
- **Cause**: Missing embedding model
- **Solution**: Ensure `embedding_model` is passed to `query_with_user_files()`

### Issue: "File extraction returned empty text"
- **Cause**: Empty file or OCR failure
- **Solution**: Verify file format, check tesseract installation for images

### Issue: Slow hybrid queries
- **Cause**: Large number of user documents
- **Solution**: Reduce `top_k`, increase `chunk_overlap` in params.yaml

### Issue: "No relevant documents found"
- **Cause**: Query terms not in documents
- **Solution**: Try more general search terms, check documents were uploaded

---

## Future Enhancements

1. **Persistent User Sessions**: Store uploaded documents beyond session
2. **Re-ranking**: Add cross-encoder for better result ranking
3. **Query Expansion**: Expand queries for broader searches
4. **Caching**: Cache embeddings of frequently uploaded documents
5. **Batch Requests**: Support async batch queries
6. **Custom Prompts**: Allow user-defined prompt templates
7. **Citation Tracking**: More detailed citation in answers
8. **Feedback Loop**: Learn from user query feedback

---

## File Summary

| File | Purpose |
|------|---------|
| `config_entity.py` | UserUploadProcessorConfig dataclass |
| `user_upload_processor.py` | Processes uploads → chunks |
| `query_component.py` | Enhanced with hybrid_query method |
| `routes/query.py` | API endpoints (/ask, /hybrid) |
| `models/requests.py` | HybridQueryRequest model |
| `models/responses.py` | HybridQueryResponse model |
| `streamlit_app.py` | UI with upload + hybrid query |
| `config.yaml` | Upload processor config |
| `params.yaml` | Chunking parameters |

---

## Contact & Support

For issues or questions about the hybrid implementation:
- Check logs in `logs/` directory
- Review error messages in UI/API responses
- Verify config files are properly formatted
- Ensure all dependencies are installed
