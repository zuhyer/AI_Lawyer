# Hybrid Temporary Context: Step-by-Step Implementation Guide

## Phase 1: Prepare Environment

### Step 1.1: Add Dependencies to requirements.txt
**What:** Add libraries to support DOCX, image extraction, and OCR.

**File:** `requirements.txt`

**Action:**
- Open `requirements.txt`
- Add the following lines:
  ```
  python-docx
  Pillow
  pytesseract
  ```
- Save the file
- Run: `pip install python-docx Pillow pytesseract`

**System-level dependency (for OCR):**
- **Linux (Ubuntu/Debian):**
  ```bash
  sudo apt-get install tesseract-ocr
  ```
- **Mac (Homebrew):**
  ```bash
  brew install tesseract
  ```
- **Windows:** Download installer from https://github.com/UB-Mannheim/tesseract/wiki

**Why:** 
- python-docx: extracts text from Microsoft Word (.docx) files
- Pillow: image processing (load PNG, JPEG, etc.)
- pytesseract: OCR (Optical Character Recognition) to extract text from images
- tesseract-ocr: system binary that pytesseract uses for OCR

**Verification:**
```bash
python -c "import docx; import PIL; import pytesseract; print('All libraries installed successfully')"
tesseract --version
```

---

## Phase 2: Build File Extraction Module

### Step 2.1: Create file_extractor.py
**What:** A module that extracts text from PDFs, DOCX, TXT, and images (with OCR).

**File:** `src/AI_Lawyer/components/file_extractor.py`

**Logic to implement:**

```
Class FileExtractor:
  
  Method: extract_pdf(file_path: str) → str
    - Use pdfplumber.open(file_path)
    - Loop through pages, extract text from each
    - Join all text with page separators
    - Handle errors (corrupted PDFs)
    - Return: full document text
  
  Method: extract_docx(file_path: str) → str
    - Use docx.Document(file_path)
    - Loop through paragraphs
    - Extract text from each paragraph
    - Join with newlines
    - Handle errors (corrupted DOCX)
    - Return: full document text
  
  Method: extract_txt(file_path: str) → str
    - Read file with encoding detection (utf-8, fallback to latin-1)
    - Return: raw text
  
  Method: extract_image_ocr(file_path: str) → str
    - Use Pillow (PIL) to open image (PNG, JPEG, JPG, BMP, TIFF)
    - Use pytesseract.image_to_string(image) to extract text via OCR
    - Handle cases where image contains no text
    - Log confidence level if available
    - Handle errors (corrupted images, tesseract not installed)
    - Return: extracted text from image
    - Note: First attempt pytesseract; if fails, return error message
  
  Method: extract_from_uploaded_file(file_path: str, file_name: str) → str
    - Detect file type from extension (.pdf, .docx, .txt, .png, .jpg, .jpeg, .bmp, .tiff)
    - Call appropriate extract_* method based on type
    - Return extracted text
    - Log which method was used
    - Handle unsupported formats gracefully (raise helpful error)
  
  Method: extract_batch(file_list: List[UploadFile]) → Dict[str, str]
    - Loop through files
    - Call extract_from_uploaded_file for each
    - Return dict: {filename: extracted_text}
    - Track any errors per file, log them but don't fail on single bad file
```

**Key points:**
- Use try/except for each file to avoid one bad file breaking the batch
- Return filename + extracted text so user knows which text came from which file
- Log extraction for debugging
- For images: include note in metadata that text came from OCR (may have errors)
- Don't store files to disk yet (keep in memory)
- Graceful degradation: if OCR fails, log warning but return empty string rather than crash

**Supported file types:**
- Documents: PDF, DOCX, DOC, TXT
- Images: PNG, JPEG, JPG, BMP, TIFF

**Testing:** Create 4 small test files (sample.pdf, sample.docx, sample.txt, sample.png with text) and test extraction manually.

---

### Step 2.2: Validate file_extractor.py Works
**Action:**
1. Create small test files in `/tmp/test_files/`:
   - sample.pdf (a legal document or any PDF)
   - sample.docx (a Word document with text)
   - sample.txt (plain text file)
   - sample.png (image with text, e.g., screenshot of document)
   
2. **Before running tests**, ensure Tesseract is installed:
   ```bash
   tesseract --version
   ```
   If not installed, follow Step 1.1 system-level dependency instructions.

3. Write a test script:
   ```python
   from AI_Lawyer.components.file_extractor import FileExtractor
   
   extractor = FileExtractor()
   
   # Test PDF
   try:
       pdf_text = extractor.extract_pdf("/tmp/test_files/sample.pdf")
       print(f"✓ PDF extracted: {len(pdf_text)} chars")
   except Exception as e:
       print(f"✗ PDF failed: {e}")
   
   # Test DOCX
   try:
       docx_text = extractor.extract_docx("/tmp/test_files/sample.docx")
       print(f"✓ DOCX extracted: {len(docx_text)} chars")
   except Exception as e:
       print(f"✗ DOCX failed: {e}")
   
   # Test TXT
   try:
       txt_text = extractor.extract_txt("/tmp/test_files/sample.txt")
       print(f"✓ TXT extracted: {len(txt_text)} chars")
   except Exception as e:
       print(f"✗ TXT failed: {e}")
   
   # Test Image (OCR)
   try:
       image_text = extractor.extract_image_ocr("/tmp/test_files/sample.png")
       if image_text:
           print(f"✓ Image OCR extracted: {len(image_text)} chars")
       else:
           print(f"⚠ Image OCR returned empty (no text detected in image)")
   except Exception as e:
       print(f"✗ Image OCR failed: {e}")
   
   # Test batch
   try:
       files = {
           "sample.pdf": "...",
           "sample.docx": "...",
           "sample.txt": "...",
           "sample.png": "..."
       }
       # This would use actual batch method
       print(f"✓ Batch processing test passed")
   except Exception as e:
       print(f"✗ Batch failed: {e}")
   ```

4. Run and verify all extract methods work
5. Keep this test script for reference

**Troubleshooting OCR:**
- If pytesseract fails with "tesseract is not installed": Install tesseract binary (see Step 1.1)
- If image OCR returns empty: Image may not contain readable text or quality is too low
- For scanned documents: OCR works best with high-resolution images (300+ DPI)

---

## Phase 3: Build Upload Processor Module

### Step 3.1: Create user_upload_processor.py
**What:** Takes extracted text and chunks it the same way as your Stage 2 pipeline.

**File:** `src/AI_Lawyer/components/user_upload_processor.py`

**Logic to implement:**

```
Class UserUploadProcessor:
  
  Method: __init__()
    - Import ConfigurationManager
    - Load chunking_config from config (chunk_size, chunk_overlap, etc.)
    - Initialize RecursiveCharacterTextSplitter (same as Stage 2)
  
  Method: process_single_file(filename: str, extracted_text: str) → List[Document]
    - Split text into chunks using splitter
    - For each chunk, create LangChain Document with metadata:
      metadata = {
        "source": filename,
        "chunk_index": i,
        "chunk_text_length": len(chunk)
      }
    - Return list of Document objects
  
  Method: process_uploaded_files(files_dict: Dict[str, str]) → List[Document]
    # files_dict is from FileExtractor.extract_batch()
    - Initialize empty list: all_documents = []
    - For each filename and text:
      - Call process_single_file(filename, text)
      - Extend all_documents with returned chunks
    - Log: f"Created {len(all_documents)} chunks from {len(files_dict)} files"
    - Return all_documents
```

**Key points:**
- Use the SAME chunking config as your Stage 2 (from config/config.yaml)
- Add "source_filename" to metadata so answers can cite which uploaded file
- Return LangChain Document objects (same format as your existing pipeline)
- This makes it compatible with your existing FAISS + embedding code

**Testing:** 
1. Use output from file_extractor test
2. Verify chunks are created with correct metadata
3. Print sample chunk: `print(documents[0].metadata, documents[0].page_content[:100])`

---

## Phase 4: Enhance QueryComponent

### Step 4.1: Add query_with_user_files() Method to QueryComponent
**What:** Extends QueryComponent to handle both FAISS search + temporary user file search.

**File:** `src/AI_Lawyer/components/query_component.py`

**Logic to implement:**

```
# Add NEW METHOD to QueryComponent class:

  Method: query_with_user_files(question: str, user_documents: List[Document], top_k: int = 5) → Dict
    
    # Step 1: Create temporary FAISS index from user documents
    temp_embeddings = self.embedding_model  # Use existing SentenceTransformer
    temp_faiss_db = FAISS.from_documents(
      user_documents,
      temp_embeddings
    )
    logger.info(f"Created temporary FAISS with {len(user_documents)} chunks")
    
    # Step 2: Search PERMANENT FAISS (legal database)
    permanent_results = self.faiss_db.similarity_search_with_scores(question, k=top_k)
    logger.info(f"Found {len(permanent_results)} results from legal DB")
    
    # Step 3: Search TEMPORARY FAISS (user uploads)
    user_results = temp_faiss_db.similarity_search_with_scores(question, k=top_k)
    logger.info(f"Found {len(user_results)} results from user uploads")
    
    # Step 4: Merge and rank results
    # Combine both lists, sort by score (higher = more relevant)
    combined = permanent_results + user_results
    combined_sorted = sorted(combined, key=lambda x: x[1], reverse=True)[:top_k]
    
    # Step 5: Extract documents and build context string
    combined_documents = [doc for doc, score in combined_sorted]
    context = "\n\n".join([
      f"[From: {doc.metadata.get('source', 'legal_db')}]\n{doc.page_content}"
      for doc in combined_documents
    ])
    
    # Step 6: Build response with LLM (reuse existing answer_query logic)
    chain = self.prompt_template | self.llm
    response = chain.invoke({"question": question, "context": context})
    
    # Step 7: Return answer + sources with metadata
    sources = []
    for doc, score in combined_sorted:
      source_info = {
        "text": doc.page_content[:200],  # First 200 chars
        "score": float(score),
        "source_type": "user_upload" if "source" in doc.metadata else "legal_db",
        "source_name": doc.metadata.get("source", "Legal Database")
      }
      sources.append(source_info)
    
    return {
      "answer": response.content if hasattr(response, 'content') else str(response),
      "sources": sources,
      "total_chunks_searched": len(combined)
    }
```

**Key points:**
- Reuse your existing embedding model (SentenceTransformer from app.state or self)
- Create temporary FAISS (only in memory, discarded after response)
- Merge results intelligently (rank by relevance score)
- Include source attribution in response
- Don't modify existing `answer_query()` method; this is a new method

**Testing:**
1. Load a real FAISS + create mock user documents
2. Call `query_with_user_files("test question", [mock_docs])`
3. Verify response includes sources from both legal DB and uploads

---

## Phase 5: Create Pydantic Models (for API/Streamlit)

### Step 5.1: Add Upload Request/Response Models
**What:** Define data structures for file upload + query requests.

**File:** `src/AI_Lawyer/entity/upload_models.py` (create new file)

**Logic to implement:**

```
from pydantic import BaseModel, Field
from typing import List, Optional

class SourceInfo(BaseModel):
    text: str = Field(..., description="Extracted chunk text")
    score: float = Field(..., ge=0, le=1, description="Relevance score")
    source_type: str = Field(..., description="'legal_db' or 'user_upload'")
    source_name: str = Field(..., description="Filename or database name")

class QueryWithFilesResponse(BaseModel):
    answer: str = Field(..., description="LLM-generated answer")
    sources: List[SourceInfo] = Field(..., description="Sources used for answer")
    total_chunks_searched: int = Field(..., description="Total chunks from both sources")
    query: str = Field(..., description="Original question asked")
    
class FileUploadBatch(BaseModel):
    files_processed: int
    total_chunks_created: int
    files: List[str] = Field(..., description="List of filenames processed")
```

**Key points:**
- Use for Streamlit display and FastAPI responses
- Add validation (scores 0-1, required fields)
- Make responses clear and serializable

---

## Phase 6: Integrate into Streamlit

### Step 6.1: Add File Uploader to app.py
**What:** Add a file upload widget to your Streamlit interface.

**File:** `app.py`

**Changes to make:**

**Location:** In the `main()` function, after the header section, add:

```python
# Add this BEFORE the "Ask a Question" header

st.divider()
st.header("📎 Optional: Upload Legal Documents & Images")
st.write(
    "Upload your own PDFs, Word documents, text files, or images (with text) to search them alongside the legal database."
)

uploaded_files = st.file_uploader(
    "Choose files to analyze:",
    type=["pdf", "docx", "doc", "txt", "png", "jpg", "jpeg", "bmp", "tiff"],
    accept_multiple_files=True,
    help="Supported: PDF, DOCX, TXT, PNG, JPEG, JPG, BMP, TIFF (images will use OCR)"
)

if uploaded_files:
    st.info(f"ℹ️ {len(uploaded_files)} file(s) selected. Images will be processed with OCR text extraction.")

st.divider()
```

**Location:** Modify the query execution section to handle uploaded files:

**Old code:**
```python
if search_button and user_question.strip():
    try:
        with st.spinner("Searching documents..."):
            response = query_engine.execute_query(user_question)
```

**New code:**
```python
if search_button and user_question.strip():
    try:
        if uploaded_files:
            with st.spinner("Processing uploaded files..."):
                # Step 1: Extract text from uploaded files
                from AI_Lawyer.components.file_extractor import FileExtractor
                from AI_Lawyer.components.user_upload_processor import UserUploadProcessor
                
                extractor = FileExtractor()
                processor = UserUploadProcessor()
                
                # Convert Streamlit UploadedFile to BytesIO, then extract
                extracted_texts = {}
                for uploaded_file in uploaded_files:
                    file_bytes = uploaded_file.read()
                    # Save temp file (FAISS.from_documents needs Document objects, not text)
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
                        tmp.write(file_bytes)
                        tmp_path = tmp.name
                    
                    extracted_text = extractor.extract_from_uploaded_file(tmp_path, uploaded_file.name)
                    extracted_texts[uploaded_file.name] = extracted_text
                
                # Step 2: Chunk and embed
                user_documents = processor.process_uploaded_files(extracted_texts)
                st.info(f"✅ Processed {len(uploaded_files)} files into {len(user_documents)} chunks")
                
                # Note if any files had extraction warnings
                extraction_warnings = []
                for filename in uploaded_files:
                    if filename.name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        extraction_warnings.append(f"⚠️ {filename.name}: Text extracted via OCR (may contain errors)")
                if extraction_warnings:
                    for warning in extraction_warnings:
                        st.warning(warning)
                
                # Step 3: Query with user files
                with st.spinner("Searching documents and generating response..."):
                    response_dict = query_engine.query_with_user_files(
                        user_question,
                        user_documents,
                        top_k=5
                    )
                    response = response_dict["answer"]
                    sources = response_dict["sources"]
        else:
            # No uploaded files, use regular query
            with st.spinner("Searching documents..."):
                response = query_engine.execute_query(user_question)
                sources = []
```

**Location:** Update response display to show sources:

```python
        # Display answer
        st.markdown(
            '<div class="response-box">'
            '<strong>📋 Response:</strong><br><br>'
            + response.replace("\n", "<br>")
            + '</div>',
            unsafe_allow_html=True
        )
        
        # Display sources (NEW)
        if uploaded_files or sources:
            st.divider()
            st.subheader("📚 Sources")
            for i, source in enumerate(sources, 1):
                with st.expander(f"Source {i}: {source['source_name']} (Score: {source['score']:.2f})"):
                    st.write(f"**Type:** {source['source_type']}")
                    st.write(f"**Score:** {source['score']:.2f}")
                    st.write(f"**Text:**\n{source['text'][:500]}...")
```

**Key points:**
- Use st.file_uploader() for multiple file selection
- Extract + chunk in-memory (don't persist to disk)
- Call new `query_with_user_files()` method
- Display sources with badges ("legal_db" vs "user_upload")
- Handle file reading safely

---

### Step 6.2: Test Streamlit Integration
**Action:**
1. Run: `streamlit run app.py`
2. Upload a small test PDF or DOCX
3. Ask a question
4. Verify:
   - File is processed without error
   - Chunks are created
   - Response includes sources from both legal DB and upload
   - Source badges show correct type

**Expected output:**
```
✅ Processed 1 files into 12 chunks
📋 Response: [answer text]
📚 Sources:
  Source 1: Legal Database (Score: 0.92) [from legal_db]
  Source 2: your_file.pdf (Score: 0.88) [from user_upload]
```

---

## Phase 7: Optional - FastAPI Endpoint

### Step 7.1: Create FastAPI Endpoint (if adding API)
**File:** `api/routers/query_with_files.py` (create new)

**Logic:**
```
@router.post("/query-with-files")
async def query_with_files(
    question: str = Form(...),
    files: List[UploadFile] = File(...),
    top_k: int = Form(default=5)
) → QueryWithFilesResponse:
    
    # Step 1: Extract files
    extractor = FileExtractor()
    extracted = {}
    for file in files:
        content = await file.read()
        # Save to temp and extract
        extracted[file.filename] = await extractor.extract_from_uploaded_file(...)
    
    # Step 2: Process
    processor = UserUploadProcessor()
    user_docs = processor.process_uploaded_files(extracted)
    
    # Step 3: Query
    query_engine = Depends(get_query_engine)
    result = query_engine.query_with_user_files(question, user_docs, top_k)
    
    return QueryWithFilesResponse(**result)
```

---

## Phase 8: Optional - Audit Trail

### Step 8.1: Add Audit Logging
**File:** `src/AI_Lawyer/components/audit_logger.py` (create new)

**Logic:**
```
Class AuditLogger:
  
  Method: log_file_upload(filename: str, extracted_text_length: int, chunks_created: int)
    - Save to: artifacts/audit_logs/uploads_[date].json
    - Log format: {timestamp, filename, text_length, chunk_count}
  
  Method: log_query(question: str, sources_searched: int, response_length: int)
    - Save to: artifacts/audit_logs/queries_[date].json
```

**Usage in Streamlit:**
```python
# After file processing
from AI_Lawyer.components.audit_logger import AuditLogger
audit = AuditLogger()
for file in uploaded_files:
    audit.log_file_upload(file.name, len(extracted_text), len(user_documents))
```

---

## Summary Checklist

### ✅ Must Do (Core Feature)
- [ ] Step 1.1: Add python-docx, Pillow, pytesseract to requirements + install Tesseract binary
- [ ] Step 2.1: Create file_extractor.py (PDF, DOCX, TXT, **Images with OCR**)
- [ ] Step 3.1: Create user_upload_processor.py
- [ ] Step 4.1: Add query_with_user_files() to QueryComponent
- [ ] Step 6.1: Add file uploader to Streamlit app.py (support images)
- [ ] Step 6.2: Test end-to-end with images

### ⭐ Nice to Have (Enhancements)
- [ ] Step 5.1: Create Pydantic models (for API)
- [ ] Step 7.1: Add FastAPI endpoint
- [ ] Step 8.1: Add audit logging

---

## Testing Commands

After each phase, run:

```bash
# Phase 1 test - dependencies
pip list | grep -E "python-docx|Pillow|pytesseract"

# Verify Tesseract installation (required for OCR)
tesseract --version

# Phase 2 test
python -c "from AI_Lawyer.components.file_extractor import FileExtractor; print('✓ FileExtractor imports')"

# Phase 3 test
python -c "from AI_Lawyer.components.user_upload_processor import UserUploadProcessor; print('✓ UserUploadProcessor imports')"

# Phase 4 test
python -c "from AI_Lawyer.components.query_component import QueryComponent; print('✓ query_with_user_files exists')"

# Full Streamlit test
streamlit run app.py
```

**Testing with sample files:**
```bash
# Create test directory
mkdir -p /tmp/test_files

# Download a sample PDF (e.g., Constitution of India)
wget -O /tmp/test_files/sample.pdf "https://example.com/sample.pdf"

# Create a test image with text (screenshot or use an online tool)
# Or use ImageMagick to create one:
# convert -size 400x100 xc:white -pointsize 30 -draw "text 20,50 'This is a test'" /tmp/test_files/sample.png

# Test extraction
python -c "
from AI_Lawyer.components.file_extractor import FileExtractor
extractor = FileExtractor()
print('Testing file extraction...')
# Add extraction tests here
"
```

---

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| "pdfplumber not reading text" | PDF may be scanned image; add OCR or skip |
| "DOCX extraction empty" | Check .docx is not encrypted; test with sample file |
| "Image OCR returns nothing" | Image may have no readable text or low resolution; try 300+ DPI |
| "pytesseract error: tesseract not installed" | Install Tesseract binary via: `sudo apt-get install tesseract-ocr` (Linux) or `brew install tesseract` (Mac) |
| "OCR results contain gibberish" | Image quality too low or language not English; try upscaling image first |
| "FAISS merge errors" | Ensure embedding dimensions match (384 for all-MiniLM-L6-v2) |
| "Streamlit file uploader slow" | Uploads are in-memory; limit to 50MB max per file |
| "Sources not showing correctly" | Check metadata is being set in user_upload_processor |
| "Image file type not accepted" | Ensure Streamlit file_uploader includes: ["pdf", "docx", "doc", "txt", "png", "jpg", "jpeg", "bmp", "tiff"] |

---

## Estimated Time

- **Phase 2 (file_extractor):** 1-2 hours
- **Phase 3 (processor):** 1 hour
- **Phase 4 (QueryComponent):** 1-2 hours
- **Phase 6 (Streamlit):** 1-2 hours
- **Testing:** 1 hour
- **Total: 5-8 hours for full working prototype**

---

## Next: Pick Your First Step

Ready to start?

1. Start with **Step 1.1** (add dependency)?
2. Or jump to **Step 2.1** (create file_extractor)?
3. Or have me code a specific step for you?

Let me know which step you'd like to tackle first, and I can help code it!
