# 🚀 FastAPI Detailed Guide - AI Lawyer Project

This document provides a comprehensive, line-by-line explanation of the FastAPI implementation in the AI Lawyer project, including all concepts, endpoints, data flows, and configurations.

---

## Table of Contents

1. [FastAPI Concepts](#fastapi-concepts)
2. [Project Structure](#project-structure)
3. [Application Initialization](#application-initialization)
4. [Middleware System](#middleware-system)
5. [API Routes & Endpoints](#api-routes--endpoints)
6. [Request/Response Models](#requestresponse-models)
7. [Exception Handling](#exception-handling)
8. [Dependency Injection](#dependency-injection)
9. [Running the Server](#running-the-server)

---

## FastAPI Concepts

### What is FastAPI?

FastAPI is a modern Python web framework for building APIs with:
- **Fast**: Very high performance (comparable to Node.js and Go)
- **Easy to Code**: 40% less code than Flask/Django
- **Fast to Code**: 2-3x faster development
- **Type Hints**: Built-in data validation using Python type hints
- **Automatic Documentation**: Auto-generated interactive API docs (Swagger UI)
- **Async Support**: Native async/await support for high concurrency

### Key Features Used in AI Lawyer

1. **Pydantic Models** - Data validation and serialization
2. **Dependency Injection** - Reusable dependencies and singleton patterns
3. **Middleware** - Request/response processing
4. **Error Handling** - Custom exception handling with proper HTTP status codes
5. **Async Endpoints** - Non-blocking I/O operations
6. **OpenAPI Integration** - Auto-generated API documentation

---

## Project Structure

```
src/AI_Lawyer/api/
├── __init__.py                 # Package initialization
├── app.py                      # Main FastAPI application factory
├── main.py                     # Entry point for running server
├── dependencies.py             # Dependency injection & service management
├── exceptions.py               # Custom exception classes
├── utils.py                    # Utility functions
├── models/
│   ├── __init__.py
│   ├── requests.py            # Pydantic request models (input validation)
│   └── responses.py           # Pydantic response models (output serialization)
└── routes/
    ├── __init__.py
    ├── health.py              # Health check endpoints
    ├── extraction.py          # File text extraction endpoints
    ├── ingestion.py           # Document ingestion endpoints
    └── query.py               # Query/RAG endpoints
```

---

## Application Initialization

### 1. Entry Point: `main.py`

**File:** `src/AI_Lawyer/api/main.py`

```python
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src to path - Makes modules importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
```

**Explanation:**
- Adds the `src` directory to Python's module search path
- Allows imports like `from AI_Lawyer.api.app import app`

```python
def load_environment():
    """Load environment variables from .env file."""
    env_file = Path(__file__).parent.parent.parent.parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
```

**Explanation:**
- Loads environment variables from `.env` file
- Used for configuration like `HOST`, `PORT`, `LOG_LEVEL`, etc.
- Must be called BEFORE importing the app to set up environment

```python
def main():
    """Run the FastAPI server with production-grade configuration."""
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    reload = os.getenv("RELOAD", "false").lower() == "true"
```

**Explanation:**
- `host = "0.0.0.0"`: Listen on all network interfaces (accessible from anywhere)
- `port = 8000`: Default port (can override with `PORT` env variable)
- `reload = True`: Auto-reload server on code changes (development only)

```python
uvicorn.run(
    app,
    host=host,
    port=port,
    reload=reload,
    log_level=log_level,
    access_log=log_level in ["debug", "info"],
)
```

**Explanation:**
- `uvicorn.run()`: Starts the ASGI server (Uvicorn handles async requests)
- **ASGI** = Asynchronous Server Gateway Interface (modern successor to WSGI)
- Supports concurrent requests via async/await

### 2. Application Factory: `app.py`

**File:** `src/AI_Lawyer/api/app.py`

```python
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZIPMiddleware
```

**Explanation:**
- **FastAPI**: Main framework class
- **Request**: Type for accessing HTTP request context
- **CORSMiddleware**: Handles Cross-Origin Resource Sharing (allows frontend to call API)
- **TrustedHostMiddleware**: Only allows requests from trusted domains
- **GZIPMiddleware**: Compresses responses to reduce bandwidth

```python
class RequestIDMiddleware:
    """Middleware to add request IDs for tracking."""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
```

**Explanation:**
- Custom middleware processes EVERY request/response
- `request.headers.get("X-Request-ID")`: Gets request ID from header if provided
- `str(uuid.uuid4())`: Generates unique ID if not provided
- `request.state.request_id`: Stores in request object for later use
- Used for logging and tracking requests through the system

```python
response = await call_next(request)
response.headers["X-Request-ID"] = request_id
return response
```

**Explanation:**
- `await call_next(request)`: Calls the next middleware/route handler (ASYNC)
- Adds request ID to response headers so client can track request
- Returns modified response

---

## Middleware System

Middleware runs in **layers** (like an onion):

```
Request → RequestIDMiddleware → LoggingMiddleware → CORSMiddleware 
          → TrustedHostMiddleware → GZIPMiddleware → Route Handler
                                                            ↓
Response ← RequestIDMiddleware ← LoggingMiddleware ← CORSMiddleware 
           ← TrustedHostMiddleware ← GZIPMiddleware ← Handler Response
```

### Middleware Stack Explanation

```python
def create_app():
    app = FastAPI(
        title="AI Lawyer API",
        description="Production-grade legal document analysis",
        version="1.0.0",
        docs_url="/docs",              # Swagger UI path
        redoc_url="/redoc",            # ReDoc path
        openapi_url="/openapi.json",   # OpenAPI schema path
        lifespan=lifespan_manager,     # Startup/shutdown events
    )
    
    # Middleware order matters - first added = inner layer
    app.add_middleware(RequestIDMiddleware)  # Added first = runs first
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(CORSMiddleware, ...)
```

**Middleware Order (Execution Flow):**

1. **RequestIDMiddleware** (first)
   - Adds request ID to track requests
   - Input: HTTP request
   - Output: Same request with `request.state.request_id` set

2. **LoggingMiddleware**
   - Logs all requests and responses
   - Input: Request from RequestIDMiddleware
   - Output: Logs + forwarded request

3. **CORSMiddleware**
   - Handles cross-origin requests
   - Input: Request from LoggingMiddleware
   - Output: Request with CORS headers

4. **TrustedHostMiddleware**
   - Validates host header (security)
   - Input: Request from CORSMiddleware
   - Output: Request or 400 error if untrusted

5. **GZIPMiddleware**
   - Compresses response body
   - Input: Response from handler
   - Output: Compressed response

6. **Route Handler** (innermost)
   - Actual endpoint logic

### CORS Configuration

```python
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,      # Which domains can call this API
    allow_credentials=True,             # Allow cookies/auth
    allow_methods=["*"],                # Allow all HTTP methods
    allow_headers=["*"],                # Allow all headers
    expose_headers=[                    # Headers visible to frontend
        "X-Request-ID",
        "X-Process-Time",
    ],
    max_age=3600,                       # Cache CORS check for 1 hour
)
```

**Explanation:**
- **CORS** = Cross-Origin Resource Sharing
- Without CORS, browser blocks requests from different domains
- `allow_origins="*"`: Allow requests from ANY domain (development only!)
- `allow_credentials=True`: Allow sending authentication cookies
- `expose_headers`: Frontend can read these custom headers

---

## API Routes & Endpoints

### Health Check Routes

**File:** `src/AI_Lawyer/api/routes/health.py`

#### Endpoint 1: GET `/health`
**Full Path:** `http://localhost:8000/health`

```python
@router.get("/", response_model=HealthResponse)
async def health_check():
    """
    Comprehensive health check endpoint.
    
    INPUT: None (GET request)
    
    OUTPUT: HealthResponse with:
    - success: bool (True if healthy)
    - status: str (healthy/degraded/unhealthy)
    - uptime_seconds: float
    - components: List of component statuses
    """
```

**What it does:**
1. Checks if API is responding ✓
2. Checks if configuration is loaded ✓
3. Checks if embedding model is available ✓
4. Checks if vector store (FAISS) is connected ✓
5. Checks if query engine is ready ✓
6. Checks if extraction engine is ready ✓

**Response Example:**
```json
{
  "success": true,
  "status": "healthy",
  "message": "System is healthy",
  "uptime_seconds": 3600.5,
  "version": "1.0.0",
  "components": [
    {
      "name": "api",
      "status": "ok",
      "message": "API is responding"
    },
    {
      "name": "vector_store",
      "status": "ok",
      "message": "Vector store connected",
      "response_time_ms": 45.2
    }
  ],
  "database_connected": true,
  "vector_store_available": true,
  "llm_available": true,
  "timestamp": "2024-12-24T10:30:00"
}
```

#### Endpoint 2: GET `/health/ready` (Kubernetes Readiness)

```python
@router.get("/ready", response_model=dict)
async def readiness_check():
    """
    Kubernetes-style readiness check.
    
    Returns 200 if service is ready to accept traffic.
    Returns 503 if it's still initializing or degraded.
    
    INPUT: None
    OUTPUT: {"ready": True/False}
    """
```

**Purpose:** Used by Kubernetes to determine if pod should receive traffic

---

### File Extraction Routes

**File:** `src/AI_Lawyer/api/routes/extraction.py`

#### Endpoint 1: POST `/extraction/extract`

**Input:** Multipart form data with file uploads

```python
@router.post("/extract", response_model=ExtractionResponse)
async def extract_files(
    files: List[UploadFile] = File(..., description="Files to extract text from")
):
    """
    Extract text from uploaded files.
    
    INPUT:
    - files: List of uploaded files (required)
      Supported formats: PDF, DOCX, TXT, PNG, JPG, JPEG, BMP, TIFF
    
    PROCESS:
    1. Receive uploaded files
    2. Call FileExtractionComponent
    3. Extract text from each file
    4. Return results mapped by filename
    
    OUTPUT: ExtractionResponse
    """
```

**Step-by-Step Execution:**

```python
component = get_extraction_component()  # Initialize extractor
results = component.extract_from_uploads(files)  # Extract from each file
```

**Response Structure:**
```json
{
  "success": true,
  "message": "Extracted from 2 files",
  "data": {
    "document.pdf": "This is extracted text from PDF...",
    "report.docx": "This is extracted text from DOCX..."
  },
  "errors": {},
  "file_count": 2,
  "successful_count": 2,
  "failed_count": 0,
  "processing_time_seconds": 2.345,
  "timestamp": "2024-12-24T10:30:00"
}
```

#### Endpoint 2: POST `/extraction/extract-path`

**Input:** JSON body with file paths

```python
@router.post("/extract-path", response_model=ExtractionResponse)
async def extract_from_path(request: ExtractionRequest):
    """
    Extract text from file(s) at specified path(s).
    
    INPUT (ExtractionRequest):
    - file_path: str (optional) - Single file path
    - file_paths: List[str] (optional) - Multiple file paths
    - directory_path: str (optional) - Directory to extract all files from
    
    EXAMPLE REQUEST:
    {
      "file_path": "/path/to/document.pdf",
      "extract_images": false,
      "preserve_formatting": true
    }
    
    OUTPUT: ExtractionResponse (same as /extract endpoint)
    """
```

**Three Modes:**

**Mode 1 - Single File:**
```python
if request.file_path:
    text = component.extract(request.file_path)
    return ExtractionResponse(data={request.file_path: text}, file_count=1)
```

**Mode 2 - Multiple Files:**
```python
elif request.file_paths:
    results = component.extract_multiple(request.file_paths)
    # Returns dict: {filename: extracted_text}
```

**Mode 3 - Directory:**
```python
elif request.directory_path:
    results = component.extract_directory(request.directory_path)
    # Extracts all supported files in directory
```

#### Endpoint 3: GET `/extraction/supported-formats`

```python
@router.get("/supported-formats", response_model=dict)
async def get_supported_formats():
    """
    Get list of supported file formats.
    
    INPUT: None
    
    OUTPUT:
    {
      "supported_formats": ["pdf", "docx", "txt", "png", "jpg"],
      "ocr_enabled": true,
      "total_formats": 5
    }
    """
```

**Purpose:** Client can check what formats are supported before uploading

---

### Data Ingestion Routes

**File:** `src/AI_Lawyer/api/routes/ingestion.py`

#### Endpoint 1: POST `/ingestion/documents`

**Purpose:** Ingest documents into vector store for RAG

```python
@router.post("/documents", response_model=IngestionResponse)
async def ingest_documents(request: DataIngestionRequest):
    """
    Ingest documents into the vector store.
    
    INPUT (DataIngestionRequest):
    - documents: List[str] (required) - Document texts to ingest
    - collection_name: str - Category name (default: "default")
    - chunk_size: int - Size of text chunks (default: 512)
    - chunk_overlap: int - Overlap between chunks (default: 128)
    - metadata: Dict - Additional metadata
    - reindex: bool - Reindex existing documents
    
    EXAMPLE REQUEST:
    {
      "documents": [
        "This is document 1 about contracts...",
        "This is document 2 about torts..."
      ],
      "collection_name": "legal_precedents",
      "chunk_size": 512,
      "chunk_overlap": 128,
      "metadata": {"source": "court_database"}
    }
    
    PROCESS:
    1. Validate documents (must have at least 1)
    2. Split documents into chunks
    3. Generate embeddings for each chunk
    4. Store in FAISS vector database
    5. Save metadata
    
    OUTPUT: IngestionResponse
    """
```

**Chunking Explanation:**
```
Original Document (2000 tokens):
"The quick brown fox jumps over the lazy dog. The dog was sleeping..."

With chunk_size=512, chunk_overlap=128:

Chunk 1 (512 tokens): "The quick brown fox... [512 tokens]"
                 ↓ overlap 128 tokens ↓
Chunk 2 (512 tokens): "[Last 128 from Chunk1]... [next 384 new tokens]"
                 ↓ overlap 128 tokens ↓
Chunk 3 (512 tokens): "[Last 128 from Chunk2]... [remaining tokens]"
```

**Why Overlap?**
- Ensures semantic continuity
- Prevents important information from being split at chunk boundaries

**Response Structure:**
```json
{
  "success": true,
  "message": "Successfully ingested 2 documents",
  "document_count": 2,
  "chunk_count": 8,
  "collection_name": "legal_precedents",
  "index_size": 4096,
  "processing_time_seconds": 3.456,
  "timestamp": "2024-12-24T10:30:00"
}
```

#### Endpoint 2: POST `/ingestion/batch`

```python
@router.post("/batch", response_model=List[IngestionResponse])
async def batch_ingest(request: BulkIngestionRequest):
    """
    Perform bulk ingestion of multiple batches.
    
    INPUT (BulkIngestionRequest):
    - ingestion_requests: List[DataIngestionRequest]
    
    EXAMPLE:
    {
      "ingestion_requests": [
        {
          "documents": ["Doc1..."],
          "collection_name": "collection1"
        },
        {
          "documents": ["Doc2..."],
          "collection_name": "collection2"
        }
      ]
    }
    
    OUTPUT: List[IngestionResponse] (one for each batch)
    
    FEATURE: Continues processing even if one batch fails
    """
```

**Error Handling:**
```python
for idx, ingestion_req in enumerate(request.ingestion_requests):
    try:
        response = await ingest_documents(ingestion_req)
        results.append(response)
    except Exception as e:
        results.append(  # Add error response, continue processing
            IngestionResponse(success=False, message=f"Batch {idx} failed")
        )
```

#### Endpoint 3: GET `/ingestion/collections`

```python
@router.get("/collections", tags=["ingestion"])
async def list_collections():
    """
    List all document collections in vector store.
    
    INPUT: None
    
    OUTPUT:
    {
      "success": true,
      "collections": [
        {
          "name": "legal_precedents",
          "document_count": 45,
          "chunk_count": 234,
          "created_at": "2024-01-01T00:00:00",
          "size_mb": 12.5
        }
      ],
      "total_count": 1
    }
    """
```

#### Endpoint 4: DELETE `/ingestion/collections/{collection_name}`

```python
@router.delete("/collections/{collection_name}")
async def delete_collection(collection_name: str):
    """
    Delete a document collection.
    
    INPUT:
    - collection_name: str (path parameter) - Name of collection to delete
    
    EXAMPLE: DELETE /ingestion/collections/legal_precedents
    
    OUTPUT:
    {
      "success": true,
      "message": "Collection 'legal_precedents' deleted"
    }
    """
```

---

### Query/RAG Routes

**File:** `src/AI_Lawyer/api/routes/query.py`

#### Endpoint 1: POST `/query/ask`

**Purpose:** Ask questions about legal documents using RAG

```python
@router.post("/ask", response_model=QueryResponse)
async def ask_query(request: QueryRequest):
    """
    Submit a standard query to the RAG system (permanent legal DB only).
    
    INPUT (QueryRequest):
    - query: str (required) - User question/query
    - mode: QueryModeEnum - Query mode (standard/hybrid/semantic)
    - top_k: int - Number of results to return (default: 5, max: 50)
    - score_threshold: float - Min similarity score (0-1)
    - use_reranker: bool - Use reranking for results
    
    EXAMPLE REQUEST:
    {
      "query": "What are the requirements for a valid contract?",
      "top_k": 5,
      "score_threshold": 0.7,
      "mode": "standard"
    }
    
    PROCESS:
    1. Take user question: "What are the requirements?"
    2. Convert to embedding (vector)
    3. Search FAISS for similar documents
    4. Retrieve top 5 results
    5. Pass results + query to LLM
    6. LLM generates answer
    
    OUTPUT: QueryResponse
    """
```

**Step-by-Step Technical Flow:**

```
User Query: "What are contract requirements?"
    ↓
[Embedding Model] converts to vector: [0.45, -0.23, 0.87, ...]
    ↓
[FAISS] searches vector database
    ↓
Returns top 5 similar documents:
  - "A contract requires offer, acceptance, consideration..." (score: 0.92)
  - "Valid contracts must have legal purpose..." (score: 0.87)
  - ...
    ↓
[LLM (Groq)] receives:
  Query: "What are contract requirements?"
  Context: [Retrieved documents above]
    ↓
LLM generates answer:
"Based on legal precedents, a contract requires:
1. Offer - Clear proposal
2. Acceptance - Agreement to terms
3. Consideration - Exchange of value
..."
```

**Response Structure:**
```json
{
  "success": true,
  "query": "What are contract requirements?",
  "answer": "Based on legal precedents, a contract requires...",
  "results": [
    {
      "text": "A contract requires offer, acceptance, consideration...",
      "source": "contract_law_db",
      "score": 0.92,
      "rank": 1,
      "source_type": "legal_db",
      "page_number": 45,
      "metadata": {"chapter": "Formation of Contracts"}
    },
    {
      "text": "Valid contracts must have legal purpose...",
      "source": "contract_law_db",
      "score": 0.87,
      "rank": 2,
      "source_type": "legal_db",
      "page_number": 50,
      "metadata": {}
    }
  ],
  "result_count": 2,
  "processing_time_seconds": 2.345,
  "embedding_time_seconds": 0.234,
  "retrieval_time_seconds": 0.156,
  "generation_time_seconds": 1.955,
  "timestamp": "2024-12-24T10:30:00"
}
```

#### Endpoint 2: POST `/query/hybrid`

**Purpose:** Query combining permanent legal DB + user uploads

```python
@router.post("/hybrid")
async def hybrid_query(
    query: str = Form(...),
    files: List[UploadFile] = File(default=[]),
    top_k: int = Form(5)
):
    """
    Submit a hybrid query with optional file uploads.
    
    INPUT (Multipart Form Data):
    - query: str (required) - User question
    - files: List[UploadFile] (optional) - Files to search
    - top_k: int - Results per search
    
    EXAMPLE REQUEST:
    POST /query/hybrid
    Content-Type: multipart/form-data
    
    query: "What are liability limitations?"
    files: [user_contract.pdf]
    top_k: 5
    
    PROCESS:
    1. Upload files: user_contract.pdf
    2. Extract text from uploaded files
    3. Chunk text into searchable pieces
    4. Generate embeddings
    5. Search PERMANENT legal DB
    6. Search USER UPLOADS
    7. Combine and rank results
    8. Generate answer from all sources
    
    EXAMPLE RESULTS:
    Permanent DB: "Liability is limited by..." (score: 0.95)
    User Upload: "Our liability cap is $1M" (score: 0.89)
    
    OUTPUT: HybridQueryResponse
    """
```

**Hybrid Search Architecture:**

```
User Query: "What are liability limitations?"
├── Search 1: Permanent Legal DB
│   └── FAISS Search → Results
│
├── Search 2: User Uploads
│   ├── Extract text from user_contract.pdf
│   ├── Chunk text
│   ├── Generate embeddings
│   └── FAISS Search → Results
│
└── Combine Results
    ├── Rank by relevance score
    ├── Remove duplicates
    └── Return top results
```

**File Processing:**
```python
for uploaded_file in files:
    content = await uploaded_file.read()  # Read bytes from upload
    
    with tempfile.NamedTemporaryFile(...) as temp_file:
        temp_file.write(content)  # Save temporarily
        
        extracted_text = file_extractor.extract_from_file(temp_path)
        # Extract text from PDF/DOCX/etc
        
        os.remove(temp_path)  # Clean up temp file
```

**Response:**
```json
{
  "success": true,
  "query": "What are liability limitations?",
  "answer": "Liability limitations are...",
  "results": [...],
  "permanent_db_results": 3,
  "user_upload_results": 2,
  "processing_time_seconds": 4.567,
  "timestamp": "2024-12-24T10:30:00"
}
```

---

## Request/Response Models

### What are Pydantic Models?

Pydantic models are Python classes that:
1. **Validate** incoming data automatically
2. **Serialize** outgoing data to JSON
3. **Generate** OpenAPI documentation
4. **Provide** type hints for IDE support

### Example: QueryRequest

**File:** `src/AI_Lawyer/api/models/requests.py`

```python
from pydantic import BaseModel, Field, validator

class QueryRequest(BaseRequest):
    """Request model for query/RAG endpoint."""
    
    query: str = Field(
        ...,                        # ... means required
        min_length=1,               # Validate: at least 1 character
        max_length=5000,            # Validate: max 5000 characters
        description="User query/question"
    )
    
    mode: QueryModeEnum = Field(
        QueryModeEnum.STANDARD,     # Default value
        description="Query processing mode"
    )
    
    top_k: int = Field(
        5,                          # Default: 5
        ge=1,                       # Validate: >= 1
        le=50,                      # Validate: <= 50
        description="Number of top results to return"
    )
    
    score_threshold: float = Field(
        0.0,
        ge=0.0,                     # >= 0.0
        le=1.0,                     # <= 1.0
        description="Minimum similarity score threshold"
    )
```

### Validation Flow

**INVALID REQUEST:**
```json
{
  "query": "",                    // ❌ min_length=1
  "top_k": 1000,                  // ❌ le=50
  "score_threshold": 1.5          // ❌ le=1.0
}
```

**FastAPI Auto-validates and returns 422 error:**
```json
{
  "detail": [
    {
      "loc": ["body", "query"],
      "msg": "ensure this value has at least 1 character",
      "type": "value_error.string.too_short"
    },
    {
      "loc": ["body", "top_k"],
      "msg": "ensure this value is less than or equal to 50",
      "type": "value_error.number.not_le"
    }
  ]
}
```

**VALID REQUEST:**
```json
{
  "query": "What are contract requirements?",
  "top_k": 5,
  "score_threshold": 0.7,
  "mode": "standard"
}
```

### Response Models

```python
class QueryResponse(BaseResponse):
    """Response model for query endpoint."""
    
    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer")
    results: List[QueryResult] = Field(
        default_factory=list,
        description="Retrieved context chunks"
    )
    result_count: int = Field(
        ...,
        description="Number of results returned"
    )
    processing_time_seconds: float = Field(
        ...,
        description="Total processing time"
    )
    embedding_time_seconds: float = Field(
        0.0,
        description="Time for embedding step"
    )
    retrieval_time_seconds: float = Field(
        0.0,
        description="Time for retrieval step"
    )
    generation_time_seconds: float = Field(
        0.0,
        description="Time for LLM generation"
    )
```

### Nested Models

```python
class QueryResult(BaseResponse):
    """Single result item in query response."""
    
    text: str                       # Retrieved text snippet
    source: str                     # Source document name
    score: float                    # Relevance score (0-1)
    rank: int = 0                   # Position in results
    source_type: str = "legal_db"   # legal_db, user_upload, cached
    metadata: Dict[str, Any] = {}   # Extra info (page, chapter, etc)

# Response contains list of results
class QueryResponse(BaseResponse):
    results: List[QueryResult]      # ← List of QueryResult objects
```

**JSON Output Structure:**
```json
{
  "results": [
    {
      "text": "...",
      "source": "contract_law.pdf",
      "score": 0.92,
      "rank": 1,
      "source_type": "legal_db",
      "metadata": {"page": 45}
    }
  ]
}
```

---

## Exception Handling

### Custom Exception System

**File:** `src/AI_Lawyer/api/exceptions.py`

**Why Custom Exceptions?**
- Standard FastAPI HTTPException returns plain text errors
- Custom exceptions return structured JSON errors
- Better for API consumers to understand and handle errors

```python
class ErrorCode(str, Enum):
    """Standard error codes."""
    VALIDATION_ERROR = "VALIDATION_ERROR"      # 400
    INVALID_INPUT = "INVALID_INPUT"            # 400
    FILE_NOT_FOUND = "FILE_NOT_FOUND"          # 404
    AUTHENTICATION_FAILED = "AUTH_FAILED"      # 401
    PERMISSION_DENIED = "PERMISSION_DENIED"    # 403
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED" # 429
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE" # 503
    INTERNAL_SERVER_ERROR = "INTERNAL_ERROR"   # 500
```

### Custom Exception Class

```python
class APIException(HTTPException):
    """Base custom exception for API errors."""
    
    def __init__(
        self,
        status_code: int,           # HTTP status code
        error_code: ErrorCode,      # Enum error code
        message: str,               # Human-readable message
        details: Optional[Dict] = None,
        headers: Optional[Dict] = None,
    ):
        """
        Initialize exception with structured error response.
        """
        content = {
            "success": False,
            "error_code": error_code.value,
            "message": message,
            "details": details or {}
        }
        
        super().__init__(
            status_code=status_code,
            detail=content,     # ← Returned as JSON
            headers=headers
        )
```

### Specific Exception Subclasses

```python
class ValidationError(APIException):
    """Raised when request validation fails."""
    
    def __init__(self, message: str, details: Dict = None):
        super().__init__(
            status_code=400,                    # HTTP 400 Bad Request
            error_code=ErrorCode.VALIDATION_ERROR,
            message=message,
            details=details
        )

class InvalidInputError(APIException):
    """Raised when input is invalid."""
    
    def __init__(self, message: str, field: str = None):
        super().__init__(
            status_code=400,
            error_code=ErrorCode.INVALID_INPUT,
            message=message,
            details={"field": field}
        )

class FileNotFoundError(APIException):
    """Raised when file cannot be found."""
    
    def __init__(self, filepath: str):
        super().__init__(
            status_code=404,                    # HTTP 404 Not Found
            error_code=ErrorCode.FILE_NOT_FOUND,
            message=f"File not found: {filepath}"
        )
```

### Error Response Examples

**Invalid Input:**
```json
HTTP 400 Bad Request
{
  "success": false,
  "error_code": "VALIDATION_ERROR",
  "message": "No documents provided",
  "details": {
    "documents": "At least one document is required"
  }
}
```

**File Not Found:**
```json
HTTP 404 Not Found
{
  "success": false,
  "error_code": "FILE_NOT_FOUND",
  "message": "File not found: /path/to/document.pdf"
}
```

**Service Error:**
```json
HTTP 503 Service Unavailable
{
  "success": false,
  "error_code": "SERVICE_UNAVAILABLE",
  "message": "Vector store is temporarily unavailable",
  "details": {
    "service": "FAISS",
    "retry_after": 30
  }
}
```

### Using Exceptions in Routes

```python
@router.post("/extract")
async def extract_files(files: List[UploadFile] = File(...)):
    try:
        if not files:
            raise ValidationError(
                "No files provided",
                {"files": "At least one file is required"}
            )
        
        # Extract files...
        
    except FileNotFoundError as e:
        # ← Caught, converted to HTTP response automatically
        raise  # Re-raise, FastAPI handles conversion
    except Exception as e:
        raise APIException(
            status_code=500,
            error_code=ErrorCode.INTERNAL_SERVER_ERROR,
            message="Extraction failed",
            details={"error_type": type(e).__name__}
        )
```

---

## Dependency Injection

**What is Dependency Injection?**
- Design pattern where dependencies are "injected" into functions
- Avoids creating dependencies inside functions
- Makes testing easier (can inject mock objects)
- Promotes reusable, testable code

### Service Manager (Singleton Pattern)

```python
class ServiceManager:
    """
    Manages all service instances as singletons.
    
    Singleton Pattern: Only one instance of ServiceManager exists
    for the entire application lifetime.
    """
    
    _instance: Optional['ServiceManager'] = None
    
    def __init__(self):
        """Initialize (called only once)."""
        if ServiceManager._instance is not None:
            raise RuntimeError("ServiceManager is singleton")
        
        self.config_manager = None
        self.query_component = None
        self.embedding_model = None
        self.faiss_db = None
    
    @classmethod
    def get_instance(cls) -> 'ServiceManager':
        """
        Get or create the singleton instance.
        
        First call: Creates instance
        Subsequent calls: Returns same instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
```

### Lazy Initialization

```python
def initialize_embedding_model(self):
    """Lazy initialize - only load when first needed."""
    
    if self.embedding_model is None:  # Check if already loaded
        try:
            from sentence_transformers import SentenceTransformer
            
            # Only load if first time
            config = self.initialize_config()
            embedding_config = config.get_embeddings_config()
            
            self.embedding_model = SentenceTransformer(
                embedding_config.model  # e.g., "all-MiniLM-L6-v2"
            )
            logger.info(f"✓ Embedding model loaded")
        except Exception as e:
            logger.error(f"✗ Failed to initialize: {e}")
            raise
    
    return self.embedding_model  # Return (new or cached)
```

**Why Lazy Initialization?**
- Embedding models are large (100+ MB)
- Only load if actually used
- Faster startup time
- Saves memory if not needed

### Dependency Functions for FastAPI

```python
async def get_config_manager() -> ConfigurationManager:
    """
    FastAPI dependency for ConfigurationManager.
    
    @router.post("/ingestion/documents")
    async def ingest_documents(
        config = Depends(get_config_manager)  # ← Injected here
    ):
        # config is ConfigurationManager instance
    """
    service_manager = ServiceManager.get_instance()
    return service_manager.initialize_config()

async def get_query_component():
    """FastAPI dependency for QueryComponent."""
    service_manager = ServiceManager.get_instance()
    return service_manager.initialize_query_component()
```

### FastAPI Depends System

```python
from fastapi import Depends

@router.post("/ingest")
async def ingest_documents(
    request: DataIngestionRequest,
    config: ConfigurationManager = Depends(get_config_manager)  # ← Injected
):
    """
    FastAPI:
    1. Sees Depends(get_config_manager)
    2. Calls get_config_manager()
    3. Gets ConfigurationManager instance
    4. Passes to this function
    """
    # Use config
    embeddings_config = config.get_embeddings_config()
```

**Benefits:**
- ✅ Single instance (singleton) across requests
- ✅ Easy to test (inject mock config)
- ✅ Automatic initialization
- ✅ Lazy loading (only when needed)

### Lifespan Manager (Startup/Shutdown)

```python
@asynccontextmanager
async def lifespan_manager(app: FastAPI):
    """
    Async context manager for app lifecycle.
    
    Runs BEFORE first request (startup).
    Runs AFTER last request (shutdown).
    """
    # ===== STARTUP =====
    logger.info("Starting up...")
    
    # Initialize critical components
    try:
        service_manager = ServiceManager.get_instance()
        service_manager.initialize_config()
        service_manager.initialize_embedding_model()
        service_manager.initialize_vector_store()
        logger.info("✓ All components initialized")
    except Exception as e:
        logger.error(f"✗ Startup failed: {e}")
        raise
    
    yield  # ← App runs here, serving requests
    
    # ===== SHUTDOWN =====
    logger.info("Shutting down...")
    
    try:
        await service_manager.shutdown()
        logger.info("✓ Cleanup complete")
    except Exception as e:
        logger.error(f"✗ Shutdown error: {e}")

# Pass to FastAPI
app = FastAPI(lifespan=lifespan_manager)
```

**Lifecycle Timeline:**
```
Server Start
    ↓
lifespan_manager - STARTUP section
    ↓
Initialize config, embedding model, vector store
    ↓
yield - App starts serving requests
    ↓
[Requests processed]
    ↓
KeyboardInterrupt or shutdown signal
    ↓
Resume after yield - SHUTDOWN section
    ↓
Clean up resources
    ↓
Server stops
```

---

## Running the Server

### Method 1: Direct Python

```bash
# From project root
python api_server.py
```

**What happens:**
1. Loads `.env` file
2. Gets config from env variables
3. Imports FastAPI app
4. Starts Uvicorn server
5. Listens on `http://localhost:8000`

### Method 2: Uvicorn Direct

```bash
uvicorn src.AI_Lawyer.api.app:app --reload --host 0.0.0.0 --port 8000
```

**Parameters:**
- `src.AI_Lawyer.api.app:app` - Module path : app instance
- `--reload` - Auto-reload on code changes
- `--host 0.0.0.0` - Listen on all interfaces
- `--port 8000` - Port number

### Method 3: Python Module

```bash
python -m AI_Lawyer.api.main
```

### Method 4: Docker

```bash
docker-compose up
```

**Configuration from Environment Variables:**

```bash
# .env file
HOST=0.0.0.0
PORT=8000
RELOAD=true
LOG_LEVEL=info
ENVIRONMENT=development
ALLOWED_ORIGINS=*
TRUSTED_HOSTS=*
```

### Accessing the API

| Feature | URL |
|---------|-----|
| **Swagger UI** | `http://localhost:8000/docs` |
| **ReDoc** | `http://localhost:8000/redoc` |
| **OpenAPI JSON** | `http://localhost:8000/openapi.json` |
| **Health Check** | `http://localhost:8000/health` |
| **Query Endpoint** | `http://localhost:8000/query/ask` |
| **Extract Endpoint** | `http://localhost:8000/extraction/extract` |

### Testing an Endpoint with cURL

```bash
# Health check (no auth)
curl http://localhost:8000/health

# Query endpoint
curl -X POST http://localhost:8000/query/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are contract requirements?",
    "top_k": 5
  }'

# File extraction
curl -X POST http://localhost:8000/extraction/extract \
  -F "files=@document.pdf" \
  -F "files=@report.docx"
```

### Testing with Python Requests

```python
import requests

# Query endpoint
response = requests.post(
    "http://localhost:8000/query/ask",
    json={
        "query": "What are contract requirements?",
        "top_k": 5
    }
)

print(response.json())
```

---

## Production Deployment

### Best Practices

1. **Don't use `reload=True`** - Disables auto-reload in production

```bash
# Development
uvicorn src.AI_Lawyer.api.app:app --reload

# Production
uvicorn src.AI_Lawyer.api.app:app --workers 4
```

2. **Use environment variables** for sensitive config

```bash
# .env (never commit to git!)
GROQ_API_KEY=gsk_xxxxxxxxxxxxx
DATABASE_URL=postgresql://user:pass@host/db
```

3. **Enable HTTPS**

```python
# In app.py
app.add_middleware(
    HTTPSRedirectMiddleware  # Forces HTTPS
)
```

4. **Set proper CORS**

```python
# Production - specific origins only
allowed_origins = [
    "https://myapp.com",
    "https://api.myapp.com",
]
```

5. **Enable compression**

```python
app.add_middleware(GZIPMiddleware, minimum_size=1000)
```

6. **Use production ASGI server**

```bash
# Gunicorn with Uvicorn workers
gunicorn src.AI_Lawyer.api.app:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

---

## Summary

### FastAPI Core Concepts in AI Lawyer

1. **Request Models (Pydantic)** - Validate and document inputs
2. **Response Models (Pydantic)** - Serialize and document outputs
3. **Routes** - Define endpoints that handle requests
4. **Middleware** - Process requests/responses globally
5. **Exceptions** - Return structured error responses
6. **Dependencies** - Inject components with Depends()
7. **Lifespan** - Startup/shutdown logic
8. **Async** - Non-blocking I/O for concurrency

### Request Lifecycle

```
HTTP Request
    ↓
Middleware 1 (RequestIDMiddleware)
    ↓
Middleware 2 (LoggingMiddleware)
    ↓
Middleware 3 (CORSMiddleware)
    ↓
Route Handler
    ↓ (dependency injection)
    ServiceManager.get_instance() → initialize components
    ↓
Process request (extract, ingest, query)
    ↓
Return Response Model (serialized to JSON)
    ↓
Middleware (reverse order)
    ↓
HTTP Response
```

### Key Files Reference

| File | Purpose |
|------|---------|
| `main.py` | Entry point, loads env, starts server |
| `app.py` | Application factory, middleware setup |
| `dependencies.py` | Service injection, singleton manager |
| `exceptions.py` | Custom exception classes |
| `models/requests.py` | Request validation models |
| `models/responses.py` | Response serialization models |
| `routes/health.py` | Health check endpoints |
| `routes/extraction.py` | File extraction endpoints |
| `routes/ingestion.py` | Document ingestion endpoints |
| `routes/query.py` | RAG query endpoints |

---

This comprehensive guide covers every aspect of the FastAPI implementation in the AI Lawyer project. Each endpoint, middleware, and concept is explained with examples and diagrams.
