AI_Lawyer — API & Testing README
=================================

This document explains how to run, test, and integrate the FastAPI service for the AI_Lawyer project, plus testing guidance for the codebase and API endpoints.

Goal
----
Provide a runnable API and test plan that exposes the project's functionality (FAISS vector search, temporary file uploads with OCR, embedding, and query) safely and reliably.

Contents
--------
- Quick start (install + run)
- API surface (endpoints + schemas)
- Running the API (development & production)
- Testing (unit, integration, API tests)
- CI suggestions (GitHub Actions)
- Docker & deployment notes
- Observability & monitoring
- Security & secrets
- Debugging tips

Quick start
-----------
1. Set up Python environment (recommended: venv)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Ensure system dependency for OCR (optional, for image uploads):

```bash
# Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y tesseract-ocr
# Mac
brew install tesseract
```

3. Build embeddings and FAISS index (if not already done):

```bash
python main.py
```

4. Run Streamlit UI (optional):

```bash
streamlit run app.py
```

5. (Optional) Run the API server locally (dev):

```bash
# If you scaffold api/main.py with FastAPI app
uvicorn api.main:app --reload --port 8000
```

API Surface
-----------
Below are recommended endpoints to implement based on the hybrid temporary-context approach. These are expected endpoints (names and inputs can be adapted to your implementation):

- GET /health
  - Purpose: Liveness/readiness
  - Response: { "status": "ok" }

- GET /status
  - Purpose: Index stats (num vectors, last_updated)
  - Response: { "num_vectors": int, "last_updated": str }

- POST /query
  - Purpose: Ask a question against permanent FAISS index
  - Input JSON: { "question": str, "top_k": int (optional, default 5) }
  - Response JSON: { "answer": str, "sources": [ {metadata} ] }

- POST /query-with-files (multipart)
  - Purpose: Single-shot query that includes uploaded files (PDF/DOCX/TXT/Images)
  - Inputs: Form field `question`, Files `files[]` (multipart), optional `top_k`
  - Behaviour: Extract text from uploaded files, chunk, embed temporarily, search permanent FAISS and temporary index, merge results, call LLM for final answer.
  - Response JSON: { "answer": str, "sources": [{"source_type":"legal_db|user_upload","source_name":"...","score":float,"text": "..."}], "total_chunks_searched": int }

- POST /embed
  - Purpose: Batch embeddings for testing
  - Input JSON: { "texts": ["...", "..."] }
  - Response JSON: embeddings array (list of float vectors)

- POST /upload (two-step option)
  - Purpose: Upload files and get a batch_id to reference later
  - Response JSON: { "batch_id": str, "files_processed": int, "chunks_created": int }

- POST /rebuild (admin)
  - Purpose: Trigger FAISS rebuild (long-running – should be background job)
  - Response: { "task_id": str }

- GET /tasks/{task_id}
  - Purpose: Check status of background tasks (rebuild, ingestion)

- GET /metrics
  - Purpose: Prometheus scrape endpoint (if instrumented)

Pydantic models (example)
-------------------------
Create a single source of truth for request/response schemas under `api/models.py`.

Example models:

```python
from pydantic import BaseModel
from typing import List

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

class SourceInfo(BaseModel):
    source_type: str
    source_name: str
    score: float
    text: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]
    total_chunks_searched: int
```

Running the API
---------------
Development
```
uvicorn api.main:app --reload --port 8000
```
Production (simple)
```
gunicorn -k uvicorn.workers.UvicornWorker -w 4 api.main:app
```
Containerized (recommended in prod)
Use the Dockerfile and `docker-compose.yml` (see Docker section).

Concurrency notes
-----------------
- Embedding and FAISS operations are CPU-bound and/or blocking. Avoid running them directly on the FastAPI event loop.
- Use `asyncio.get_running_loop().run_in_executor(None, blocking_fn, *args)` to offload blocking operations to the thread pool, or run endpoints as synchronous functions (FastAPI runs them in a threadpool).
- For long running tasks (rebuilds, large uploads), use background workers (Celery/RQ) and return a task id.

Testing Strategy
----------------
Testing should cover unit tests for components and integration tests for endpoints.

1. Unit tests (fast, no external calls)
- Test `FileExtractor` methods with sample files placed in `tests/data/`.
- Test `UserUploadProcessor` chunking logic for consistency.
- Test `local_embedding` wrapper by generating embeddings for small texts (mocking heavy models when needed).
- Use dependency injection to replace `QueryComponent.llm` with a mock LLM that returns fixed text.

2. Integration tests (API-level)
- Use FastAPI `TestClient` or `httpx.AsyncClient` to test routers.
- Mock the FAISS and embedding model or use a small in-memory FAISS instance with a few documents.
- Test `/query-with-files` by uploading small PDF/DOCX/TXT/PNG files included under `tests/data/` and assert expected structure in response.

3. End-to-end tests (optional, slower)
- Use real embeddings & FAISS index; run in CI with a dedicated runner and sample dataset.

Example pytest structure
```
tests/
  unit/
    test_file_extractor.py
    test_user_upload_processor.py
    test_local_embedding.py
  integration/
    test_api_query.py
    test_api_query_with_files.py
```

Example unit test (pytest)
```python
from AI_Lawyer.components.file_extractor import FileExtractor

def test_extract_txt(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("Hello world")
    ext = FileExtractor()
    assert "Hello" in ext.extract_txt(str(p))
```

Mocking heavy dependencies
--------------------------
- Replace `app.state.faiss_db` with a small FAISS index or a mocked object that implements `similarity_search`.
- Replace the LLM client (`ChatGroq`) with a dummy class whose `__call__` returns a fixed response or a simple echo.
- Use `pytest` fixtures to inject these mocks into the FastAPI `app` during tests.

Running tests locally
---------------------
Install test dependencies (pytest, httpx):
```
pip install pytest pytest-asyncio httpx
```
Run tests:
```
pytest -q
```

API test example with httpx
```python
import pytest
from httpx import AsyncClient
from api.main import app

@pytest.mark.asyncio
async def test_health():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        r = await ac.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
```

CI Integration (GitHub Actions)
------------------------------
Create `.github/workflows/ci.yml` with steps:
- Install Python
- pip install -r requirements.txt
- Run linter (ruff/flake8)
- Run tests
- Optionally build Docker image

Example snippet
```yaml
name: CI
on: [push, pull_request]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest -q
```

Docker & Deployment
-------------------
Dockerfile (example)
```
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "2", "api.main:app", "--bind", "0.0.0.0:8000"]
```

docker-compose.yml (example)
```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - CONFIG_PATH=/app/config/config.yaml
    volumes:
      - ./models/vector_store:/app/models/vector_store
  redis:
    image: redis:7
    ports:
      - "6379:6379"
```

Observability & Monitoring
--------------------------
- Structured logging: configure `logging` to include timestamps and request IDs
- Metrics: instrument key operations (embedding time, FAISS search time, LLM latency) using `prometheus_client` and expose `/metrics`
- Tracing: optional OpenTelemetry integration
- Error reporting: Sentry for production exception aggregation

Security & Secrets
------------------
- Do not commit `config/secret.yaml` to git. Use environment variables or a secret manager in production.
- Provide `check_groq_key.py` to help verify the Groq API key.
- Protect `/rebuild`, `/upload` and admin endpoints with API key or OAuth2 scopes.
- Add rate limiting (e.g. `fastapi-limiter` with Redis backend) on `/query` endpoints to avoid abuse.

Debugging Tips
--------------
- If FAISS load fails with pydantic errors, rebuild the index with `rebuild_faiss.py`.
- If embeddings mismatch dims: confirm model in `config/config.yaml` matches the embeddings used to build FAISS (all-MiniLM-L6-v2 -> 384 dims).
- For OCR issues: verify `tesseract --version` and try improving image DPI/resolution.
- To test LLM behavior locally, mock the LLM client or use a development API key with low quotas.

Appendix: Example curl requests
------------------------------
POST /query
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key provisions of the IPC?", "top_k": 5}'
```

POST /query-with-files (multipart)
```bash
curl -X POST http://localhost:8000/query-with-files \
  -F "question=Summarize liability clauses" \
  -F "files=@/path/to/contract.pdf" \
  -F "files=@/path/to/scan.png"
```

Closing notes
-------------
This README is a practical reference to implement, run and test the API and ingestion pipeline for AI_Lawyer. If you want, I can now scaffold the FastAPI app skeleton, the Pydantic models, and a couple of example tests (mocked) to get you started quickly.
