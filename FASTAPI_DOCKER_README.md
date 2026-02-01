# FastAPI & Docker Integration Guide

This document explains what was added to the project to provide a production-ready web API and a Docker image. It is written so non-coders can understand, and it documents the code for developers.

## Why this was added
- A web server (FastAPI) lets the project accept file uploads and queries over the network.
- Docker packages the app and all dependencies so it runs the same on any machine or in the cloud.

## What the API does (simple)
- Accepts files (PDF, DOCX, TXT, images) and returns extracted text.
- Supports telling the server to extract files already on disk.
- Provides a query endpoint (placeholder) to plug-in your RAG pipeline.
- Has health and readiness endpoints for monitoring.

## Key Files and Plain English Explanations
- File: [src/AI_Lawyer/api/app.py](src/AI_Lawyer/api/app.py)
  - Sets up the web server (think: building the front desk and rules for handling incoming visitors).
  - Main parts:
    - `create_app()` — assembles the app, registers routes, and configures middleware.
    - `lifespan` — runs simple startup/shutdown logs.
    - Request timing middleware — measures how long each request takes and adds headers so callers can see processing time.
    - Error handlers — return tidy error messages instead of raw errors.

- File: [src/AI_Lawyer/api/main.py](src/AI_Lawyer/api/main.py)
  - Small runner that reads environment variables and starts the server.

- File: [src/AI_Lawyer/api/routes/extraction.py](src/AI_Lawyer/api/routes/extraction.py)
  - The extraction endpoints:
    - `POST /extraction/extract` — upload files directly. The server processes each file and returns the text or an error message for each file.
    - `POST /extraction/extract-path` — tell the server to extract files already on the server by giving paths or a directory.
    - `GET /extraction/supported-formats` — returns which file types are supported and whether OCR is enabled.

- File: [src/AI_Lawyer/api/routes/health.py](src/AI_Lawyer/api/routes/health.py)
  - Health checks:
    - `GET /health/` — returns server health and uptime.
    - `GET /health/ready` and `/health/live` — quick checks used by deployment systems.

- File: [src/AI_Lawyer/api/routes/query.py](src/AI_Lawyer/api/routes/query.py)
  - Query/RAG endpoint (placeholder). Use this file to connect your RAG/vector store/LLM.

- File: [src/AI_Lawyer/api/models/requests.py](src/AI_Lawyer/api/models/requests.py)
  - Descriptions of expected inputs — like form templates the server expects.

- File: [src/AI_Lawyer/api/models/responses.py](src/AI_Lawyer/api/models/responses.py)
  - Describes how the server formats its replies.

- File: [src/AI_Lawyer/components/file_extractor.py](src/AI_Lawyer/components/file_extractor.py)
  - The core logic that reads files and extracts text.
  - Main methods:
    - `extract_pdf(path)` — pulls text from each page of a PDF.
    - `extract_docx(path)` — reads paragraphs from Word files.
    - `extract_txt(path)` — reads plain text files (tries UTF-8 then latin-1 encoding).
    - `extract_image_ocr(path)` — runs OCR (Tesseract) on images to get text.
    - `extract_from_file(path)` — determines the file type and calls the correct extractor.
    - `extract_batch(list)` — accepts a list of files (paths or uploaded file objects) and returns a mapping of filename → extracted text or error message.

- File: [src/AI_Lawyer/components/extraction_component.py](src/AI_Lawyer/components/extraction_component.py)
  - A wrapper around the extractor that loads configuration automatically and exposes friendly methods for the API to call.

- File: [config/config.yaml](config/config.yaml)
  - Settings for the extractor (which file types to accept, whether OCR is enabled, path to Tesseract binary, etc.).

## Docker (what and why)
- `Dockerfile`: Builds a container image that includes Python, system packages (including `tesseract-ocr`), Python dependencies, and the project files. The image runs the API server when started.
- `docker-compose.yml`: Lets you run the service locally with a single command and mounts project directories into the container so persistent files (like downloaded PDFs and model files) are accessible.

Why Tesseract is installed in Docker
- Tesseract is an external OCR engine used by `pytesseract`.
- It is a system binary (not a Python package), so it must be installed on the OS level inside the container.

## How to run (step-by-step)
Local (development):
```bash
pip install -r requirements.txt
python api_server.py
```

Run with Uvicorn directly (recommended during development):
```bash
uvicorn AI_Lawyer.api.app:app --host 0.0.0.0 --port 8000
```

Docker (self-contained, recommended for deployment):
```bash
docker-compose up --build -d
# then visit http://localhost:8000/docs
```

## Simple examples
Upload files using `curl`:
```bash
curl -X POST "http://localhost:8000/extraction/extract" \
  -F "files=@/path/to/document.pdf" \
  -F "files=@/path/to/image.png"
```

Extract from a server directory:
```bash
curl -X POST "http://localhost:8000/extraction/extract-path" \
  -H "Content-Type: application/json" \
  -d '{"directory_path": "artifacts/data/pdfs/"}'
```

Query the system (placeholder):
```bash
curl -X POST "http://localhost:8000/query/ask" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are fundamental rights?", "top_k": 5}'
```

## Production tips (short)
- Use HTTPS and authentication (API keys or OAuth).
- Configure `ALLOWED_ORIGINS` in `.env` to restrict access.
- Persist vector stores and models with Docker volumes or managed services.
- Monitor CPU and memory; LLM inference is often resource-intensive.

---

If you want, I can also:
- Wire the `POST /query/ask` endpoint to your RAG pipeline and vector store.
- Add authentication and role-based access.
- Create a small web UI for uploading files and asking questions.

Tell me which of these you'd like next and I'll implement it.
