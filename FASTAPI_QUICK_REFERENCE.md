# 🚀 FastAPI Quick Reference Guide

## Installation & Setup (5 minutes)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy configuration
cp .env.production .env

# 3. Edit .env with your keys
export GROQ_API_KEY="your-api-key"

# 4. Start server
python api_server.py

# 5. Visit documentation
# Swagger: http://localhost:8000/docs
# ReDoc: http://localhost:8000/redoc
```

## Common Commands

```bash
# Start API
python api_server.py

# Check health
curl http://localhost:8000/health

# Validate setup
python validate_api.py

# Docker build
docker build -t ai-lawyer-api .

# Docker run
docker run -p 8000:8000 --env-file .env ai-lawyer-api

# Production with Gunicorn
gunicorn src.AI_Lawyer.api.app:app --workers 4
```

## API Endpoints Quick Reference

### Health Checks
```bash
# Full health check
curl http://localhost:8000/health

# Readiness probe (K8s)
curl http://localhost:8000/health/ready

# Liveness probe (K8s)
curl http://localhost:8000/health/live
```

### File Extraction
```bash
# Extract from file
curl -X POST http://localhost:8000/extraction/extract \
  -F "files=@document.pdf"
```

### Query/RAG
```bash
# Standard query
curl -X POST http://localhost:8000/query/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are fundamental rights?",
    "top_k": 5,
    "score_threshold": 0.0
  }'

# Hybrid search
curl -X POST http://localhost:8000/query/hybrid \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are my obligations?",
    "use_permanent_db": true,
    "use_user_uploads": true
  }'
```

### Data Ingestion
```bash
# Ingest documents
curl -X POST http://localhost:8000/ingestion/documents \
  -H "Content-Type: application/json" \
  -d '{
    "documents": ["document text 1", "document text 2"],
    "collection_name": "my_collection",
    "chunk_size": 512
  }'

# List collections
curl http://localhost:8000/ingestion/collections

# Rebuild index
curl -X POST http://localhost:8000/ingestion/reindex

# Check ingestion status
curl http://localhost:8000/ingestion/status
```

## Environment Variables (Essential)

```bash
# LLM Configuration
LLM_PROVIDER=groq
LLM_MODEL=mixtral-8x7b-32768
GROQ_API_KEY=your-key-here

# Embeddings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu

# Vector Store
VECTOR_STORE_PATH=./models/vector_store
VECTOR_STORE_TYPE=faiss

# Server
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=info

# Security
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8501
```

## Error Responses Format

```json
{
  "success": false,
  "error_code": "VALIDATION_ERROR",
  "message": "Invalid request parameters",
  "errors": [
    {
      "field": "query",
      "message": "Field required"
    }
  ],
  "request_id": "req-123456789",
  "timestamp": "2024-01-13T10:30:00Z"
}
```

## File Locations

```
API Files:
├── api/app.py                    Main FastAPI app
├── api/main.py                   Entry point
├── api/dependencies.py           Dependency injection
├── api/exceptions.py             Error handling
├── api/utils.py                  Utilities

Config Files:
├── .env.production               Config template
├── .env                          Your config (create from template)

Documentation:
├── PRODUCTION_API_GUIDE.md       Comprehensive guide
├── FASTAPI_PRODUCTION_IMPLEMENTATION.md  Technical details
├── IMPLEMENTATION_SUMMARY.md     This document

Validation:
├── validate_api.py               Validation script
```

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| `Module not found` | Run `pip install -r requirements.txt` |
| `API key not found` | Set `GROQ_API_KEY` in `.env` |
| `Vector store not found` | Ensure FAISS index exists at `VECTOR_STORE_PATH` |
| `Port 8000 in use` | Change `PORT` in `.env` or kill process: `lsof -i :8000` |
| `CORS errors` | Add origin to `ALLOWED_ORIGINS` in `.env` |
| `Health check failed` | Check logs: `tail -f logs/api.log` |

## Health Check Response Example

```json
{
  "success": true,
  "message": "System is healthy",
  "status": "healthy",
  "uptime_seconds": 3600.5,
  "version": "1.0.0",
  "components": [
    {
      "name": "api",
      "status": "ok"
    },
    {
      "name": "vector_store",
      "status": "ok",
      "response_time_ms": 12.5
    },
    {
      "name": "embedding_model",
      "status": "ok",
      "response_time_ms": 45.2
    }
  ],
  "database_connected": true,
  "vector_store_available": true,
  "llm_available": true
}
```

## Query Response Example

```json
{
  "success": true,
  "query": "What are fundamental rights?",
  "answer": "The fundamental rights in the Indian Constitution include...",
  "results": [
    {
      "text": "Article 14 grants equality before law...",
      "source": "COI.pdf",
      "score": 0.92,
      "rank": 1,
      "source_type": "legal_db",
      "page_number": 5
    }
  ],
  "result_count": 1,
  "processing_time_seconds": 0.234,
  "confidence_score": 0.87,
  "mode": "standard"
}
```

## Performance Tips

1. **Use batch endpoints** for multiple operations
2. **Adjust top_k** - smaller values = faster results
3. **Enable caching** - set `ENABLE_QUERY_CACHING=true`
4. **Use smaller embedding models** for speed
5. **Run with Gunicorn** - 4+ workers for production
6. **Enable compression** - `GZIP` is on by default

## Monitoring Checklist

- [ ] Health check endpoint responding
- [ ] All components showing "ok" status
- [ ] Logs being written to `logs/api.log`
- [ ] API documentation accessible at `/docs`
- [ ] Requests being logged with request IDs
- [ ] Error responses formatted correctly
- [ ] Performance timing being tracked

## Kubernetes Deployment

```bash
# Check readiness
kubectl exec pod -c api -- curl localhost:8000/health/ready

# Check liveness
kubectl exec pod -c api -- curl localhost:8000/health/live

# View logs
kubectl logs -f deployment/ai-lawyer-api

# Port forward
kubectl port-forward svc/ai-lawyer-api 8000:8000
```

## Debugging Commands

```bash
# Check logs
tail -f logs/api.log

# Check running processes
ps aux | grep uvicorn

# Test endpoint
curl -v http://localhost:8000/health

# Check environment
env | grep -E "GROQ|LLM|EMBEDDING"

# Validate Python syntax
python -m py_compile src/AI_Lawyer/api/*.py

# Check imports
python validate_api.py
```

## Key Files to Know

| File | Purpose |
|------|---------|
| `app.py` | Main FastAPI application |
| `main.py` | Server entry point |
| `dependencies.py` | Service initialization and injection |
| `exceptions.py` | Custom error handling |
| `utils.py` | Helper functions |
| `models/requests.py` | Request data models |
| `models/responses.py` | Response data models |
| `routes/health.py` | Health check endpoints |
| `routes/extraction.py` | File extraction endpoints |
| `routes/ingestion.py` | Data ingestion endpoints |
| `routes/query.py` | Query/RAG endpoints |

## Support Resources

1. **API Documentation** (when running): `http://localhost:8000/docs`
2. **Deployment Guide**: `PRODUCTION_API_GUIDE.md`
3. **Technical Reference**: `FASTAPI_PRODUCTION_IMPLEMENTATION.md`
4. **Configuration Reference**: `.env.production`
5. **Validation Script**: `python validate_api.py`

## Production Checklist

- [ ] Update `.env` with production values
- [ ] Set `ENVIRONMENT=production`
- [ ] Set `DEBUG=false`
- [ ] Configure `ALLOWED_ORIGINS` for frontend
- [ ] Setup database (if needed)
- [ ] Configure monitoring (Prometheus/Sentry)
- [ ] Enable HTTPS (`REQUIRE_HTTPS=true`)
- [ ] Setup logging aggregation
- [ ] Configure backup for vector store
- [ ] Test health endpoints
- [ ] Load testing with realistic data
- [ ] Security audit
- [ ] Documentation for your team

---

**For detailed information**, refer to:
- `PRODUCTION_API_GUIDE.md` - Full deployment guide
- `FASTAPI_PRODUCTION_IMPLEMENTATION.md` - Technical details
- API Docs at `http://localhost:8000/docs` (when running)

Last Updated: January 13, 2024
