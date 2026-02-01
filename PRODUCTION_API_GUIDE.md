# AI Lawyer API - Production Deployment Guide

## Overview

This is a production-grade FastAPI implementation for the AI Lawyer project. The API provides:

- Legal document extraction and processing
- Vector-based semantic search (RAG)
- Hybrid search combining permanent and user-uploaded documents
- Comprehensive health monitoring
- Production-grade error handling
- Request/response validation
- API documentation

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.production .env
# Edit .env with your configuration
```

### Running the Server

```bash
# Development mode
python api_server.py

# Production mode with Gunicorn
gunicorn src.AI_Lawyer.api.app:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120

# Docker
docker build -t ai-lawyer-api .
docker run -p 8000:8000 ai-lawyer-api
```

## API Endpoints

### Health Check

```
GET /health
GET /health/ready     # Kubernetes readiness probe
GET /health/live      # Kubernetes liveness probe
```

### File Extraction

```
POST /extraction/extract
- Extract text from uploaded files
- Supported formats: PDF, DOCX, TXT, PNG, JPG, JPEG, BMP, TIFF

Request:
{
  "files": [file1, file2, ...],
  "extract_images": false,
  "preserve_formatting": true
}

Response:
{
  "success": true,
  "data": {
    "filename.pdf": "extracted text...",
    ...
  },
  "errors": {},
  "processing_time_seconds": 1.234
}
```

### Data Ingestion

```
POST /ingestion/documents
- Index documents into vector store

Request:
{
  "documents": ["document text 1", "document text 2", ...],
  "collection_name": "my_collection",
  "chunk_size": 512,
  "chunk_overlap": 128
}

Response:
{
  "success": true,
  "document_count": 2,
  "chunk_count": 8,
  "processing_time_seconds": 2.345
}
```

### Query/RAG

```
POST /query/ask
- Query the vector store using RAG

Request:
{
  "query": "What are fundamental rights?",
  "mode": "standard",
  "top_k": 5,
  "score_threshold": 0.0
}

Response:
{
  "success": true,
  "query": "What are fundamental rights?",
  "answer": "Fundamental rights in the Indian Constitution include...",
  "results": [
    {
      "text": "Article 14 grants equality before law...",
      "source": "COI.pdf",
      "score": 0.92,
      "rank": 1
    }
  ],
  "processing_time_seconds": 0.567,
  "confidence_score": 0.87
}
```

### Hybrid Query

```
POST /query/hybrid
- Query both permanent database and user uploads

Request:
{
  "query": "What are my obligations?",
  "use_permanent_db": true,
  "use_user_uploads": true
}

Response:
{
  "success": true,
  "answer": "Based on the documents, your obligations are...",
  "permanent_db_results": 3,
  "user_upload_results": 2,
  "processing_time_seconds": 0.789
}
```

## Configuration

### Environment Variables

See `.env.production` for a complete list of configuration options.

Key configurations:

- `HOST`, `PORT`: Server binding
- `LOG_LEVEL`: Logging verbosity
- `EMBEDDING_MODEL`: Sentence transformer model
- `LLM_PROVIDER`, `LLM_MODEL`: LLM configuration
- `CHUNK_SIZE`, `CHUNK_OVERLAP`: Text chunking
- `TOP_K_RESULTS`: Default retrieval count

### API Keys

Store sensitive credentials in environment variables:

```bash
GROQ_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
```

## Deployment

### Docker Deployment

```dockerfile
# See Dockerfile for complete setup
docker build -t ai-lawyer-api .
docker run -p 8000:8000 --env-file .env ai-lawyer-api
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-lawyer-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-lawyer-api
  template:
    metadata:
      labels:
        app: ai-lawyer-api
    spec:
      containers:
      - name: api
        image: ai-lawyer-api:latest
        ports:
        - containerPort: 8000
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
        env:
        - name: LOG_LEVEL
          value: "info"
        - name: ENVIRONMENT
          value: "production"
```

### Docker Compose

```yaml
# See docker-compose.yml for multi-service setup
docker-compose up -d
```

## Monitoring

### Health Checks

The API provides multiple health check endpoints:

- `/health` - Full health status with component checks
- `/health/ready` - Kubernetes readiness probe
- `/health/live` - Kubernetes liveness probe
- `/health/startup` - Startup verification

### Logging

Logs are configured via environment variables:

```bash
LOG_FORMAT=json              # JSON structured logging
LOG_LEVEL=info               # Log level
LOG_FILE=./logs/api.log      # Log file path
SENTRY_DSN=https://...       # Error tracking
```

### Metrics

Optional Prometheus metrics:

```bash
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=8001
```

Access metrics at `http://localhost:8001/metrics`

## Performance Tuning

### Uvicorn Workers

```bash
# Use Gunicorn with multiple workers
gunicorn \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  src.AI_Lawyer.api.app:app
```

### Vector Store

- Use FAISS for smaller datasets (~1M documents)
- Use Pinecone for larger datasets with cloud infrastructure
- Enable caching for frequently accessed queries

### LLM Optimization

- Use streaming responses for large answers
- Implement prompt caching for similar queries
- Use smaller models for latency-sensitive operations

## Error Handling

### Error Response Format

All errors follow a consistent format:

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

### Common Error Codes

- `VALIDATION_ERROR` (400): Invalid input
- `AUTHENTICATION_FAILED` (401): Auth required
- `PERMISSION_DENIED` (403): Insufficient permissions
- `RESOURCE_NOT_FOUND` (404): Resource not found
- `EXTRACTION_ERROR` (422): File extraction failed
- `QUERY_ERROR` (422): Query processing failed
- `RATE_LIMIT_EXCEEDED` (429): Too many requests
- `INTERNAL_SERVER_ERROR` (500): Server error

## Testing

### Unit Tests

```bash
pytest tests/ -v
```

### Integration Tests

```bash
# Start the server in test mode
ENVIRONMENT=test python api_server.py

# Run integration tests
pytest tests/integration -v
```

### Load Testing

```bash
# Using Apache Bench
ab -n 1000 -c 10 http://localhost:8000/health

# Using wrk
wrk -t12 -c400 -d30s http://localhost:8000/health

# Using Locust
locust -f locustfile.py --host=http://localhost:8000
```

## Security

### CORS Configuration

Configure allowed origins:

```bash
ALLOWED_ORIGINS=https://example.com,https://app.example.com
```

### Authentication (Optional)

Enable JWT authentication:

```bash
AUTH_ENABLED=true
JWT_SECRET_KEY=your-secret-key
JWT_ALGORITHM=HS256
```

### HTTPS

Enable HTTPS redirect in production:

```bash
REQUIRE_HTTPS=true
```

## Troubleshooting

### Vector Store Errors

```bash
# Rebuild vector store index
curl -X POST http://localhost:8000/ingestion/reindex

# Check vector store status
curl http://localhost:8000/ingestion/status
```

### LLM Service Errors

```bash
# Check LLM availability
curl http://localhost:8000/health

# Verify API keys
echo $GROQ_API_KEY
```

### Performance Issues

```bash
# Check health and component status
curl http://localhost:8000/health

# Monitor logs
tail -f logs/api.log

# Check system resources
top
df -h
```

## API Documentation

Interactive API documentation is available at:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`

## Support

For issues and questions:

- GitHub Issues: [project/issues](https://github.com/ZuhairAnsari17/AI_Lawyer/issues)
- Email: support@ailawyer.com
- Documentation: https://docs.ailawyer.com

## Changelog

### v1.0.0 (2024-01-13)

- Initial production release
- Comprehensive request/response models
- Advanced error handling
- Health monitoring system
- File extraction endpoints
- Data ingestion system
- Query/RAG endpoints
- Hybrid search support
- Docker deployment
- Kubernetes support

---

Last updated: January 13, 2024
