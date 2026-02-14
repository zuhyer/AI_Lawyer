# 🏛️ FastAPI Beginner's Guide - Detailed Code Walkthrough

This is a **comprehensive, beginner-friendly guide** for learning FastAPI through the AI Lawyer project. We'll start with basic concepts and gradually build up to understanding the complete codebase.

---

## Table of Contents

1. [Part 1: Introduction & Basics](#part-1-introduction--basics)
2. [Part 2: Getting Started](#part-2-getting-started)
3. [Part 3: Project Structure Deep Dive](#part-3-project-structure-deep-dive)
4. [Part 4: FastAPI Fundamentals](#part-4-fastapi-fundamentals)
5. [Part 5: Detailed Code Explanation](#part-5-detailed-code-explanation)
6. [Part 6: Data Models & Validation](#part-6-data-models--validation)
7. [Part 7: Routing & Endpoints](#part-7-routing--endpoints)
8. [Part 8: Request/Response Flow](#part-8-requestresponse-flow)
9. [Part 9: Advanced Concepts](#part-9-advanced-concepts)

---

# Part 1: Introduction & Basics

## What is This Project?

### Simple Explanation

Imagine you have a bookshelf with 100 law books. Instead of manually searching through each book for information, you want a smart assistant that:
- Remembers all the books
- Understands your questions
- Instantly finds the right answers

**That's what AI Lawyer does!**

### What Problem Does It Solve?

**Before:** Lawyers spend hours reading through legal documents
**After:** AI Lawyer answers legal questions in seconds

### Real-World Example

```
You: "What is the definition of contract in Indian law?"
     ↓
AI Lawyer: Searches through uploaded legal documents
     ↓
AI Lawyer: "According to section 2(h) of Indian Contract Act..."
     ↓
You: Get instant answer with sources
```

---

## What is an API?

### Understanding APIs with Real Examples

**API = Application Programming Interface**

Think of it like ordering food at a restaurant:

```
Customer (You)           Waiter (API)           Kitchen (Server)
   "I want a              Takes order to          Prepares the
    coffee"   --------->     kitchen     ------>  coffee
                                          <------
                           Brings back             Ready
                           the coffee   <-------
   Gets coffee <---------
```

### In Code Terms

```
Your Computer (Client)    FastAPI Server         Database/Logic
   Sends Request    ------>    Receives Request
                               |
                               ↓
                          Processes Request
                               |
                               ↓
   Gets Response    <------    Sends Response
```

### API Request & Response Structure

**What You Send (Request):**
```json
{
  "question": "What is a contract?",
  "timeout": 30
}
```

**What You Get Back (Response):**
```json
{
  "question": "What is a contract?",
  "answer": "A contract is a legally binding agreement...",
  "confidence": 0.95,
  "sources": ["document1.pdf", "document2.pdf"]
}
```

---

## What is FastAPI?

### Why FastAPI is Perfect for Beginners

| Feature | Why It Matters | For Beginners |
|---------|----------------|--------------------|
| **Simple Syntax** | Code is easy to read | Less confusion |
| **Automatic Validation** | Checks data automatically | Don't write validation code manually |
| **Interactive Docs** | Built-in Swagger UI | Test without writing client code |
| **Fast** | Handles many requests | Scalable from day one |
| **Modern** | Uses Python 3.7+ features | Learn best practices |

### FastAPI vs Other Frameworks

```
Framework   Learning Curve   Speed   Built-in Docs   Modern
─────────────────────────────────────────────────────────────
FastAPI        Easy ⬆️      Very Fast✅   Yes ✅       Yes ✅
Flask          Easy ⬆️      Medium ✓     Manual       Older
Django         Hard ↑↑      Medium ✓     Manual       Yes ✅
```

### What Makes FastAPI Special?

**1. Automatic Documentation**
```
Just write your code → Swagger UI generates docs automatically!
```

**2. Type Hints**
```python
# FastAPI can validate types automatically
def get_user(user_id: int):  # user_id MUST be an integer
    return {"user_id": user_id}
```

**3. Automatic Validation**
```python
# If someone sends text instead of number, FastAPI rejects it
GET /user/notanumber  → Error: "user_id must be an integer"
GET /user/123         → Success!
```

---

# Part 2: Getting Started

## Installation & Setup

### Step 1: Install Dependencies

```bash
# Navigate to project directory
cd /workspaces/AI_Lawyer

# Install all required packages
pip install -r requirements.txt
```

**What This Does:**
1. Reads `requirements.txt` (list of all packages needed)
2. Downloads each package from PyPI (Python Package Index)
3. Installs them in your Python environment

### Step 2: Understand requirements.txt

```bash
# View the requirements file
cat requirements.txt
```

**What You'll See:**

```
# ===== CORE DEPENDENCIES =====
fastapi>=0.104.0          # FastAPI framework
uvicorn[standard]>=0.24.0 # Web server to run FastAPI
pydantic>=2.0.0           # Data validation
pydantic-settings>=2.0.0  # Configuration management

# ===== DOCUMENT PROCESSING =====
faiss-cpu>=1.7.0          # Vector database for searches
pdfplumber>=0.10.0        # Extract text from PDFs

# ===== EMBEDDINGS & NLP =====
sentence-transformers>=2.2.0  # Convert text to numbers (embeddings)

# ===== ENVIRONMENT & CONFIG =====
python-dotenv>=1.0.0     # Load .env file
```

**What Each Package Does:**

| Package | Purpose | In Simple Terms |
|---------|---------|-----------------|
| `fastapi` | Web framework | The engine that runs your API |
| `uvicorn` | Web server | The kitchen that serves requests |
| `pydantic` | Data validation | Checks if requests are correct |
| `faiss-cpu` | Vector search | Finds similar documents fast |
| `sentence-transformers` | Text embeddings | Converts text to numbers for searching |

### Step 3: Create Environment Variables

```bash
# Create .env file with settings
cat > .env << 'EOF'
# Server Settings
HOST=0.0.0.0              # Accept connections from any IP
PORT=8000                 # Port number
RELOAD=true               # Auto-restart when code changes
LOG_LEVEL=info            # Log level (debug, info, warning, error)
ENVIRONMENT=development   # Development or production

# API Keys (example - add your actual keys)
GROQ_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
EOF
```

**What Each Setting Means:**
- `HOST=0.0.0.0` → Accept requests from anywhere
- `PORT=8000` → Server listens on port 8000
- `RELOAD=true` → Auto-reload when you change code (great for development!)
- `LOG_LEVEL=info` → Show info, warning, and error messages

### Step 4: Start the Server

```bash
# Method 1: Using the main entry point
python -m AI_Lawyer.api.main

# Method 2: Using uvicorn directly
uvicorn AI_Lawyer.api.app:app --reload

# Method 3: Using the api_server.py shortcut
python api_server.py
```

**What You Should See:**

```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete
```

**What This Means:**
- ✅ Server is running
- ✅ You can now access the API at `http://localhost:8000`
- ✅ Swagger UI is available at `http://localhost:8000/docs`

### Step 5: Test with Swagger UI

1. **Open your browser and go to:**
   ```
   http://localhost:8000/docs
   ```

2. **You'll see:**
   - A list of all endpoints (API entry points)
   - Blue buttons for GET requests
   - Green buttons for POST requests
   - Red buttons for DELETE requests

3. **To test an endpoint:**
   - Click on any endpoint (e.g., `GET /health`)
   - Click "Try it out"
   - Click "Execute"
   - See the response!

---

# Part 3: Project Structure Deep Dive

## Understanding the Folder Layout

```
/AI_Lawyer/
│
├── src/                                    ← Source code folder
│   └── AI_Lawyer/                         ← Main package
│       │
│       ├── api/                           ← FastAPI code (what we focus on)
│       │   ├── main.py                    ← Entry point (starts server)
│       │   ├── app.py                     ← FastAPI app setup + middleware
│       │   │
│       │   ├── models/                    ← Data validation models
│       │   │   ├── requests.py            ← Request data structure
│       │   │   └── responses.py           ← Response data structure
│       │   │
│       │   ├── routes/                    ← API endpoints (grouped by feature)
│       │   │   ├── health.py              ← System health endpoint
│       │   │   ├── query.py               ← Legal question endpoint
│       │   │   ├── extraction.py          ← Data extraction endpoint
│       │   │   └── ingestion.py           ← Document upload endpoint
│       │   │
│       │   ├── dependencies.py            ← Shared code between endpoints
│       │   └── exceptions.py              ← Error handling
│       │
│       ├── components/                    ← Business logic (non-API)
│       │   ├── query_component.py         ← Question answering logic
│       │   ├── file_extractor.py          ← File reading logic
│       │   └── embedding.py               ← Text embedding logic
│       │
│       ├── config/                        ← Settings and configuration
│       │   └── configuration.py           ← Load and manage settings
│       │
│       └── utils/                         ← Helper functions
│           ├── logging_setup.py           ← Logging configuration
│           └── common.py                  ← Utility functions
│
├── requirements.txt                       ← Package list
└── README.md                              ← Project documentation
```

## What Each Folder Does

### 1. `api/` - The FastAPI Application

**Purpose:** Contains all FastAPI-related code

**Key Files:**
- `main.py` → Starts the server
- `app.py` → Configures FastAPI app
- `models/` → Data validation
- `routes/` → API endpoints
- `dependencies.py` → Shared code

### 2. `models/` - Data Validation

**Purpose:** Define what data can be sent and received

**Files:**
- `requests.py` → "When you call my API, you must send THIS data"
- `responses.py` → "When you call my API, you'll always get THIS data back"

### 3. `routes/` - API Endpoints

**Purpose:** Define "doors" or entry points to your API

**Files:**
- `health.py` → Check if system is healthy
- `query.py` → Ask legal questions
- `extraction.py` → Extract data from documents
- `ingestion.py` → Upload documents

### 4. `components/` - Business Logic

**Purpose:** Contains actual logic (not API-related)

**This is where the "smart" work happens:**
- Convert questions to embeddings
- Search vector database
- Extract answers
- Read PDF files

### 5. `config/` - Settings

**Purpose:** Load and manage configuration

**Example:** API keys, database paths, model names

### 6. `utils/` - Helper Functions

**Purpose:** Reusable code snippets

**Example:** Logging, common functions, utilities

---

# Part 4: FastAPI Fundamentals

## Concept 1: What is `FastAPI()`?

### The Absolute Basics

```python
from fastapi import FastAPI

# This creates an empty API
app = FastAPI()
```

**What This Does:**
- Creates a new FastAPI application object
- This object will hold all your endpoints

**Real-World Analogy:**
```
FastAPI() = Opening a new restaurant
- You have the building
- You have an empty kitchen
- You haven't added any tables or menu yet
```

### Adding Features to FastAPI

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware (allow other websites to call your API)
app.add_middleware(CORSMiddleware, allow_origins=["*"])

# Add routes (create the menu)
@app.get("/")
def read_root():
    return {"Hello": "World"}
```

---

## Concept 2: Decorators (The @ Symbol)

### What is a Decorator?

A **decorator** is like adding a sticker to something that changes how it works.

**Non-FastAPI Example:**

```python
# Without decorator - just a function
def hello():
    print("Hello!")

# With decorator - the function behaves differently
@print_before_after  # This sticker makes it print before and after
def hello():
    print("Hello!")

# When you call it:
# Output:
# [Before]
# Hello!
# [After]
```

### Decorators in FastAPI

```python
from fastapi import FastAPI

app = FastAPI()

# This decorator says: "Make this function an API endpoint"
# Method: GET (read something)
# Path: /greeting
@app.get("/greeting")
def get_greeting():
    return {"message": "Hi there!"}
```

**What the decorator does:**
1. Takes your function
2. Turns it into an API endpoint
3. Handles HTTP requests automatically
4. Converts return value to JSON

**Without the decorator:**
```python
def get_greeting():  # Just a normal function
    return {"message": "Hi there!"}

# Calling it:
result = get_greeting()  # Returns a dictionary
print(result)  # {"message": "Hi there!"}
```

**With the decorator:**
```python
@app.get("/greeting")
def get_greeting():  # Now it's an endpoint!
    return {"message": "Hi there!"}

# Calling it via HTTP:
GET http://localhost:8000/greeting
# Returns JSON response: {"message": "Hi there!"}
```

---

## Concept 3: HTTP Methods

### The Four Main Methods

| Method | Operation | Use Case | Real Example |
|--------|-----------|----------|--------------|
| **GET** | Read | Fetch data | Get a user's profile |
| **POST** | Create | Send data to create | Submit a form |
| **PUT** | Replace | Full update | Update entire user profile |
| **DELETE** | Remove | Delete data | Delete a user account |

### PUT vs PATCH (Bonus)

```
PUT:   Replace the ENTIRE resource
PATCH: Replace ONLY the fields you specify

# PUT example
PUT /user/123
{
  "name": "John",
  "email": "john@example.com",
  "age": 30
}
→ Replaces user's entire profile

# PATCH example
PATCH /user/123
{
  "email": "newemail@example.com"
}
→ Only changes the email, keeps name and age
```

### FastAPI Example for Each Method

```python
from fastapi import FastAPI

app = FastAPI()

# ===== GET: Read data =====
@app.get("/users/{user_id}")
def get_user(user_id: int):
    """Get a user by ID."""
    return {"user_id": user_id, "name": "John"}

# ===== POST: Create new data =====
@app.post("/users")
def create_user(name: str, email: str):
    """Create a new user."""
    return {
        "message": "User created",
        "name": name,
        "email": email
    }

# ===== PUT: Replace data =====
@app.put("/users/{user_id}")
def update_user(user_id: int, name: str, email: str):
    """Replace entire user data."""
    return {
        "message": "User updated",
        "user_id": user_id,
        "name": name,
        "email": email
    }

# ===== DELETE: Remove data =====
@app.delete("/users/{user_id}")
def delete_user(user_id: int):
    """Delete a user."""
    return {"message": f"User {user_id} deleted"}
```

---

# Part 5: Detailed Code Explanation

## File 1: `src/AI_Lawyer/api/main.py` - Server Entry Point

### Complete Code With Explanations

```python
"""
Entry point for running the AI Lawyer FastAPI server.
Includes comprehensive logging and configuration management.

This is the file you run to start the server!

Usage:
    python -m AI_Lawyer.api.main
    python api_server.py
    uvicorn AI_Lawyer.api.app:app
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from AI_Lawyer.utils.logging_setup import logger


def load_environment():
    """
    Load environment variables from .env file.
    
    What this does:
    1. Looks for .env file in the project root
    2. Reads all settings from that file
    3. Makes them available via os.getenv()
    
    Why we need this:
    - Keep sensitive data (API keys) out of code
    - Have different settings for dev, test, production
    - Make configuration easy to change
    """
    # Build path to .env file
    env_file = Path(__file__).parent.parent.parent.parent / ".env"
    
    # Check if .env file exists
    if env_file.exists():
        # Load environment variables from the file
        load_dotenv(env_file)
        logger.info(f"✓ Environment loaded from: {env_file}")
    else:
        # Warn if .env doesn't exist
        logger.warning(f"⚠ .env file not found at: {env_file}")


def main():
    """
    Run the FastAPI server with production-grade configuration.
    
    This is the main function that starts everything!
    """
    import uvicorn
    
    # Step 1: Load environment variables from .env file
    load_environment()
    
    # Step 2: Read configuration from environment variables
    # If variable doesn't exist, use the default value after comma
    host = os.getenv("HOST", "0.0.0.0")  # Default: accept any IP
    port = int(os.getenv("PORT", 8000))   # Default: port 8000
    reload = os.getenv("RELOAD", "false").lower() == "true"  # Default: don't reload
    log_level = os.getenv("LOG_LEVEL", "info")  # Default: info level
    environment = os.getenv("ENVIRONMENT", "development")  # Default: development
    
    # Step 3: Print startup information
    logger.info("=" * 70)
    logger.info("🚀 AI LAWYER API - STARTING UP")
    logger.info(f"Environment: {environment}")
    logger.info(f"Host: {host}")
    logger.info(f"Port: {port}")
    logger.info(f"Reload: {reload}")
    logger.info("=" * 70)
    
    # Step 4: Import the FastAPI app
    # We import it here (not at the top) so environment is loaded first
    from AI_Lawyer.api.app import app
    
    # Step 5: Start the server using uvicorn
    # uvicorn = the server software that runs FastAPI
    uvicorn.run(
        app,                    # The FastAPI application
        host=host,              # Listen on this host
        port=port,              # Listen on this port
        reload=reload,          # Auto-restart when code changes?
        log_level=log_level     # How detailed should logs be?
    )


if __name__ == "__main__":
    # When this file is run directly, execute main()
    main()
```

### Key Concepts Explained

**1. What does `Path` do?**
```python
from pathlib import Path

# Build absolute path to .env file
env_file = Path(__file__).parent.parent.parent.parent / ".env"

# Breaking it down:
# __file__                  = /workspaces/AI_Lawyer/src/AI_Lawyer/api/main.py
# .parent                   = /workspaces/AI_Lawyer/src/AI_Lawyer/api
# .parent.parent            = /workspaces/AI_Lawyer/src/AI_Lawyer
# .parent.parent.parent     = /workspaces/AI_Lawyer/src
# .parent.parent.parent.parent = /workspaces/AI_Lawyer
# / ".env"                  = /workspaces/AI_Lawyer/.env
```

**2. What does `load_dotenv()` do?**
```python
from dotenv import load_dotenv

load_dotenv(".env")  # Read .env file and load variables

# Now you can access them:
import os
api_key = os.getenv("GROQ_API_KEY")  # Gets value from .env
```

**3. What is `uvicorn.run()`?**
```python
import uvicorn

# This line does ALL the magic:
uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

# It:
# 1. Takes your FastAPI app
# 2. Starts a web server
# 3. Listens for HTTP requests
# 4. Routes them to your endpoints
# 5. Sends responses back
```

---

## File 2: `src/AI_Lawyer/api/app.py` - FastAPI Setup

### Complete Explanation

```python
"""
Production-grade FastAPI application with comprehensive middleware,
error handling, security, and observability.

This file sets up the entire FastAPI application with all
middleware, error handlers, and route registration.
"""

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZIPMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import time
import logging
import os
import uuid
from datetime import datetime

# Import all route modules
from AI_Lawyer.api.routes import health, extraction, query, ingestion
from AI_Lawyer.api.exceptions import APIException, ErrorCode
from AI_Lawyer.api.dependencies import lifespan_manager
from AI_Lawyer.utils.logging_setup import logger


# ===== STEP 1: MIDDLEWARE =====

class RequestIDMiddleware:
    """
    Middleware to add request IDs for tracking.
    
    What this does:
    - Every request gets a unique ID
    - ID is stored in the request object
    - Returned in response headers
    - Used for logging and debugging
    
    Why we need it:
    - Track requests through the system
    - Debug issues by finding specific requests
    - Monitor which requests are slow
    """
    
    def __init__(self, app):
        """Initialize middleware with FastAPI app."""
        self.app = app
    
    async def __call__(self, request: Request, call_next):
        """
        Process each request and add request ID.
        
        Steps:
        1. Check if request header has X-Request-ID
        2. If not, generate a new random ID
        3. Store it in request state
        4. Pass request to next middleware/route
        5. Add ID to response headers
        6. Return response
        """
        # Get Request ID from headers, or generate new one
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        
        # Store in request state for later use
        request.state.request_id = request_id
        request.state.start_time = time.time()
        
        # Continue processing the request
        response = await call_next(request)
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        
        return response


class LoggingMiddleware:
    """
    Middleware for request/response logging.
    
    What this does:
    - Logs every incoming request
    - Logs every outgoing response
    - Records execution time
    - Records status codes
    """
    
    def __init__(self, app):
        """Initialize middleware."""
        self.app = app
    
    async def __call__(self, request: Request, call_next):
        """
        Log request details and response.
        """
        start_time = time.time()
        request_id = getattr(request.state, "request_id", "unknown")
        
        # Log incoming request
        logger.info(
            f"[{request_id}] {request.method} {request.url.path} - "
            f"Client: {request.client.host}"
        )
        
        # Process the request (call next middleware/route)
        response = await call_next(request)
        
        # Calculate execution time
        process_time = time.time() - start_time
        
        # Log outgoing response
        logger.info(
            f"[{request_id}] {response.status_code} - "
            f"Execution time: {process_time:.3f}s"
        )
        
        return response


# ===== STEP 2: CREATE FASTAPI APP =====

# Create the main FastAPI application
app = FastAPI(
    title="AI Lawyer API",
    description="Intelligent legal assistant API",
    version="1.0.0",
)


# ===== STEP 3: ADD MIDDLEWARE =====

# Middleware runs on EVERY request/response
# Order matters - they execute from bottom to top on requests

# Add request ID middleware
app.add_middleware(RequestIDMiddleware)

# Add logging middleware
app.add_middleware(LoggingMiddleware)

# Add CORS middleware (allow cross-origin requests)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Allow requests from any origin
    allow_methods=["*"],      # Allow all HTTP methods
    allow_headers=["*"],      # Allow all headers
)

# Add trusted host middleware (security)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "0.0.0.0"]
)

# Add GZIP middleware (compress responses)
app.add_middleware(GZIPMiddleware, minimum_size=1000)


# ===== STEP 4: ERROR HANDLERS =====

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle validation errors.
    
    When a request fails validation (wrong data type, missing field, etc.),
    this handler creates a nice error response.
    """
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": exc.errors(),
            "request_id": getattr(request.state, "request_id", "unknown")
        },
    )


@app.exception_handler(APIException)
async def api_exception_handler(request: Request, exc: APIException):
    """Handle custom API exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "request_id": getattr(request.state, "request_id", "unknown")
        },
    )


# ===== STEP 5: REGISTER ROUTES =====

# Routes are organized in separate files for clarity
# Register each router here

app.include_router(health.router)      # Health check endpoints
app.include_router(query.router)       # Query/search endpoints
app.include_router(ingestion.router)   # Document upload endpoints
app.include_router(extraction.router)  # Data extraction endpoints


# ===== STEP 6: ROOT ENDPOINT =====

@app.get("/")
async def root():
    """
    Root endpoint - welcome message.
    
    When someone visits http://localhost:8000/
    They get this response.
    """
    return {
        "message": "Welcome to AI Lawyer API",
        "docs_url": "/docs",
        "openapi_url": "/openapi.json"
    }


# ===== STEP 7: LIFESPAN EVENTS =====

@app.on_event("startup")
async def startup_event():
    """
    Run when server starts.
    
    Use this to:
    - Initialize database connections
    - Load models
    - Warm up caches
    - Print startup messages
    """
    logger.info("🚀 Server startup complete!")


@app.on_event("shutdown")
async def shutdown_event():
    """
    Run when server shuts down.
    
    Use this to:
    - Close database connections
    - Clean up resources
    - Print shutdown messages
    """
    logger.info("🛑 Server shutdown!")
```

### Understanding Middleware

**What is Middleware?**

Middleware is code that runs on EVERY request and response. Think of it like security guards at a building entrance:

```
Request comes in
    ↓
[Guard 1: Check ID] (Trusted Host Middleware)
    ↓
[Guard 2: Log entry] (Logging Middleware)
    ↓
[Guard 3: Add pass] (Request ID Middleware)
    ↓
Request enters building (your endpoint code)
    ↓
[Guard 3: Log exit] (Logging Middleware)
    ↓
[Guard 2: Add stamp] (Request ID Middleware)
    ↓
Response leaves building
```

**Middleware Order Matters:**

```python
# Middleware is executed in REVERSE order on requests
app.add_middleware(A)  # Runs 3rd
app.add_middleware(B)  # Runs 2nd
app.add_middleware(C)  # Runs 1st (last one added, first executed)

# Request flow: C → B → A → endpoint
# Response flow: A → B → C
```

---

## File 3: `src/AI_Lawyer/api/routes/health.py`

### Complete Explanation

```python
"""
Health check and system status endpoints.
Production-grade health monitoring with component status checks.

Health checks are important because:
- Monitoring systems (like Kubernetes) need to know if API is alive
- Load balancers use health checks to route traffic
- You need to know if critical components are working
"""

from fastapi import APIRouter, HTTPException, status
from datetime import datetime
import time

from AI_Lawyer.api.models.responses import (
    HealthResponse, ComponentHealth, ComponentStatusEnum
)
from AI_Lawyer.utils.logging_setup import logger

# Create a router for health-related endpoints
# prefix="/health" means all endpoints here start with /health
# tags=["health"] groups them together in Swagger UI
router = APIRouter(prefix="/health", tags=["health"])

# Track server start time so we can calculate uptime
SERVER_START_TIME = time.time()


@router.get("/", response_model=HealthResponse)
async def health_check():
    """
    Comprehensive health check endpoint.
    
    HTTP Method: GET
    URL: http://localhost:8000/health or http://localhost:8000/health/
    
    What this endpoint does:
    1. Checks if API is up and running
    2. Checks various system components
    3. Returns detailed status information
    
    When to use it:
    - Monitoring systems (like Kubernetes) call this regularly
    - Load balancers use it to decide where to route traffic
    - You can use it to debug system issues
    
    What it returns (example):
    {
      "status": "healthy",
      "timestamp": "2024-01-15T10:30:45.123456",
      "uptime_seconds": 3600,
      "components": {
        "api": "healthy",
        "database": "healthy",
        "embeddings": "healthy"
      }
    }
    """
    
    try:
        # Calculate uptime (how long server has been running)
        current_time = time.time()
        uptime_seconds = current_time - SERVER_START_TIME
        
        # Check each component and build component status
        components = {
            "api": ComponentHealth(
                status=ComponentStatusEnum.HEALTHY,
                details="API is running"
            ),
            # Add more components as needed
        }
        
        # Build response using HealthResponse model
        response = HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            uptime_seconds=uptime_seconds,
            components=components
        )
        
        return response
        
    except Exception as e:
        # If anything goes wrong, log error and return unhealthy status
        logger.error(f"Health check failed: {str(e)}")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Health check failed"
        )


@router.get("/ready")
async def readiness_check():
    """
    Readiness check - Can the API handle requests?
    
    URL: http://localhost:8000/health/ready
    
    Unlike health_check (is it alive?),
    readiness_check asks: is it ready to accept requests?
    
    Used by Kubernetes to decide when to route traffic to this pod.
    """
    return {
        "ready": True,
        "timestamp": datetime.now().isoformat()
    }


@router.get("/alive")
async def liveness_check():
    """
    Liveness check - Is the API still running?
    
    URL: http://localhost:8000/health/alive
    
    Used by container orchestration systems
    (like Kubernetes, Docker Swarm) to check if container is alive.
    
    If this fails, the container will be restarted.
    """
    return {
        "alive": True,
        "timestamp": datetime.now().isoformat()
    }
```

### Key Concepts

**1. Route Decorator**
```python
@router.get("/")
# This means:
# - HTTP Method: GET
# - Full path: /health/ (prefix + route)
# - When someone visits /health, this function runs
```

**2. Async Function**
```python
async def health_check():
    # 'async' means this function can handle concurrent requests
    # It won't block other requests if it takes time
    # Important for scalability!
```

**3. Response Model**
```python
@router.get("/", response_model=HealthResponse)
# response_model tells FastAPI to validate response against HealthResponse
# If response doesn't match model, error is raised
# Swagger UI uses model to show what response looks like
```

---

## File 4: `src/AI_Lawyer/api/routes/query.py` - Query Endpoint

### Key Sections Explained

```python
"""
Query/RAG endpoint - supports both standard and hybrid queries.

This endpoint handles the main functionality:
- User sends a question
- API searches through documents
- API returns answer with sources
"""

from fastapi import APIRouter, HTTPException
from typing import Optional, List
import time

from AI_Lawyer.api.models.requests import QueryRequest, HybridQueryRequest
from AI_Lawyer.api.models.responses import QueryResponse, HybridQueryResponse
from AI_Lawyer.components.query_component import QueryComponent
from AI_Lawyer.config.configuration import ConfigurationManager
from AI_Lawyer.utils.logging_setup import logger

# Create router for query endpoints
router = APIRouter(prefix="/query", tags=["query"])

# Initialize components globally for efficiency
# (Singleton pattern - create once, use many times)
_config_manager = None
_query_component = None


def get_components():
    """
    Lazy initialization of components.
    
    What this does:
    1. First time: Initialize components
    2. Next times: Reuse same components
    
    Why we do this:
    - Loading embedding models is SLOW
    - We don't want to reload them on every request
    - Load once, use forever
    
    This is called "lazy loading" or "singleton pattern"
    """
    global _config_manager, _query_component
    
    if _config_manager is None:
        try:
            # Load configuration from files
            _config_manager = ConfigurationManager()
            logger.info("✅ ConfigurationManager initialized")
        except Exception as e:
            logger.error(f"✗ Failed to initialize ConfigurationManager: {e}")
            raise
    
    if _query_component is None:
        try:
            # Load the query component (this might take a few seconds)
            from langchain_community.vectorstores import FAISS
            from sentence_transformers import SentenceTransformer
            
            # Get embedding configuration
            embedding_config = _config_manager.get_embeddings_config()
            vector_store_path = embedding_config.vector_store_path
            
            # Load embedding model
            _query_component = QueryComponent(
                config=embedding_config,
                vector_store_path=vector_store_path
            )
            logger.info("✅ QueryComponent initialized")
            
        except Exception as e:
            logger.error(f"✗ Failed to initialize QueryComponent: {e}")
            raise
    
    return _config_manager, _query_component


@router.post("/ask", response_model=QueryResponse)
async def ask_question(query: QueryRequest):
    """
    Standard query endpoint - Ask a legal question.
    
    HTTP Method: POST
    URL: http://localhost:8000/query/ask
    
    Request Format (QueryRequest):
    {
      "question": "What is a contract?",
      "timeout": 30,
      "document_type": "pdf"
    }
    
    Response Format (QueryResponse):
    {
      "question": "What is a contract?",
      "answer": "A contract is a legally binding agreement...",
      "confidence": 0.95,
      "sources": ["document1.pdf", "section2.pdf"],
      "processing_time": 1.23
    }
    
    Step-by-step execution:
    1. FastAPI validates request (is it a QueryRequest?)
    2. Components are initialized (lazy loading)
    3. Question is converted to embeddings
    4. Vector database is searched
    5. Most relevant documents are returned
    6. Answer is extracted
    7. Response is validated against QueryResponse model
    8. JSON is sent to client
    """
    
    try:
        # Get components (lazy load if needed)
        config_manager, query_component = get_components()
        
        # Time how long this takes
        start_time = time.time()
        
        # Log what we're doing
        logger.info(f"Processing query: {query.question[:50]}...")
        
        # Call the component that does the actual work
        result = query_component.process_query(
            question=query.question,
            filters={"document_type": query.document_type} if query.document_type else None
        )
        
        # Calculate how long it took
        processing_time = time.time() - start_time
        
        # Build response
        response = QueryResponse(
            question=query.question,
            answer=result["answer"],
            confidence=result.get("confidence", 0.5),
            sources=result.get("sources", []),
            processing_time=processing_time
        )
        
        logger.info(f"✅ Query processed in {processing_time:.2f}s")
        return response
        
    except ValueError as e:
        # If something is wrong with the request data
        logger.error(f"❌ Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
        
    except TimeoutError:
        # If request takes too long
        logger.error(f"❌ Query timeout")
        raise HTTPException(
            status_code=504,
            detail="Query processing timeout"
        )
        
    except Exception as e:
        # If anything else goes wrong
        logger.error(f"❌ Query error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )


@router.post("/hybrid", response_model=HybridQueryResponse)
async def hybrid_query(query: HybridQueryRequest):
    """
    Hybrid query endpoint - Advanced search with multiple methods.
    
    This endpoint uses multiple search strategies:
    1. Vector similarity search (embedding-based)
    2. BM25 (keyword-based)
    3. Combines results for best answer
    
    Request:
    {
      "question": "Legal question?",
      "search_method": "hybrid",  # Can be 'vector', 'bm25', or 'hybrid'
      "top_k": 5  # Return top 5 results
    }
    
    Response:
    {
      "question": "...",
      "result": "...",
      "sources": [...],
      "search_method_used": "hybrid",
      "confidence": 0.92
    }
    """
    
    try:
        config_manager, query_component = get_components()
        
        start_time = time.time()
        
        # Hybrid search uses multiple methods
        result = query_component.hybrid_search(
            question=query.question,
            search_method=query.search_method,
            top_k=query.top_k
        )
        
        processing_time = time.time() - start_time
        
        response = HybridQueryResponse(
            question=query.question,
            result=result["answer"],
            sources=result["sources"],
            search_method_used=query.search_method,
            confidence=result.get("confidence", 0.5),
            processing_time=processing_time
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Hybrid query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
```

---

# Part 6: Data Models & Validation

## File: `src/AI_Lawyer/api/models/requests.py`

### Detailed Explanation

```python
"""
Pydantic request models for API endpoints - Production Grade.

Pydantic models are like contracts:
- Client: "I will send THIS format"
- Server: "I expect THIS format"
- If mismatch → Error message
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional
from enum import Enum


# ===== STEP 1: ENUMS (Fixed Choices) =====

class DocumentTypeEnum(str, Enum):
    """
    Enum defines allowed values for document_type.
    
    Instead of allowing any string,
    limit to these specific choices:
    """
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    IMAGE = "image"
    ALL = "all"


class QueryModeEnum(str, Enum):
    """Allowed query modes."""
    STANDARD = "standard"
    HYBRID = "hybrid"
    SEMANTIC = "semantic"


# ===== STEP 2: BASE MODELS =====

class BaseRequest(BaseModel):
    """
    Base request model with common fields.
    
    This is a parent class for all requests.
    All request models inherit from this,
    so they all have these fields:
    """
    
    request_id: Optional[str] = Field(
        None,
        # description: shown in Swagger UI
        description="Unique request identifier for tracking"
    )
    
    timeout: int = Field(
        30,  # Default value
        ge=5,   # Greater than or equal to 5
        le=300, # Less than or equal to 300
        description="Request timeout in seconds (5-300)"
    )


# ===== STEP 3: SPECIFIC REQUEST MODELS =====

class QueryRequest(BaseRequest):
    """
    Request model for legal questions.
    
    When client sends JSON to POST /query/ask:
    {
      "question": "What is a contract?",
      "timeout": 30,
      "document_type": "pdf"
    }
    
    FastAPI:
    1. Checks if JSON matches this model
    2. Validates each field
    3. Converts to Python object
    4. Passes to endpoint function
    """
    
    question: str = Field(
        ...,  # ... means "required, no default"
        min_length=1,
        max_length=1000,
        description="The legal question to ask"
    )
    
    document_type: Optional[DocumentTypeEnum] = Field(
        None,  # Optional field
        description="Filter results by document type"
    )
    
    search_depth: int = Field(
        1,
        ge=1,
        le=3,
        description="How deep to search (1=shallow, 3=deep)"
    )
    
    @validator('question')
    def question_not_empty(cls, v):
        """
        Custom validator - check question is not just whitespace.
        
        This runs AFTER type checking but BEFORE model creation.
        If validation fails, raise ValueError.
        """
        if not v.strip():
            raise ValueError('question cannot be empty or whitespace')
        return v.strip()


class HybridQueryRequest(QueryRequest):
    """
    Request model for hybrid queries (inherits from QueryRequest).
    
    Inherits all fields from QueryRequest,
    plus adds hybrid-specific fields:
    """
    
    search_method: QueryModeEnum = Field(
        QueryModeEnum.HYBRID,
        description="Search method to use"
    )
    
    top_k: int = Field(
        5,
        ge=1,
        le=20,
        description="Return top K results"
    )
```

### How Validation Works

**Example 1: Valid Request**
```python
# Valid JSON sent to /query/ask
{
  "question": "What is a contract?",
  "timeout": 30,
  "document_type": "pdf"
}

# FastAPI validates:
# ✅ question is string? YES
# ✅ question not empty? YES
# ✅ timeout between 5-300? YES (30 is in range)
# ✅ document_type is 'pdf'? YES (valid enum value)

# Result: Accepted! Function is called
```

**Example 2: Invalid - Missing Required Field**
```python
{
  "timeout": 30
  # Missing "question" field!
}

# FastAPI validates:
# ❌ question is required but missing!

# Result: Error response
{
  "detail": [
    {
      "loc": ["body", "question"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

**Example 3: Invalid - Wrong Type**
```python
{
  "question": 12345,  # Should be string, not number!
  "timeout": 30
}

# FastAPI validates:
# ❌ question should be string, got int!

# Result: Error response
{
  "detail": [
    {
      "loc": ["body", "question"],
      "msg": "str type expected",
      "type": "type_error.str"
    }
  ]
}
```

**Example 4: Invalid - Value Out of Range**
```python
{
  "question": "What is law?",
  "timeout": 500  # Should be between 5 and 300!
}

# FastAPI validates:
# ❌ timeout 500 > max 300!

# Result: Error response
{
  "detail": [
    {
      "loc": ["body", "timeout"],
      "msg": "ensure this value is less than or equal to 300",
      "type": "value_error.number.not_le"
    }
  ]
}
```

---

## File: `src/AI_Lawyer/api/models/responses.py`

### Response Model Explanation

```python
"""
Pydantic response models for API endpoints.

Response models ensure clients always know what to expect.
They also show up in Swagger UI documentation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class ComponentStatusEnum(str, Enum):
    """Status of a system component."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthResponse(BaseModel):
    """Response from /health endpoint."""
    
    status: str = Field(
        ...,
        description="Overall health status"
    )
    
    timestamp: str = Field(
        ...,
        description="ISO format timestamp"
    )
    
    uptime_seconds: float = Field(
        ...,
        description="How long server has been running"
    )
    
    components: dict = Field(
        ...,
        description="Status of each component"
    )


class QueryResponse(BaseModel):
    """Response from /query/ask endpoint."""
    
    question: str = Field(
        ...,
        description="Echo back the question"
    )
    
    answer: str = Field(
        ...,
        description="The answer to the question"
    )
    
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score (0.0-1.0)"
    )
    
    sources: List[str] = Field(
        default_factory=list,
        description="Document sources used for answer"
    )
    
    processing_time: float = Field(
        ...,
        description="Time taken to process (seconds)"
    )


# In Swagger UI, this model shows clients exactly
# what response they'll get:
"""
{
  "question": "What is a contract?",
  "answer": "A contract is a...",
  "confidence": 0.95,
  "sources": ["doc1.pdf"],
  "processing_time": 1.23
}
"""
```

---

# Part 7: Routing & Endpoints

## How Routing Works

### Route Resolution

When a request comes in, FastAPI must figure out which function to call.

```
GET /health
   ↓
[Check all routes]
   ↓
Found: @app.get("/health")
   ↓
Found: @router.get("/") in health.py with prefix="/health"
   ↓
Since prefix="/health" + route="/" = "/health" ✅
   ↓
Call health_check() function
```

### Path Parameters

```python
@router.get("/documents/{doc_id}")
async def get_document(doc_id: int):
    """
    Get specific document by ID.
    
    URL: http://localhost:8000/documents/123
    
    FastAPI extracts:
    - {doc_id} from URL → doc_id = 123
    - Validates: is it an int? YES
    - Passes to function: get_document(doc_id=123)
    """
    return {"doc_id": doc_id}


# More complex example:
@router.get("/users/{user_id}/documents/{doc_id}")
async def get_user_document(user_id: int, doc_id: int):
    """Multiple path parameters."""
    return {
        "user_id": user_id,
        "doc_id": doc_id
    }
```

### Query Parameters

```python
@router.get("/search")
async def search(query: str, limit: int = 10, doc_type: str = None):
    """
    Query parameters are optional values in URL.
    
    Examples:
    /search?query=contract                    → query="contract", limit=10, doc_type=None
    /search?query=contract&limit=20           → query="contract", limit=20, doc_type=None
    /search?query=contract&limit=20&doc_type=pdf → query="contract", limit=20, doc_type="pdf"
    
    Rules:
    - Parameters with defaults are optional
    - Parameters without defaults are required
    - FastAPI validates types automatically
    """
    return {
        "query": query,
        "limit": limit,
        "doc_type": doc_type
    }
```

### Request Body (POST)

```python
@router.post("/documents")
async def create_document(doc: DocumentModel):
    """
    Request body parameters - full JSON object.
    
    Client sends:
    POST /documents
    {
      "title": "Contract 2024",
      "content": "..."
    }
    
    FastAPI:
    1. Reads JSON from request body
    2. Validates against DocumentModel
    3. Converts to Python object
    4. Calls function: create_document(doc=DocumentModel(...))
    """
    return {"created": True}
```

---

# Part 8: Request/Response Flow

## Complete Flow Diagram

```
Client                          FastAPI Server
   |                                  |
   | 1. Send HTTP Request             |
   |─────────────────────────────────>|
   |                                  |
   |     POST /query/ask              |
   |     { "question": "..." }        |
   |                                  |
   |                                  |
   |        2. Request reaches FastAPI|
   |        Check routing             |
   |        (which handler?)          |
   |        ↓                          |
   |   3. Check Request Validation    |
   |      Is query a QueryRequest?    |
   |      ✓ Yes, all fields valid     |
   |      ↓                            |
   |   4. Resolve Dependencies        |
   |      Load QueryComponent         |
   |      Load ConfigManager          |
   |      ↓                            |
   |   5. Call Endpoint Function      |
   |      ask_question(query)         |
   |      ↓                            |
   |   6. Function Processes Request  |
   |      - Convert question          |
   |      - Search documents          |
   |      - Extract answer            |
   |      ↓                            |
   |   7. Build Response              |
   |      QueryResponse object        |
   |      ↓                            |
   |   8. Validate Response           |
   |      Does it match model?        |
   |      ✓ Yes                       |
   |      ↓                            |
   |   9. Convert to JSON             |
   |                                  |
   | <─────────────────────────────────|
   | 10. Receive HTTP Response        |
   |                                  |
   | { "answer": "...", ... }         |
   |                                  |
   | Done!                            |
```

## Code Walkthrough of Complete Flow

### 1. Client Sends Request

```javascript
// JavaScript client example
fetch('http://localhost:8000/query/ask', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({
        question: "What is a contract?",
        timeout: 30,
        document_type: "pdf"
    })
})
.then(response => response.json())
.then(data => console.log(data))
```

### 2. FastAPI Receives Request

```python
# Inside FastAPI app.py
@router.post("/ask", response_model=QueryResponse)
async def ask_question(query: QueryRequest):
    # We're here now!
    # FastAPI has already:
    # 1. Received HTTP request
    # 2. Parsed JSON body
    # 3. Validated against QueryRequest model
    # 4. Created QueryRequest object
    pass
```

### 3. Validation Happens

```python
# FastAPI checks:
# - Is body valid JSON? ✅
# - Does it match QueryRequest model? ✅
# - question is string? ✅
# - timeout between 5-300? ✅
# - document_type valid enum? ✅

# If ANY validation fails, error response is sent
# (never reaches the endpoint function)
```

### 4. Dependencies Are Resolved

```python
@router.post("/ask", response_model=QueryResponse)
async def ask_question(query: QueryRequest):
    # If we had used: depends(get_stuff)
    # FastAPI would call get_stuff() first
    # And pass result to this function
    
    config_manager, query_component = get_components()
    # This is manual, but same concept
```

### 5. Endpoint Function Runs

```python
@router.post("/ask", response_model=QueryResponse)
async def ask_question(query: QueryRequest):
    # The actual "business logic" runs here
    
    config_manager, query_component = get_components()
    
    # This is where the real work happens:
    result = query_component.process_query(
        question=query.question,
        filters={"document_type": query.document_type}
    )
    
    # result might be:
    # {
    #   "answer": "A contract is...",
    #   "confidence": 0.95,
    #   "sources": ["doc1.pdf"]
    # }
```

### 6. Response Is Created

```python
# Create response object
response = QueryResponse(
    question=query.question,
    answer=result["answer"],
    confidence=result.get("confidence", 0.5),
    sources=result.get("sources", []),
    processing_time=processing_time
)

# response is now a QueryResponse instance:
# QueryResponse(
#     question="What is a contract?",
#     answer="A contract is...",
#     confidence=0.95,
#     sources=["doc1.pdf"],
#     processing_time=1.23
# )
return response
```

### 7. FastAPI Validates Response

```python
# FastAPI checks:
# - Does response match response_model=QueryResponse ? ✅
# - question is string? ✅
# - confidence between 0.0-1.0? ✅
# - sources is list? ✅

# If validation fails, 500 error is returned
# (developer's fault for wrong return value)
```

### 8. Convert to JSON

```python
# FastAPI converts the object:
# QueryResponse → JSON

# From:
QueryResponse(
    question="What is a contract?",
    answer="A contract is...",
    confidence=0.95,
    sources=["doc1.pdf"],
    processing_time=1.23
)

# To:
{
    "question": "What is a contract?",
    "answer": "A contract is...",
    "confidence": 0.95,
    "sources": ["doc1.pdf"],
    "processing_time": 1.23
}
```

### 9. Send HTTP Response

```
HTTP/1.1 200 OK
Content-Type: application/json
X-Request-ID: abc123

{
    "question": "What is a contract?",
    "answer": "A contract is...",
    "confidence": 0.95,
    "sources": ["doc1.pdf"],
    "processing_time": 1.23
}
```

---

# Part 9: Advanced Concepts

## Dependency Injection Deep Dive

### Problem Without Dependencies

```python
# Bad approach - code duplication

@app.get("/documents")
async def get_documents():
    # Repeat this setup code in EVERY endpoint:
    config = ConfigurationManager()
    vector_store = load_vector_store(config)
    embedding_model = load_embeddings(config)
    
    # ... do stuff with them
    return documents


@app.get("/users")
async def get_users():
    # Repeat AGAIN:
    config = ConfigurationManager()
    vector_store = load_vector_store(config)
    embedding_model = load_embeddings(config)
    
    # ... do stuff with them
    return users


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: int):
    # Repeat AGAIN:
    config = ConfigurationManager()
    vector_store = load_vector_store(config)
    embedding_model = load_embeddings(config)
    
    # ... do stuff with them
    return {"deleted": True}
```

**Problems:**
- ❌ Code duplication (lots of repeat)
- ❌ Hard to update (change in 3+ places)
- ❌ Hard to test (components mixed in)
- ❌ Slow (reloading components every time!)

### Solution With Dependencies

```python
from fastapi import Depends

class Services:
    """Container for all shared services."""
    
    def __init__(self):
        self.config = ConfigurationManager()
        self.vector_store = load_vector_store(self.config)
        self.embedding_model = load_embeddings(self.config)


# Global services instance (created once)
_services = None


def get_services():
    """Dependency provider - give me services!"""
    global _services
    
    if _services is None:
        _services = Services()
    
    return _services


# Now use it in endpoints:

@app.get("/documents")
async def get_documents(services: Services = Depends(get_services)):
    # services is automatically provided!
    # FastAPI calls get_services() and passes result here
    return documents


@app.get("/users")
async def get_users(services: Services = Depends(get_services)):
    # Same services object!
    # Same component instances!
    return users


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: int, services: Services = Depends(get_services)):
    # Same services!
    return {"deleted": True}
```

**Benefits:**
- ✅ No code duplication
- ✅ Easy to update (one place)
- ✅ Easy to test (mock services)
- ✅ Fast (reuse components)

### How `Depends()` Works

```python
from fastapi import Depends

def get_items():
    """Dependency: get a list of items."""
    return ["item1", "item2", "item3"]


@app.get("/items")
async def items(items = Depends(get_items)):
    """
    When this endpoint is called:
    
    1. FastAPI sees: Depends(get_items)
    2. FastAPI calls: get_items()
    3. FastAPI gets: ["item1", "item2", "item3"]
    4. FastAPI passes: items=["item1", "item2", "item3"]
    5. Function runs with items available
    """
    return items  # Returns the list


# Flow diagram:
#
# Client Request
#   ↓
# FastAPI sees: Depends(get_items)
#   ↓
# FastAPI calls get_items()
#   ↓
# get_items() returns ["item1", "item2", "item3"]
#   ↓
# FastAPI passes to endpoint:
# items(items=["item1", "item2", "item3"])
#   ↓
# Endpoint function runs
#   ↓
# Response sent to client
```

---

## Testing Endpoints with Swagger UI

### Step 1: Make Sure Server is Running

```bash
python -m AI_Lawyer.api.main
```

### Step 2: Open Swagger UI

```
http://localhost:8000/docs
```

### Step 3: Try an Endpoint

**Example: Test /health Endpoint**

1. Find the endpoint "GET /health"
2. Click to expand it
3. Click "Try it out"
4. Click "Execute"
5. Scroll down to see response

**Expected Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:45.123456",
  "uptime_seconds": 3600,
  "components": {
    "api": {
      "status": "healthy",
      "details": "API is running"
    }
  }
}
```

**Example: Test /query/ask Endpoint**

1. Find "POST /query/ask"
2. Click to expand
3. Click "Try it out"
4. Edit the JSON body:
```json
{
  "question": "What is a contract?",
  "timeout": 30,
  "document_type": "pdf"
}
```
5. Click "Execute"
6. See the answer!

---

## Error Handling In Detail

### HTTP Status Codes

```
2xx Success
200: OK (request worked)
201: Created (new resource created)

4xx Client Error (your fault)
400: Bad Request (invalid data)
401: Unauthorized (need login)
404: Not Found (resource doesn't exist)
422: Unprocessable Entity (validation error)

5xx Server Error (our fault)
500: Internal Server Error (bug in code)
503: Service Unavailable (server down)
```

### Raising Errors in FastAPI

```python
from fastapi import HTTPException, status

@app.get("/documents/{doc_id}")
async def get_document(doc_id: int):
    """
    Get a document - can fail in multiple ways.
    """
    
    # Error 1: Document doesn't exist
    if doc_id not in documents:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {doc_id} not found"
        )
    
    # Error 2: No permission
    if not user_has_permission(doc_id):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="You don't have permission"
        )
    
    # Error 3: Server error
    try:
        content = read_document(doc_id)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to read document"
        )
    
    return {"content": content}


# Responses:
# 404 → {"detail": "Document 999 not found"}
# 401 → {"detail": "You don't have permission"}
# 500 → {"detail": "Failed to read document"}
```

### Try-Except Error Handling

```python
@app.post("/upload")
async def upload_file(file: UploadFile):
    """
    Upload and process a file.
    """
    try:
        # Try to do the work
        content = await file.read()
        process_file(content)
        
        return {"status": "success", "filename": file.filename}
        
    except ValueError as e:
        # Specific error - wrong format
        logger.error(f"Format error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail="Invalid file format"
        )
    
    except TimeoutError as e:
        # Specific error - took too long
        logger.error(f"Timeout: {str(e)}")
        raise HTTPException(
            status_code=504,
            detail="File processing timeout"
        )
    
    except Exception as e:
        # Catch-all for unexpected errors
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )
```

---

## Summary

You've now learned:

1. ✅ What APIs and FastAPI are
2. ✅ How to set up and run a FastAPI project
3. ✅ Project structure and file organization
4. ✅ FastAPI fundamentals (decorators, methods, etc.)
5. ✅ Detailed code explanation
6. ✅ Data validation with Pydantic
7. ✅ Request and response models
8. ✅ Routing and endpoints
9. ✅ Complete request/response flow
10. ✅ Advanced concepts (dependencies, error handling)

---

## Next Steps

### 🎓 Practice

1. **Run the server locally**
   ```bash
   python -m AI_Lawyer.api.main
   ```

2. **Test endpoints with Swagger UI**
   ```
   http://localhost:8000/docs
   ```

3. **Make small changes**
   - Add a new field to a request model
   - Add a new endpoint
   - Test it in Swagger UI

### 📚 Learn More

- [Official FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [HTTP Status Codes](https://httpstatuses.com/)
- [REST API Best Practices](https://restfulapi.net/)

### 🚀 Advanced Topics

- Async/await programming
- Database integration
- Authentication & security
- Testing
- Deployment
- Performance optimization
- Monitoring & logging

---

**Happy Learning! 🎉**
