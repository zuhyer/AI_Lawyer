"""Minimal FastAPI app for testing."""

from fastapi import FastAPI

# Super minimal app
minimal_app = FastAPI(
    title="AI Lawyer API - Test",
    docs_url="/docs",
    openapi_url="/openapi.json",
)

@minimal_app.get("/")
async def root():
    return {"status": "ok", "message": "Minimal API is working"}

@minimal_app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    print("Starting minimal test API on 0.0.0.0:8001...")
    uvicorn.run(minimal_app, host="0.0.0.0", port=8001)
