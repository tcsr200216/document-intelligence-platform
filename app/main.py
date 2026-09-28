from fastapi import FastAPI

app = FastAPI(
    title="Document Intelligence Platform",
    version="0.1.0",
    description="Production-oriented document ingestion and retrieval API.",
)


@app.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    """Return a lightweight liveness response for local and container health checks."""
    return {"status": "ok"}


@app.get("/", tags=["system"])
async def root() -> dict[str, str]:
    return {
        "service": "document-intelligence-platform",
        "status": "running",
    }
