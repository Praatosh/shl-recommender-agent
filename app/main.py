"""
SHL Assessment Recommender - FastAPI Application
==================================================
Production-ready FastAPI app with two endpoints:
  GET  /health - Health check
  POST /chat   - Conversational assessment recommender

Startup sequence:
1. Load configuration from environment
2. Load SHL catalog from JSON
3. Build or load FAISS vector index
4. Initialize LLM client and recommendation engine
5. Start accepting requests

Design choices:
- Lifespan context manager for clean startup/shutdown (FastAPI best practice)
- CORS middleware enabled for frontend integration
- Request timing middleware for performance monitoring
- Error handling with structured error responses
- Async endpoint with to_thread for blocking operations
"""

import time
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from app.config import get_settings
from app.schemas import ChatRequest, ChatResponse, HealthResponse
from app.logger import setup_logging, get_logger
from app.scraper import load_and_preprocess_catalog
from app.embeddings import get_vector_store
from app.engine import get_engine, reset_engine


# Initialize logging
setup_logging()
logger = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    Application lifespan manager.
    Handles startup initialization and shutdown cleanup.
    
    Startup sequence:
    1. Load catalog data
    2. Build or load FAISS index
    3. Initialize recommendation engine
    
    Uses lifespan context manager (preferred over on_event decorators)
    per FastAPI best practices.
    """
    settings = get_settings()
    logger.info("=" * 60)
    logger.info("SHL Assessment Recommender - Starting up")
    logger.info("=" * 60)

    # Step 1: Load catalog
    catalog_path = settings.catalog_path
    if not os.path.isabs(catalog_path):
        catalog_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), catalog_path)

    catalog = load_and_preprocess_catalog(catalog_path)
    if not catalog:
        logger.error("No catalog items loaded! Check catalog_path in settings.")
        logger.error(f"Looked at: {catalog_path}")

    # Step 2: Build or load FAISS index
    vector_store = get_vector_store()
    index_path = settings.faiss_index_path
    if not os.path.isabs(index_path):
        index_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), index_path)

    if not vector_store.load_index(index_path):
        logger.info("Building FAISS index from catalog...")
        vector_store.build_index(catalog)
        vector_store.save_index(index_path)
        logger.info("FAISS index built and saved")
    else:
        logger.info("FAISS index loaded from disk")

    # Step 3: Initialize engine
    reset_engine()
    engine = get_engine()
    logger.info(f"Recommendation engine initialized with {len(vector_store.catalog_items)} assessments")
    logger.info(f"LLM provider: {settings.llm_provider}")
    logger.info("=" * 60)
    logger.info("SHL Assessment Recommender - Ready!")
    logger.info("=" * 60)

    yield

    # Shutdown
    logger.info("SHL Assessment Recommender - Shutting down")


# Create FastAPI app
app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational AI agent that helps recruiters discover relevant SHL assessments through dialogue.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Request Timing Middleware
# ============================================================
@app.middleware("http")
async def request_timing_middleware(request: Request, call_next):
    """
    Log request timing for performance monitoring.
    
    Why middleware instead of per-endpoint timing?
    - Captures ALL requests uniformly, including errors
    - Includes middleware processing time in the measurement
    - Follows the FastAPI skill recommendation for middleware-based logging
    """
    start_time = time.time()
    response = await call_next(request)
    elapsed = time.time() - start_time

    # Only log non-health requests to avoid noise
    if request.url.path != "/health":
        logger.info(
            f"{request.method} {request.url.path} -> {response.status_code} ({elapsed:.2f}s)"
        )

    # Add timing header for debugging
    response.headers["X-Response-Time"] = f"{elapsed:.3f}s"
    return response


# ============================================================
# API Endpoints
# ============================================================

@app.get("/", include_in_schema=False)
async def root():
    """Redirect base URL to the Swagger documentation."""
    return RedirectResponse(url="/docs")

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    Returns service status and version.
    Used by deployment platforms (Render, etc.) for health monitoring.
    """
    return HealthResponse(status="healthy", version="1.0.0")


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Conversational assessment recommender endpoint.
    
    Accepts full conversation history (stateless design).
    Returns:
    - reply: Agent's response text
    - recommendations: List of SHL assessments (empty if clarifying)
    - end_of_conversation: True only after final confirmation
    
    The API is stateless - every request must contain the full conversation history.
    
    Design note: Uses async handler with to_thread internally to avoid
    blocking the event loop during LLM API calls and embedding operations.
    """
    try:
        # Get the recommendation engine
        engine = get_engine()

        # Process the chat asynchronously (wraps blocking calls)
        response = await engine.process_chat_async(request.messages)

        logger.info(
            f"Chat | Messages: {len(request.messages)} | "
            f"Recommendations: {len(response.recommendations)} | "
            f"EOC: {response.end_of_conversation}"
        )

        return response

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)

        # Return a graceful error response instead of 500
        return ChatResponse(
            reply="I apologize, but I encountered an issue processing your request. Could you please try rephrasing your question about SHL assessments?",
            recommendations=[],
            end_of_conversation=False,
        )


# ============================================================
# Error Handlers
# ============================================================

@app.exception_handler(422)
async def validation_error_handler(request, exc):
    """Handle Pydantic validation errors with helpful messages."""
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Invalid request format. Please ensure your request matches the expected schema.",
            "error": str(exc),
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle internal server errors gracefully."""
    logger.error(f"Internal error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error. Please try again.",
        }
    )
