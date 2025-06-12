"""
Cerby Identity Automation Platform - Main Application Entry Point

This module initializes the FastAPI application with comprehensive observability
through Logfire, sets up middleware, and configures the core API structure.
"""

import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import logfire

# Load environment variables
load_dotenv()

# Configure Logfire for observability
logfire.configure(
    token=os.getenv("LOGFIRE_TOKEN", ""),
    project_name=os.getenv("LOGFIRE_PROJECT_NAME", "cerby-identity-automation"),
    service_name=os.getenv("LOGFIRE_SERVICE_NAME", "cerby-api"),
    environment=os.getenv("LOGFIRE_ENVIRONMENT", "development")
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle events.

    Handles startup and shutdown procedures including database connections,
    background tasks, and resource cleanup.
    """
    # Startup
    logfire.info(
        "Application starting up",
        environment=os.getenv("ENVIRONMENT", "development"),
        version="1.0.0"
    )

    # Initialize database connections
    # TODO: Add database initialization

    # Start background tasks
    # TODO: Add background task initialization

    yield

    # Shutdown
    logfire.info("Application shutting down")

    # Clean up resources
    # TODO: Add cleanup procedures


# Create FastAPI application instance
app = FastAPI(
    title="Cerby Identity Automation Platform",
    description=(
        "A demonstration platform showcasing automated identity management "
        "across disconnected SaaS applications with genetic algorithm policy optimization"
    ),
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan
)

# Enable Logfire instrumentation
logfire.instrument_fastapi(app, capture_headers=True)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time", "X-Request-ID"]
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """
    Add processing time header and request tracking.

    Measures request processing time and adds it to response headers
    for performance monitoring.
    """
    import time
    import uuid

    request_id = str(uuid.uuid4())
    start_time = time.time()

    # Add request ID to Logfire context
    with logfire.span(
        "HTTP Request",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        client_host=request.client.host if request.client else None
    ):
        response = await call_next(request)

        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Request-ID"] = request_id

        logfire.info(
            "Request completed",
            request_id=request_id,
            status_code=response.status_code,
            process_time=process_time
        )

        return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Handle HTTP exceptions with proper logging.
    """
    logfire.error(
        "HTTP Exception",
        status_code=exc.status_code,
        detail=exc.detail,
        path=request.url.path
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    Handle unexpected exceptions with proper error logging.
    """
    import traceback

    logfire.error(
        "Unhandled Exception",
        error=str(exc),
        error_type=type(exc).__name__,
        path=request.url.path,
        traceback=traceback.format_exc()
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# Root endpoint
@app.get("/", tags=["Health"])
async def root() -> Dict[str, Any]:
    """
    Root endpoint providing basic API information.
    """
    return {
        "message": "Cerby Identity Automation Platform API",
        "status": "operational",
        "version": "1.0.0",
        "docs": "/api/docs",
        "health": "/health"
    }


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, Any]:
    """
    Comprehensive health check endpoint.

    Returns detailed health status including service dependencies.
    """
    with logfire.span("Health check"):
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": os.getenv("LOGFIRE_SERVICE_NAME", "cerby-api"),
            "environment": os.getenv("ENVIRONMENT", "development"),
            "version": "1.0.0",
            "checks": {
                "api": "operational",
                "database": "operational",  # TODO: Add actual database check
                "cache": "operational",  # TODO: Add Redis check
                "genetic_algorithm": "operational"  # TODO: Add Darwin check
            }
        }

        # Check if any component is not operational
        if any(status != "operational" for status in health_status["checks"].values()):
            health_status["status"] = "degraded"
            logfire.warning("Health check failed", checks=health_status["checks"])
        else:
            logfire.info("Health check passed")

        return health_status


# Import and include routers
# TODO: Add routers as they are implemented
# from src.api.v1 import identity, policy, analytics, ingestion
# app.include_router(identity.router, prefix="/api/v1/identity", tags=["Identity"])
# app.include_router(policy.router, prefix="/api/v1/policy", tags=["Policy"])
# app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["Analytics"])
# app.include_router(ingestion.router, prefix="/api/v1/ingestion", tags=["Ingestion"])


if __name__ == "__main__":
    import uvicorn

    # Development server configuration
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("DEBUG", "true").lower() == "true",
        log_level="info"
    )
