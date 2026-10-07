"""StepFun ASR API — FastAPI application entry point.

Usage:
    python -m app.main
    uvicorn app.main:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.config import settings
from app.database import close_database, get_session_factory, init_database, is_database_ready
from app.middleware import RequestLoggingMiddleware
from app.routes.health import router as health_router
from app.routes.transcription import limiter, router as transcription_router
from app.routes.corpus import router as corpus_router
from app.services.transcriber import TranscriptionWorker
from app.worker import TaskScheduler


# --- Logging Setup ---


def setup_logging() -> None:
    """Configure structlog for structured JSON logging in production
    or colored console output in development.
    """
    timestamper = structlog.processors.TimeStamper(fmt="iso")

    if settings.log_format == "console":
        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.ConsoleRenderer(),
        ]
    else:
        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            timestamper,
            structlog.processors.JSONRenderer(),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Silence noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)


# --- Lifespan ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: init DB, start cloud worker and scheduler, cleanup."""
    logger = structlog.get_logger(__name__)

    logger.info("Starting StepFun ASR API server")

    # --- Database ---
    if settings.database_enabled:
        logger.info("Initializing database")
        await init_database()
        logger.info("Database connected")
    else:
        logger.info("Database not configured — running in stateless mode")

    # --- Cloud worker (the app can start without a key) ---
    worker = TranscriptionWorker(settings=settings)
    await worker.start()
    app.state.worker = worker

    # --- Task Scheduler (DB-backed mode only) ---
    scheduler = None
    if settings.database_enabled:
        session_factory = get_session_factory()
        if session_factory is not None:
            scheduler = TaskScheduler(session_factory, worker, settings=settings)
            await scheduler.start()
            app.state.scheduler = scheduler
            logger.info("Task scheduler started")
        else:
            logger.warning("Database configured but session factory not available")

    # --- Log startup ---
    logger.info(
        "Server ready",
        host=settings.host,
        port=settings.port,
        model=settings.asr_model,
        cloud_configured=settings.cloud_configured,
        database_enabled=settings.database_enabled,
        rate_limit=f"{settings.rate_limit_rpm}/min" if settings.rate_limit_rpm > 0 else "disabled",

    )

    try:
        yield
    finally:
        logger.info("Shutting down server")
        if scheduler is not None:
            await scheduler.stop()
        await worker.stop()
        if settings.database_enabled:
            await close_database()
        logger.info("Server stopped")


# --- App Factory ---


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    setup_logging()

    app = FastAPI(
        title="StepFun ASR API",
        description="Speech-to-text API backed by StepFun cloud speech",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.log_format == "console" else None,
        redoc_url=None,
    )

    # --- Middleware ---
    # Order matters: last added = outermost (executed first)

    # CORS — explicit origins only; W3C forbids allow_credentials=True with wildcard
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True if settings.cors_origins_list != ["*"] else False,
        allow_methods=["GET", "POST", "OPTIONS"] if settings.cors_origins_list != ["*"] else ["*"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"] if settings.cors_origins_list != ["*"] else ["*"],
    )

    # Request logging (innermost — wraps the actual handler)
    app.add_middleware(RequestLoggingMiddleware)

    # --- Rate Limiter ---
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # --- Routes ---
    app.include_router(health_router)
    app.include_router(transcription_router)
    app.include_router(corpus_router)

    # --- Global Exception Handler ---
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Catch-all exception handler for unhandled errors."""
        logger = structlog.get_logger(__name__)
        logger.exception(
            "Unhandled exception",
            path=request.url.path,
            method=request.method,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": "Internal server error",
                    "type": "internal_error",
                    "code": None,
                    "param": None,
                }
            },
        )

    return app


# --- Module-level app instance (for uvicorn) ---

app = create_app()


# --- Direct run ---

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        reload=False,
    )
