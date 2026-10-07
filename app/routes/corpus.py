"""Corpus & task management endpoints — the DB-backed ASR workflow.

All endpoints return 503 when DATABASE_URL is not configured.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db, is_database_ready
from app.schemas.requests import CorpusCreateRequest, CorpusListParams, TaskListParams
from app.schemas.responses import (
    AsrTaskResponse,
    AsrTaskSummary,
    CorpusListResponse,
    CorpusResponse,
    ErrorResponse,
    RecognizeResponse,
    TaskListResponse,
)
from app.services.audio import cleanup_temp_file
from app.services.upload_pipeline import process_upload, validate_content_length
from app.services.corpus import (
    compute_md5,
    create_corpus,
    create_task,
    find_corpus_by_md5,
    find_successful_task,
    get_corpus,
    get_task,
    list_corpora,
    list_tasks,
    store_file,
    update_corpus_status,
)

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/asr", tags=["corpus"])


def _require_db():
    """Raise 503 if database is not configured."""
    if not is_database_ready():
        raise HTTPException(
            status_code=503,
            detail="Database not configured. Set DATABASE_URL to enable.",
        )


def _build_corpus_response(corpus) -> CorpusResponse:
    """Build a CorpusResponse from a Corpus ORM instance."""
    tasks = []
    if corpus.asr_tasks:
        tasks = [
            AsrTaskSummary(
                id=t.id,
                status=t.status,
                asr_engine=t.asr_engine,
                result_text=t.result_text,
                confidence=float(t.confidence) if t.confidence is not None else None,
                error_message=t.error_message,
                created_at=t.created_at,
                completed_at=t.completed_at,
            )
            for t in corpus.asr_tasks
        ]
    return CorpusResponse(
        id=corpus.id,
        file_md5=corpus.file_md5,
        file_name=corpus.file_name,
        file_size=corpus.file_size,
        duration=corpus.duration,
        sample_rate=corpus.sample_rate,
        channels=corpus.channels,
        language=corpus.language,
        status=corpus.status,
        text_content=corpus.text_content,
        business_id=corpus.business_id,
        business_type=corpus.business_type,
        tags=corpus.tags or [],
        is_deleted=corpus.is_deleted,
        created_at=corpus.created_at,
        updated_at=corpus.updated_at,
        tasks=tasks,
    )


# ---- Corpus Endpoints ----


@router.post(
    "/corpus",
    response_model=RecognizeResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def upload_corpus(
    request: Request,
    file: UploadFile = File(..., description="Audio file to transcribe"),
    language: str = Form(default="zh-CN", description="Language code (e.g. 'zh-CN', 'en-US')."),
    business_id: str | None = Form(default=None, description="Business ID (e.g., patient ID)."),
    business_type: str | None = Form(default=None, description="Business type."),
    tags: str = Form(default="", description="Comma-separated tags."),
    asr_engine: str = Form(default="STEPFUN", description="ASR engine; this fork supports STEPFUN only."),
    db: AsyncSession = Depends(get_db),
) -> RecognizeResponse:
    """Upload an audio file for ASR processing.

    Creates a corpus record and a PENDING ASR task. If the same file (by MD5)
    has already been processed successfully, returns the cached result immediately.

    The task will be picked up by the background TaskScheduler for processing.
    """
    _require_db()
    if asr_engine != "STEPFUN":
        raise HTTPException(status_code=400, detail="Only STEPFUN is supported for new tasks.")
    if not request.app.state.worker.is_ready:
        raise HTTPException(status_code=503, detail="Cloud speech is not ready. Set OPENAI_API_KEY.")

    # Parse tags
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

    # Pre-check Content-Length before reading body
    validate_content_length(request)

    # Save upload to temp, probe duration, and decode (keep temp file for storage)
    try:
        decoded = await process_upload(file, cleanup=False)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Upload processing failed")
        raise HTTPException(status_code=500, detail="Failed to process audio upload")

    try:
        # Compute MD5
        md5 = await compute_md5(decoded.temp_path)

        # Dedup check
        existing = await find_corpus_by_md5(db, md5)
        if existing is not None:
            # Check for an existing successful task
            cached_task = await find_successful_task(db, existing.id, settings.asr_model)
            if cached_task is not None:
                cleanup_temp_file(decoded.temp_path)
                await db.commit()
                logger.info("Dedup hit — returning cached result", corpus_id=existing.id, md5=md5)
                return RecognizeResponse(
                    corpus_id=existing.id,
                    task_id=cached_task.id,
                    file_md5=md5,
                    cached=True,
                    status="SUCCESS",
                    result_text=cached_task.result_text,
                )

            # MD5 exists but no successful task — create a new task for retry
            corpus = existing
            logger.info("Dedup hit but no success — creating retry task", corpus_id=corpus.id, md5=md5)
        else:
            # New file — use duration from process_upload
            duration_ms = int(decoded.duration * 1000)

            # Store file in permanent storage
            storage_dir = settings.storage_path_resolved
            storage_dir.mkdir(parents=True, exist_ok=True)
            dest_file_name = f"{md5}{decoded.temp_path.suffix}"
            dest_path = await store_file(decoded.temp_path, storage_dir, dest_file_name)

            file_size = dest_path.stat().st_size

            # Create corpus
            corpus = await create_corpus(
                db,
                file_md5=md5,
                file_name=file.filename or "unknown",
                file_path=str(dest_path),
                file_size=file_size,
                duration=duration_ms,
                channels=1,
                language=language,
                business_id=business_id,
                business_type=business_type,
                tags=tag_list,
            )

            # Update status to UPLOADED (file is now in permanent storage)
            await update_corpus_status(db, corpus, "UPLOADED")

        # Create ASR task
        task = await create_task(
            db,
            corpus_id=corpus.id,
            asr_engine="STEPFUN",
            engine_config={"model": settings.asr_model, "protocol": "chat_audio"},
        )

        await db.commit()

        logger.info(
            "Corpus upload complete",
            corpus_id=corpus.id,
            task_id=task.id,
            md5=md5,
            cached=False,
        )

        return RecognizeResponse(
            corpus_id=corpus.id,
            task_id=task.id,
            file_md5=md5,
            cached=False,
            status="PENDING",
        )

    finally:
        # Clean up temp file only if it still exists (may have been moved)
        cleanup_temp_file(decoded.temp_path)


@router.get(
    "/corpus",
    response_model=CorpusListResponse,
    responses={503: {"model": ErrorResponse}},
)
async def list_corpora_endpoint(
    business_id: str | None = Query(default=None),
    business_type: str | None = Query(default=None),
    status: str | None = Query(default=None),
    is_deleted: bool = Query(default=False),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> CorpusListResponse:
    """List corpora with optional filters and pagination."""
    _require_db()

    params = CorpusListParams(
        business_id=business_id,
        business_type=business_type,
        status=status,
        is_deleted=is_deleted,
        page=page,
        page_size=page_size,
    )
    items, total = await list_corpora(db, params)

    return CorpusListResponse(
        items=[_build_corpus_response(c) for c in items],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/corpus/{corpus_id}",
    response_model=CorpusResponse,
    responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def get_corpus_endpoint(
    corpus_id: int,
    db: AsyncSession = Depends(get_db),
) -> CorpusResponse:
    """Get a corpus by ID with its ASR task history."""
    _require_db()

    corpus = await get_corpus(db, corpus_id)
    if corpus is None:
        raise HTTPException(status_code=404, detail=f"Corpus {corpus_id} not found")

    return _build_corpus_response(corpus)


# ---- Task Endpoints ----


@router.get(
    "/tasks",
    response_model=TaskListResponse,
    responses={503: {"model": ErrorResponse}},
)
async def list_tasks_endpoint(
    status: str | None = Query(default=None),
    corpus_id: int | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> TaskListResponse:
    """List ASR tasks with optional filters and pagination."""
    _require_db()

    items, total = await list_tasks(db, status=status, corpus_id=corpus_id, page=page, page_size=page_size)

    return TaskListResponse(
        items=[
            AsrTaskSummary(
                id=t.id,
                status=t.status,
                asr_engine=t.asr_engine,
                result_text=t.result_text,
                confidence=float(t.confidence) if t.confidence is not None else None,
                error_message=t.error_message,
                created_at=t.created_at,
                completed_at=t.completed_at,
            )
            for t in items
        ],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/tasks/{task_id}",
    response_model=AsrTaskResponse,
    responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def get_task_endpoint(
    task_id: int,
    db: AsyncSession = Depends(get_db),
) -> AsrTaskResponse:
    """Get a task by ID with full details."""
    _require_db()

    task = await get_task(db, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    return AsrTaskResponse(
        id=task.id,
        corpus_id=task.corpus_id,
        status=task.status,
        asr_engine=task.asr_engine,
        engine_config=task.engine_config or {},
        result_text=task.result_text,
        confidence=float(task.confidence) if task.confidence is not None else None,
        result_detail=task.result_detail,
        processing_time=task.processing_time,
        error_message=task.error_message,
        started_at=task.started_at,
        completed_at=task.completed_at,
        created_at=task.created_at,
        updated_at=task.updated_at,
    )
