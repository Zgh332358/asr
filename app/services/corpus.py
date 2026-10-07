"""Corpus business logic — MD5 hashing, dedup, file storage, CRUD operations."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import structlog
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.corpus import Corpus
from app.models.asr_task import AsrTask
from app.schemas.requests import CorpusListParams

logger = structlog.get_logger(__name__)

# 1 MB read buffer for MD5 computation
_MD5_CHUNK_SIZE = 1024 * 1024


async def compute_md5(file_path: Path) -> str:
    """Compute the MD5 hash of a file in 1 MB chunks.

    Avoids loading the entire file into memory — safe for large audio files.
    """
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(_MD5_CHUNK_SIZE):
            md5.update(chunk)
    return md5.hexdigest()


async def find_corpus_by_md5(db: AsyncSession, md5: str) -> Corpus | None:
    """Find an existing corpus record by MD5 hash. Excludes soft-deleted records."""
    result = await db.execute(
        select(Corpus).where(
            Corpus.file_md5 == md5,
            Corpus.is_deleted == False,  # noqa: E712
        )
    )
    return result.scalar_one_or_none()


async def find_successful_task(db: AsyncSession, corpus_id: int, model: str) -> AsrTask | None:
    """Find a successful result for this corpus and the configured provider/model."""
    result = await db.execute(
        select(AsrTask)
        .where(
            AsrTask.corpus_id == corpus_id,
            AsrTask.status == "SUCCESS",
            AsrTask.asr_engine == "STEPFUN",
            AsrTask.engine_config["model"].astext == model,
        )
        .order_by(AsrTask.completed_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def create_corpus(
    db: AsyncSession,
    file_md5: str,
    file_name: str,
    file_path: str | None = None,
    file_size: int | None = None,
    duration: int | None = None,
    sample_rate: int | None = None,
    channels: int = 1,
    language: str = "zh-CN",
    business_id: str | None = None,
    business_type: str | None = None,
    tags: list | None = None,
) -> Corpus:
    """Create a new corpus record with status UPLOADING."""
    corpus = Corpus(
        file_md5=file_md5,
        file_name=file_name,
        file_path=file_path,
        file_size=file_size,
        duration=duration,
        sample_rate=sample_rate,
        channels=channels,
        language=language,
        status="UPLOADING",
        business_id=business_id,
        business_type=business_type,
        tags=tags or [],
    )
    db.add(corpus)
    await db.flush()
    await db.refresh(corpus)
    logger.info("Corpus created", corpus_id=corpus.id, file_md5=file_md5)
    return corpus


async def update_corpus_status(
    db: AsyncSession,
    corpus: Corpus,
    status: str,
    **extra,
) -> Corpus:
    """Update a corpus record's status and optional extra fields."""
    corpus.status = status
    for key, value in extra.items():
        if hasattr(corpus, key) and value is not None:
            setattr(corpus, key, value)
    await db.flush()
    logger.info("Corpus updated", corpus_id=corpus.id, status=status)
    return corpus


async def sync_text_content(db: AsyncSession, corpus_id: int, text: str) -> None:
    """Denormalize the ASR result text onto the corpus record for easy queries."""
    corpus = await db.get(Corpus, corpus_id)
    if corpus:
        corpus.text_content = text
        await db.flush()


async def get_corpus(db: AsyncSession, corpus_id: int) -> Corpus | None:
    """Get a corpus by ID, excluding soft-deleted records."""
    result = await db.execute(
        select(Corpus).where(
            Corpus.id == corpus_id,
            Corpus.is_deleted == False,  # noqa: E712
        )
    )
    return result.scalar_one_or_none()


async def list_corpora(
    db: AsyncSession,
    params: CorpusListParams,
) -> tuple[list[Corpus], int]:
    """List corpora with optional filters and pagination.

    Returns:
        Tuple of (items, total_count).
    """
    # Build base query
    conditions = [Corpus.is_deleted == params.is_deleted]  # noqa: E712
    if params.business_id:
        conditions.append(Corpus.business_id == params.business_id)
    if params.business_type:
        conditions.append(Corpus.business_type == params.business_type)
    if params.status:
        conditions.append(Corpus.status == params.status)

    # Count total
    count_query = select(func.count()).select_from(Corpus)
    for cond in conditions:
        count_query = count_query.where(cond)
    total = (await db.execute(count_query)).scalar() or 0

    # Fetch page
    query = select(Corpus).order_by(Corpus.created_at.desc())
    for cond in conditions:
        query = query.where(cond)
    offset = (params.page - 1) * params.page_size
    query = query.offset(offset).limit(params.page_size)
    items = list((await db.execute(query)).scalars().all())

    return items, total


async def create_task(
    db: AsyncSession,
    corpus_id: int,
    asr_engine: str = "STEPFUN",
    engine_config: dict | None = None,
) -> AsrTask:
    """Create a new ASR task in PENDING status."""
    task = AsrTask(
        corpus_id=corpus_id,
        status="PENDING",
        asr_engine=asr_engine,
        engine_config=engine_config or {},
    )
    db.add(task)
    await db.flush()
    await db.refresh(task)
    logger.info("ASR task created", task_id=task.id, corpus_id=corpus_id)
    return task


async def get_task(db: AsyncSession, task_id: int) -> AsrTask | None:
    """Get an ASR task by ID."""
    return await db.get(AsrTask, task_id)


async def list_tasks(
    db: AsyncSession,
    status: str | None = None,
    corpus_id: int | None = None,
    page: int = 1,
    page_size: int = 20,
) -> tuple[list[AsrTask], int]:
    """List ASR tasks with optional filters and pagination."""
    conditions = []
    if status:
        conditions.append(AsrTask.status == status)
    if corpus_id:
        conditions.append(AsrTask.corpus_id == corpus_id)

    # Count
    count_query = select(func.count()).select_from(AsrTask)
    for cond in conditions:
        count_query = count_query.where(cond)
    total = (await db.execute(count_query)).scalar() or 0

    # Fetch
    query = select(AsrTask).order_by(AsrTask.created_at.desc())
    for cond in conditions:
        query = query.where(cond)
    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size)
    items = list((await db.execute(query)).scalars().all())

    return items, total


async def claim_pending_task(db: AsyncSession) -> AsrTask | None:
    """Atomically claim the oldest PENDING task.

    Uses SELECT ... FOR UPDATE SKIP LOCKED for safe horizontal scaling.
    Returns None if no PENDING tasks exist.
    """
    result = await db.execute(
        select(AsrTask)
        .where(AsrTask.status == "PENDING")
        .order_by(AsrTask.created_at.asc())
        .limit(1)
        .with_for_update(skip_locked=True)
    )
    task = result.scalar_one_or_none()
    if task is None:
        return None

    import datetime
    task.status = "PROCESSING"
    task.started_at = datetime.datetime.now(datetime.timezone.utc)
    await db.flush()
    return task


async def complete_task(
    db: AsyncSession,
    task_id: int,
    result_text: str,
    confidence: float | None = None,
    result_detail: dict | None = None,
    processing_time: int | None = None,
) -> AsrTask:
    """Mark a task as SUCCESS with results.

    Args:
        db: Database session.
        task_id: The task ID to complete.
    """
    import datetime
    task = await db.get(AsrTask, task_id)
    if task is None:
        raise ValueError(f"Task {task_id} not found")
    task.status = "SUCCESS"
    task.result_text = result_text
    task.confidence = confidence
    task.result_detail = result_detail
    task.processing_time = processing_time
    task.completed_at = datetime.datetime.now(datetime.timezone.utc)
    await db.flush()
    logger.info("Task completed", task_id=task_id, corpus_id=task.corpus_id)
    return task


async def fail_task(
    db: AsyncSession,
    task_id: int,
    error_message: str,
) -> AsrTask:
    """Mark a task as FAILED with an error message.

    Args:
        db: Database session.
        task_id: The task ID to fail.
    """
    import datetime
    task = await db.get(AsrTask, task_id)
    if task is None:
        raise ValueError(f"Task {task_id} not found")
    task.status = "FAILED"
    task.error_message = error_message
    task.completed_at = datetime.datetime.now(datetime.timezone.utc)
    await db.flush()
    logger.warning("Task failed", task_id=task_id, error=error_message)
    return task


async def store_file(tmp_path: Path, storage_dir: Path, file_name: str) -> Path:
    """Move a file from temp location to permanent storage.

    Returns the destination path. Preserves the original file extension.
    """
    storage_dir.mkdir(parents=True, exist_ok=True)
    dest_path = storage_dir / file_name
    shutil.move(str(tmp_path), str(dest_path))
    logger.info("File stored", dest=str(dest_path))
    return dest_path
