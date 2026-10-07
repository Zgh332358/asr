"""Task processor — handles the full ASR task lifecycle: claim → load → decode → transcribe → store → callback."""

from __future__ import annotations

import asyncio
import os
import time

import structlog
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from app.config import Settings
from app.models.corpus import Corpus
from app.services.audio import decode_audio_ffmpeg
from app.services.callback import notify_main_backend
from app.services.corpus import (
    claim_pending_task,
    complete_task,
    fail_task,
    sync_text_content,
)
from app.services.transcriber import SpeechServiceError, TranscriptionWorker

logger = structlog.get_logger(__name__)


class TaskProcessor:
    """Processes ASR tasks end-to-end.

    Handles the full pipeline: claim → load audio → decode → transcribe → store → callback.
    This is the deep module behind the scheduler's polling seam.

    Usage:
        processor = TaskProcessor(session_factory, worker, settings)
        claimed = await processor.try_claim()          # → (task_id, corpus_id, audio_path, language) | None
        await processor.process(task_id, corpus_id, audio_path)
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        worker: TranscriptionWorker,
        settings: Settings | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._worker = worker
        self._settings = Settings.resolve(settings)

    async def try_claim(self) -> tuple[int, int, str, str] | None:
        """Atomically claim the oldest PENDING task and return its details.

        Returns:
            (task_id, corpus_id, audio_path, language) if a task was claimed, None otherwise.
        """
        try:
            async with self._session_factory() as db:
                task = await claim_pending_task(db)
                if task is None:
                    await db.commit()
                    return None

                # Do not run a newly configured model under an old task identity.
                if (task.asr_engine != "STEPFUN" or
                        (task.engine_config or {}).get("model") != self._settings.asr_model):
                    error = "Task provider/model no longer matches server configuration; upload again to create a new task."
                    await fail_task(db, task.id, error)
                    await db.commit()
                    await self._notify(task.id, task.corpus_id, "FAILED", error=error)
                    return None

                # Load corpus to get audio file path
                corpus = await db.get(Corpus, task.corpus_id)
                if corpus is None or corpus.file_path is None:
                    await fail_task(db, task.id, "Corpus record missing or file_path is null")
                    await db.commit()
                    await self._notify(task.id, task.corpus_id, "FAILED", error="Corpus missing")
                    return None

                task_id = task.id
                corpus_id = corpus.id
                audio_path = corpus.file_path
                language = corpus.language
                await db.commit()

            return (task_id, corpus_id, audio_path, language)

        except Exception:
            logger.exception("Failed to claim task")
            return None

    async def process(
        self,
        task_id: int,
        corpus_id: int,
        audio_path: str,
        language: str = "",
    ) -> None:
        """Load audio, submit to cloud worker, store results, trigger callback."""
        try:
            # Load and decode audio
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            audio = await asyncio.to_thread(decode_audio_ffmpeg, audio_path)

            # Submit to cloud worker (reuses the same asyncio.Queue)
            start = time.monotonic()
            result = await self._worker.submit(audio, language=language)
            elapsed_ms = int((time.monotonic() - start) * 1000)

            # Store results in a fresh session
            async with self._session_factory() as db:
                await complete_task(
                    db,
                    task_id=task_id,
                    result_text=result.text,
                    confidence=result.confidence,
                    result_detail={
                        "language": result.language,
                        "duration": result.duration,
                        "segments": result.segments,
                    },
                    processing_time=elapsed_ms,
                )
                await sync_text_content(db, corpus_id, result.text)
                await db.commit()

        except asyncio.CancelledError:
            await self._record_failure(task_id, corpus_id, "ASR task interrupted by service shutdown; please retry.", notify=False)
            raise
        except Exception as exc:
            logger.warning("Task processing failed", task_id=task_id, error_type=type(exc).__name__)
            # Only adapter-created errors may cross persistence/callback boundaries.
            error_msg = str(exc) if isinstance(exc, SpeechServiceError) else "ASR task failed or was interrupted; please retry."
            await self._record_failure(task_id, corpus_id, error_msg)
            return

        # A callback interruption must not turn an already committed SUCCESS into FAILED.
        await self._notify(task_id, corpus_id, "SUCCESS", result_text=result.text, confidence=result.confidence)

    async def _record_failure(self, task_id: int, corpus_id: int, error_msg: str, *, notify: bool = True) -> None:
        try:
            async with self._session_factory() as db:
                await fail_task(db, task_id=task_id, error_message=error_msg)
                await db.commit()
        except Exception:
            logger.warning("Failed to update task as FAILED", task_id=task_id)
        if notify:
            await self._notify(task_id, corpus_id, "FAILED", error=error_msg)

    async def _notify(
        self,
        task_id: int,
        corpus_id: int,
        status: str,
        result_text: str | None = None,
        confidence: float | None = None,
        error: str | None = None,
    ) -> None:
        """Fire-and-forget callback to main backend."""
        if not self._settings.main_backend_callback_url.strip():
            return
        try:
            await notify_main_backend(
                task_id=task_id,
                corpus_id=corpus_id,
                status=status,
                result_text=result_text,
                confidence=confidence,
                error_message=error,
                callback_url=self._settings.main_backend_callback_url,
                max_retries=self._settings.callback_max_retries,
                retry_base_delay=self._settings.callback_retry_base_delay,
            )
        except Exception:
            logger.exception("Callback failed unexpectedly", task_id=task_id)