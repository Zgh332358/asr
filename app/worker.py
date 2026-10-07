"""Task scheduler — polls the database for PENDING ASR tasks and processes them.

The TaskScheduler runs as a background asyncio task. It periodically queries
the database for PENDING tasks, atomically claims them, and delegates the full
processing pipeline to TaskProcessor.

This sits ABOVE the TranscriptionWorker — the worker remains a bounded cloud queue.
The scheduler is the DB-aware orchestrator that feeds it.
"""

from __future__ import annotations

import asyncio

import structlog
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from app.config import Settings
from app.services.task_processor import TaskProcessor
from app.services.transcriber import TranscriptionWorker

logger = structlog.get_logger(__name__)


class TaskScheduler:
    """Background task that polls the DB for PENDING ASR tasks.

    For each pending task found:
    1. Atomically claim (UPDATE status='PROCESSING', started_at=NOW())
    2. Delegate to TaskProcessor for audio load → decode → transcribe → store → callback
    3. On failure: mark FAILED; a new upload can create a retry task

    The scheduler respects max_concurrent_tasks — it won't claim more tasks
    than the configured limit.

    Usage:
        scheduler = TaskScheduler(session_factory, worker)
        await scheduler.start()
        ...
        await scheduler.stop()
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
        self._processor = TaskProcessor(session_factory, worker, settings)
        self._running = False
        self._task: asyncio.Task | None = None
        self._claim_task: asyncio.Task | None = None
        self._active_count = 0
        self._active_tasks: set[asyncio.Task] = set()
        self._max_concurrent = max(1, self._settings.max_concurrent_tasks)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def active_count(self) -> int:
        return self._active_count

    async def start(self) -> None:
        """Start the polling loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        logger.info(
            "Task scheduler started",
            poll_interval=self._settings.task_poll_interval,
            max_concurrent=self._max_concurrent,
        )

    async def stop(self) -> None:
        """Gracefully stop the scheduler. Waits for in-flight tasks to complete."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        # A committed claim must be registered before we cancel processing jobs.
        # The polling task awaits this critical section through shield().
        if self._claim_task is not None:
            await self._claim_task
            self._claim_task = None
        # Let newly created jobs enter the processor's cancellation handler.
        await asyncio.sleep(0)
        # Cancel and await jobs so their processor can persist interruption as FAILED.
        for task in list(self._active_tasks):
            task.cancel()
        if self._active_tasks:
            await asyncio.gather(*self._active_tasks, return_exceptions=True)
        logger.info("Task scheduler stopped", remaining_tasks=self._active_count)

    async def _poll_loop(self) -> None:
        """Main loop: poll DB, claim tasks, process them."""
        while self._running:
            try:
                # Check for PENDING tasks if we have capacity
                while self._worker.is_ready and self._active_count < self._max_concurrent:
                    task_claimed = await self._try_claim_and_process()
                    if not task_claimed:
                        break  # No more PENDING tasks

                await asyncio.sleep(self._settings.task_poll_interval)

            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Scheduler poll error — will retry")
                await asyncio.sleep(self._settings.task_poll_interval)

    async def _try_claim_and_process(self) -> bool:
        """Try to claim one PENDING task and start processing it.

        Returns True if a task was claimed, False if none available.
        """
        if not self._worker.is_ready:
            return False
        self._claim_task = asyncio.create_task(self._claim_and_register())
        try:
            return await asyncio.shield(self._claim_task)
        finally:
            if self._claim_task is not None and self._claim_task.done():
                self._claim_task = None

    async def _claim_and_register(self) -> bool:
        """Never cancel between committing a claim and tracking its processor."""
        claimed = await self._processor.try_claim()
        if claimed is None:
            return False

        task_id, corpus_id, audio_path, language = claimed
        self._active_count += 1
        task = asyncio.create_task(self._process_and_track(task_id, corpus_id, audio_path, language))
        self._active_tasks.add(task)
        task.add_done_callback(self._active_tasks.discard)
        return True

    async def _process_and_track(
        self,
        task_id: int,
        corpus_id: int,
        audio_path: str,
        language: str,
    ) -> None:
        """Delegate to TaskProcessor and manage active_count tracking."""
        try:
            await self._processor.process(task_id, corpus_id, audio_path, language)
        finally:
            self._active_count -= 1
