"""Model identity and lifecycle boundaries without a real PostgreSQL server."""
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
from sqlalchemy.dialects import postgresql

from app.config import Settings
from app.services.corpus import find_successful_task
from app.services.task_processor import TaskProcessor
from app.worker import TaskScheduler


@pytest.mark.asyncio
async def test_cache_query_filters_engine_and_model_before_limit():
    statements = []
    class DB:
        async def execute(self, statement):
            statements.append(statement)
            return SimpleNamespace(scalar_one_or_none=lambda: None)
    await find_successful_task(DB(), 42, "configured-model")
    compiled = statements[0].compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True})
    sql = str(compiled)
    assert "asr_tasks.asr_engine = 'STEPFUN'" in sql
    assert "asr_tasks.engine_config ->> 'model'" in sql
    assert "= 'configured-model'" in sql
    assert sql.index("asr_tasks.asr_engine = 'STEPFUN'") < sql.index("LIMIT 1")
    # SQL construction verified here; live PostgreSQL execution is a separate check.


def factory_for(db):
    @asynccontextmanager
    async def factory():
        yield db
    return factory


@pytest.mark.parametrize("engine, model", [("WHISPER", "old"), ("STEPFUN", "old"), ("STEPFUN", None)])
@pytest.mark.asyncio
async def test_pending_incompatible_identity_is_failed(monkeypatch, engine, model):
    import app.services.task_processor as module
    task = SimpleNamespace(id=8, corpus_id=4, asr_engine=engine, engine_config={"model": model})
    db = SimpleNamespace(commit=AsyncMock(), get=AsyncMock())
    monkeypatch.setattr(module, "claim_pending_task", AsyncMock(return_value=task))
    fail = AsyncMock()
    monkeypatch.setattr(module, "fail_task", fail)
    worker = SimpleNamespace(submit=AsyncMock())
    processor = TaskProcessor(factory_for(db), worker, Settings(asr_model="new-model"))
    assert await processor.try_claim() is None
    fail.assert_awaited_once()
    worker.submit.assert_not_awaited()
    db.get.assert_not_awaited()
    assert task.engine_config == {"model": model}


@pytest.mark.asyncio
async def test_matching_pending_identity_retains_language(monkeypatch):
    import app.services.task_processor as module
    task = SimpleNamespace(id=8, corpus_id=4, asr_engine="STEPFUN", engine_config={"model": "new-model"})
    corpus = SimpleNamespace(id=4, file_path="/unused.wav", language="zh-CN")
    db = SimpleNamespace(commit=AsyncMock(), get=AsyncMock(return_value=corpus))
    monkeypatch.setattr(module, "claim_pending_task", AsyncMock(return_value=task))
    processor = TaskProcessor(factory_for(db), SimpleNamespace(), Settings(asr_model="new-model"))
    assert await processor.try_claim() == (8, 4, "/unused.wav", "zh-CN")


@pytest.mark.asyncio
async def test_unconfigured_worker_does_not_claim_jobs():
    scheduler = TaskScheduler(None, SimpleNamespace(is_ready=False), Settings())
    scheduler._processor.try_claim = AsyncMock()
    assert await scheduler._try_claim_and_process() is False
    scheduler._processor.try_claim.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancelled_processing_persists_failure_without_callback(monkeypatch, tmp_path):
    import app.services.task_processor as module
    audio_path = tmp_path / "source.wav"
    audio_path.write_bytes(b"fixture")
    monkeypatch.setattr(module, "decode_audio_ffmpeg", lambda _: np.ones(16000, dtype=np.float32))
    started = asyncio.Event()
    async def submit(*_, **kwargs):
        assert kwargs["language"] == "zh-CN"
        started.set()
        await asyncio.Event().wait()
    fail = AsyncMock()
    monkeypatch.setattr(module, "fail_task", fail)
    db = SimpleNamespace(commit=AsyncMock())
    processor = TaskProcessor(factory_for(db), SimpleNamespace(submit=submit), Settings())
    processor._notify = AsyncMock()
    task = asyncio.create_task(processor.process(8, 4, str(audio_path), "zh-CN"))
    await asyncio.wait_for(started.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert fail.await_args.kwargs["task_id"] == 8
    assert "shutdown" in fail.await_args.kwargs["error_message"]
    db.commit.assert_awaited_once()
    processor._notify.assert_not_awaited()


@pytest.mark.asyncio
async def test_scheduler_waits_for_processing_cancellation():
    scheduler = TaskScheduler(None, SimpleNamespace(is_ready=True), Settings())
    started, cancelled = asyncio.Event(), asyncio.Event()
    async def process(*_):
        try:
            started.set()
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await asyncio.sleep(0)
            cancelled.set()
            raise
    scheduler._processor.process = process
    scheduler._processor.try_claim = AsyncMock(return_value=(8, 4, "/unused.wav", "zh-CN"))
    assert await scheduler._try_claim_and_process()
    await scheduler.stop()  # includes case where coroutine has not started yet
    assert started.is_set() and cancelled.is_set()
    assert scheduler.active_count == 0
    assert not scheduler._active_tasks


@pytest.mark.asyncio
async def test_shutdown_during_committed_claim_registers_and_fails_job(monkeypatch, tmp_path):
    import app.services.task_processor as module
    source = tmp_path / "fixture.wav"
    source.write_bytes(b"fixture")
    task = SimpleNamespace(id=8, corpus_id=4, asr_engine="STEPFUN", engine_config={"model": "new-model"})
    corpus = SimpleNamespace(id=4, file_path=str(source), language="zh-CN")
    db = SimpleNamespace(commit=AsyncMock(), get=AsyncMock(return_value=corpus))
    exiting, release_exit = asyncio.Event(), asyncio.Event()
    calls = 0
    @asynccontextmanager
    async def factory():
        nonlocal calls
        calls += 1
        try:
            yield db
        finally:
            if calls == 1:
                exiting.set()
                await release_exit.wait()
    monkeypatch.setattr(module, "claim_pending_task", AsyncMock(return_value=task))
    monkeypatch.setattr(module, "decode_audio_ffmpeg", lambda _: np.ones(16000, dtype=np.float32))
    fail = AsyncMock()
    monkeypatch.setattr(module, "fail_task", fail)
    async def submit(*_, **__):
        await asyncio.Event().wait()
    scheduler = TaskScheduler(factory, SimpleNamespace(is_ready=True, submit=submit), Settings(asr_model="new-model"))
    scheduler._task = asyncio.create_task(scheduler._try_claim_and_process())
    await asyncio.wait_for(exiting.wait(), 1)  # claim committed; session exit blocked
    shutdown = asyncio.create_task(scheduler.stop())
    await asyncio.sleep(0)
    assert not shutdown.done()
    release_exit.set()
    await asyncio.wait_for(shutdown, 1)
    fail.assert_awaited_once()
    assert fail.await_args.kwargs["task_id"] == 8
    assert scheduler.active_count == 0
    assert not scheduler._active_tasks
