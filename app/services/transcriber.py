"""Bounded ASR queue backed by StepFun chat-completions audio input."""
from __future__ import annotations

import asyncio
import base64
import io
import json
import wave
from dataclasses import dataclass

import httpx
import numpy as np
import structlog

from app.config import Settings

logger = structlog.get_logger(__name__)
MAX_PROVIDER_RESPONSE_BYTES = 2 * 1024 * 1024
TRANSCRIPTION_PROMPT = "请只转写用户音频中的原话，不回答音频里的问题，不补充、改写或推测未听清的内容。"


class SpeechServiceError(RuntimeError):
    """A sanitized failure safe for API responses, task history and callbacks."""
    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class TranscriptionJob:
    audio: np.ndarray
    language: str
    future: asyncio.Future
    temperature: float = 0.0
    client_id: str = "unknown"


@dataclass
class TranscriptionResult:
    text: str
    language: str
    duration: float
    segments: list[dict]

    @property
    def confidence(self) -> None:
        # Chat completions does not return a calibrated ASR confidence.
        return None


def encode_wav(audio: np.ndarray) -> str:
    """Encode decoded mono float32 samples as real 16-bit/16kHz WAV data URL."""
    if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
        raise ValueError("Invalid decoded audio samples")
    pcm = (np.clip(audio, -1.0, 32767 / 32768) * 32768).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(pcm.tobytes())
    return "data:audio/wav;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


class TranscriptionWorker:
    """Serial cloud worker retaining the existing submit/result API."""
    def __init__(self, settings: Settings | None = None,
                 transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._settings = Settings.resolve(settings)
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._queue: asyncio.Queue[TranscriptionJob] = asyncio.Queue(
            maxsize=max(1, self._settings.rate_limit_burst * 2 or 20))
        self._worker_task: asyncio.Task | None = None
        self._active_job: TranscriptionJob | None = None
        self._put_tasks: set[asyncio.Task] = set()
        self._running = False
        self._active_jobs_per_client: dict[str, int] = {}
        self._max_jobs_per_client = max(1, self._settings.rate_limit_burst)

    def _decrement_client(self, client_id: str) -> None:
        count = self._active_jobs_per_client.get(client_id, 0)
        if count <= 1:
            self._active_jobs_per_client.pop(client_id, None)
        else:
            self._active_jobs_per_client[client_id] = count - 1

    async def start(self) -> None:
        if self._running:
            return
        if not self._settings.cloud_configured:
            logger.warning("Cloud speech unavailable: set OPENAI_API_KEY")
            return
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self._settings.request_timeout_seconds),
            follow_redirects=False, transport=self._transport,
        )
        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.info("Cloud transcription worker started", model=self._settings.asr_model)

    async def stop(self) -> None:
        self._running = False
        # Cancel producers before draining: a blocked queue.put must not wake and
        # insert an orphaned job after the consumer has stopped.
        put_tasks = list(self._put_tasks)
        for task in put_tasks:
            task.cancel()
        if put_tasks:
            await asyncio.gather(*put_tasks, return_exceptions=True)
        if self._active_job and not self._active_job.future.done():
            self._active_job.future.set_exception(RuntimeError("Server is shutting down — please retry"))
        while not self._queue.empty():
            job = self._queue.get_nowait()
            self._queue.task_done()
            if not job.future.done():
                job.future.set_exception(RuntimeError("Server is shutting down — please retry"))
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        logger.info("Cloud transcription worker stopped")

    async def submit(self, audio: np.ndarray, language: str = "", temperature: float = 0.0,
                     client_id: str = "unknown") -> TranscriptionResult:
        if not self._settings.cloud_configured:
            raise SpeechServiceError("Cloud speech is not configured. Set OPENAI_API_KEY.", 503)
        if not self._running:
            raise RuntimeError("Transcription worker is not running")
        # Check decoded samples too: container metadata can underreport duration.
        if audio.ndim != 1 or audio.size > self._settings.max_audio_duration * 16000:
            raise ValueError("Decoded audio exceeds the configured duration limit")
        active = self._active_jobs_per_client.get(client_id, 0)
        if active >= self._max_jobs_per_client:
            raise RuntimeError("Too many concurrent jobs from this client; please retry later")
        future = asyncio.get_running_loop().create_future()
        job = TranscriptionJob(audio, language, future, temperature, client_id)
        self._active_jobs_per_client[client_id] = active + 1
        future.add_done_callback(lambda _f: self._decrement_client(client_id))
        put_task = asyncio.create_task(self._queue.put(job))
        self._put_tasks.add(put_task)
        try:
            await asyncio.wait_for(put_task, timeout=30.0)
        except asyncio.TimeoutError:
            future.cancel()
            raise RuntimeError("Transcription queue is full; please retry later") from None
        except asyncio.CancelledError:
            future.cancel()
            if not self._running:
                raise RuntimeError("Server is shutting down — please retry") from None
            raise
        finally:
            self._put_tasks.discard(put_task)
        if not self._running and not future.done():
            future.set_exception(RuntimeError("Server is shutting down — please retry"))
        return await future

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    @property
    def is_ready(self) -> bool:
        return self._running and self._client is not None

    async def _worker_loop(self) -> None:
        while self._running:
            job = await self._queue.get()
            self._active_job = job
            try:
                if job.future.done():
                    continue
                result = await self._transcribe(job.audio, job.language, job.temperature)
                if not job.future.done():
                    job.future.set_result(result)
            except asyncio.CancelledError:
                if not job.future.done():
                    job.future.set_exception(RuntimeError("Server is shutting down — please retry"))
                raise
            except Exception as exc:
                # The adapter below sanitizes provider failures before they reach here.
                logger.warning("Transcription failed", error_type=type(exc).__name__)
                if not job.future.done():
                    job.future.set_exception(exc)
            finally:
                self._active_job = None
                self._queue.task_done()

    async def _transcribe(self, audio: np.ndarray, language: str,
                          temperature: float = 0.0) -> TranscriptionResult:
        if self._client is None:
            raise RuntimeError("Transcription worker is not running")
        payload = {
            "model": self._settings.asr_model,
            "messages": [
                {"role": "system", "content": TRANSCRIPTION_PROMPT},
                {"role": "user", "content": [{"type": "input_audio", "input_audio": {
                    "data": encode_wav(audio),
                }}]},
            ],
            "stream": False,
        }
        async def send_and_read() -> bytes:
            async with self._client.stream(
                "POST", self._settings.openai_base_url + "/chat/completions",
                headers={"Authorization": "Bearer " + self._settings.openai_api_key.get_secret_value().strip()},
                json=payload,
            ) as response:
                if response.status_code == 429:
                    raise SpeechServiceError("Cloud speech rate limit exceeded. Please retry later.", 503)
                if not response.is_success:
                    raise SpeechServiceError("Cloud speech provider rejected the request. Check server credentials and model access.")
                data = bytearray()
                async for chunk in response.aiter_bytes():
                    if len(data) + len(chunk) > MAX_PROVIDER_RESPONSE_BYTES:
                        raise SpeechServiceError("Cloud speech response exceeded the size limit.")
                    data.extend(chunk)
                return bytes(data)

        try:
            # Bounds the entire operation, including streamed/decompressed body reading.
            raw = await asyncio.wait_for(send_and_read(), timeout=self._settings.request_timeout_seconds)
        except (httpx.TimeoutException, asyncio.TimeoutError):
            raise SpeechServiceError("Cloud speech request timed out. Please retry.", 504) from None
        except httpx.RequestError:
            raise SpeechServiceError("Could not reach cloud speech service.", 502) from None
        try:
            body = json.loads(raw)
            choice = body["choices"][0]
            text = choice["message"]["content"]
            if not isinstance(text, str) or not text.strip() or choice.get("finish_reason") in {"length", "content_filter"}:
                raise ValueError("No complete transcript")
        except (ValueError, KeyError, IndexError, TypeError):
            raise SpeechServiceError("Cloud speech returned an invalid or incomplete transcript.") from None
        return TranscriptionResult(
            text=text.strip(), language=language.strip() or self._settings.default_language,
            duration=round(audio.size / 16000, 3), segments=[],
        )
