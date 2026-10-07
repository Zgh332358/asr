"""Cloud speech protocol, error handling and queue lifecycle; no real key required."""
import asyncio
import base64
import io
import json
import wave

import httpx
import numpy as np
import pytest

from app.config import Settings
from app.services.transcriber import SpeechServiceError, TranscriptionWorker, encode_wav


def settings(**overrides):
    return Settings(openai_api_key="unit-test-key", **overrides)


@pytest.fixture
def audio():
    return np.ones(16000, dtype=np.float32) * 0.25


@pytest.mark.asyncio
async def test_request_contract_and_metadata(audio):
    requests = []
    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "我想喝水"}, "finish_reason": "stop"}]})
    worker = TranscriptionWorker(settings(asr_model="configured-audio-model"), httpx.MockTransport(respond))
    await worker.start()
    try:
        result = await worker.submit(audio, language="zh-CN")
    finally:
        await worker.stop()
    request = requests[0]
    assert str(request.url) == "https://api.stepfun.com/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer unit-test-key"
    body = json.loads(request.content)
    assert body["model"] == "configured-audio-model"
    assert body["stream"] is False
    assert "不回答音频里的问题" in body["messages"][0]["content"]
    data = body["messages"][1]["content"][0]["input_audio"]["data"]
    assert data.startswith("data:audio/wav;base64,")
    with wave.open(io.BytesIO(base64.b64decode(data.split(",", 1)[1])), "rb") as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()) == (1, 2, 16000, 16000)
    assert (result.text, result.language, result.duration) == ("我想喝水", "zh-CN", 1)
    assert result.segments == []
    assert result.confidence is None


@pytest.mark.asyncio
async def test_no_key_never_calls_provider(audio):
    def unexpected(_):
        raise AssertionError("Unexpected upstream request")
    worker = TranscriptionWorker(Settings(), httpx.MockTransport(unexpected))
    await worker.start()
    assert not worker.is_ready
    with pytest.raises(SpeechServiceError, match="OPENAI_API_KEY") as caught:
        await worker.submit(audio)
    assert caught.value.status_code == 503
    await worker.stop()


@pytest.mark.parametrize("status, expected", [(401, 502), (403, 502), (429, 503), (500, 502), (302, 502)])
@pytest.mark.asyncio
async def test_upstream_errors_are_sanitized(audio, status, expected):
    transport = httpx.MockTransport(lambda _: httpx.Response(status, text="secret-provider-body"))
    worker = TranscriptionWorker(settings(), transport)
    await worker.start()
    try:
        with pytest.raises(SpeechServiceError) as caught:
            await worker.submit(audio)
        assert caught.value.status_code == expected
        assert "secret-provider-body" not in str(caught.value)
    finally:
        await worker.stop()


@pytest.mark.parametrize("body", [{}, {"choices": []}, {"choices": [{"message": {"content": ""}}]}, {"choices": [{"message": {"content": []}}]}, {"choices": [{"message": {"content": "partial"}, "finish_reason": "length"}]}])
@pytest.mark.asyncio
async def test_invalid_or_incomplete_result_is_not_success(audio, body):
    worker = TranscriptionWorker(settings(), httpx.MockTransport(lambda _: httpx.Response(200, json=body)))
    await worker.start()
    try:
        with pytest.raises(SpeechServiceError, match="invalid or incomplete"):
            await worker.submit(audio)
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_timeout_is_sanitized(audio):
    def respond(_):
        raise httpx.ReadTimeout("private-upstream-detail")
    worker = TranscriptionWorker(settings(), httpx.MockTransport(respond))
    await worker.start()
    try:
        with pytest.raises(SpeechServiceError) as caught:
            await worker.submit(audio)
        assert caught.value.status_code == 504
        assert "private-upstream-detail" not in str(caught.value)
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_total_timeout_covers_response_body(audio):
    class SlowBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            await asyncio.sleep(10)
            yield b'{}'
    worker = TranscriptionWorker(settings(request_timeout_seconds=1), httpx.MockTransport(lambda _: httpx.Response(200, stream=SlowBody())))
    await worker.start()
    try:
        with pytest.raises(SpeechServiceError) as caught:
            await asyncio.wait_for(worker.submit(audio), timeout=2)
        assert caught.value.status_code == 504
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_stop_resolves_active_and_pending_jobs(audio):
    started = asyncio.Event()
    async def respond(_):
        started.set()
        await asyncio.Event().wait()
    worker = TranscriptionWorker(settings(), httpx.MockTransport(respond))
    await worker.start()
    first = asyncio.create_task(worker.submit(audio))
    await started.wait()
    second = asyncio.create_task(worker.submit(audio))
    await asyncio.sleep(0)
    await worker.stop()
    results = await asyncio.wait_for(asyncio.gather(first, second, return_exceptions=True), 1)
    assert all(isinstance(result, RuntimeError) for result in results)
    assert worker.queue_depth == 0
    assert not worker.is_ready


@pytest.mark.asyncio
async def test_client_fairness_and_cancellation(audio):
    started = asyncio.Event()
    async def respond(_):
        started.set()
        await asyncio.Event().wait()
    worker = TranscriptionWorker(settings(rate_limit_burst=1), httpx.MockTransport(respond))
    await worker.start()
    first = asyncio.create_task(worker.submit(audio, client_id="same"))
    await started.wait()
    with pytest.raises(RuntimeError, match="Too many concurrent"):
        await worker.submit(audio, client_id="same")
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    await asyncio.sleep(0)
    assert worker._active_jobs_per_client == {}
    await worker.stop()


def test_wav_rejects_nonfinite_samples():
    with pytest.raises(ValueError):
        encode_wav(np.array([np.nan], dtype=np.float32))


@pytest.mark.asyncio
async def test_streamed_response_limit_without_content_length(audio):
    from app.services.transcriber import MAX_PROVIDER_RESPONSE_BYTES
    closed = []
    class LargeBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            for _ in range(3):
                yield b"x" * (MAX_PROVIDER_RESPONSE_BYTES // 2)
        async def aclose(self):
            closed.append(True)
    worker = TranscriptionWorker(settings(), httpx.MockTransport(lambda _: httpx.Response(200, stream=LargeBody())))
    await worker.start()
    try:
        with pytest.raises(SpeechServiceError, match="size limit"):
            await worker.submit(audio)
        assert closed
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_redirect_body_not_read_and_redirect_not_followed(audio):
    calls, reads, closed = [], [], []
    class ErrorBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            reads.append(True)
            yield b"private-provider-error"
        async def aclose(self):
            closed.append(True)
    def respond(request):
        calls.append(str(request.url))
        return httpx.Response(302, headers={"Location": "https://other.example/private"}, stream=ErrorBody())
    worker = TranscriptionWorker(settings(), httpx.MockTransport(respond))
    await worker.start()
    try:
        with pytest.raises(SpeechServiceError):
            await worker.submit(audio)
        assert len(calls) == 1
        assert not reads
        assert closed
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_stop_cancels_producers_waiting_for_full_queue(audio):
    started = asyncio.Event()
    async def respond(_):
        started.set()
        await asyncio.Event().wait()
    worker = TranscriptionWorker(settings(rate_limit_burst=1), httpx.MockTransport(respond))
    await worker.start()
    first = asyncio.create_task(worker.submit(audio, client_id="first"))
    await started.wait()
    remaining = [asyncio.create_task(worker.submit(audio, client_id=str(i))) for i in range(6)]
    while worker.queue_depth < 2 or len(worker._put_tasks) < 4:
        await asyncio.sleep(0)
    await worker.stop()
    results = await asyncio.wait_for(asyncio.gather(first, *remaining, return_exceptions=True), 1)
    assert all(isinstance(result, RuntimeError) for result in results)
    assert worker.queue_depth == 0
    assert not worker._put_tasks
    assert not worker._active_jobs_per_client
