"""Local ASGI requests exercise the upload route; StepFun is mocked."""
import io
import wave

import httpx
import numpy as np
import pytest

from app.config import Settings
from app.services.transcriber import TranscriptionWorker


def wav_bytes():
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes((np.sin(np.arange(4000) / 8) * 4000).astype("<i2").tobytes())
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_no_key_app_starts_but_transcription_is_503(monkeypatch):
    import app.main as main
    monkeypatch.setattr(main, "settings", Settings())
    app = main.create_app()
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            assert (await client.get("/health/live")).status_code == 200
            assert (await client.get("/health/ready")).status_code == 503
            assert (await client.get("/health/gpu")).status_code == 404
            response = await client.post("/v1/audio/transcriptions", files={"file": ("invalid.wav", b"not audio")})
            assert response.status_code == 503  # checked before ffmpeg work
            assert "OPENAI_API_KEY" in response.text
            assert (await client.get("/api/v1/asr/corpus")).status_code == 503


@pytest.mark.parametrize("response_format", ["json", "verbose_json"])
@pytest.mark.asyncio
async def test_upload_success_with_real_wav_decode(monkeypatch, response_format):
    import app.main as main
    def respond(_):
        return httpx.Response(200, json={"choices": [{"message": {"content": "测试转写"}}]})
    worker = TranscriptionWorker(Settings(openai_api_key="test-only"), httpx.MockTransport(respond))
    await worker.start()
    app = main.create_app()
    app.state.worker = worker
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/v1/audio/transcriptions", files={"file": ("tone.wav", wav_bytes(), "audio/wav")}, data={"response_format": response_format, "language": "zh", "model": "ignored-client-model"})
            assert response.status_code == 200, response.text
            assert response.json()["text"] == "测试转写"
            if response_format == "verbose_json":
                assert response.json()["segments"] == []
                assert response.json()["duration"] == 0.25
            ready = await client.get("/health/ready")
            assert ready.status_code == 200
            assert ready.json()["gpu_available"] is False
    finally:
        await worker.stop()
