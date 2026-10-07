"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def clean_settings(monkeypatch):
    """Ensure each test starts with a clean settings state.

    Unsets all WHISPER/APP-related env vars that might leak from the host.
    """
    env_vars_to_clear = [
        "OPENAI_API_KEY", "OPENAI_BASE_URL", "ASR_MODEL", "REQUEST_TIMEOUT_SECONDS", "DATABASE_URL",
        "HOST", "PORT", "CORS_ORIGINS", "MODEL_PATH", "MODEL_COMPUTE_TYPE",
        "MODEL_DEVICE", "MODEL_DEVICE_INDEX", "MODEL_DOWNLOAD_URL", "HF_MODEL_ID",
        "MAX_UPLOAD_BYTES", "MAX_AUDIO_DURATION", "DEFAULT_LANGUAGE",
        "VAD_ENABLED", "VAD_THRESHOLD", "VAD_MIN_SILENCE_DURATION_MS",
        "RATE_LIMIT_RPM", "RATE_LIMIT_BURST",
        "LOG_LEVEL", "LOG_FORMAT", "TEMP_DIR", "HTTP_PROXY", "HTTPS_PROXY",
    ]
    import os
    for key in env_vars_to_clear:
        monkeypatch.delenv(key, raising=False)
