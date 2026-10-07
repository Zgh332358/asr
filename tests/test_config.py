"""Tests for application configuration (pydantic-settings).

Covers: env var loading, defaults, property resolution, extra="forbid" behavior.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings


class TestSettingsDefaults:
    """Verify all config fields have sensible defaults."""

    def test_server_defaults(self):
        s = Settings()
        assert s.host == "0.0.0.0"
        assert s.port == 8080
        assert s.cors_origins == ""

    def test_model_defaults(self):
        s = Settings()
        assert s.asr_model == "stepaudio-3-chat-preview"
        assert s.openai_base_url == "https://api.stepfun.com/v1"
        assert s.cloud_configured is False
        assert s.request_timeout_seconds == 120

    def test_audio_limits_defaults(self):
        s = Settings()
        assert s.max_upload_bytes == 524_288_000
        assert s.max_audio_duration == 600
        assert s.default_language == "zh"

    def test_rate_limit_defaults(self):
        s = Settings()
        assert s.rate_limit_rpm == 60
        assert s.rate_limit_burst == 10

    def test_logging_defaults(self):
        s = Settings()
        assert s.log_level == "info"
        assert s.log_format == "json"

    def test_temp_dir_defaults(self):
        s = Settings()
        assert s.temp_dir == "/tmp/whisper_api"


class TestSettingsProperties:
    """Verify computed properties."""

    def test_cors_origins_list_empty(self):
        s = Settings(cors_origins="")
        assert s.cors_origins_list == ["*"]

    def test_cors_origins_list_single(self):
        s = Settings(cors_origins="https://app.example.com")
        assert s.cors_origins_list == ["https://app.example.com"]

    def test_cors_origins_list_multiple(self):
        s = Settings(cors_origins="https://a.com, https://b.com")
        assert s.cors_origins_list == ["https://a.com", "https://b.com"]

    def test_temp_dir_resolved(self):
        s = Settings(temp_dir="/tmp/test_whisper")
        resolved = s.temp_dir_resolved
        assert isinstance(resolved, Path)
        assert resolved.is_absolute()


class TestSettingsEnvOverride:
    """Verify environment variables override defaults."""

    def test_env_override_port(self, monkeypatch):
        monkeypatch.setenv("PORT", "9090")
        s = Settings()
        assert s.port == 9090

    def test_env_override_model(self, monkeypatch):
        monkeypatch.setenv("ASR_MODEL", "configured-model")
        s = Settings()
        assert s.asr_model == "configured-model"

    def test_env_override_log_level(self, monkeypatch):
        monkeypatch.setenv("LOG_LEVEL", "debug")
        s = Settings()
        assert s.log_level == "debug"

    def test_env_override_cors_origins(self, monkeypatch):
        monkeypatch.setenv("CORS_ORIGINS", "https://prod.example.com")
        s = Settings()
        assert s.cors_origins_list == ["https://prod.example.com"]


class TestSettingsExtraForbid:
    """Verify extra="forbid" prevents accidental field typos in code (B-8 fix)."""

    def test_extra_field_rejected_at_init(self):
        """Passing an unknown kwarg to Settings() should raise ValidationError."""
        with pytest.raises(Exception):  # pydantic ValidationError
            Settings(model_paths="/wrong/path")  # typo: model_paths vs model_path


def test_api_key_is_redacted():
    s = Settings(openai_api_key="test-secret-must-not-print")
    assert s.cloud_configured
    assert "test-secret-must-not-print" not in repr(s)
    assert "test-secret-must-not-print" not in s.model_dump_json()


@pytest.mark.parametrize("url", ["http://api.stepfun.com/v1", "https://user:pass@api.stepfun.com/v1", "https://api.stepfun.com/v1?key=secret", "https://api.stepfun.com/v1#x"])
def test_base_url_rejects_unsafe_config(url):
    with pytest.raises(ValueError):
        Settings(openai_base_url=url)


def test_template_is_valid():
    s = Settings(_env_file=".env.example")
    assert not s.cloud_configured
    assert s.asr_model == "stepaudio-3-chat-preview"
