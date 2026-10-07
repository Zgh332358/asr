"""Application configuration via pydantic-settings.

All values can be overridden via environment variables or a .env file.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from pydantic import Field, SecretStr, field_validator
from urllib.parse import urlsplit
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """StepFun ASR API settings.

    Loaded from .env file and environment variables (env vars take precedence).
    """

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="forbid",
    )

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8080
    cors_origins: str = ""  # comma-separated, e.g. "https://app.example.com,https://admin.example.com"

    # --- Cloud speech model (server-only credentials) ---
    openai_api_key: SecretStr = SecretStr("")
    openai_base_url: str = "https://api.stepfun.com/v1"
    asr_model: str = "stepaudio-3-chat-preview"
    request_timeout_seconds: float = Field(default=120.0, ge=1, le=600)

    # --- Audio Limits ---
    max_upload_bytes: int = 524_288_000  # 500 MB
    max_audio_duration: int = 600  # 10 minutes
    default_language: str = "zh"

    # --- Rate Limiting ---
    rate_limit_rpm: int = 60  # requests per minute, 0 = disabled
    rate_limit_burst: int = 10

    # --- Logging ---
    log_level: str = "info"
    log_format: str = "json"  # "json" or "console"

    # --- Temp Files ---
    temp_dir: str = "/tmp/whisper_api"

    # --- Database (empty = stateless mode for backward compatibility) ---
    database_url: str = ""
    database_pool_size: int = 5
    database_pool_overflow: int = 10

    # --- File Storage ---
    storage_path: str = "/data/asr_storage"

    # --- Task Scheduler ---
    task_poll_interval: int = 5  # seconds between DB polls for PENDING tasks
    task_max_retries: int = 3    # max retries for failed tasks before giving up
    max_concurrent_tasks: int = 2  # max in-flight DB tasks processed concurrently

    # --- Callback ---
    main_backend_callback_url: str = ""  # e.g. http://main:8080/api/v1/callback/asr
    callback_max_retries: int = 5
    callback_retry_base_delay: float = 1.0  # exponential backoff base (seconds)

    # --- Proxy ---
    http_proxy: str = ""
    https_proxy: str = ""

    # --- Validators ---

    @field_validator("openai_base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        parsed = urlsplit(value)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username
                or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("OPENAI_BASE_URL must be an HTTPS URL without credentials, query or fragment")
        return value

    @field_validator("asr_model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("ASR_MODEL must not be empty")
        return value.strip()

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"debug", "info", "warning", "error", "critical"}
        if v.lower() not in valid:
            raise ValueError(f"log_level must be one of {valid}, got '{v}'")
        return v.lower()

    @field_validator("log_format")
    @classmethod
    def validate_log_format(cls, v: str) -> str:
        valid = {"json", "console"}
        if v not in valid:
            raise ValueError(f"log_format must be one of {valid}, got '{v}'")
        return v

    # --- Properties ---

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse comma-separated CORS origins into a list.

        Returns ["*"] if no origins are configured (allow all, credentials disabled).
        """
        if not self.cors_origins.strip():
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def cloud_configured(self) -> bool:
        return bool(self.openai_api_key.get_secret_value().strip())

    @property
    def temp_dir_resolved(self) -> Path:
        """Resolve temp directory path."""
        return Path(self.temp_dir).expanduser().resolve()

    @property
    def storage_path_resolved(self) -> Path:
        """Resolve permanent storage directory path."""
        return Path(self.storage_path).expanduser().resolve()

    @property
    def database_enabled(self) -> bool:
        """Whether database integration is configured."""
        return bool(self.database_url.strip())

    @classmethod
    def resolve(cls, settings: Settings | None) -> Settings:
        """Return the given settings instance or the default singleton.

        Use this to accept optional settings in constructors:
            self._settings = Settings.resolve(settings)
        """
        return settings if settings is not None else _settings_singleton


# Singleton — use Settings.resolve() instead of importing this directly
_settings_singleton = Settings()
settings = _settings_singleton  # backward-compatible alias
