"""Pydantic request schemas for API endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CorpusCreateRequest(BaseModel):
    """Metadata for corpus creation submitted alongside the audio file upload."""

    language: str = Field(
        default="zh-CN",
        description="Language code (e.g. 'zh-CN', 'en-US').",
    )
    business_id: str | None = Field(
        default=None,
        description="Business ID (e.g., patient ID).",
    )
    business_type: str | None = Field(
        default=None,
        description="Business type (e.g., PATIENT_VOICE, SAMPLE_AUDIO).",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for categorization and filtering.",
    )
    asr_engine: str = Field(
        default="STEPFUN",
        pattern=r"^STEPFUN$",
        description="ASR engine to use.",
    )
    engine_config: dict = Field(
        default_factory=dict,
        description="Engine-specific configuration parameters.",
    )


class CorpusListParams(BaseModel):
    """Query parameters for listing corpora."""

    business_id: str | None = Field(default=None, description="Filter by business ID.")
    business_type: str | None = Field(default=None, description="Filter by business type.")
    status: str | None = Field(default=None, description="Filter by status (UPLOADING, UPLOADED, FAILED).")
    is_deleted: bool = Field(default=False, description="Include soft-deleted records.")
    page: int = Field(default=1, ge=1, description="Page number (1-based).")
    page_size: int = Field(default=20, ge=1, le=100, description="Items per page.")


class TaskListParams(BaseModel):
    """Query parameters for listing ASR tasks."""

    status: str | None = Field(default=None, description="Filter by status (PENDING, PROCESSING, SUCCESS, FAILED).")
    corpus_id: int | None = Field(default=None, description="Filter by corpus ID.")
    page: int = Field(default=1, ge=1, description="Page number (1-based).")
    page_size: int = Field(default=20, ge=1, le=100, description="Items per page.")
