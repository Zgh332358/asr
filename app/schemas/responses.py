"""Pydantic response models — OpenAI-compatible + corpus/task management."""

from __future__ import annotations

import datetime
from typing import Optional

from pydantic import BaseModel, Field


# --- Health ---


class HealthResponse(BaseModel):
    """Basic health check response."""

    status: str = "ok"


class ReadyResponse(BaseModel):
    """Readiness check response.

    NOTE: queue_depth is intentionally excluded from this unauthenticated probe.
    Kubernetes readiness only needs a binary healthy/unhealthy signal; runtime
    metrics should live behind an authenticated endpoint.
    """

    status: str
    model_loaded: bool
    gpu_available: bool
    provider: str = "stepfun"
    configured: bool = False
    model: str = ""


class GPUInfoResponse(BaseModel):
    """GPU memory and utilization info."""

    device_name: str
    device_index: int
    total_memory_mb: int = Field(..., allow_inf_nan=False)
    used_memory_mb: int = Field(..., allow_inf_nan=False)
    free_memory_mb: int = Field(..., allow_inf_nan=False)
    utilization_pct: float = Field(..., allow_inf_nan=False)


# --- Transcription (OpenAI-compatible) ---


class TranscriptionSegment(BaseModel):
    """A single transcribed segment with timestamps."""

    id: int
    seek: int = 0
    start: float = Field(..., allow_inf_nan=False)
    end: float = Field(..., allow_inf_nan=False)
    text: str
    tokens: list[int] = Field(default_factory=list)
    temperature: float = Field(default=0.0, allow_inf_nan=False)
    avg_logprob: float = Field(default=0.0, allow_inf_nan=False)
    compression_ratio: float = Field(default=0.0, allow_inf_nan=False)
    no_speech_prob: float = Field(default=0.0, allow_inf_nan=False)


class TranscriptionResponse(BaseModel):
    """Simple text-only transcription (response_format=json)."""

    text: str


class VerboseTranscriptionResponse(BaseModel):
    """Verbose transcription with segments and metadata (response_format=verbose_json)."""

    text: str
    language: str
    duration: float = Field(..., allow_inf_nan=False)
    segments: list[TranscriptionSegment] = Field(default_factory=list)


# --- Error ---


class ErrorDetail(BaseModel):
    """Detailed error information."""

    message: str
    type: str
    param: Optional[str] = None
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    """OpenAI-compatible error response."""

    error: ErrorDetail


# --- Corpus & Task Management ---


class AsrTaskSummary(BaseModel):
    """Lightweight task info for embedding in corpus responses."""

    id: int
    status: str
    asr_engine: str
    result_text: str | None = None
    confidence: float | None = None
    error_message: str | None = None
    created_at: datetime.datetime
    completed_at: datetime.datetime | None = None


class CorpusResponse(BaseModel):
    """Corpus details with embedded task summaries."""

    id: int
    file_md5: str
    file_name: str
    file_size: int | None = None
    duration: int | None = None
    sample_rate: int | None = None
    channels: int
    language: str
    status: str
    text_content: str | None = None
    business_id: str | None = None
    business_type: str | None = None
    tags: list = Field(default_factory=list)
    is_deleted: bool = False
    created_at: datetime.datetime
    updated_at: datetime.datetime
    tasks: list[AsrTaskSummary] = Field(default_factory=list)


class CorpusListResponse(BaseModel):
    """Paginated corpus list."""

    items: list[CorpusResponse]
    total: int
    page: int
    page_size: int


class AsrTaskResponse(BaseModel):
    """Full task details including results."""

    id: int
    corpus_id: int
    status: str
    asr_engine: str
    engine_config: dict = Field(default_factory=dict)
    result_text: str | None = None
    confidence: float | None = None
    result_detail: dict | None = None
    processing_time: int | None = None
    error_message: str | None = None
    started_at: datetime.datetime | None = None
    completed_at: datetime.datetime | None = None
    created_at: datetime.datetime
    updated_at: datetime.datetime


class TaskListResponse(BaseModel):
    """Paginated task list."""

    items: list[AsrTaskSummary]
    total: int
    page: int
    page_size: int


class RecognizeResponse(BaseModel):
    """Response for corpus upload + task creation."""

    corpus_id: int
    task_id: int
    file_md5: str
    cached: bool = False
    status: str  # "PENDING" or "SUCCESS" (if cached)
    result_text: str | None = None  # only set when cached=True

