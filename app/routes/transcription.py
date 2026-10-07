"""Transcription endpoint — OpenAI-compatible POST /v1/audio/transcriptions."""

from typing import Literal

import structlog
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import settings
from app.schemas.responses import (
    ErrorResponse,
    TranscriptionResponse,
    TranscriptionSegment,
    VerboseTranscriptionResponse,
)
from app.services.transcriber import SpeechServiceError
from app.services.upload_pipeline import process_upload, validate_content_length

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["transcription"])

# Rate limiter — keyed by client IP
limiter = Limiter(key_func=get_remote_address)


@router.post(
    "/v1/audio/transcriptions",
    response_model=TranscriptionResponse | VerboseTranscriptionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad request"},
        413: {"model": ErrorResponse, "description": "File too large"},
        429: {"model": ErrorResponse, "description": "Rate limited"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
)
@limiter.limit(
    f"{settings.rate_limit_rpm}/minute" if settings.rate_limit_rpm > 0 else "100000/minute"
)
async def transcribe(
    request: Request,
    file: UploadFile = File(..., description="Audio file to transcribe"),
    language: str = Form(
        default="",
        description="Language code (e.g. 'zh', 'en'). Empty for auto-detect.",
    ),
    temperature: float = Form(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Sampling temperature (0-1). Lower = more deterministic.",
    ),
    response_format: Literal["json", "verbose_json"] = Form(
        default="json",
        description="Response format: 'json' for text only, 'verbose_json' for segments.",
    ),
) -> TranscriptionResponse | VerboseTranscriptionResponse:
    """Transcribe audio to text.

    OpenAI-compatible endpoint. Accepts multipart form data with an audio file.
    Supports all formats that ffmpeg can decode (wav, mp3, m4a, ogg, flac, etc.).

    Authentication is handled by the upstream API gateway — no Bearer token
    validation is performed at this layer.
    """
    worker = request.app.state.worker
    if not worker.is_ready:
        raise HTTPException(status_code=503, detail="Cloud speech is not configured or not ready. Set OPENAI_API_KEY.")

    # Validate language
    lang = language.strip()

    # Pre-check Content-Length before reading body
    validate_content_length(request)

    # Process upload through pipeline
    decoded = await process_upload(file)
    audio = decoded.audio

    # Submit to the cloud worker
    client_id = request.client.host if request.client else "unknown"
    try:
        result = await worker.submit(
            audio, language=lang, temperature=temperature, client_id=client_id
        )
    except SpeechServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from None
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid or oversized decoded audio") from None
    except RuntimeError as exc:
        # Worker is shutting down or not running — map to 503 (C-1)
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        )

    # Build response
    if response_format == "verbose_json":
        segments = [
            TranscriptionSegment(
                id=seg["id"],
                seek=seg["seek"],
                start=seg["start"],
                end=seg["end"],
                text=seg["text"],
                tokens=seg["tokens"],
                temperature=seg["temperature"],
                avg_logprob=seg["avg_logprob"],
                compression_ratio=seg["compression_ratio"],
                no_speech_prob=seg["no_speech_prob"],
            )
            for seg in result.segments
        ]
        return VerboseTranscriptionResponse(
            text=result.text,
            language=result.language,
            duration=result.duration,
            segments=segments,
        )
    else:
        return TranscriptionResponse(text=result.text)
