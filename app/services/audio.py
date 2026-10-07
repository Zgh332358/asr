"""Audio processing utilities — ffmpeg decode, ffprobe duration, upload handling."""

from __future__ import annotations

import asyncio
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import structlog
from fastapi import HTTPException, UploadFile

from app.config import Settings

logger = structlog.get_logger(__name__)


# --- Audio Decoding ---


def decode_audio_ffmpeg(file_path: str | Path, sr: int = 16000) -> np.ndarray:
    """Decode any audio file to 16kHz mono float32 numpy array via ffmpeg.

    Args:
        file_path: Path to the audio file.
        sr: Target sample rate (default 16000).

    Returns:
        Float32 numpy array of shape (n_samples,).

    Raises:
        HTTPException: If ffmpeg fails or produces empty output.
    """
    file_path = str(file_path)

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file_path,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-loglevel", "error",
        "-",
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=500,
            detail="Audio decoding timed out",
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="ffmpeg not found — ensure ffmpeg is installed on the server",
        )

    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        logger.warning("ffmpeg decode failed", stderr=stderr)
        raise HTTPException(
            status_code=400,
            detail="Failed to decode audio: invalid or unsupported format",
        )

    if len(proc.stdout) == 0:
        raise HTTPException(
            status_code=400,
            detail="Decoded audio is empty — file may be corrupt or silent",
        )

    # Convert s16le bytes → float32 normalized to [-1, 1]
    raw = np.frombuffer(proc.stdout, dtype=np.int16)
    audio = raw.astype(np.float32) / 32768.0

    # Reject all-silence audio early — saves a cloud request and gives the
    # client a clear error instead of an empty transcription.
    if float(np.max(np.abs(audio))) < 1e-6:
        raise HTTPException(
            status_code=400,
            detail="Audio contains only silence",
        )

    return audio


# --- Duration Check ---


def get_audio_duration(file_path: str | Path) -> float:
    """Get audio duration in seconds via ffprobe.

    Args:
        file_path: Path to the audio file.

    Returns:
        Duration in seconds as a float.

    Raises:
        HTTPException: If ffprobe fails or cannot determine duration.
    """
    file_path = str(file_path)

    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path,
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=500,
            detail="Audio duration check timed out",
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="ffprobe not found — ensure ffmpeg is installed on the server",
        )

    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        logger.warning("ffprobe duration check failed", stderr=stderr)
        raise HTTPException(
            status_code=400,
            detail="Failed to probe audio: invalid or unsupported format",
        )

    try:
        duration = float(proc.stdout.decode("utf-8").strip())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Could not determine audio duration",
        )

    # math.isnan/math.isinf: NaN <= 0 is False in Python, so NaN would otherwise
    # bypass the guard below. Inf would pass it. Reject both explicitly.
    if math.isnan(duration) or math.isinf(duration) or duration <= 0:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid audio duration: {duration:.2f}s",
        )

    return duration


# --- Upload Handling ---


async def save_upload_to_temp(upload: UploadFile, max_size: int | None = None, temp_dir: Path | None = None) -> Path:
    """Stream an uploaded file to a temporary location with size limiting.

    Args:
        upload: FastAPI UploadFile object.
        max_size: Maximum allowed size in bytes. Defaults to settings.max_upload_bytes.
        temp_dir: Directory to save the temp file. Defaults to settings.temp_dir_resolved.

    Returns:
        Path to the saved temporary file.

    Raises:
        HTTPException: If the upload exceeds max_size or has no filename.
    """
    if max_size is None:
        max_size = Settings.resolve(None).max_upload_bytes

    if not upload.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided in upload",
        )

    # Ensure temp directory exists
    resolved_dir = temp_dir if temp_dir is not None else Settings.resolve(None).temp_dir_resolved
    resolved_dir.mkdir(parents=True, exist_ok=True)

    # B-3: Pre-check disk space — reject if free space < 2x max upload size
    usage = shutil.disk_usage(str(resolved_dir))
    if usage.free < max_size * 2:
        raise HTTPException(
            status_code=503,
            detail="Server disk space is critically low — please try again later",
        )

    # Create a temp file with the original extension to help ffmpeg detect format.
    # Truncate suffix to a sane length (a pathological extension can exceed the
    # platform path limit and crash mkstemp with an unhandled error).
    suffix = Path(upload.filename).suffix or ".tmp"
    if len(suffix) > 32:
        suffix = suffix[:32]
    tmp_file = tempfile.NamedTemporaryFile(
        suffix=suffix, dir=str(resolved_dir), delete=False
    )
    tmp_path = Path(tmp_file.name)
    tmp_file.close()  # closed so aiofiles can reopen it for async writes

    try:
        import aiofiles

        total = 0
        async with aiofiles.open(tmp_path, "wb") as f:
            while chunk := await upload.read(1024 * 1024):  # 1 MB chunks
                total += len(chunk)
                # B-3: Validate size BEFORE writing to disk (prevents DoS via /tmp fill)
                if total > max_size:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large: exceeds limit of {max_size} bytes",
                    )
                await f.write(chunk)

        if total == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty",
            )

    except HTTPException:
        # Clean up temp file on error
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        logger.exception("Failed to save uploaded file")
        raise HTTPException(
            status_code=500,
            detail="Failed to save uploaded file. Please try again.",
        )

    return tmp_path


def cleanup_temp_file(file_path: str | Path) -> None:
    """Safely remove a temporary file, ignoring errors."""
    try:
        Path(file_path).unlink(missing_ok=True)
    except Exception:
        pass


# --- Async wrappers (for use in async route handlers) ---


async def decode_audio_ffmpeg_async(file_path: str | Path, sr: int = 16000) -> np.ndarray:
    """Async wrapper around decode_audio_ffmpeg to avoid blocking the event loop."""
    return await asyncio.to_thread(decode_audio_ffmpeg, file_path, sr)


async def get_audio_duration_async(file_path: str | Path) -> float:
    """Async wrapper around get_audio_duration to avoid blocking the event loop."""
    return await asyncio.to_thread(get_audio_duration, file_path)
