"""HTTP callback service — notify the main backend of task completion."""

from __future__ import annotations

import asyncio

import httpx
import structlog

from app.config import Settings

logger = structlog.get_logger(__name__)


async def notify_main_backend(
    task_id: int,
    corpus_id: int,
    status: str,
    result_text: str | None = None,
    confidence: float | None = None,
    error_message: str | None = None,
    callback_url: str | None = None,
    max_retries: int | None = None,
    retry_base_delay: float | None = None,
) -> bool:
    """POST task completion result to the main backend with exponential retry.

    Args:
        task_id: The ASR task ID.
        corpus_id: The associated corpus ID.
        status: Final task status (SUCCESS or FAILED).
        result_text: Transcription result text (if SUCCESS).
        confidence: Overall confidence score (if SUCCESS).
        error_message: Error description (if FAILED).
        callback_url: Override settings.main_backend_callback_url.
        max_retries: Override settings.callback_max_retries.
        retry_base_delay: Override settings.callback_retry_base_delay.

    Returns:
        True if the callback succeeded, False after exhausting all retries.
    """
    _defaults = Settings.resolve(None)
    _url = callback_url if callback_url is not None else _defaults.main_backend_callback_url
    _max_retries = max_retries if max_retries is not None else _defaults.callback_max_retries
    _base_delay = retry_base_delay if retry_base_delay is not None else _defaults.callback_retry_base_delay

    if not _url.strip():
        logger.debug("No callback URL configured, skipping")
        return True  # not a failure — callback is optional

    payload = {
        "task_id": task_id,
        "corpus_id": corpus_id,
        "status": status,
    }
    if status == "SUCCESS":
        payload["result_text"] = result_text
        payload["confidence"] = float(confidence) if confidence is not None else None
    elif status == "FAILED":
        payload["error_message"] = error_message

    last_exc: Exception | None = None

    async with httpx.AsyncClient(timeout=30.0) as client:
        for attempt in range(_max_retries + 1):
            try:
                response = await client.post(
                    _url,
                    json=payload,
                )
                if response.status_code < 500:
                    # 2xx/3xx/4xx: don't retry — 4xx means bad request, not transient
                    if response.status_code >= 400:
                        logger.warning(
                            "Callback rejected by main backend",
                            task_id=task_id,
                            status_code=response.status_code,
                        )
                    else:
                        logger.info(
                            "Callback succeeded",
                            task_id=task_id,
                            status_code=response.status_code,
                        )
                    return response.is_success
            except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError) as exc:
                last_exc = exc

            if attempt < _max_retries:
                delay = _base_delay * (2 ** attempt)
                logger.warning(
                    "Callback attempt failed, retrying",
                    task_id=task_id,
                    attempt=attempt + 1,
                    max_retries=_max_retries,
                    delay=delay,
                    error_type=type(last_exc).__name__ if last_exc else "HTTPError",
                )
                await asyncio.sleep(delay)

    logger.error(
        "Callback exhausted all retries",
        task_id=task_id,
        max_retries=_max_retries,
        last_error_type=type(last_exc).__name__ if last_exc else "HTTPError",
    )
    return False
