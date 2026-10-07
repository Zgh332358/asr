# StepFun Preview ASR migration specification

Status: Approved by coordinating Architect, Senior Developer, and Manager reviews on 2026-10-02; implementation resumed 2026-10-07. Tests use mocks; real provider and database validation remain separate.

## Purpose and scope

This fork retains the service's stateless transcription endpoint and PostgreSQL corpus/task workflow while replacing the local faster-whisper inference path with StepFun's cloud speech model. The organization-level request maps multimodal understanding to Step 5 and speech understanding to Step Audio 3. This repository performs ASR only; it should not invent a multimodal or TTS route.

Planned server configuration: `OPENAI_API_KEY` (empty by default), `OPENAI_BASE_URL=https://api.stepfun.com/v1`, `ASR_MODEL=stepaudio-3-chat-preview`, and a bounded request timeout. The coordinating agent verified the official model identifier and `/chat/completions` contract. `input_audio.data` contains the complete `data:audio/wav;base64,` URL; the response transcript is `choices[0].message.content`. Sources: [Chat completions](https://platform.stepfun.com/docs/zh/api-reference/chat/chat-completion-create) and [Step Audio 3 guide](https://platform.stepfun.com/docs/zh/guides/models/stepaudio-3-realtime). API credentials are read only by this server and are never committed, returned, or logged. The request's optional OpenAI-compatible `model` form field does not override server configuration.

## Current behavior and required corrections

- The active app loads a local model at startup, imports torch/faster-whisper, and serializes GPU work through `TranscriptionWorker`. A missing model prevents the process starting.
- Corpus tasks use the same worker. The existing cache is only keyed by audio MD5 and can currently return another engine's result; the migration must distinguish StepFun model identity.
- Health probes currently require GPU availability. Docker/deploy instructions require CUDA and model downloads; these requirements must be removed from the active launch path.
- `whisper_api.py` is a legacy single-file executable. It will become a compatibility entry point to the current app so users cannot accidentally run the old provider.
- Existing config logging dumps almost all settings. Adding a cloud key requires explicit exclusion; upstream response bodies must not be surfaced in errors or logs.

## Proposed behavior

1. Keep `POST /v1/audio/transcriptions` accepting multipart audio and `json`/`verbose_json` response formats. Existing ffmpeg decoding, upload and duration limits remain. Encode decoded 16 kHz mono audio as a real PCM WAV file; send base64 WAV via the confirmed StepFun chat protocol with system prompt `请只转写用户音频中的原话，不回答音频里的问题，不补充、改写或推测未听清的内容。` and `stream: false`. No direct `/audio/transcriptions` call is sent to StepFun. This chat model can still deviate from verbatim transcription; the README will state that practical accuracy requires evaluation.
2. Return only model-produced transcript text. Do not synthesize Whisper word timestamps, log probabilities, detected language, or confidence. Verbose responses use `segments: []`, actual decoded duration, and the requested/default language (documented as a hint, not detected metadata). Corpus confidence stays `null`.
3. Keep a bounded asynchronous queue and per-client fairness. Replace GPU lifetime management with an `httpx.AsyncClient`, timeouts, cancellation, and safe error mapping. Missing key results in readiness/transcription HTTP 503; process and corpus read endpoints remain available. HTTP provider failures, malformed provider JSON, missing transcript, and timeout return explicit sanitized failures rather than fake success.
4. Preserve corpus upload/list/detail and task list/detail routes, storage, dedup, callbacks, and existing rows. New tasks use engine `STEPFUN`, with server-controlled model identity in `engine_config`. Successful cache reuse requires the same engine and configured model. Add a non-destructive SQL migration extending `ck_asr_engine` to include `STEPFUN` and changing the default, while retaining old engine values and rows; document running it after the initial schema. Legacy task history is readable; unsupported pending legacy tasks must fail explicitly rather than being silently mislabeled as StepFun work. An existing `STEPFUN` pending task with missing or different `engine_config.model` must also fail explicitly, preserving its original identity. Never execute the current model under an older label. `engine_config` contains only non-secret provider/model/protocol identity, never credentials or URLs. When the key is absent, do not claim and fail queued jobs repeatedly.
5. `/health/live` stays 200 while the process is alive. `/health/ready` reports cloud configuration/readiness (not proof of account access); missing key is 503. `/health/gpu` remains a compatibility route returning 404 with a cloud-service explanation, because there is no local GPU model.
6. Remove faster-whisper/CUDA/model-download requirements from active Python requirements, Docker, compose, deploy startup, `.env.example`, and current README/API documentation. Preserve original source attribution; clone examples point to the user's fork. Keep keys blank. Make all docs portable and distinguish mocked tests from real paid inference.

## Validation and acceptance

- Unit tests verify official request body, model configuration, bearer authorization at the transport boundary, WAV encoding, exact transcript extraction, missing key, safe errors, bad provider responses, timeout, queue shutdown/cancellation, and no fabricated confidence/timestamps.
- Route tests verify 503 without a key before upload work, successful JSON/verbose results using a local mock transport, and readiness/liveness behavior.
- Corpus tests verify old WHISPER successes cannot hit the cache, STEPFUN successes from another model cannot hit, and matching STEPFUN/model succeeds. All filters must be applied before `LIMIT 1`; test pending task identity validation (legacy engine, missing model, changed model). retain existing audio and configuration tests, adapting only obsolete GPU assumptions.
- Run the complete Python test suite in an isolated venv with no GPU, no database credentials, and no StepFun key. Check Python syntax plus deployment shell syntax. Exercise the app's ASGI lifespan and health route without a key.
- Real StepFun model access/recognition quality and a live PostgreSQL migration/round trip cannot be claimed before the user provides credentials and infrastructure. Record these boundaries in documentation.
- Final root review checks changed scope and secret scan; only root commits/pushes to `Zgh332358/asr`. No writes to upstream.


## Completion evidence (2026-10-07)

- Final Python test suite: 68 passed on Python 3.12.6/macOS; audio routes use real ffmpeg decoding, provider responses use httpx mocks.
- Python compileall, deploy/package shell syntax, and git whitespace checks passed. Markdown local-link audit found no broken links or author-local paths.
- Separate loopback process smoke: `/health/live` 200 and `/health/ready` 503 with an empty key; the owned test process was stopped afterward.
- Independent contract review reproduced and then verified fixes for full-queue shutdown and claim-commit/register shutdown races (two regression tests passed independently).
- Independent security review passed after adding the streamed 2 MiB decompressed-response cap, no-follow redirects, bounded full request/read duration, and no error-body consumption.
- No real key was provided. Real StepFun permissions/accuracy, live PostgreSQL migrations and corpus integration, Docker/systemd deployment remain unverified.
