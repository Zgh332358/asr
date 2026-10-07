# Resonance ASR · StepFun Preview

This fork provides the speech recognition and corpus service for Revoice Resonance. It keeps the FastAPI upload/task API and replaces local Whisper inference with **`stepaudio-3-chat-preview`** at **`https://api.stepfun.com/v1`**. A CPU machine is sufficient; no CUDA, GPU, downloaded model, or local Whisper package is required.

The service converts uploaded audio into 16 kHz mono WAV, sends a transcription-only instruction with audio to StepFun Chat Completions, and returns the model's text. It preserves the optional PostgreSQL corpus registry, task history, file storage, and completion callback. Multimodal Step 5 integration belongs to the application/backend repositories; this repository implements ASR only.

## Start locally

Requirements: Python 3.10 or later (tested with 3.12), `ffmpeg` and `ffprobe`, and a StepFun account/key with access to the configured model. On macOS, `brew install ffmpeg` supplies both audio tools.

```bash
git clone https://github.com/Zgh332358/asr.git
cd asr
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set OPENAI_API_KEY when your key is available.
python -m app.main
```

The template binds to `127.0.0.1:8080`. A missing key does not prevent startup: `/health/live` returns 200, while `/health/ready` and transcription return **503**. After adding a key, restart the process. Readiness confirms local configuration/client initialization; it does not verify your balance, model access, or provider availability.

```dotenv
OPENAI_API_KEY=
OPENAI_BASE_URL=https://api.stepfun.com/v1
ASR_MODEL=stepaudio-3-chat-preview
REQUEST_TIMEOUT_SECONDS=120
```

Keep the real key in the server's private `.env` or deployment secret store. This file is ignored by Git. Never put the provider key in a browser or client build. `OPENAI_BASE_URL` and `ASR_MODEL` are configurable server settings; an uploaded `model` form field cannot change them.

```bash
curl http://127.0.0.1:8080/health/live
curl -X POST http://127.0.0.1:8080/v1/audio/transcriptions \
  -F 'file=@audio.wav' -F 'language=zh' -F 'response_format=json'
```

The response is `{"text":"转写文本"}`. The local endpoint accepts common ffmpeg-decodable formats. Its name is OpenAI-compatible, but the upstream request is **`POST /chat/completions`**, using a complete `data:audio/wav;base64,...` value in `input_audio.data`; it does not send this model to StepFun's `/audio/transcriptions` endpoint. See [API.md](API.md).

## Corpus mode and existing data

Leave `DATABASE_URL` empty for stateless mode. To enable corpus and task routes:

1. Prepare a PostgreSQL database and apply [001_init.sql](app/migrations/001_init.sql), then [002_stepfun_engine.sql](app/migrations/002_stepfun_engine.sql). An existing database initialized with 001 only needs **002**. Back up existing data before running a migration.
2. Set `DATABASE_URL=postgresql+asyncpg://...` in the private server configuration, and provide a writable `STORAGE_PATH`.
3. Restart the service. The scheduler runs when the cloud worker is configured.

Migration 002 allows the `STEPFUN` engine and changes its database default. It preserves all historical rows and old engine names. New uploads use `STEPFUN` and record the configured model identity; reuse of a cached result requires matching audio MD5, engine, and model. A historical Whisper result or a result from another StepFun model is not reused. Old pending tasks, or pending tasks whose model identity is missing/different, are explicitly failed; upload again to create a new task under the current configuration. Historical completed tasks remain readable.

API routes are unchanged: upload/list/detail under `/api/v1/asr/corpus`, list/detail under `/api/v1/asr/tasks`. New task creation supports `asr_engine=STEPFUN` only. A missing database returns 503 for these routes. A missing provider key blocks new transcription uploads and prevents the scheduler claiming pending jobs. This repository contains schema and service code, **not a populated speech corpus or a training pipeline**.

## Deployment and migration from the original service

- Back up the old `.env` and create a new one from the current `.env.example`. Copy only settings still present in the template. The application rejects unknown `.env` fields; old `MODEL_*`, `HF_MODEL_ID`, `VAD_*`, and deployment-script-only variables must be removed.
- `bash deploy.sh setup/start/stop/restart/status/logs` remains available. Setup creates an isolated environment and requires installed ffmpeg tools; it no longer downloads models. Deployment script options such as `GIT_REMOTE` are supplied through shell environment variables, not `.env`.
- `docker compose up --build -d` runs the CPU gateway and persists corpus audio in a volume. Its published port is loopback-only by default. Docker readiness stays unhealthy until a key is configured.
- Existing `whisper_api.py` and `whisper-asr.service` names are compatibility entry points. They launch the same cloud app. `package_model.sh` is retired and exits with migration guidance; `MODEL_URL` is no longer used.
- `deploy.sh update` fetches the configured remote and uses a fast-forward pull; the configured remote is checked and must be `Zgh332358/asr`. Automatic updates are opt-in. No deployment or scheduled updater is enabled by these source changes.
- The local API relies on an upstream authenticated gateway and has no API-key validation of its own. Keep it on loopback/private networking, or supply your gateway's authentication and request limits before exposing it.

## Behavior and limits

- Step Audio 3 is a chat audio-understanding model instructed to transcribe. It may paraphrase, omit content, or misunderstand speech; changing the model is not evidence of improved dysarthric-speech accuracy. Evaluate it on consented real samples before use.
- `verbose_json` returns decoded duration, the requested/default **language hint**, and `segments: []`. No word timestamps, detected-language claim, or calibrated confidence is fabricated. Corpus confidence is `null`. The legacy `temperature` form parameter is accepted for client compatibility but is not sent to the provider.
- The retained local admission defaults are 500 MB upload and 600 seconds. These are local ceilings, not a promise that the provider accepts that payload. Start with short recordings and lower the settings to your deployment/provider limits. Base64 WAV increases the outbound size; requests are bounded by `REQUEST_TIMEOUT_SECONDS` including reading the response. Successful provider bodies are capped at 2 MiB after decompression; error bodies are not read, and redirects are not followed.
- Provider/network/invalid-output failures are explicit sanitized errors. There is no mock fallback or fake successful transcript, and provider response bodies are not included in task errors/callbacks.
- `/health/gpu` returns 404 because inference is remote. `/health/ready` retains compatibility fields `model_loaded` (client initialized) and `gpu_available=false`; GPU availability no longer determines readiness.
- Existing `TASK_MAX_RETRIES` is retained for configuration compatibility; the scheduler does not automatically retry failed inference jobs. A repeated upload creates a new task when no matching successful result exists. Callback retries remain configurable.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
python -m compileall -q app tests whisper_api.py
bash -n deploy.sh package_model.sh
```

Tests cover the request/data-URL/WAV contract, transcript/error handling, timeout through response-body reading, queue cancellation, no-key startup, multipart uploads with ffmpeg, model-aware cache query construction, task-identity rejection, and scheduler shutdown. They use HTTP mocks and do not spend API credits. PostgreSQL query construction and mocked task lifecycles do not replace a live database migration/integration test.

Real StepFun requests, account/model permissions, target-user recognition quality, and a live PostgreSQL migration remain to be verified with your credentials/infrastructure. Docker and systemd deployment require separate testing on their target hosts.

Official contract references: [Chat Completions](https://platform.stepfun.com/docs/zh/api-reference/chat/chat-completion-create), [Step Audio 3 guide](https://platform.stepfun.com/docs/zh/guides/models/stepaudio-3-realtime). The original project is [revoice-resonance/asr](https://github.com/revoice-resonance/asr); this fork retains its [MIT license](LICENSE).
