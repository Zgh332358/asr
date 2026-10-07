# Repository guidance

This fork is the StepFun cloud ASR gateway for Revoice Resonance. Read [README.md](README.md), [API.md](API.md), and [the approved migration specification](docs/specs/stepfun-preview.md) before changing model integration.

- Active entry point: `python -m app.main`; `whisper_api.py` is a compatibility wrapper.
- `app/config.py` uses pydantic-settings and rejects unknown `.env` fields. Keep `OPENAI_API_KEY` server-only, blank in templates, and out of responses/logging/commits.
- `TranscriptionWorker` preserves a bounded queue and per-client limits; it now calls StepFun `/chat/completions` with a real WAV data URL. No local GPU/Whisper dependencies are required.
- `DATABASE_URL` empty means stateless. Corpus mode requires SQL migrations 001 and 002, local file storage, and the existing scheduler/callback flow. Never rewrite historical task identities. Cache reuse must match both engine and model before selecting a result.
- Model output does not provide timestamps or calibrated confidence: use empty segments/null confidence and describe language as a hint.
- Missing key is an explicit 503 for inference and readiness. Do not fabricate successful output or claim live provider validation from mocked tests.
- Run `python -m pytest -q`, `python -m compileall -q app tests whisper_api.py`, and `bash -n deploy.sh package_model.sh` after relevant changes. Tests require ffmpeg but no GPU, real key, or database.
- Source publication targets the user's fork. Do not publish or deploy to the original organization. Existing deployment/updater commands are operator-invoked tools; tests must not execute them against a real host.

Issue-tracker/domain conventions remain in [docs/agents](docs/agents/domain.md). `TWIN-REVIEW-REPORT.md` is historical context for the former Whisper implementation, not current validation evidence.
