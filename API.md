# StepFun-backed ASR API

Base URL for local use: `http://127.0.0.1:8080`. Provider credentials remain on this server. The API itself relies on a private network/authenticated upstream gateway.

| Method | Route | Behavior |
|---|---|---|
| GET | `/health/live` | 200 if the process is running |
| GET | `/health/ready` | 200 if cloud client is initialized; 503 without a key; does not perform paid inference |
| GET | `/health/gpu` | 404: cloud inference has no local GPU |
| POST | `/v1/audio/transcriptions` | Stateless multipart upload and transcription |
| POST | `/api/v1/asr/corpus` | Store an upload, create a task or reuse a matching result; requires DB and configured key |
| GET | `/api/v1/asr/corpus` | Paginated corpus list; requires DB |
| GET | `/api/v1/asr/corpus/{id}` | Corpus details and task summaries; requires DB |
| GET | `/api/v1/asr/tasks` | Paginated task list; requires DB |
| GET | `/api/v1/asr/tasks/{id}` | Full task result/history; requires DB |

## Stateless transcription

```bash
curl -X POST http://127.0.0.1:8080/v1/audio/transcriptions \
  -F 'file=@recording.wav' -F 'language=zh' -F 'response_format=verbose_json'
```

`file` is required; any ffmpeg-decodable audio is normalized to mono 16 kHz WAV. `language` is optional metadata, falling back to `DEFAULT_LANGUAGE`. `response_format` is `json` (default) or `verbose_json`. Legacy `temperature` (0–1) is accepted but ignored. An optional client `model` field is ignored; `ASR_MODEL` is controlled by the server.

```json
{"text":"我想喝水","language":"zh","duration":1.25,"segments":[]}
```

For `json`, only `text` is returned. `duration` is calculated from decoded samples. `language` is a hint, not detected metadata. No fabricated timestamps or probability fields are supplied.

The server calls `${OPENAI_BASE_URL}/chat/completions` using `ASR_MODEL` (default `stepaudio-3-chat-preview`), a transcription-only system instruction, `messages[1].content[0].input_audio.data` containing `data:audio/wav;base64,...`, and `stream:false`. It reads `choices[0].message.content`. This chat model requires evaluation for faithful transcription; it is not a guarantee of word-for-word accuracy.

Error status: 400 for invalid audio, 413 for oversized uploads, 422 for invalid form fields, 429 for local rate limit, 503 for missing cloud configuration/queue capacity/provider rate limit, 504 for provider timeout, and 502 for provider/network/invalid-transcript failures. Provider error text and secrets are not returned.

## Corpus upload and query

Initialize the database as described in [README.md](README.md), then:

```bash
curl -X POST http://127.0.0.1:8080/api/v1/asr/corpus \
  -F 'file=@recording.wav' -F 'language=zh-CN' \
  -F 'asr_engine=STEPFUN' -F 'business_id=example-id' -F 'tags=practice'
```

`business_id`, `business_type`, and comma-separated `tags` are optional. The only supported new-task engine is `STEPFUN`. The response contains `corpus_id`, `task_id`, `file_md5`, `cached`, `status`, and optionally cached `result_text`. New tasks progress PENDING → PROCESSING → SUCCESS/FAILED. Poll `/api/v1/asr/tasks/{task_id}` for `result_text` or `error_message`.

Dedup uses the file MD5. Successful result reuse additionally requires engine `STEPFUN` and an exact configured model match. Changing `ASR_MODEL` does not relabel or reuse old results. Old pending engine/model identities fail explicitly; upload again to create a new task. `engine_config` records only non-secret model/protocol identity. A successful chat transcript has `confidence:null` and empty segments in `result_detail`.

Corpus list filters: `business_id`, `business_type`, `status`, `is_deleted`, `page` (≥1), `page_size` (1–100). Task list filters: `status`, `corpus_id`, `page`, `page_size`. Missing DB returns 503. Missing records return 404.

Optional `MAIN_BACKEND_CALLBACK_URL` receives task/corpus IDs, status, and text/null confidence on success or a sanitized error on failure. Controlled shutdown persists interrupted tasks as FAILED; callbacks are skipped for shutdown interruptions. Read task history after restart. Callback delivery is best effort and is not backed by a durable outbox.
