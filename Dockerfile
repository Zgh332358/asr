# CPU-only gateway; inference runs on StepFun.
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*
RUN groupadd -r asr && useradd -r -g asr -d /app asr
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ ./app/
RUN mkdir -p /tmp/whisper_api /data/asr_storage && \
    chown -R asr:asr /app /tmp/whisper_api /data/asr_storage
USER asr
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health/ready || exit 1
EXPOSE 8080
CMD ["python", "-m", "app.main"]
