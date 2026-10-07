-- Apply after 001_init.sql on new databases, or once to an existing database.
-- Keeps all historical engine values and rows; does not rewrite task identity.
BEGIN;
ALTER TABLE asr_tasks DROP CONSTRAINT IF EXISTS ck_asr_engine;
ALTER TABLE asr_tasks ADD CONSTRAINT ck_asr_engine CHECK (
    asr_engine IN ('STEPFUN', 'WHISPER', 'AZURE', 'ALIYUN', 'TENCENT', 'HUAWEI')
);
ALTER TABLE asr_tasks ALTER COLUMN asr_engine SET DEFAULT 'STEPFUN';
COMMENT ON COLUMN asr_tasks.asr_engine IS 'ASR引擎: STEPFUN (new tasks), historical engines retained';
COMMIT;
