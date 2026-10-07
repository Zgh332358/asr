"""Compatibility entry point; local Whisper inference has been retired.

Use python -m app.main for the StepFun service. This legacy filename forwards
both direct execution and `uvicorn whisper_api:app` to the same cloud app.
"""
from app.main import app

if __name__ == "__main__":
    import runpy
    runpy.run_module("app.main", run_name="__main__")
