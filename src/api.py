import os
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse
import sys

# Ensure the library is in the path
sys.path.append("/workspace/finetune-repo")
from src.cli import main as cli_main

app = FastAPI(title="Qwen3-TTS API")

# Default settings from environment variables
CHECKPOINT = os.getenv("INFERENCE_CHECKPOINT", "")
SPEAKER = os.getenv("INFERENCE_SPEAKER", "my_speaker")
OUTPUT_DIR = "/workspace/output/api_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

class TTSRequest(BaseModel):
    text: str
    language: str = "English"

@app.post("/infer")
async def infer(request: TTSRequest):
    if not CHECKPOINT:
        raise HTTPException(status_code=500, detail="INFERENCE_CHECKPOINT env var not set.")
    
    filename = f"api_{uuid.uuid4().hex}.wav"
    output_path = os.path.join(OUTPUT_DIR, filename)
    
    try:
        # Programmatically calling your CLI logic
        args = [
            "infer",
            "--checkpoint", CHECKPOINT,
            "--speaker", SPEAKER,
            "--text", request.text,
            "--output", output_path,
            "--language", request.language
        ]
        cli_main(args)
        return FileResponse(output_path, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "online", "checkpoint": CHECKPOINT}