import os
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse
from src.cli import main as cli_main # Import the existing CLI logic

app = FastAPI(title="Qwen3-TTS Inference Service")

# Configuration via Environment Variables
CHECKPOINT = os.getenv("INFERENCE_CHECKPOINT", "/workspace/output/maxpayne/checkpoint-step-200")
SPEAKER = os.getenv("INFERENCE_SPEAKER", "my_speaker")
OUTPUT_DIR = "/workspace/output/api_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

class TTSRequest(BaseModel):
    text: str
    language: str = "English"

@app.post("/generate")
async def generate_audio(request: TTSRequest):
    output_filename = f"{uuid.uuid4()}.wav"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    # We call the CLI logic programmatically 
    # This ensures we use the exact same preprocessing/inference pipeline
    try:
        # Construct arguments as if they were passed via terminal
        args = [
            "infer",
            "--checkpoint", CHECKPOINT,
            "--speaker", SPEAKER,
            "--text", request.text,
            "--output", output_path,
            "--language", request.language
        ]
        
        # Note: In a high-load scenario, you'd want to refactor cli.py 
        # to load the model into a global variable instead of re-running main()
        cli_main(args) 
        
        return FileResponse(output_path, media_type="audio/wav", filename=output_filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ready", "model": CHECKPOINT}