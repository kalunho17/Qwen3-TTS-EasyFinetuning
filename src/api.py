import os
import uuid
import torch
import soundfile as sf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager

# Import the model class and utilities used in cli.py
from qwen_tts import Qwen3TTSModel
from utils import resolve_speaker_choice, resolve_path

# --- Configuration ---
CHECKPOINT = os.getenv("INFERENCE_CHECKPOINT", "/workspace/output/maxpayne/checkpoint-step-200")
SPEAKER = os.getenv("INFERENCE_SPEAKER", "my_speaker")
GPU_DEVICE = os.getenv("GPU_DEVICE", "cuda:0")
OUTPUT_DIR = "/workspace/output/api_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global model variable
MODEL = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown events."""
    global MODEL
    print(f"🚀 Loading Qwen3-TTS model from: {CHECKPOINT}")
    try:
        # Exact instantiation used in your cli.py
        MODEL = Qwen3TTSModel.from_pretrained(
            CHECKPOINT,
            device_map=GPU_DEVICE,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if "cuda" in GPU_DEVICE else None,
        )
        print("✅ Model loaded and ready for inference.")
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Failed to load model: {e}")
    yield
    # Cleanup logic (if any) would go here
    del MODEL

app = FastAPI(title="Qwen3-TTS Dedicated API", lifespan=lifespan)

class TTSRequest(BaseModel):
    text: str
    language: str = "English"
    instruct: str = None

@app.post("/generate")
async def generate_audio(request: TTSRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    output_filename = f"{uuid.uuid4().hex}.wav"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    try:
        # Replicating the logic from cmd_infer in cli.py
        supported_speakers = MODEL.get_supported_speakers() if hasattr(MODEL, 'get_supported_speakers') else []
        resolved_speaker = resolve_speaker_choice(SPEAKER, supported_speakers)
        
        # Generate audio
        wavs, sr = MODEL.generate_custom_voice(
            text=request.text,
            speaker=resolved_speaker,
            language=request.language,
            instruct=request.instruct
        )
        
        # Save to disk
        sf.write(output_path, wavs[0], sr)
        
        return FileResponse(
            output_path, 
            media_type="audio/wav", 
            filename=f"generated_{request.language}.wav"
        )
        
    except Exception as e:
        print(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {
        "status": "ready" if MODEL else "loading",
        "checkpoint": CHECKPOINT,
        "speaker": SPEAKER
    }