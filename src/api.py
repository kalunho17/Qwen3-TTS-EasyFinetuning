import gc
import json
import os
import threading
import uuid
from typing import Optional, Tuple

import torch
import soundfile as sf
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager

# Import the model class and utilities used in cli.py
from qwen_tts import Qwen3TTSModel
from utils import resolve_speaker_choice

# --- Configuration ---
def _parse_inference_checkpoint(raw: str) -> Tuple[str, Optional[str]]:
    """
    Returns (output_root_or_resolved_base, initial_checkpoint_full_path_or_none).

    If INFERENCE_CHECKPOINT points at a checkpoint directory (basename starts with
    'checkpoint-'), output root is its parent's parent (.../<project>/<ckpt> -> root .../output).
    Initial load uses that checkpoint path.

    Otherwise the path is treated as the output root (e.g. /workspace/output) and no model
    is loaded until POST /checkpoint/load.
    """
    expanded = os.path.abspath(os.path.expanduser(raw.strip()))
    if not os.path.isdir(expanded):
        return expanded, None
    base = os.path.basename(expanded)
    if base.startswith("checkpoint-"):
        output_root = str(Path(expanded).parent.parent)
        return output_root, expanded
    return expanded, None


_RAW_CHECKPOINT = os.getenv(
    "INFERENCE_CHECKPOINT", "/workspace/output/maxpayne/checkpoint-step-200"
)
OUTPUT_ROOT, _INITIAL_CHECKPOINT = _parse_inference_checkpoint(_RAW_CHECKPOINT)
SPEAKER = os.getenv("INFERENCE_SPEAKER", "my_speaker")
GPU_DEVICE = os.getenv("GPU_DEVICE", "cuda:0")
OUTPUT_DIR = "/workspace/output/api_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL = None
LOADED_CHECKPOINT_PATH: Optional[str] = None
MODEL_LOCK = threading.Lock()


def _checkpoint_sort_key(output_root: str, project: str, checkpoint_name: str):
    checkpoint_dir = os.path.join(output_root, project, checkpoint_name)
    trainer_state_path = os.path.join(checkpoint_dir, "trainer_state.json")
    global_step = -1
    epoch = -1
    if os.path.exists(trainer_state_path):
        try:
            with open(trainer_state_path, "r", encoding="utf-8") as f:
                trainer_state = json.load(f)
            global_step = int(trainer_state.get("global_step", -1))
            epoch = int(trainer_state.get("epoch", -1))
        except Exception:
            pass
    return (global_step, epoch, checkpoint_name)


def _is_checkpoint_dir(name: str) -> bool:
    return name.startswith("checkpoint-step-") or name.startswith("checkpoint-epoch-")


def list_inventory() -> dict:
    """Projects under OUTPUT_ROOT and checkpoint-* dirs per project."""
    if not os.path.isdir(OUTPUT_ROOT):
        return {"output_root": OUTPUT_ROOT, "projects": []}
    projects_out = []
    for name in sorted(os.listdir(OUTPUT_ROOT)):
        exp_dir = os.path.join(OUTPUT_ROOT, name)
        if not os.path.isdir(exp_dir):
            continue
        ckpts = [
            item
            for item in os.listdir(exp_dir)
            if os.path.isdir(os.path.join(exp_dir, item)) and _is_checkpoint_dir(item)
        ]
        ckpts.sort(
            key=lambda c: _checkpoint_sort_key(OUTPUT_ROOT, name, c), reverse=True
        )
        projects_out.append({"name": name, "checkpoints": ckpts})
    return {"output_root": OUTPUT_ROOT, "projects": projects_out}


def _realpath_under_root(root: str, *relative_parts: str) -> str:
    root_real = os.path.realpath(root)
    candidate = os.path.realpath(os.path.join(root_real, *relative_parts))
    root_prefix = root_real if root_real.endswith(os.sep) else root_real + os.sep
    if candidate != root_real and not candidate.startswith(root_prefix):
        raise HTTPException(status_code=400, detail="Checkpoint path escapes output root")
    return candidate


def _unload_model_locked():
    global MODEL, LOADED_CHECKPOINT_PATH
    old = MODEL
    MODEL = None
    LOADED_CHECKPOINT_PATH = None
    del old
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_model_at(checkpoint_path: str):
    global MODEL, LOADED_CHECKPOINT_PATH
    _unload_model_locked()
    new_model = Qwen3TTSModel.from_pretrained(
        checkpoint_path,
        device_map=GPU_DEVICE,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if "cuda" in GPU_DEVICE else None,
    )
    MODEL = new_model
    LOADED_CHECKPOINT_PATH = checkpoint_path


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, LOADED_CHECKPOINT_PATH
    if _INITIAL_CHECKPOINT:
        print(f"🚀 Loading Qwen3-TTS model from: {_INITIAL_CHECKPOINT}")
        try:
            with MODEL_LOCK:
                _load_model_at(_INITIAL_CHECKPOINT)
            print("✅ Model loaded and ready for inference.")
        except Exception as e:
            print(f"❌ CRITICAL ERROR: Failed to load model: {e}")
    else:
        print(
            f"📂 Output root is {OUTPUT_ROOT!r} (no initial checkpoint). "
            "Use POST /checkpoint/load when ready."
        )
    yield
    with MODEL_LOCK:
        _unload_model_locked()


app = FastAPI(title="Qwen3-TTS Dedicated API", lifespan=lifespan)


class TTSRequest(BaseModel):
    text: str
    language: str = "English"
    instruct: str = None


class LoadCheckpointRequest(BaseModel):
    """Load a checkpoint under OUTPUT_ROOT."""

    project: str = Field(..., description="Experiment folder, e.g. maxpayne")
    checkpoint: str = Field(
        ..., description="Checkpoint folder name, e.g. checkpoint-step-200"
    )


class LoadCheckpointByRelPathRequest(BaseModel):
    """Alternative: single relative path project/checkpoint-step-200."""

    relative_path: str = Field(
        ...,
        description="Path under output root, e.g. maxpayne/checkpoint-step-200",
    )


@app.post("/generate")
async def generate_audio(request: TTSRequest):
    output_filename = f"{uuid.uuid4().hex}.wav"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    loaded_path: Optional[str] = None

    try:
        with MODEL_LOCK:
            if MODEL is None:
                raise HTTPException(status_code=503, detail="Model not loaded")
            model = MODEL
            loaded_path = LOADED_CHECKPOINT_PATH

            supported_speakers = (
                model.get_supported_speakers()
                if hasattr(model, "get_supported_speakers")
                else []
            )
            resolved_speaker = resolve_speaker_choice(SPEAKER, supported_speakers)

            wavs, sr = model.generate_custom_voice(
                text=request.text,
                speaker=resolved_speaker,
                language=request.language,
                instruct=request.instruct,
            )

        sf.write(output_path, wavs[0], sr)

        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename=f"generated_{request.language}.wav",
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Inference error ({loaded_path}): {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    with MODEL_LOCK:
        loaded = LOADED_CHECKPOINT_PATH
        ready = MODEL is not None
    return {
        "status": "ready" if ready else "no_model",
        "output_root": OUTPUT_ROOT,
        "loaded_checkpoint": loaded,
        "inference_checkpoint_env": _RAW_CHECKPOINT,
        "speaker": SPEAKER,
    }


@app.get("/checkpoint/inventory")
async def checkpoint_inventory():
    """List projects and checkpoint-* folders under INFERENCE_CHECKPOINT (output root)."""
    return list_inventory()


@app.post("/checkpoint/load")
async def checkpoint_load(body: LoadCheckpointRequest):
    rel = os.path.normpath(os.path.join(body.project, body.checkpoint))
    if rel.startswith("..") or os.path.isabs(rel):
        raise HTTPException(status_code=400, detail="Invalid project/checkpoint")
    ckpt_path = _realpath_under_root(OUTPUT_ROOT, body.project, body.checkpoint)
    if not os.path.isdir(ckpt_path):
        raise HTTPException(status_code=404, detail=f"Not a directory: {ckpt_path}")
    if not _is_checkpoint_dir(os.path.basename(ckpt_path)):
        raise HTTPException(
            status_code=400,
            detail="Folder must be named checkpoint-step-* or checkpoint-epoch-*",
        )
    try:
        with MODEL_LOCK:
            print(f"🚀 Loading checkpoint: {ckpt_path}")
            _load_model_at(ckpt_path)
        print("✅ Model loaded.")
        return {"status": "ok", "loaded_checkpoint": ckpt_path}
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Load failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/checkpoint/load-path")
async def checkpoint_load_path(body: LoadCheckpointByRelPathRequest):
    rel = os.path.normpath(body.relative_path.strip().lstrip("/"))
    if not rel or rel.startswith(".."):
        raise HTTPException(status_code=400, detail="Invalid relative_path")
    parts = rel.split(os.sep)
    if len(parts) < 2:
        raise HTTPException(
            status_code=400,
            detail="relative_path must be like project/checkpoint-step-200",
        )
    ckpt_path = _realpath_under_root(OUTPUT_ROOT, *parts)
    if not os.path.isdir(ckpt_path):
        raise HTTPException(status_code=404, detail=f"Not a directory: {ckpt_path}")
    if not _is_checkpoint_dir(os.path.basename(ckpt_path)):
        raise HTTPException(
            status_code=400,
            detail="Leaf folder must be checkpoint-step-* or checkpoint-epoch-*",
        )
    try:
        with MODEL_LOCK:
            print(f"🚀 Loading checkpoint: {ckpt_path}")
            _load_model_at(ckpt_path)
        print("✅ Model loaded.")
        return {"status": "ok", "loaded_checkpoint": ckpt_path}
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Load failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/checkpoint/unload")
async def checkpoint_unload():
    with MODEL_LOCK:
        _unload_model_locked()
    print("📤 Model unloaded.")
    return {"status": "ok", "loaded_checkpoint": None}
