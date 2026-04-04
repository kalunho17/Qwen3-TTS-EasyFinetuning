FROM nvcr.io/nvidia/pytorch:25.01-py3

# Override when building: docker build --build-arg FINETUNE_REPO_URL=https://github.com/you/Qwen3-TTS-EasyFinetuning.git .
ARG FINETUNE_REPO_URL=https://github.com/kalunho17/Qwen3-TTS-EasyFinetuning.git

ENV TORCH_CUDA_ARCH_LIST="12.0" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/hf-cache \
    MODELSCOPE_CACHE=/workspace/ms-cache \
    PIP_NO_CACHE_DIR=1 \
    MAX_JOBS=4 \
    NVCC_THREADS=1 \
    USE_FLASH_ATTN=1 \
    FLASH_ATTENTION_FORCE_BUILD=1 \
    FINETUNE_BASE=/workspace/finetune-repo \
    PYTHONPATH="/workspace/finetune-repo:/workspace/finetune-repo/src"

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ffmpeg sox libsox-dev libsndfile1 git build-essential ninja-build \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN mkdir -p \
    /workspace/models/Qwen \
    /workspace/raw-dataset \
    /workspace/final-dataset \
    /workspace/output \
    /workspace/hf-cache \
    /workspace/ms-cache

RUN git clone "${FINETUNE_REPO_URL}" /workspace/finetune-repo

# Symlink so the webui's internal relative paths still resolve
RUN ln -s /workspace/final-dataset /workspace/finetune-repo/final-dataset

RUN pip install --upgrade pip "setuptools<81.0.0" wheel ninja

RUN pip uninstall -y torch torchvision torchaudio flash-attn 2>/dev/null || true && \
    pip install --pre torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/nightly/cu128

RUN pip install --no-deps qwen-tts qwen-asr qwen-omni-utils
RUN pip install \
    "transformers>=4.48.0" "accelerate>=1.1.0" \
    librosa soundfile vector-quantize-pytorch vocos gradio \
    funasr modelscope openpyxl tensorboard \
    peft bitsandbytes sentencepiece openai-whisper

WORKDIR /workspace/finetune-repo
RUN sed -i '/torch/d; /flash-attn/d' requirements.txt && \
    pip install -r requirements.txt

RUN pip install flash-attn --no-build-isolation --no-cache-dir

# Verify flash-attn and that dataset.py has no syntax errors
RUN python3 -c "import flash_attn; print('flash_attn', flash_attn.__version__)"
RUN python3 -c "import py_compile; py_compile.compile('src/dataset.py', doraise=True); print('dataset.py syntax OK')"

# Inline entrypoint: locks cwd before anything runs so every subprocess
# inherits the correct working directory and relative paths resolve properly.
RUN printf '#!/bin/bash\nset -e\ncd "${FINETUNE_BASE:-/workspace/finetune-repo}"\nexec "$@"\n' \
    > /entrypoint.sh && chmod +x /entrypoint.sh

EXPOSE 7860 6006

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python3", "src/webui.py"]
