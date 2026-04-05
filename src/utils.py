import os
import re
import shutil


def get_project_root():
    '''Detect the project root directory.
    FINETUNE_BASE overrides everything when set (e.g. Docker clone at /workspace/finetune-repo).
    In Docker without FINETUNE_BASE, default is /workspace (legacy layout).
    Otherwise, it's the parent directory of this src file.
    '''
    env_base = os.environ.get("FINETUNE_BASE")
    if env_base:
        env_base = os.path.abspath(env_base)
        if os.path.isdir(env_base):
            return env_base
    if os.path.exists('/.dockerenv') or os.environ.get('IS_DOCKER'):
        return '/workspace'
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_path(path):
    '''Normalize a path to be absolute within the project root if it is relative.'''
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(get_project_root(), path))


def get_models_root():
    """Root directory for downloaded models (repo id → <root>/Qwen/...).

    Docker Compose typically mounts host ./models at /workspace/models while FINETUNE_BASE
    points at /workspace/finetune-repo. Without this, models land under finetune-repo/models
    and disagree with ASR loaded from /workspace/models/....

    Override with FINETUNE_MODELS_DIR or MODELS_DIR (absolute or relative to cwd is not used —
    non-absolute values are resolved from project root).
    """
    for key in ("FINETUNE_MODELS_DIR", "MODELS_DIR"):
        raw = os.environ.get(key)
        if raw:
            if os.path.isabs(raw):
                return os.path.abspath(raw)
            return resolve_path(raw)

    # Dockerfile sets FINETUNE_BASE=/workspace/finetune-repo; weights belong in sibling
    # /workspace/models (compose bind mount). This does not depend on string prefix checks
    # that can fail with odd abspath/cwd combinations.
    finetune_base_raw = os.environ.get("FINETUNE_BASE", "").strip()
    if finetune_base_raw:
        fb_abs = os.path.normpath(os.path.abspath(finetune_base_raw))
        if os.path.isdir(fb_abs) and os.path.basename(fb_abs) == "finetune-repo":
            return os.path.join(os.path.dirname(fb_abs), "models")

    project_root = os.path.normpath(get_project_root())
    workspace = os.path.normpath("/workspace")
    sep = os.sep
    if project_root == workspace or project_root.startswith(workspace + sep):
        return os.path.join(workspace, "models")
    return os.path.join(project_root, "models")


def get_outputs_root(cli_override=None):
    """Root for training checkpoints: <root>/<experiment_name>/checkpoint-epoch-*/...

    Resolution order:
    1. ``cli_override`` (train --output_root) if non-empty
    2. ``FINETUNE_OUTPUT_DIR`` (absolute or relative to project root)
    3. If ``FINETUNE_BASE`` is .../finetune-repo, sibling ``.../output`` (Docker: /workspace/output)
    4. Else ``<project_root>/output``
    """
    if cli_override is not None and str(cli_override).strip():
        raw = str(cli_override).strip()
        if os.path.isabs(raw):
            return os.path.abspath(raw)
        return resolve_path(raw)

    env_raw = os.environ.get("FINETUNE_OUTPUT_DIR", "").strip()
    if env_raw:
        if os.path.isabs(env_raw):
            return os.path.abspath(env_raw)
        return resolve_path(env_raw)

    finetune_base_raw = os.environ.get("FINETUNE_BASE", "").strip()
    if finetune_base_raw:
        fb_abs = os.path.normpath(os.path.abspath(finetune_base_raw))
        if os.path.isdir(fb_abs) and os.path.basename(fb_abs) == "finetune-repo":
            return os.path.join(os.path.dirname(fb_abs), "output")

    return os.path.join(get_project_root(), "output")


def resolve_audio_file_path(path: str) -> str:
    """Resolve a JSONL audio path for librosa: absolute path under project root, file must exist."""
    if not path:
        raise ValueError(
            "Audio path is empty in JSONL. "
            "Re-run the ASR and tokenisation steps and check tts_train.jsonl."
        )
    if path.startswith(("http://", "https://")):
        return path
    root = get_project_root()
    if not os.path.isabs(path):
        path = os.path.join(root, path)
    path = os.path.normpath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path!r}")
    return path


def get_model_local_dir(model_id):
    return os.path.join(get_models_root(), model_id)



def is_model_dir_ready(path):
    if not path or not os.path.isdir(path):
        return False
    marker_files = (
        'config.json',
        'tokenizer_config.json',
        'preprocessor_config.json',
        'model.safetensors.index.json',
        'pytorch_model.bin.index.json',
    )
    if any(os.path.exists(os.path.join(path, marker)) for marker in marker_files):
        return True
    try:
        entries = os.listdir(path)
    except OSError:
        return False
    return any(entry.endswith(('.safetensors', '.bin', '.json', '.model')) for entry in entries)



def _cleanup_empty_dir(path):
    if not os.path.isdir(path) or os.path.islink(path):
        return
    try:
        if not os.listdir(path):
            os.rmdir(path)
    except OSError:
        pass



def _ensure_shared_model_dir(model_id, downloaded_path):
    local_dir = get_model_local_dir(model_id)
    resolved_downloaded_path = os.path.realpath(downloaded_path) if downloaded_path else downloaded_path
    resolved_local_dir = os.path.realpath(local_dir)

    if is_model_dir_ready(local_dir):
        return local_dir

    if not downloaded_path or not is_model_dir_ready(downloaded_path):
        return downloaded_path

    if resolved_downloaded_path == resolved_local_dir:
        return local_dir

    os.makedirs(os.path.dirname(local_dir), exist_ok=True)
    _cleanup_empty_dir(local_dir)

    if not os.path.exists(local_dir):
        try:
            os.symlink(downloaded_path, local_dir, target_is_directory=True)
            print(f'Linked shared model directory {local_dir} -> {downloaded_path}')
            return local_dir
        except OSError:
            pass

    if not is_model_dir_ready(local_dir):
        shutil.copytree(downloaded_path, local_dir, dirs_exist_ok=True)
        print(f'Copied model into shared directory {local_dir}')

    return local_dir if is_model_dir_ready(local_dir) else downloaded_path



def is_model_downloaded(model_id):
    local_dir = get_model_local_dir(model_id)
    return is_model_dir_ready(local_dir)



def get_model_path(model_id, use_hf=False):
    resolved_input = resolve_path(model_id)
    if os.path.exists(resolved_input):
        return resolved_input
    output_candidate = os.path.join(get_outputs_root(), model_id)
    if os.path.exists(output_candidate):
        print(f'Found local checkpoint at {output_candidate}')
        return output_candidate
    local_dir = get_model_local_dir(model_id)
    if is_model_downloaded(model_id):
        print(f'Found local model at {local_dir}, skipping download!')
        return local_dir
    print(f'Downloading model {model_id} into {local_dir}...')
    os.makedirs(os.path.dirname(local_dir), exist_ok=True)
    try:
        if use_hf:
            from huggingface_hub import snapshot_download
            downloaded_path = snapshot_download(repo_id=model_id, local_dir=local_dir)
        else:
            from modelscope import snapshot_download
            downloaded_path = snapshot_download(model_id, cache_dir=get_models_root())
        return _ensure_shared_model_dir(model_id, downloaded_path)
    except Exception as e:
        print(f'Warning: Download failed, falling back to id: {e}')
        return model_id


def speaker_key(value):
    return re.sub(r'[^a-z0-9]+', '', str(value).lower())

def resolve_speaker_choice(speaker, supported_speakers):
    if not speaker or not supported_speakers:
        return speaker
    if speaker in supported_speakers:
        return speaker
    lower_map = {str(s).lower(): s for s in supported_speakers}
    lowered = str(speaker).lower()
    if lowered in lower_map:
        return lower_map[lowered]
    normalized = speaker_key(speaker)
    normalized_map = {}
    for s in supported_speakers:
        normalized_map.setdefault(speaker_key(s), s)
    return normalized_map.get(normalized, speaker)
