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
    root = get_project_root()
    return os.path.join(root, 'models', model_id)



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
    output_candidate = resolve_path(os.path.join('output', model_id))
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
            downloaded_path = snapshot_download(model_id, cache_dir=os.path.join(get_project_root(), 'models'))
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
