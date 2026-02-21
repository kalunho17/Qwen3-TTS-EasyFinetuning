import gradio as gr
import os
import subprocess
import threading
import json
import time
from data_pipeline import run_pipeline

# Check if model is running
def is_process_running(keyword):
    try:
        output = subprocess.check_output(["pgrep", "-f", keyword]).decode("utf-8").strip()
        return output != ""
    except:
        return False

# ----- Data Pipeline Tab -----
def run_data_prep(input_dir, ref_audio, output_dir, model_id, batch_size, model_source):
    if not os.path.exists(input_dir):
        return "Input directory does not exist!"
        
    os.makedirs(output_dir, exist_ok=True)
    
    # We ignore model_source for ASR for now if they are using ModelScope, else we can set ENV VAR
    if model_source == "HuggingFace":
        os.environ["USE_HF"] = "1"
    else:
        os.environ.pop("USE_HF", None)
        
    try:
        success, msg = run_pipeline(
            input_dir=input_dir,
            ref_audio=ref_audio,
            output_dir=output_dir,
            model_id=model_id,
            batch_size=batch_size
        )
        return msg
    except Exception as e:
        return f"Error during data preparation: {e}"

# ----- Training Tab -----
training_process = None

def start_training(raw_jsonl, speaker_name, init_model, model_source, batch_size, lr, epochs, grad_acc):
    global training_process
    
    if training_process is not None and training_process.poll() is None:
        return "Training is already running!"
    
    if not os.path.exists(raw_jsonl):
        return f"JSONL file {raw_jsonl} not found. Please run data prep first."
        
    # Launch training as a subprocess
    output_dir = "output"
    train_jsonl = raw_jsonl.replace(".jsonl", "_with_codes.jsonl")
    
    # Run prepare_data.py to extract codec
    prep_cmd = [
        "python", "prepare_data.py", 
        "--device", "cuda:0", 
        "--tokenizer_model_path", "Qwen/Qwen3-TTS-Tokenizer-12Hz", 
        "--input_jsonl", raw_jsonl, 
        "--output_jsonl", train_jsonl
    ]
    
    try:
        if not os.path.exists(train_jsonl):
            print("Running prepare_data.py for audio codecs...")
            subprocess.run(prep_cmd, check=True)
    except subprocess.CalledProcessError as e:
        return f"Error running prepare_data.py: {e}"
    
    # Construct sft_12hz.py command
    cmd = [
        "python", "sft_12hz.py",
        "--init_model_path", init_model,
        "--output_model_path", output_dir,
        "--train_jsonl", train_jsonl,
        "--batch_size", str(batch_size),
        "--lr", str(lr),
        "--num_epochs", str(epochs),
        "--speaker_name", speaker_name,
        "--gradient_accumulation_steps", str(grad_acc),
        "--resume_from_checkpoint", "latest"
    ]
    
    # Set HF or MS
    env = os.environ.copy()
    if model_source == "ModelScope":
        env["VLLM_USE_MODELSCOPE"] = "true"
        # Since snapshot_download is used in code, they rely on env vars or manual. 
        # But Qwen3TTSModel.from_pretrained works with HF directly, or MS if switched
    
    with open("training_log.txt", "w") as log_file:
        training_process = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    
    # Start Tensorboard if not running
    if not is_process_running("tensorboard --logdir logs"):
        subprocess.Popen(["tensorboard", "--logdir", "logs", "--port", "6006"])
        
    return f"Training started! PID: {training_process.pid}. Tensorboard running at port 6006."

def stop_training():
    global training_process
    if training_process is not None and training_process.poll() is None:
        training_process.terminate()
        return "Training stopped."
    return "No training process running."

def read_logs():
    if os.path.exists("training_log.txt"):
        with open("training_log.txt", "r") as f:
            lines = f.readlines()
            return "".join(lines[-30:]) # return last 30 lines
    return "No logs available."

# Presets mapping based on GitHub issues screenshots
presets = {
    "0.6B Model": {
        "init_model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "lr": 1e-7,
        "epochs": 2,
        "batch_size": 2,
        "grad_acc": 4
    },
    "1.7B Model": {
        "init_model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "lr": 2e-6,
        "epochs": 3,
        "batch_size": 2,
        "grad_acc": 1
    }
}

def apply_preset(preset_name):
    if preset_name in presets:
        p = presets[preset_name]
        return p["init_model"], p["lr"], p["epochs"], p["batch_size"], p["grad_acc"]
    return "Qwen/Qwen3-TTS-12Hz-0.6B-Base", 1e-7, 2, 2, 4

# ----- Inference Tab -----
def run_inference(model_path, text, speaker="crypto"):
    # Save a temporary prompt
    cmd = [
        "python", "quicktest.py",
        "--model_path", model_path,
        "--text", text,
        "--speaker", speaker,
        "--output", "webui_output.wav"
    ]
    try:
        subprocess.run(cmd, check=True)
        if os.path.exists("webui_output.wav"):
            return "webui_output.wav", "Success"
        else:
            return None, "Output file not generated."
    except subprocess.CalledProcessError as e:
        return None, f"Error during inference: {e}"

# Build Gradio UI
with gr.Blocks(title="Qwen3-TTS Easy Finetuning") as app:
    gr.Markdown("# 🎙️ Qwen3-TTS Easy Finetuning WebUI")
    
    with gr.Tabs():
        # DATA PREPERATION TAB
        with gr.Tab("1. Data Preparation"):
            gr.Markdown("Auto split, transcribe (ASR), clean, and resample your dataset to 24kHz.")
            
            with gr.Row():
                with gr.Column():
                    input_dir = gr.Textbox(label="Raw WAVs Directory", value="/home/mozi/train-qwen3-tts/dataset(nomark)")
                    ref_audio = gr.Textbox(label="Reference Audio Path (For TTS)", value="/home/mozi/train-qwen3-tts/finetune_data/audio/ref.wav", placeholder="Clear sounding audio clip (~3-10s)")
                    output_dir = gr.Textbox(label="Output Dataset Directory", value="data")
                
                with gr.Column():
                    asr_model = gr.Textbox(label="ASR Model for Transcription", value="Qwen/Qwen3-ASR-1.7B")
                    asr_source = gr.Radio(["ModelScope", "HuggingFace"], label="ASR Download Source", value="ModelScope")
                    batch_size_asr = gr.Slider(minimum=1, maximum=32, value=16, step=1, label="ASR Batch Size")
                    
                    prep_btn = gr.Button("Start Data Processing", variant="primary")
                    prep_out = gr.Textbox(label="Processing Output", lines=5)
                    
            prep_btn.click(
                fn=run_data_prep,
                inputs=[input_dir, ref_audio, output_dir, asr_model, batch_size_asr, asr_source],
                outputs=[prep_out]
            )

        # TRAINING TAB
        with gr.Tab("2. Training (Fine-tuning)"):
            gr.Markdown("Finetune Qwen3-TTS model with your processed dataset.")
            
            with gr.Row():
                with gr.Column():
                    raw_jsonl = gr.Textbox(label="Data JSONL Path", value="data/tts_train.jsonl")
                    speaker_name = gr.Textbox(label="Train Speaker Name", value="my_speaker")
                    
                    preset_dropdown = gr.Dropdown(list(presets.keys()), label="Training Preset", value="0.6B Model")
                    
                    init_model = gr.Textbox(label="Initial Model", value="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
                    model_source = gr.Radio(["ModelScope", "HuggingFace"], label="Model Download Source", value="HuggingFace")
                    
                with gr.Column():
                    t_lr = gr.Number(label="Learning Rate", value=1e-7)
                    t_epochs = gr.Slider(minimum=1, maximum=100, step=1, value=2, label="Epochs")
                    t_batch = gr.Slider(minimum=1, maximum=16, step=1, value=2, label="Batch Size")
                    t_grad = gr.Slider(minimum=1, maximum=16, step=1, value=4, label="Gradient Accumulation")
                    
            preset_dropdown.change(
                fn=apply_preset,
                inputs=[preset_dropdown],
                outputs=[init_model, t_lr, t_epochs, t_batch, t_grad]
            )
            
            with gr.Row():
                train_btn = gr.Button("Start Training", variant="primary")
                stop_btn = gr.Button("Stop Training", variant="stop")
            
            train_status = gr.Textbox(label="Training Status", lines=2)
            log_box = gr.Textbox(label="Live Training Logs", lines=10)
            refresh_btn = gr.Button("Refresh Logs")
            
            train_btn.click(
                fn=start_training,
                inputs=[raw_jsonl, speaker_name, init_model, model_source, t_batch, t_lr, t_epochs, t_grad],
                outputs=[train_status]
            )
            stop_btn.click(fn=stop_training, outputs=[train_status])
            refresh_btn.click(fn=read_logs, outputs=[log_box])

        # INFERENCE TAB
        with gr.Tab("3. Inference / Testing"):
            gr.Markdown("Test your trained checkpoints here.")
            
            with gr.Row():
                test_model = gr.Textbox(label="Checkpoint Path", value="output/checkpoint-epoch-0")
                test_speaker = gr.Textbox(label="Speaker Name", value="my_speaker")
                
            test_text = gr.Textbox(label="Text to Synthesize", value="Hello, this is a test from my custom voice.", lines=3)
            
            test_btn = gr.Button("Synthesize Audio", variant="primary")
            inference_status = gr.Textbox(label="Status")
            audio_out = gr.Audio(label="Generated Audio")
            
            test_btn.click(
                fn=run_inference,
                inputs=[test_model, test_text, test_speaker],
                outputs=[audio_out, inference_status]
            )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
