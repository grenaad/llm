import gradio as gr
import nemo.collections.asr as nemo_asr
import torch
import os
import subprocess
import tempfile
import threading

# 1. Setup Device
# 1080 Ti is perfectly fine for 1.1B, but we'll use Float32 for stability
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"--- Initializing Parakeet XXL on {device.upper()} ---")

# 2. Load the 1.1B Model
# This will download ~2GB the first time you run it.
# We use the TDT variant because it is the fastest high-accuracy model.
model_name = "nvidia/parakeet-tdt-1.1b"
model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)

# 1080 Ti Optimization: Force Float32
# Older GTX cards can sometimes get 'NaN' errors with half-precision (FP16)
model = model.to(device).float()
model.eval()

# Video extensions that need audio extraction
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v", ".flv", ".wmv"}

# Cancellation flag
cancel_flag = threading.Event()


def extract_audio_from_video(video_path):
    """Extract audio from video file using ffmpeg, returns path to temp wav file."""
    temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_audio.close()
    
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        temp_audio.name
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {result.stderr}")
    
    return temp_audio.name


def transcribe_single_file(file_path):
    """Transcribe a single audio/video file."""
    # Check if it's a video file that needs audio extraction
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext in VIDEO_EXTENSIONS:
        print(f"Extracting audio from video: {file_path}")
        audio_path = extract_audio_from_video(file_path)
        cleanup_temp = True
    else:
        audio_path = file_path
        cleanup_temp = False
    
    # Perform transcription
    print(f"Processing file: {audio_path}")
    with torch.no_grad():
        transcriptions = model.transcribe([audio_path])
    
    # Clean up temp file if we created one
    if cleanup_temp and os.path.exists(audio_path):
        os.unlink(audio_path)

    # Extract text from Hypothesis object if needed
    result = transcriptions[0]
    if hasattr(result, 'text'):
        return result.text
    return str(result)


def transcribe_files(files):
    """Transcribe multiple files sequentially with cancellation support."""
    global cancel_flag
    cancel_flag.clear()
    
    if files is None or len(files) == 0:
        yield "No files uploaded."
        return

    results = []
    total = len(files)
    
    for i, file in enumerate(files):
        # Check for cancellation
        if cancel_flag.is_set():
            results.append(f"\n--- CANCELLED ---\nProcessed {i}/{total} files before cancellation.")
            yield "\n\n".join(results)
            return
        
        file_path = file.name if hasattr(file, 'name') else file
        file_name = os.path.basename(file_path)
        
        try:
            # Show progress
            progress_msg = f"[{i+1}/{total}] Processing: {file_name}..."
            if results:
                yield "\n\n".join(results) + f"\n\n{progress_msg}"
            else:
                yield progress_msg
            
            # Transcribe
            transcription = transcribe_single_file(file_path)
            results.append(f"### {file_name}\n{transcription}")
            
        except Exception as e:
            results.append(f"### {file_name}\nError: {str(e)}")
        
        # Yield current results
        yield "\n\n".join(results)
    
    # Final result
    yield "\n\n".join(results)


def cancel_transcription():
    """Set the cancellation flag."""
    global cancel_flag
    cancel_flag.set()
    return "Cancellation requested... Will stop after current file."


# 3. Create the UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Local Parakeet XXL (1.1B)")
    gr.Markdown(
        "Running locally on **GTX 1080 Ti**. Your data never leaves this machine."
    )

    with gr.Row():
        file_input = gr.File(
            label="Upload Audio or Video Files",
            file_types=[".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".mp4", ".mkv", ".webm", ".mov", ".avi"],
            file_count="multiple",
        )

    with gr.Row():
        transcription_output = gr.Textbox(
            label="Transcription",
            placeholder="Results will appear here...",
            lines=15,
        )

    with gr.Row():
        submit_btn = gr.Button("Transcribe", variant="primary")
        cancel_btn = gr.Button("Cancel", variant="stop")

    # Auto-transcribe when files are uploaded
    file_input.change(
        fn=transcribe_files,
        inputs=file_input,
        outputs=transcription_output,
    )
    
    # Transcribe button - also available for manual re-run
    submit_btn.click(
        fn=transcribe_files,
        inputs=file_input,
        outputs=transcription_output,
    )
    
    # Cancel button
    cancel_btn.click(
        fn=cancel_transcription,
        inputs=None,
        outputs=transcription_output,
    )

if __name__ == "__main__":
    # Launch on all interfaces so WSL can talk to Windows
    demo.launch(server_name="0.0.0.0", server_port=7860)
