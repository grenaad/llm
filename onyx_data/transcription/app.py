import gradio as gr
import nemo.collections.asr as nemo_asr
import torch
import os

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


def transcribe_audio(audio_path):
    if audio_path is None:
        return "No file uploaded."

    try:
        # Perform transcription
        # The TDT model handles punctuation and capitalization automatically
        print(f"Processing file: {audio_path}")
        with torch.no_grad():
            transcriptions = model.transcribe([audio_path])

        return transcriptions[0]
    except Exception as e:
        return f"Error during transcription: {str(e)}"


# 3. Create the UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# Local Parakeet XXL (1.1B)")
    gr.Markdown(
        f"Running locally on **GTX 1080 Ti**. Your data never leaves this machine."
    )

    with gr.Row():
        audio_input = gr.Audio(type="filepath", label="Upload Audio File")

    with gr.Row():
        transcription_output = gr.Textbox(
            label="Transcription",
            placeholder="Result will appear here...",
            lines=10,
            show_copy_button=True,
        )

    submit_btn = gr.Button("Transcribe", variant="primary")
    submit_btn.click(
        fn=transcribe_audio, inputs=audio_input, outputs=transcription_output
    )

if __name__ == "__main__":
    # Launch on all interfaces so WSL can talk to Windows
    demo.launch(server_name="0.0.0.0", server_port=7860)
