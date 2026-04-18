# Parakeet ASR Transcription Service

A local audio transcription web application using NVIDIA's **Parakeet-TDT-1.1B** model. This service provides fast, accurate speech-to-text transcription that runs entirely on your local machine.

## Overview

- **Model**: [nvidia/parakeet-tdt-1.1b](https://huggingface.co/nvidia/parakeet-tdt-1.1b) (~2GB download on first run)
- **Interface**: Gradio web UI
- **Port**: 7860
- **Features**:
  - Automatic punctuation and capitalization
  - Handles overlapping speech and background noise
  - Does not hallucinate during silence or music (unlike Whisper)
  - All processing happens locally - your data never leaves your machine

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM  | 8GB     | 11GB (GTX 1080 Ti) |
| System RAM | 16GB   | 32GB |
| Shared Memory | 8GB | 8GB+ |

### GTX 1080 Ti Compatibility

This application is optimized for the GTX 1080 Ti (Pascal architecture):
- Uses **Float32** precision for stability (avoids NaN errors common with FP16 on 10-series cards)
- Peak VRAM usage: ~5.5GB to 6.5GB
- Expected performance: ~100x to 200x real-time (2-minute audio transcribes in ~1 second)

## Quick Start

### 1. Build the Docker Image

```bash
cd transcription
docker build -t parakeet-webui .
```

### 2. Run the Container

```bash
docker run --gpus all \
  -p 7860:7860 \
  --shm-size=8GB \
  --name parakeet-web \
  -v $(pwd):/app \
  parakeet-webui
```

**Important flags:**
- `--gpus all` - Enable GPU access
- `--shm-size=8GB` - Required for the 1.1B model's internal tensors
- `-p 7860:7860` - Expose the web UI port

### 3. Access the Web UI

Open your browser and navigate to:

```
http://localhost:7860
```

## Usage

1. Upload an audio file (supports WAV, MP3, FLAC, and other common formats)
2. Click **Transcribe**
3. View and copy the transcription result

## Docker Commands Reference

### Stop the container
```bash
docker stop parakeet-web
```

### Start an existing container
```bash
docker start parakeet-web
```

### Remove the container
```bash
docker rm parakeet-web
```

### View logs
```bash
docker logs -f parakeet-web
```

## Troubleshooting

### "CUDA out of memory" error
- Ensure no other GPU-intensive applications are running
- Try restarting the Docker container

### "NaN" or blank output
- The Float32 mode should prevent this, but if it occurs:
  - Restart the container
  - Check that your NVIDIA drivers are up to date

### Container won't start with GPU
- Verify NVIDIA Container Toolkit is installed:
  ```bash
  nvidia-smi
  docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
  ```

### WSL2 specific
- Ensure WSL2 GPU passthrough is enabled
- Use `0.0.0.0` as the server address (already configured in app.py)

## Model Information

**Parakeet-TDT-1.1B** is a Transducer-based ASR model from NVIDIA:
- State-of-the-art accuracy for English transcription
- Efficient architecture optimized for real-time inference
- Built-in text normalization (punctuation, capitalization)
- Robust handling of noisy audio environments

For more information, see the [NVIDIA NeMo documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/intro.html).
