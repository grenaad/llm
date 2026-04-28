import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from typing import Callable, Optional

from faster_whisper import WhisperModel, BatchedInferencePipeline

log = logging.getLogger("whisper")

# Video extensions that need audio extraction
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v", ".flv", ".wmv"}

# Model configuration
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3")
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "float16")
BATCH_SIZE = int(os.environ.get("WHISPER_BATCH_SIZE", "8"))

# Singleton model instance
_model = None
_batched_model = None
_device = None
_model_ready = threading.Event()


def _fmt_size(nbytes: int) -> str:
    """Format byte count as human-readable string."""
    if nbytes < 1024:
        return f"{nbytes} B"
    if nbytes < 1024 * 1024:
        return f"{nbytes / 1024:.1f} KB"
    if nbytes < 1024 * 1024 * 1024:
        return f"{nbytes / (1024 * 1024):.1f} MB"
    return f"{nbytes / (1024 * 1024 * 1024):.2f} GB"


def _fmt_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s"


def get_device() -> str:
    global _device
    if _device is None:
        try:
            import torch
            _device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            # faster-whisper uses ctranslate2 which has its own CUDA detection
            _device = "cuda"
    return _device


def is_model_ready() -> bool:
    """Check if the model has finished loading."""
    return _model_ready.is_set()


def load_model():
    """Load the Whisper model using faster-whisper. Called once at startup."""
    global _model, _batched_model
    device = get_device()
    log.info("Loading faster-whisper %s on %s (compute_type=%s, batch_size=%d)",
             WHISPER_MODEL, device.upper(), COMPUTE_TYPE, BATCH_SIZE)
    t0 = time.monotonic()

    _model = WhisperModel(
        WHISPER_MODEL,
        device=device,
        compute_type=COMPUTE_TYPE,
    )

    # Create batched inference pipeline for faster transcription
    _batched_model = BatchedInferencePipeline(model=_model)

    elapsed = time.monotonic() - t0
    log.info("Model loaded in %s", _fmt_duration(elapsed))
    _model_ready.set()


def get_audio_duration(file_path: str) -> float:
    """Get audio/video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        file_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            info = json.loads(result.stdout)
            return float(info["format"]["duration"])
    except Exception as e:
        log.warning("ffprobe failed for %s: %s", file_path, e)
    return 0.0


def extract_audio_from_video(video_path: str) -> str:
    """Extract audio from video file using ffmpeg, returns path to temp wav file."""
    t0 = time.monotonic()
    temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_audio.close()

    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        temp_audio.name,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        os.unlink(temp_audio.name)
        raise RuntimeError(f"ffmpeg error: {result.stderr}")

    elapsed = time.monotonic() - t0
    wav_size = os.path.getsize(temp_audio.name)
    log.info("Audio extracted in %s, WAV size: %s", _fmt_duration(elapsed), _fmt_size(wav_size))

    return temp_audio.name


def _log_gpu_mem():
    """Log current GPU memory usage (one line)."""
    try:
        import torch
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated(0) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
            log.info("GPU memory: %.1f GB allocated, %.1f GB reserved", alloc, reserved)
    except ImportError:
        pass


# Type for progress callback: (current_seconds, total_seconds) -> None
ProgressCallback = Callable[[float, float], None]


def transcribe_file(
    file_path: str,
    progress_callback: Optional[ProgressCallback] = None,
) -> str:
    """Transcribe a single audio/video file using faster-whisper with batched inference.

    Args:
        file_path: Path to the audio/video file.
        progress_callback: Called with (current_seconds, total_seconds) as
            segments are transcribed. Optional.
    """
    if not _model_ready.is_set():
        log.info("Model still loading, waiting...")
        if not _model_ready.wait(timeout=300):
            raise RuntimeError("Model loading timed out (5 min)")
        log.info("Model ready, proceeding")

    file_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
    log.info("Starting: %s (%s)", file_name, _fmt_size(file_size))

    ext = os.path.splitext(file_path)[1].lower()
    cleanup_audio = False
    t_total = time.monotonic()

    try:
        # Step 1: Extract audio if video
        if ext in VIDEO_EXTENSIONS:
            log.info("Extracting audio from video...")
            audio_path = extract_audio_from_video(file_path)
            cleanup_audio = True
        else:
            audio_path = file_path

        # Step 2: Get duration for progress calculation
        duration = get_audio_duration(audio_path)
        log.info("Duration: %s", _fmt_duration(duration))

        _log_gpu_mem()

        # Step 3: Transcribe using batched inference pipeline
        # faster-whisper handles chunking internally - no need for manual splitting
        log.info("Transcribing with batched inference (batch_size=%d)...", BATCH_SIZE)

        segments, info = _batched_model.transcribe(
            audio_path,
            language="en",
            batch_size=BATCH_SIZE,
            beam_size=5,
            vad_filter=True,  # Skip silence for speed
            vad_parameters=dict(min_silence_duration_ms=1000),
        )

        log.info("Detected language: %s (probability: %.2f)",
                 info.language, info.language_probability)

        # Collect segments and report progress
        texts = []
        for segment in segments:
            text = segment.text.strip()
            if text:
                texts.append(text)
                # Print segment for docker logs (same format as before)
                start_str = _format_timestamp(segment.start)
                end_str = _format_timestamp(segment.end)
                print(f"[{start_str} --> {end_str}] {text}", flush=True)

                # Report progress
                if progress_callback and duration > 0:
                    progress_callback(segment.end, duration)

        # Report final progress
        if progress_callback and duration > 0:
            progress_callback(duration, duration)

        total_elapsed = time.monotonic() - t_total
        full_text = " ".join(texts)

        # Calculate speed ratio
        if total_elapsed > 0 and duration > 0:
            speed_ratio = duration / total_elapsed
            log.info("Transcription complete: %s in %s (%d chars, %.1fx realtime)",
                     file_name, _fmt_duration(total_elapsed), len(full_text), speed_ratio)
        else:
            log.info("Transcription complete: %s in %s (%d chars)",
                     file_name, _fmt_duration(total_elapsed), len(full_text))

        _log_gpu_mem()
        return full_text

    except Exception:
        total_elapsed = time.monotonic() - t_total
        log.error("Transcription FAILED: %s after %s", file_name, _fmt_duration(total_elapsed))
        _log_gpu_mem()
        traceback.print_exc()
        raise

    finally:
        # Clean up extracted audio
        if cleanup_audio:
            try:
                os.unlink(audio_path)
            except OSError:
                pass


def _format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS.mmm or HH:MM:SS.mmm timestamp."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)

    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}.{millis:03d}"
    return f"{minutes:02d}:{secs:02d}.{millis:03d}"


def get_gpu_info() -> dict:
    """Return GPU information for the status endpoint."""
    info = {
        "device": get_device(),
        "model_loaded": _model_ready.is_set(),
        "model_name": f"faster-whisper-{WHISPER_MODEL}",
        "compute_type": COMPUTE_TYPE,
        "batch_size": BATCH_SIZE,
    }
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory
            info["gpu_memory_total"] = f"{mem / 1024**3:.1f} GB"
    except (ImportError, Exception):
        pass
    return info
