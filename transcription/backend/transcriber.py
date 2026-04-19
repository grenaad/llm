import json
import logging
import os
import subprocess
import tempfile
import threading
import time
import traceback
from typing import Callable, Optional

import torch
import nemo.collections.asr as nemo_asr

log = logging.getLogger("parakeet")

# Video extensions that need audio extraction
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v", ".flv", ".wmv"}

# Chunk config
CHUNK_DURATION = 300  # 5 minutes in seconds
CHUNK_OVERLAP = 2  # 2 second overlap to avoid cutting words

# Singleton model instance
_model = None
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
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    return _device


def load_model():
    """Load the Parakeet TDT-1.1B model. Called once at startup."""
    global _model
    device = get_device()
    log.info("Loading Parakeet TDT-0.6B-v3 on %s", device.upper())
    t0 = time.monotonic()

    model_name = "nvidia/parakeet-tdt-0.6b-v3"
    _model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)

    # 1080 Ti Optimization: Force Float32
    # Older GTX cards can sometimes get 'NaN' errors with half-precision (FP16)
    _model = _model.to(device).float()
    _model.eval()

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


def split_audio_into_chunks(wav_path: str, duration: float) -> list[str]:
    """Split a WAV file into chunks of CHUNK_DURATION seconds.

    Returns list of paths to chunk files. Chunks have a small overlap
    (CHUNK_OVERLAP seconds) so words at boundaries aren't lost.
    """
    t0 = time.monotonic()
    chunk_paths = []
    start = 0.0
    chunk_idx = 0

    while start < duration:
        chunk_file = tempfile.NamedTemporaryFile(
            suffix=f"_chunk{chunk_idx:04d}.wav", delete=False
        )
        chunk_file.close()

        # Each chunk is CHUNK_DURATION + CHUNK_OVERLAP seconds long,
        # except the last one which goes to the end.
        chunk_len = CHUNK_DURATION + CHUNK_OVERLAP

        cmd = [
            "ffmpeg", "-y",
            "-i", wav_path,
            "-ss", str(start),
            "-t", str(chunk_len),
            "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            chunk_file.name,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            # Clean up all chunk files on error
            for p in chunk_paths:
                try:
                    os.unlink(p)
                except OSError:
                    pass
            os.unlink(chunk_file.name)
            raise RuntimeError(f"ffmpeg chunk split error: {result.stderr}")

        chunk_paths.append(chunk_file.name)
        start += CHUNK_DURATION
        chunk_idx += 1

    elapsed = time.monotonic() - t0
    log.info("Split into %d chunks in %s", len(chunk_paths), _fmt_duration(elapsed))
    return chunk_paths


def _transcribe_single(audio_path: str) -> str:
    """Transcribe a single audio file (must fit in GPU memory)."""
    with torch.no_grad():
        transcriptions = _model.transcribe([audio_path])

    result = transcriptions[0]
    if hasattr(result, "text"):
        return result.text
    return str(result)


def _log_gpu_mem():
    """Log current GPU memory usage (one line)."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated(0) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
        log.info("GPU memory: %.1f GB allocated, %.1f GB reserved", alloc, reserved)


# Type for progress callback: (chunk_number, total_chunks) -> None
ProgressCallback = Callable[[int, int], None]


def transcribe_file(
    file_path: str,
    progress_callback: Optional[ProgressCallback] = None,
) -> str:
    """Transcribe a single audio/video file. Returns the transcription text.

    For files longer than CHUNK_DURATION, splits into chunks and transcribes
    each one separately to avoid GPU OOM on the GTX 1080 Ti.

    Args:
        file_path: Path to the audio/video file.
        progress_callback: Called with (current_chunk, total_chunks) after
            each chunk is transcribed. Optional.
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
    chunk_paths: list[str] = []
    t_total = time.monotonic()

    try:
        # Step 1: Extract audio if video
        if ext in VIDEO_EXTENSIONS:
            log.info("Extracting audio from video...")
            audio_path = extract_audio_from_video(file_path)
            cleanup_audio = True
        else:
            audio_path = file_path

        # Step 2: Check duration
        duration = get_audio_duration(audio_path)
        log.info("Duration: %s", _fmt_duration(duration))

        # Step 3: Decide whether to chunk
        if duration > CHUNK_DURATION + 30:
            # Long file -> split into chunks
            total_chunks = int(duration // CHUNK_DURATION) + (
                1 if duration % CHUNK_DURATION > 0 else 0
            )
            log.info("Long file, chunking into ~%d x %ds segments", total_chunks, CHUNK_DURATION)

            chunk_paths = split_audio_into_chunks(audio_path, duration)
            total_chunks = len(chunk_paths)  # actual count after splitting

            _log_gpu_mem()

            texts = []
            for i, chunk_path in enumerate(chunk_paths):
                chunk_num = i + 1

                if progress_callback:
                    progress_callback(chunk_num, total_chunks)

                t_chunk = time.monotonic()
                text = _transcribe_single(chunk_path)
                chunk_elapsed = time.monotonic() - t_chunk

                texts.append(text)
                log.info("Chunk %d/%d done in %s (%d chars)",
                         chunk_num, total_chunks, _fmt_duration(chunk_elapsed), len(text))

                # Clean up chunk immediately after transcription
                try:
                    os.unlink(chunk_path)
                except OSError:
                    pass

            # Mark chunk_paths as cleaned so finally block doesn't double-delete
            chunk_paths = []

            total_elapsed = time.monotonic() - t_total
            full_text = " ".join(texts)
            log.info("Transcription complete: %s, %d chunks in %s, %d chars total",
                     file_name, total_chunks, _fmt_duration(total_elapsed), len(full_text))
            _log_gpu_mem()
            return full_text
        else:
            # Short file -> transcribe directly
            if progress_callback:
                progress_callback(1, 1)

            _log_gpu_mem()
            t_chunk = time.monotonic()
            text = _transcribe_single(audio_path)
            chunk_elapsed = time.monotonic() - t_chunk

            total_elapsed = time.monotonic() - t_total
            log.info("Transcription complete: %s in %s (%d chars)",
                     file_name, _fmt_duration(total_elapsed), len(text))
            return text

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
        # Clean up any remaining chunk files (in case of error)
        for p in chunk_paths:
            try:
                os.unlink(p)
            except OSError:
                pass


def get_gpu_info() -> dict:
    """Return GPU information for the status endpoint."""
    info = {"device": get_device(), "model_loaded": _model_ready.is_set()}
    try:
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory
            info["gpu_memory_total"] = f"{mem / 1024**3:.1f} GB"
    except Exception:
        pass
    return info
