import json
import logging
import os
import re
import subprocess
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from faster_whisper import BatchedInferencePipeline, WhisperModel

from models import GpuInfo

log = logging.getLogger("whisper")

# Model configuration
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3")
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8_float32")
BATCH_SIZE = int(os.environ.get("WHISPER_BATCH_SIZE", "8"))

# Singleton model instance
_model = None
_batched_model = None
_device = None
_model_ready = threading.Event()


def derive_display_title(file_name: str) -> str:
    """Convert an upload filename into a human-readable display title.

    Strips the file extension, replaces underscores with spaces, collapses
    whitespace, and strips a trailing YouTube-style `[xxxxxxxxxxx]` ID tag.
    Used as the value embedded into the MKV container's `title` tag so
    Jellyfin can display a clean name instead of the uuid-prefixed filename.
    """
    stem = Path(file_name).stem
    # Drop trailing " [youtubeid]" style suffix produced by yt-dlp
    stem = re.sub(r"\s*\[[A-Za-z0-9_-]{6,}\]\s*$", "", stem)
    # Underscores -> spaces, collapse runs of whitespace
    stem = stem.replace("_", " ")
    stem = re.sub(r"\s+", " ", stem).strip()
    return stem or Path(file_name).stem


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


def load_model() -> None:
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



@dataclass
class Word:
    """A single word with timing from word-level timestamps."""
    start: float
    end: float
    word: str


@dataclass
class Segment:
    """A single transcription segment with timing."""
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


@dataclass
class TranscriptionResult:
    """Result of a transcription, containing text and timed segments."""
    text: str
    segments: list[Segment] = field(default_factory=list)


def has_video_stream(file_path: str) -> bool:
    """Check if a file contains a video stream using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_type",
        "-print_format", "json",
        file_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            info = json.loads(result.stdout)
            streams = info.get("streams", [])
            return len(streams) > 0
    except Exception as e:
        log.warning("ffprobe video check failed for %s: %s", file_path, e)
    return False


def _format_srt_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS,mmm for SRT format (comma separator)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


_SRT_MAX_CHARS = 84  # ~2 lines at typical subtitle width (42 chars/line)


def _regroup_words_into_chunks(segments: list[Segment]) -> list[Segment]:
    """Regroup word-level timestamps into short subtitle chunks.

    Collects all words across segments and groups them into chunks
    of at most _SRT_MAX_CHARS characters, using precise word-level
    start/end timing for each chunk.
    """
    # Flatten all words from all segments
    all_words: list[Word] = []
    for seg in segments:
        all_words.extend(seg.words)

    if not all_words:
        # No word-level data available, return segments as-is
        return segments

    chunks: list[Segment] = []
    chunk_words: list[Word] = []
    chunk_len = 0

    for word in all_words:
        word_len = len(word.word) + (1 if chunk_words else 0)  # +1 for space
        if chunk_words and chunk_len + word_len > _SRT_MAX_CHARS:
            # Flush current chunk
            chunks.append(Segment(
                start=chunk_words[0].start,
                end=chunk_words[-1].end,
                text=" ".join(w.word for w in chunk_words),
            ))
            chunk_words = []
            chunk_len = 0

        chunk_words.append(word)
        chunk_len += word_len

    # Flush remaining words
    if chunk_words:
        chunks.append(Segment(
            start=chunk_words[0].start,
            end=chunk_words[-1].end,
            text=" ".join(w.word for w in chunk_words),
        ))

    return chunks


def generate_srt(segments: list[Segment]) -> str:
    """Generate SRT subtitle content from transcription segments.

    Uses word-level timestamps to create short subtitle chunks (~2 lines max).
    Falls back to segment-level timing if word data is not available.
    """
    chunks = _regroup_words_into_chunks(segments)
    lines: list[str] = []
    for i, seg in enumerate(chunks, start=1):
        lines.append(str(i))
        lines.append(f"{_format_srt_timestamp(seg.start)} --> {_format_srt_timestamp(seg.end)}")
        lines.append(seg.text)
        lines.append("")  # blank line between entries
    return "\n".join(lines)


def mux_subtitles(
    video_path: str,
    srt_path: str,
    output_path: str,
    title: str | None = None,
) -> None:
    """Mux SRT subtitles into a video as a soft subtitle track (MKV output).

    Uses ffmpeg with -c copy (no re-encoding), so this is fast and lossless.

    Args:
        video_path: Source video file.
        srt_path: SRT subtitle file to embed.
        output_path: Output MKV path.
        title: If provided, written as the MKV container-level title tag.
            Jellyfin (with "Prefer embedded titles over filenames" enabled
            on the library) uses this for the displayed item name.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", srt_path,
        "-c", "copy",
        "-c:s", "srt",
        # Tag the embedded subtitle track for nice player labeling.
        "-metadata:s:s:0", "language=eng",
        "-metadata:s:s:0", "title=English (Whisper)",
        "-disposition:s:0", "default",
    ]
    if title:
        # Container-level title -- this is what Jellyfin reads when
        # "Prefer embedded titles over filenames" is on.
        cmd += ["-metadata", f"title={title}"]
    cmd.append(output_path)

    log.info("Muxing subtitles: %s (title=%r)", os.path.basename(output_path), title)
    t0 = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    elapsed = time.monotonic() - t0

    if result.returncode != 0:
        log.error("ffmpeg mux failed: %s", result.stderr[-500:] if result.stderr else "unknown")
        raise RuntimeError(f"ffmpeg subtitle mux failed (exit {result.returncode})")

    log.info("Muxing complete in %s", _fmt_duration(elapsed))


def retag_title(input_path: str, output_path: str, title: str) -> None:
    """Rewrite an existing MKV with a new container-level title tag.

    Uses -c copy so it's a fast remux with no quality loss. The output path
    must differ from the input; the caller is responsible for swapping the
    file in place if desired.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-map", "0",
        "-c", "copy",
        "-metadata", f"title={title}",
        # Force the Matroska muxer because callers commonly hand us a
        # temp output path (e.g. foo.mkv.retag.tmp) where ffmpeg cannot
        # infer the container from the extension.
        "-f", "matroska",
        output_path,
    ]
    t0 = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    elapsed = time.monotonic() - t0

    if result.returncode != 0:
        log.error("ffmpeg retag failed: %s",
                  result.stderr[-500:] if result.stderr else "unknown")
        raise RuntimeError(f"ffmpeg retag failed (exit {result.returncode})")

    log.info("Retag complete: %s (title=%r) in %s",
             os.path.basename(output_path), title, _fmt_duration(elapsed))


def _log_gpu_mem() -> None:
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


CancelCheck = Callable[[], bool]


def transcribe_file(
    file_path: str,
    progress_callback: ProgressCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> TranscriptionResult:
    """Transcribe a single audio/video file using faster-whisper with batched inference.

    Args:
        file_path: Path to the audio/video file.
        progress_callback: Called with (current_seconds, total_seconds) as
            segments are transcribed. Optional.
        cancel_check: Called between segments; returns True if cancelled.
            Stops processing remaining segments early. Optional.

    Returns:
        TranscriptionResult with full text and timed segments.
    """
    if not _model_ready.is_set():
        log.info("Model still loading, waiting...")
        if not _model_ready.wait(timeout=300):
            raise RuntimeError("Model loading timed out (5 min)")
        log.info("Model ready, proceeding")

    file_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
    log.info("Starting: %s (%s)", file_name, _fmt_size(file_size))

    t_total = time.monotonic()

    try:
        # Step 1: Get duration for progress calculation
        duration = get_audio_duration(file_path)
        log.info("Duration: %s", _fmt_duration(duration))

        _log_gpu_mem()

        # Step 2: Transcribe using batched inference pipeline
        # faster-whisper accepts video/audio directly via its internal ffmpeg decoder
        log.info("Transcribing with batched inference (batch_size=%d)...", BATCH_SIZE)

        if _batched_model is None:
            raise RuntimeError("Model not loaded")

        segments, info = _batched_model.transcribe(
            file_path,
            language="en",
            batch_size=BATCH_SIZE,
            beam_size=5,
            word_timestamps=True,  # Per-word timing for subtitle generation
            vad_filter=True,  # Skip silence for speed
            vad_parameters=dict(min_silence_duration_ms=1000),
        )

        log.info("Detected language: %s (probability: %.2f)",
                 info.language, info.language_probability)

        # Collect segments and report progress
        texts: list[str] = []
        timed_segments: list[Segment] = []
        cancelled = False
        for segment in segments:
            # Check for cancellation between segments
            if cancel_check and cancel_check():
                log.info("Transcription cancelled mid-stream: %s", file_name)
                cancelled = True
                break

            text = segment.text.strip()
            if text:
                texts.append(text)
                # Capture word-level timestamps for subtitle generation
                seg_words: list[Word] = []
                if segment.words:
                    for w in segment.words:
                        seg_words.append(Word(
                            start=w.start,
                            end=w.end,
                            word=w.word.strip(),
                        ))
                timed_segments.append(Segment(
                    start=segment.start,
                    end=segment.end,
                    text=text,
                    words=seg_words,
                ))
                # Print segment for docker logs (same format as before)
                start_str = _format_timestamp(segment.start)
                end_str = _format_timestamp(segment.end)
                print(f"[{start_str} --> {end_str}] {text}", flush=True)

                # Report progress
                if progress_callback and duration > 0:
                    progress_callback(segment.end, duration)

        if cancelled:
            total_elapsed = time.monotonic() - t_total
            log.info("Cancelled: %s after %s", file_name, _fmt_duration(total_elapsed))
            _log_gpu_mem()
            return TranscriptionResult(text="", segments=[])

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
        return TranscriptionResult(text=full_text, segments=timed_segments)

    except Exception:
        total_elapsed = time.monotonic() - t_total
        log.error("Transcription FAILED: %s after %s", file_name, _fmt_duration(total_elapsed))
        _log_gpu_mem()
        traceback.print_exc()
        raise



def _format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS.mmm or HH:MM:SS.mmm timestamp."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)

    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}.{millis:03d}"
    return f"{minutes:02d}:{secs:02d}.{millis:03d}"


def get_gpu_info() -> GpuInfo:
    """Return GPU information for the status endpoint."""
    gpu_name: str | None = None
    gpu_memory_total: str | None = None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            gpu_name = parts[0]
            gpu_memory_total = f"{int(parts[1]) / 1024:.1f} GB"
    except Exception:
        pass

    return GpuInfo(
        device=get_device(),
        model_loaded=_model_ready.is_set(),
        model_name=f"faster-whisper-{WHISPER_MODEL}",
        compute_type=COMPUTE_TYPE,
        batch_size=BATCH_SIZE,
        gpu_name=gpu_name,
        gpu_memory_total=gpu_memory_total,
    )
