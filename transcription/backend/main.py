import asyncio
import json
import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import aiofiles
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request

from models import (
    ActiveJob,
    ActiveJobInfo,
    DeleteCountResponse,
    DeleteResponse,
    GpuInfo,
    SavedTranscription,
    TranscribeRequest,
    TranscribeResponse,
    UploadResponse,
    WsCancelled,
    WsFileError,
    WsFileProgress,
    WsFileResult,
    WsFileStatus,
    WsMessage,
)
from transcriber import (
    _fmt_duration,
    _fmt_size,
    get_gpu_info,
    is_model_ready,
    load_model,
    transcribe_file,
)

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
# Quiet down noisy libraries
logging.getLogger("matplotlib").setLevel(logging.WARNING)

log = logging.getLogger("whisper")

# --- Configuration ---
UPLOAD_DIR = Path("/tmp/whisper_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path("/app/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
TRANSCRIPTIONS_FILE = DATA_DIR / "transcriptions.json"

STATIC_DIR = Path(__file__).parent / "static"


# --- JSON Storage Helpers ---
def load_transcriptions() -> list[SavedTranscription]:
    """Load all transcriptions from JSON file."""
    if not TRANSCRIPTIONS_FILE.exists():
        return []
    try:
        raw = json.loads(TRANSCRIPTIONS_FILE.read_text())
        return [SavedTranscription(**t) for t in raw]
    except (OSError, json.JSONDecodeError):
        return []


def save_transcription(id: str, name: str, text: str, size: int) -> SavedTranscription:
    """Save a transcription to JSON file."""
    transcriptions = load_transcriptions()
    entry = SavedTranscription(
        id=id,
        name=name,
        text=text,
        size=size,
        created_at=datetime.utcnow().isoformat() + "Z",
    )
    transcriptions.append(entry)
    TRANSCRIPTIONS_FILE.write_text(
        json.dumps([t.model_dump() for t in transcriptions], indent=2)
    )
    return entry


def delete_transcription_by_id(id: str) -> bool:
    """Delete a transcription by ID."""
    transcriptions = load_transcriptions()
    filtered = [t for t in transcriptions if t.id != id]
    if len(filtered) == len(transcriptions):
        return False
    TRANSCRIPTIONS_FILE.write_text(
        json.dumps([t.model_dump() for t in filtered], indent=2)
    )
    return True


def delete_all_transcriptions() -> int:
    """Delete all transcriptions. Returns count deleted."""
    transcriptions = load_transcriptions()
    count = len(transcriptions)
    TRANSCRIPTIONS_FILE.write_text("[]")
    return count

# --- Global state ---
# Job registry: tracks all queued/in-progress transcription jobs.
# Jobs survive WebSocket disconnects (page refresh) and continue processing.
active_jobs: dict[str, ActiveJob] = {}

# Connected WebSocket clients. Progress/status is broadcast to ALL clients.
connected_clients: set[WebSocket] = set()

# Transcription queue: only one job runs at a time on the GPU.
transcription_queue: asyncio.Queue[str] = asyncio.Queue()


async def broadcast(message: WsMessage) -> None:
    """Send a message to all connected WebSocket clients."""
    data = message.model_dump()
    disconnected = set()
    for client in connected_clients:
        try:
            await client.send_json(data)
        except Exception:
            disconnected.add(client)
    connected_clients.difference_update(disconnected)


async def global_transcription_worker() -> None:
    """Process transcription jobs one at a time, globally.

    Each job is a (file_id,) tuple. Job details are in active_jobs.
    This ensures only one file is being transcribed on the GPU at any time,
    even if multiple browsers are submitting files simultaneously.
    """
    while True:
        file_id = await transcription_queue.get()
        job = active_jobs.get(file_id)
        if not job:
            transcription_queue.task_done()
            continue

        remaining = transcription_queue.qsize()
        log.info("Transcription starting: %s (%d more in queue)",
                 job.name, remaining)
        try:
            await process_file(file_id)
        except Exception:
            pass  # process_file already handles errors internally
        finally:
            transcription_queue.task_done()


# --- App setup ---
app = FastAPI(title="Whisper Transcription")


@app.on_event("startup")
async def startup() -> None:
    import threading
    threading.Thread(target=load_model, daemon=True, name="model-loader").start()
    log.info("Server ready, model loading in background")
    asyncio.create_task(global_transcription_worker())


# --- API routes ---
@app.get("/api/status")
async def status() -> GpuInfo:
    return get_gpu_info()


@app.get("/api/transcriptions")
async def get_transcriptions() -> list[SavedTranscription]:
    """Get all saved transcriptions."""
    return load_transcriptions()


@app.get("/api/jobs")
async def get_jobs() -> list[ActiveJobInfo]:
    """Get all active (queued/in-progress) jobs."""
    return [job.to_info() for job in active_jobs.values()]


@app.delete("/api/transcriptions/{transcription_id}")
async def delete_transcription_endpoint(transcription_id: str) -> DeleteResponse:
    """Delete a specific transcription."""
    deleted = delete_transcription_by_id(transcription_id)
    return DeleteResponse(deleted=deleted)


@app.delete("/api/transcriptions")
async def delete_all_transcriptions_endpoint() -> DeleteCountResponse:
    """Delete all transcriptions."""
    count = delete_all_transcriptions()
    return DeleteCountResponse(deleted=count)


@app.post("/api/upload/{filename:path}")
async def upload_file(filename: str, request: Request) -> UploadResponse:
    """Upload a single file as raw binary stream (no multipart overhead).

    The browser sends the file body directly as application/octet-stream.
    This bypasses Starlette's multipart parser for maximum throughput.
    """
    file_id = uuid4().hex[:12]
    safe_name = Path(filename).name or f"{file_id}.bin"
    dest = UPLOAD_DIR / f"{file_id}_{safe_name}"

    t0 = time.monotonic()
    bytes_written = 0

    log.info("Upload started: %s", safe_name)

    async with aiofiles.open(dest, "wb") as f:
        async for chunk in request.stream():
            await f.write(chunk)
            bytes_written += len(chunk)

    elapsed = time.monotonic() - t0
    speed = bytes_written / elapsed if elapsed > 0 else 0
    log.info("Upload complete: %s (%s) in %s @ %s/s",
             safe_name, _fmt_size(bytes_written), _fmt_duration(elapsed),
             _fmt_size(int(speed)))

    return UploadResponse(
        id=file_id,
        name=safe_name,
        path=str(dest),
        size=bytes_written,
    )


@app.post("/api/transcribe")
async def submit_transcription(body: TranscribeRequest) -> TranscribeResponse:
    """Submit a file for transcription. Registers the job and queues it."""
    job = ActiveJob(
        id=body.file_id,
        name=body.file_name,
        path=body.file_path,
        size=body.file_size,
    )
    active_jobs[body.file_id] = job

    # Add to global queue
    await transcription_queue.put(body.file_id)
    log.info("Queued: %s (global queue depth: %d)", body.file_name, transcription_queue.qsize())

    # Broadcast status to all connected clients
    await broadcast(WsFileStatus(
        file_id=body.file_id,
        name=body.file_name,
        status="waiting",
    ))

    return TranscribeResponse(queued=True, file_id=body.file_id)


@app.delete("/api/files/{file_id}")
async def delete_file(file_id: str) -> DeleteResponse:
    """Delete an uploaded file."""
    for path in UPLOAD_DIR.glob(f"{file_id}_*"):
        path.unlink(missing_ok=True)
        return DeleteResponse(deleted=True)
    return DeleteResponse(deleted=False)


@app.websocket("/api/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    connected_clients.add(ws)
    log.info("WebSocket connected (%d clients)", len(connected_clients))

    # Send current state of all active jobs to the newly connected client
    for job in active_jobs.values():
        ws_msg: WsFileProgress | WsFileStatus
        if job.progress_seconds is not None:
            ws_msg = WsFileProgress(
                file_id=job.id,
                name=job.name,
                progress_seconds=job.progress_seconds,
                total_seconds=job.total_seconds or 0.0,
            )
        else:
            ws_msg = WsFileStatus(
                file_id=job.id,
                name=job.name,
                status=job.status,
            )
        try:
            await ws.send_json(ws_msg.model_dump())
        except Exception:
            pass

    try:
        while True:
            data = await ws.receive_json()
            msg_type = data.get("type")

            if msg_type == "cancel":
                cancel_file_id: str | None = data.get("file_id")
                if cancel_file_id is not None:
                    cancel_job = active_jobs.get(cancel_file_id)
                    if cancel_job:
                        cancel_job.cancel_flag.set()
                        log.info("Cancel requested: %s", cancel_file_id)
                        # Immediately notify all clients
                        await broadcast(WsCancelled(file_id=cancel_file_id))

            elif msg_type == "cancel_all":
                for active_job in active_jobs.values():
                    active_job.cancel_flag.set()
                log.info("Cancel all requested (%d jobs)", len(active_jobs))

    except WebSocketDisconnect:
        connected_clients.discard(ws)
        log.info("WebSocket disconnected (%d clients remain)",
                 len(connected_clients))
        # Do NOT cancel jobs -- they continue running for reconnecting clients
    except Exception as e:
        connected_clients.discard(ws)
        log.error("WebSocket error: %s", e)
        traceback.print_exc()


async def process_file(file_id: str) -> None:
    """Transcribe a single file with chunk progress reporting.

    Uses the global active_jobs registry and broadcasts status/progress
    to all connected WebSocket clients.
    """
    job = active_jobs.get(file_id)
    if not job:
        return

    file_name = job.name
    file_path = job.path

    try:
        # Check cancellation before starting
        if job.cancel_flag.is_set():
            log.info("Skipped (cancelled): %s", file_name)
            await broadcast(WsCancelled(file_id=file_id, name=file_name))
            return

        # Check if model is still loading
        if not is_model_ready():
            log.info("Waiting for model: %s", file_name)
            job.status = "loading_model"
            await broadcast(WsFileStatus(
                file_id=file_id,
                name=file_name,
                status="loading_model",
            ))

            # Wait for model in async-friendly way (allows cancellation)
            while not is_model_ready():
                if job.cancel_flag.is_set():
                    log.info("Cancelled while waiting for model: %s", file_name)
                    await broadcast(WsCancelled(file_id=file_id, name=file_name))
                    return
                await asyncio.sleep(0.5)  # Check every 500ms

            log.info("Model ready, starting: %s", file_name)

        # Update status: transcribing
        job.status = "transcribing"
        await broadcast(WsFileStatus(
            file_id=file_id,
            name=file_name,
            status="transcribing",
        ))

        # Create a progress callback that broadcasts to all clients.
        # Since transcribe_file runs in a thread pool, we need to use
        # run_coroutine_threadsafe to send messages from the thread.
        loop = asyncio.get_event_loop()

        def progress_callback(current_seconds: float, total_seconds: float) -> None:
            """Called from the transcription thread as segments complete."""
            # Update job registry for new clients connecting mid-transcription
            job.progress_seconds = current_seconds
            job.total_seconds = total_seconds

            future = asyncio.run_coroutine_threadsafe(
                broadcast(WsFileProgress(
                    file_id=file_id,
                    name=file_name,
                    progress_seconds=current_seconds,
                    total_seconds=total_seconds,
                )),
                loop,
            )
            try:
                future.result(timeout=5)
            except Exception as e:
                log.warning("Progress broadcast error: %s", e)

        # Run transcription in a thread pool
        t0 = time.monotonic()
        text = await loop.run_in_executor(
            None, transcribe_file, file_path, progress_callback
        )
        elapsed = time.monotonic() - t0

        # Check cancellation after transcription
        if job.cancel_flag.is_set():
            log.info("Cancelled after transcription: %s", file_name)
            await broadcast(WsCancelled(file_id=file_id, name=file_name))
        else:
            # Get file size before cleanup
            try:
                file_size = Path(file_path).stat().st_size
            except Exception:
                file_size = job.size

            # Save to persistent storage
            save_transcription(
                id=file_id,
                name=file_name,
                text=text,
                size=file_size,
            )

            log.info("Result sent: %s (%d chars, %s)",
                     file_name, len(text), _fmt_duration(elapsed))
            await broadcast(WsFileResult(
                file_id=file_id,
                name=file_name,
                text=text,
            ))

    except Exception as e:
        log.error("Transcription error: %s - %s", file_name, e)
        traceback.print_exc()
        await broadcast(WsFileError(
            file_id=file_id,
            name=file_name,
            error=str(e),
        ))

    finally:
        # Remove from active jobs
        active_jobs.pop(file_id, None)
        # Clean up uploaded file after processing
        try:
            Path(file_path).unlink(missing_ok=True)
        except Exception:
            pass


# --- Static file serving (SolidJS frontend) ---
if STATIC_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(STATIC_DIR / "assets")), name="assets")

    @app.get("/{path:path}")
    async def serve_frontend(path: str = "") -> FileResponse:
        file_path = STATIC_DIR / path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(STATIC_DIR / "index.html")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
