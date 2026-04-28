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

from transcriber import load_model, transcribe_file, get_gpu_info, is_model_ready, _fmt_size, _fmt_duration

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
def load_transcriptions() -> list[dict]:
    """Load all transcriptions from JSON file."""
    if not TRANSCRIPTIONS_FILE.exists():
        return []
    try:
        return json.loads(TRANSCRIPTIONS_FILE.read_text())
    except (json.JSONDecodeError, IOError):
        return []


def save_transcription(id: str, name: str, text: str, size: int) -> dict:
    """Save a transcription to JSON file."""
    transcriptions = load_transcriptions()
    entry = {
        "id": id,
        "name": name,
        "text": text,
        "size": size,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    transcriptions.append(entry)
    TRANSCRIPTIONS_FILE.write_text(json.dumps(transcriptions, indent=2))
    return entry


def delete_transcription_by_id(id: str) -> bool:
    """Delete a transcription by ID."""
    transcriptions = load_transcriptions()
    filtered = [t for t in transcriptions if t["id"] != id]
    if len(filtered) == len(transcriptions):
        return False
    TRANSCRIPTIONS_FILE.write_text(json.dumps(filtered, indent=2))
    return True


def delete_all_transcriptions() -> int:
    """Delete all transcriptions. Returns count deleted."""
    transcriptions = load_transcriptions()
    count = len(transcriptions)
    TRANSCRIPTIONS_FILE.write_text("[]")
    return count

# --- Global transcription queue ---
# All WebSocket connections share this queue so only one transcription
# runs at a time on the GPU, regardless of how many browsers are connected.
transcription_queue: asyncio.Queue = asyncio.Queue()


async def global_transcription_worker():
    """Process transcription jobs one at a time, globally.

    Each job is a tuple of (ws, file_info, cancel_flag).
    This ensures only one file is being transcribed on the GPU at any time,
    even if multiple browsers are submitting files simultaneously.
    """
    while True:
        ws, file_info, cancel_flag = await transcription_queue.get()
        remaining = transcription_queue.qsize()
        log.info("Transcription starting: %s (%d more in queue)",
                 file_info["name"], remaining)
        try:
            await process_file(ws, file_info, cancel_flag)
        except Exception:
            pass  # process_file already handles errors internally
        finally:
            transcription_queue.task_done()


# --- App setup ---
app = FastAPI(title="Whisper Transcription")


@app.on_event("startup")
async def startup():
    import threading
    threading.Thread(target=load_model, daemon=True, name="model-loader").start()
    log.info("Server ready, model loading in background")
    asyncio.create_task(global_transcription_worker())


# --- API routes ---
@app.get("/api/status")
async def status():
    return get_gpu_info()


@app.get("/api/transcriptions")
async def get_transcriptions():
    """Get all saved transcriptions."""
    return load_transcriptions()


@app.delete("/api/transcriptions/{transcription_id}")
async def delete_transcription_endpoint(transcription_id: str):
    """Delete a specific transcription."""
    deleted = delete_transcription_by_id(transcription_id)
    return {"deleted": deleted}


@app.delete("/api/transcriptions")
async def delete_all_transcriptions_endpoint():
    """Delete all transcriptions."""
    count = delete_all_transcriptions()
    return {"deleted": count}


@app.post("/api/upload/{filename:path}")
async def upload_file(filename: str, request: Request):
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

    return {
        "id": file_id,
        "name": safe_name,
        "path": str(dest),
        "size": bytes_written,
    }


@app.delete("/api/files/{file_id}")
async def delete_file(file_id: str):
    """Delete an uploaded file."""
    for path in UPLOAD_DIR.glob(f"{file_id}_*"):
        path.unlink(missing_ok=True)
        return {"deleted": True}
    return {"deleted": False}


@app.websocket("/api/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    log.info("WebSocket connected")

    # Track cancel flags for this connection's files
    cancel_flags: dict[str, asyncio.Event] = {}

    try:
        while True:
            data = await ws.receive_json()
            msg_type = data.get("type")

            if msg_type == "transcribe":
                file_info = {
                    "id": data["file_id"],
                    "name": data["file_name"],
                    "path": data["file_path"],
                }
                cancel_flag = asyncio.Event()
                cancel_flags[file_info["id"]] = cancel_flag

                # Add to global queue -- the single worker processes sequentially
                await transcription_queue.put((ws, file_info, cancel_flag))
                log.info("Queued: %s (global queue depth: %d)",
                         file_info["name"], transcription_queue.qsize())

            elif msg_type == "cancel":
                file_id = data.get("file_id")
                if file_id and file_id in cancel_flags:
                    cancel_flags[file_id].set()
                    log.info("Cancel requested: %s", file_id)
                    # Immediately notify frontend
                    await ws.send_json({
                        "type": "cancelled",
                        "file_id": file_id,
                    })

            elif msg_type == "cancel_all":
                for flag in cancel_flags.values():
                    flag.set()
                log.info("Cancel all requested (%d files)", len(cancel_flags))

    except WebSocketDisconnect:
        log.info("WebSocket disconnected, cancelling %d pending files",
                 len(cancel_flags))
        # Cancel all pending jobs for this connection so the worker skips them
        for flag in cancel_flags.values():
            flag.set()
    except Exception as e:
        log.error("WebSocket error: %s", e)
        traceback.print_exc()


async def process_file(
    ws: WebSocket,
    file_info: dict,
    cancel_flag: asyncio.Event,
):
    """Transcribe a single file with chunk progress reporting."""
    file_id = file_info["id"]
    file_name = file_info["name"]
    file_path = file_info["path"]

    try:
        # Check cancellation before starting
        if cancel_flag.is_set():
            log.info("Skipped (cancelled): %s", file_name)
            try:
                await ws.send_json({
                    "type": "cancelled",
                    "file_id": file_id,
                    "name": file_name,
                })
            except Exception:
                pass
            return

        # Check if model is still loading
        if not is_model_ready():
            log.info("Waiting for model: %s", file_name)
            await ws.send_json({
                "type": "file_status",
                "file_id": file_id,
                "name": file_name,
                "status": "loading_model",
            })

            # Wait for model in async-friendly way (allows cancellation)
            while not is_model_ready():
                if cancel_flag.is_set():
                    log.info("Cancelled while waiting for model: %s", file_name)
                    await ws.send_json({
                        "type": "cancelled",
                        "file_id": file_id,
                        "name": file_name,
                    })
                    return
                await asyncio.sleep(0.5)  # Check every 500ms

            log.info("Model ready, starting: %s", file_name)

        # Send status: transcribing
        await ws.send_json({
            "type": "file_status",
            "file_id": file_id,
            "name": file_name,
            "status": "transcribing",
        })

        # Create a progress callback that sends progress updates over WebSocket.
        # Since transcribe_file runs in a thread pool, we need to use
        # run_coroutine_threadsafe to send messages from the thread.
        loop = asyncio.get_event_loop()

        def progress_callback(current_seconds: float, total_seconds: float):
            """Called from the transcription thread as segments complete."""
            future = asyncio.run_coroutine_threadsafe(
                ws.send_json({
                    "type": "file_progress",
                    "file_id": file_id,
                    "name": file_name,
                    "progress_seconds": current_seconds,
                    "total_seconds": total_seconds,
                }),
                loop,
            )
            try:
                future.result(timeout=5)
            except Exception as e:
                log.warning("Progress send error: %s", e)

        # Run transcription in a thread pool
        t0 = time.monotonic()
        text = await loop.run_in_executor(
            None, transcribe_file, file_path, progress_callback
        )
        elapsed = time.monotonic() - t0

        # Check cancellation after transcription
        if cancel_flag.is_set():
            log.info("Cancelled after transcription: %s", file_name)
            await ws.send_json({
                "type": "cancelled",
                "file_id": file_id,
                "name": file_name,
            })
        else:
            # Get file size before cleanup
            try:
                file_size = Path(file_path).stat().st_size
            except Exception:
                file_size = 0

            # Save to persistent storage
            save_transcription(
                id=file_id,
                name=file_name,
                text=text,
                size=file_size,
            )

            log.info("Result sent: %s (%d chars, %s)",
                     file_name, len(text), _fmt_duration(elapsed))
            await ws.send_json({
                "type": "file_result",
                "file_id": file_id,
                "name": file_name,
                "text": text,
            })

    except Exception as e:
        log.error("Transcription error: %s - %s", file_name, e)
        traceback.print_exc()
        try:
            await ws.send_json({
                "type": "file_error",
                "file_id": file_id,
                "name": file_name,
                "error": str(e),
            })
        except Exception:
            pass  # WebSocket may have closed

    finally:
        # Clean up uploaded file after processing
        try:
            Path(file_path).unlink(missing_ok=True)
        except Exception:
            pass


# --- Static file serving (SolidJS frontend) ---
if STATIC_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(STATIC_DIR / "assets")), name="assets")

    @app.get("/{path:path}")
    async def serve_frontend(path: str = ""):
        file_path = STATIC_DIR / path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(STATIC_DIR / "index.html")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
