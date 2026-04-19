import asyncio
import logging
import os
import sys
import time
import traceback
from collections import deque
from pathlib import Path
from uuid import uuid4

import aiofiles
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request

from transcriber import load_model, transcribe_file, get_gpu_info, _fmt_size, _fmt_duration

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
# Quiet down noisy libraries
logging.getLogger("nemo_logger").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

log = logging.getLogger("parakeet")

# --- Configuration ---
UPLOAD_DIR = Path("/tmp/parakeet_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

STATIC_DIR = Path(__file__).parent / "static"

# --- App setup ---
app = FastAPI(title="Parakeet Transcription")


@app.on_event("startup")
async def startup():
    import threading
    threading.Thread(target=load_model, daemon=True, name="model-loader").start()
    log.info("Server ready, model loading in background")


# --- API routes ---
@app.get("/api/status")
async def status():
    return get_gpu_info()


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

    # Per-connection state: transcription queue and cancellation flags
    queue: deque[dict] = deque()
    cancel_flags: dict[str, asyncio.Event] = {}
    processing = False
    processing_lock = asyncio.Lock()

    async def process_queue():
        """Process files in the queue one at a time."""
        nonlocal processing
        async with processing_lock:
            if processing:
                return
            processing = True

        try:
            while queue:
                file_info = queue.popleft()
                remaining = len(queue)
                log.info("Transcription starting: %s (%d more in queue)",
                         file_info["name"], remaining)
                await process_file(ws, file_info, cancel_flags)
        finally:
            processing = False

    try:
        while True:
            data = await ws.receive_json()
            msg_type = data.get("type")

            if msg_type == "transcribe":
                # Add file to queue and start processing
                file_info = {
                    "id": data["file_id"],
                    "name": data["file_name"],
                    "path": data["file_path"],
                }
                cancel_flags[file_info["id"]] = asyncio.Event()
                queue.append(file_info)
                log.info("Queued: %s (queue depth: %d)", file_info["name"], len(queue))

                # Kick off processing (no-op if already running)
                asyncio.create_task(process_queue())

            elif msg_type == "cancel":
                file_id = data.get("file_id")
                if file_id and file_id in cancel_flags:
                    cancel_flags[file_id].set()
                    log.info("Cancel requested: %s", file_id)

            elif msg_type == "cancel_all":
                for flag in cancel_flags.values():
                    flag.set()
                log.info("Cancel all requested (%d files)", len(cancel_flags))

    except WebSocketDisconnect:
        log.info("WebSocket disconnected")
    except Exception as e:
        log.error("WebSocket error: %s", e)
        traceback.print_exc()


async def process_file(
    ws: WebSocket,
    file_info: dict,
    cancel_flags: dict[str, asyncio.Event],
):
    """Transcribe a single file with chunk progress reporting."""
    file_id = file_info["id"]
    file_name = file_info["name"]
    file_path = file_info["path"]

    try:
        # Check cancellation before starting
        if file_id in cancel_flags and cancel_flags[file_id].is_set():
            log.info("Skipped (cancelled): %s", file_name)
            await ws.send_json({
                "type": "cancelled",
                "file_id": file_id,
                "name": file_name,
            })
            return

        # Send status: transcribing
        await ws.send_json({
            "type": "file_status",
            "file_id": file_id,
            "name": file_name,
            "status": "transcribing",
        })

        # Create a progress callback that sends chunk updates over WebSocket.
        # Since transcribe_file runs in a thread pool, we need to use
        # run_coroutine_threadsafe to send messages from the thread.
        loop = asyncio.get_event_loop()

        def progress_callback(chunk: int, total_chunks: int):
            """Called from the transcription thread for each chunk."""
            future = asyncio.run_coroutine_threadsafe(
                ws.send_json({
                    "type": "file_progress",
                    "file_id": file_id,
                    "name": file_name,
                    "chunk": chunk,
                    "total_chunks": total_chunks,
                }),
                loop,
            )
            # Wait for the send to complete (with timeout) to ensure
            # messages are delivered in order
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
        if file_id in cancel_flags and cancel_flags[file_id].is_set():
            log.info("Cancelled after transcription: %s", file_name)
            await ws.send_json({
                "type": "cancelled",
                "file_id": file_id,
                "name": file_name,
            })
        else:
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
        cancel_flags.pop(file_id, None)
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
