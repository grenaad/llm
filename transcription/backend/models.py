"""Pydantic models for the Whisper transcription service.

Covers API request/response models, internal job state, and WebSocket messages.
"""

from __future__ import annotations

import asyncio
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class JobStatus(str, Enum):
    """Status of a transcription job in the processing pipeline."""

    WAITING = "waiting"
    LOADING_MODEL = "loading_model"
    TRANSCRIBING = "transcribing"


class ClientMsgType(str, Enum):
    """WebSocket message types sent from client to server."""

    CANCEL = "cancel"
    CANCEL_ALL = "cancel_all"


# ---------------------------------------------------------------------------
# API response models
# ---------------------------------------------------------------------------
# API response models
# ---------------------------------------------------------------------------

class GpuInfo(BaseModel):
    """Returned by GET /api/status."""
    device: str
    model_loaded: bool
    model_name: str
    compute_type: str
    batch_size: int
    gpu_name: str | None = None
    gpu_memory_total: str | None = None


class UploadResponse(BaseModel):
    """Returned by POST /api/upload/{filename}."""
    id: str
    name: str
    path: str
    size: int


class SavedTranscription(BaseModel):
    """A completed transcription stored in transcriptions.json."""
    id: str
    name: str
    text: str
    size: int
    created_at: str


class ActiveJobInfo(BaseModel):
    """Public view of an active job, returned by GET /api/jobs."""
    id: str
    name: str
    path: str
    size: int
    status: JobStatus
    progress_seconds: float | None = None
    total_seconds: float | None = None


# ---------------------------------------------------------------------------
# API request models
# ---------------------------------------------------------------------------

class TranscribeRequest(BaseModel):
    """Body for POST /api/transcribe."""
    file_id: str
    file_name: str
    file_path: str
    file_size: int = 0


# ---------------------------------------------------------------------------
# API simple response models
# ---------------------------------------------------------------------------

class TranscribeResponse(BaseModel):
    """Returned by POST /api/transcribe."""
    queued: bool
    file_id: str


class DeleteResponse(BaseModel):
    """Returned by single-item DELETE endpoints."""
    deleted: bool


class DeleteCountResponse(BaseModel):
    """Returned by bulk DELETE endpoints."""
    deleted: int


# ---------------------------------------------------------------------------
# Internal job state
# ---------------------------------------------------------------------------

class ActiveJob(BaseModel):
    """In-memory job entry in the active_jobs registry.

    Contains a cancel_flag (asyncio.Event) which is not JSON-serializable,
    so we allow arbitrary types and provide a to_info() method for the API.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    name: str
    path: str
    size: int
    status: JobStatus = JobStatus.WAITING
    cancel_flag: asyncio.Event = Field(default_factory=asyncio.Event)
    progress_seconds: float | None = None
    total_seconds: float | None = None

    def to_info(self) -> ActiveJobInfo:
        """Convert to the public API representation."""
        return ActiveJobInfo(
            id=self.id,
            name=self.name,
            path=self.path,
            size=self.size,
            status=self.status,
            progress_seconds=self.progress_seconds,
            total_seconds=self.total_seconds,
        )


# ---------------------------------------------------------------------------
# WebSocket message models
# ---------------------------------------------------------------------------

class WsFileStatus(BaseModel):
    """Broadcast when a file's status changes."""
    type: Literal["file_status"] = "file_status"
    file_id: str
    name: str
    status: JobStatus


class WsFileProgress(BaseModel):
    """Broadcast as transcription progresses."""
    type: Literal["file_progress"] = "file_progress"
    file_id: str
    name: str
    progress_seconds: float
    total_seconds: float


class WsFileResult(BaseModel):
    """Broadcast when transcription completes."""
    type: Literal["file_result"] = "file_result"
    file_id: str
    name: str
    text: str


class WsFileError(BaseModel):
    """Broadcast when transcription fails."""
    type: Literal["file_error"] = "file_error"
    file_id: str
    name: str
    error: str


class WsCancelled(BaseModel):
    """Broadcast when a job is cancelled."""
    type: Literal["cancelled"] = "cancelled"
    file_id: str
    name: str | None = None


WsMessage = Annotated[
    WsFileStatus | WsFileProgress | WsFileResult | WsFileError | WsCancelled,
    Field(discriminator="type"),
]
