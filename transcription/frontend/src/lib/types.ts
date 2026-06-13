export enum FileStatus {
  Uploading = "uploading",
  Uploaded = "uploaded",
  Waiting = "waiting",
  LoadingModel = "loading_model",
  Transcribing = "transcribing",
  Muxing = "muxing",
  Done = "done",
  Error = "error",
  Cancelled = "cancelled",
}

export const TERMINAL_STATUSES: readonly FileStatus[] = [
  FileStatus.Done,
  FileStatus.Error,
  FileStatus.Cancelled,
];

export function isTerminalStatus(status: FileStatus): boolean {
  return TERMINAL_STATUSES.includes(status);
}

export interface TranscriptionFile {
  id: string;
  name: string;
  path: string;
  size: number;
  status: FileStatus;
  text?: string;
  error?: string;
  progressSeconds?: number;
  totalSeconds?: number;
  hasVideo?: boolean;
}

export interface UploadedFile {
  id: string;
  name: string;
  path: string;
  size: number;
}

export interface WsMessage {
  type: string;
  file_id?: string;
  name?: string;
  status?: string;
  text?: string;
  error?: string;
  progress_seconds?: number;
  total_seconds?: number;
  has_video?: boolean;
}

export interface GpuInfo {
  device: string;
  model_loaded: boolean;
  gpu_name?: string;
  gpu_memory_total?: string;
}

export interface SavedTranscription {
  id: string;
  name: string;
  text: string;
  size: number;
  created_at: string;
  srt_text?: string;
  video_path?: string;
}

export interface ActiveJob {
  id: string;
  name: string;
  path: string;
  size: number;
  status: string;
  progress_seconds?: number;
  total_seconds?: number;
}
