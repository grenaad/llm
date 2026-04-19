export type FileStatus =
  | "uploading"
  | "uploaded"
  | "waiting"
  | "transcribing"
  | "done"
  | "error"
  | "cancelled";

export interface TranscriptionFile {
  id: string;
  name: string;
  path: string;
  size: number;
  status: FileStatus;
  uploadProgress: number;
  text?: string;
  error?: string;
  chunk?: number;
  totalChunks?: number;
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
  chunk?: number;
  total_chunks?: number;
}

export interface GpuInfo {
  device: string;
  model_loaded: boolean;
  gpu_name?: string;
  gpu_memory_total?: string;
}
