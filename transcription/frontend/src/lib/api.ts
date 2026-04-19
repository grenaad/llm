import type { UploadedFile, GpuInfo } from "./types";

const API_BASE = "";

export async function fetchStatus(): Promise<GpuInfo> {
  const res = await fetch(`${API_BASE}/api/status`);
  return res.json();
}

/**
 * Upload a single file as raw binary stream (no multipart overhead).
 * Sends the file body directly as application/octet-stream via PUT.
 * Calls onProgress with (loaded, total) during upload.
 */
export function uploadFile(
  file: File,
  onProgress: (loaded: number, total: number) => void,
): Promise<UploadedFile> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) {
        onProgress(e.loaded, e.total);
      }
    };

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(JSON.parse(xhr.responseText));
      } else {
        reject(new Error(`Upload failed: ${xhr.statusText}`));
      }
    };

    xhr.onerror = () => reject(new Error("Upload failed: network error"));

    const encodedName = encodeURIComponent(file.name);
    xhr.open("POST", `${API_BASE}/api/upload/${encodedName}`);
    xhr.send(file);
  });
}

export function createWebSocket(
  onMessage: (data: any) => void,
  onClose?: () => void,
): WebSocket {
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${protocol}//${location.host}/api/ws`);

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    onMessage(data);
  };

  ws.onclose = () => {
    onClose?.();
  };

  return ws;
}

export function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024)
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}
