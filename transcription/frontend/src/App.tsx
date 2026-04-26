import { createSignal, onCleanup } from "solid-js";
import type { TranscriptionFile, WsMessage } from "./lib/types";
import { uploadFile, createWebSocket } from "./lib/api";
import Header from "./components/Header";
import FileUpload from "./components/FileUpload";
import FileList from "./components/FileList";
import Transcription from "./components/Transcription";
import styles from "./App.module.css";

export default function App() {
  const [files, setFiles] = createSignal<TranscriptionFile[]>([]);
  const [ws, setWs] = createSignal<WebSocket | null>(null);

  // Ensure we have a WebSocket connection, creating one if needed
  const ensureWs = (): WebSocket => {
    let socket = ws();
    if (socket && socket.readyState === WebSocket.OPEN) {
      return socket;
    }

    socket = createWebSocket(
      (msg: WsMessage) => {
        switch (msg.type) {
          case "file_status":
            setFiles((prev) =>
              prev.map((f) =>
                f.id === msg.file_id
                  ? { ...f, status: msg.status as TranscriptionFile["status"] }
                  : f,
              ),
            );
            break;

          case "file_progress":
            setFiles((prev) =>
              prev.map((f) =>
                f.id === msg.file_id
                  ? { ...f, progressSeconds: msg.progress_seconds, totalSeconds: msg.total_seconds }
                  : f,
              ),
            );
            break;

          case "file_result":
            setFiles((prev) =>
              prev.map((f) =>
                f.id === msg.file_id
                  ? {
                      ...f,
                      status: "done",
                      text: msg.text,
                      progressSeconds: undefined,
                      totalSeconds: undefined,
                    }
                  : f,
              ),
            );
            break;

          case "file_error":
            setFiles((prev) =>
              prev.map((f) =>
                f.id === msg.file_id
                  ? {
                      ...f,
                      status: "error",
                      error: msg.error,
                      progressSeconds: undefined,
                      totalSeconds: undefined,
                    }
                  : f,
              ),
            );
            break;

          case "cancelled":
            setFiles((prev) =>
              prev.map((f) =>
                f.id === msg.file_id ? { ...f, status: "cancelled" } : f,
              ),
            );
            break;
        }
      },
      () => {
        setWs(null);
      },
    );

    setWs(socket);
    return socket;
  };

  /**
   * Request transcription for a file over WebSocket.
   * Waits for the socket to be open if it was just created.
   */
  const requestTranscription = (file: TranscriptionFile) => {
    const socket = ensureWs();

    const send = () => {
      socket.send(
        JSON.stringify({
          type: "transcribe",
          file_id: file.id,
          file_name: file.name,
          file_path: file.path,
        }),
      );
    };

    if (socket.readyState === WebSocket.OPEN) {
      send();
    } else {
      // Socket is still connecting, wait for open
      socket.addEventListener("open", send, { once: true });
    }
  };

  /**
   * Handle file selection: upload each file individually and start
   * transcription as soon as each upload completes.
   */
  const handleFilesSelected = async (selectedFiles: File[]) => {
    // Create placeholder entries for all files (status: uploading)
    const placeholders: TranscriptionFile[] = selectedFiles.map((f, i) => ({
      id: `pending_${Date.now()}_${i}`,
      name: f.name,
      path: "",
      size: f.size,
      status: "uploading" as const,
      uploadProgress: 0,
    }));

    setFiles((prev) => [...prev, ...placeholders]);

    // Upload sequentially, but don't wait for transcription.
    // Each upload completes -> sends transcribe request -> next upload starts.
    // Backend queues transcriptions and processes them one at a time.
    for (let i = 0; i < selectedFiles.length; i++) {
      const file = selectedFiles[i];
      const placeholderId = placeholders[i].id;

      await uploadSingleFile(file, placeholderId);
    }
  };

  /**
   * Upload a single file and immediately request transcription when done.
   */
  const uploadSingleFile = async (file: File, placeholderId: string) => {
    try {
      const uploaded = await uploadFile(file, (loaded, total) => {
        const pct = (loaded / total) * 100;
        setFiles((prev) =>
          prev.map((f) =>
            f.id === placeholderId ? { ...f, uploadProgress: pct } : f,
          ),
        );
      });

      // Replace placeholder with real file info
      const realFile: TranscriptionFile = {
        id: uploaded.id,
        name: uploaded.name,
        path: uploaded.path,
        size: uploaded.size,
        status: "waiting",
        uploadProgress: 100,
      };

      setFiles((prev) =>
        prev.map((f) => (f.id === placeholderId ? realFile : f)),
      );

      // Immediately request transcription
      requestTranscription(realFile);
    } catch (err) {
      console.error(`Upload failed for ${file.name}:`, err);
      setFiles((prev) =>
        prev.map((f) =>
          f.id === placeholderId
            ? {
                ...f,
                status: "error" as const,
                error: `Upload failed: ${err}`,
                uploadProgress: 0,
              }
            : f,
        ),
      );
    }
  };

  const handleCancel = (fileId: string) => {
    const socket = ws();
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: "cancel", file_id: fileId }));
    }
  };

  const handleCancelAll = () => {
    const socket = ws();
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: "cancel_all" }));
    }
  };

  const handleClear = () => {
    ws()?.close();
    setWs(null);
    setFiles([]);
  };

  const handleRemoveFile = (fileId: string) => {
    setFiles((prev) => prev.filter((f) => f.id !== fileId));
  };

  onCleanup(() => {
    ws()?.close();
  });

  return (
    <div class={styles.app}>
      <Header />
      <main class={styles.main}>
        <FileUpload
          onFilesSelected={handleFilesSelected}
          disabled={false}
        />
        <FileList
          files={files()}
          onCancel={handleCancel}
          onCancelAll={handleCancelAll}
          onClear={handleClear}
          onRemoveFile={handleRemoveFile}
        />
        <Transcription files={files()} />
      </main>
    </div>
  );
}
