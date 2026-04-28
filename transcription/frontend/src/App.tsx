import { createSignal, onCleanup, onMount } from "solid-js";
import { FileStatus, isTerminalStatus } from "./lib/types";
import type { TranscriptionFile, WsMessage } from "./lib/types";
import { uploadFile, createWebSocket, fetchTranscriptions, fetchJobs, submitTranscription, deleteTranscription, deleteAllTranscriptions } from "./lib/api";
import Header from "./components/Header";
import FileUpload from "./components/FileUpload";
import FileList from "./components/FileList";
import styles from "./App.module.css";

// Store abort functions outside of reactive state
// This avoids re-renders when storing/retrieving abort functions
const abortFunctions = new Map<string, () => void>();

export default function App() {
  const [files, setFiles] = createSignal<TranscriptionFile[]>([]);
  const [ws, setWs] = createSignal<WebSocket | null>(null);

  // Separate signal for upload progress - updates frequently without causing row re-renders
  const [uploadProgress, setUploadProgress] = createSignal<Record<string, number>>({});

  // Load saved transcriptions and active jobs on mount
  onMount(async () => {
    try {
      const [saved, jobs] = await Promise.all([
        fetchTranscriptions(),
        fetchJobs(),
      ]);

      // Convert saved transcriptions to TranscriptionFile format
      const completedFiles: TranscriptionFile[] = saved.map((t) => ({
        id: t.id,
        name: t.name,
        path: "",
        size: t.size,
        status: FileStatus.Done,
        text: t.text,
      }));

      // Convert active jobs to TranscriptionFile format
      const activeFiles: TranscriptionFile[] = jobs.map((j) => ({
        id: j.id,
        name: j.name,
        path: j.path,
        size: j.size,
        status: j.status as FileStatus,
        progressSeconds: j.progress_seconds,
        totalSeconds: j.total_seconds,
      }));

      // Deduplicate: active jobs take priority (in case a job just completed)
      const activeIds = new Set(activeFiles.map((f) => f.id));
      const deduped = completedFiles.filter((f) => !activeIds.has(f.id));

      setFiles([...deduped, ...activeFiles]);

      // Connect WebSocket to receive live updates for active jobs
      if (activeFiles.length > 0) {
        ensureWs();
      }
    } catch (err) {
      console.error("Failed to load transcriptions:", err);
    }
  });

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
                  : f
              )
            );
            break;

          case "file_progress":
            setFiles((prev) =>
              prev.map((f) =>
                f.id === msg.file_id
                  ? {
                      ...f,
                      progressSeconds: msg.progress_seconds,
                      totalSeconds: msg.total_seconds,
                    }
                  : f
              )
            );
            break;

          case "file_result":
            setFiles((prev) =>
              prev.map((f) =>
                f.id === msg.file_id
                  ? {
                      ...f,
                      status: FileStatus.Done,
                      text: msg.text,
                      progressSeconds: undefined,
                      totalSeconds: undefined,
                    }
                  : f
              )
            );
            break;

          case "file_error":
            setFiles((prev) =>
              prev.map((f) =>
                f.id === msg.file_id
                  ? {
                      ...f,
                      status: FileStatus.Error,
                      error: msg.error,
                      progressSeconds: undefined,
                      totalSeconds: undefined,
                    }
                  : f
              )
            );
            break;

          case "cancelled":
            setFiles((prev) =>
              prev.map((f) =>
                f.id === msg.file_id ? { ...f, status: FileStatus.Cancelled } : f
              )
            );
            break;
        }
      },
      () => {
        setWs(null);
      }
    );

    setWs(socket);
    return socket;
  };

  /**
   * Submit a file for transcription via REST endpoint.
   * This atomically registers the job on the backend, so it survives page refresh.
   */
  const requestTranscription = async (file: TranscriptionFile) => {
    await submitTranscription({
      id: file.id,
      name: file.name,
      path: file.path,
      size: file.size,
    });
    // Ensure WebSocket is connected to receive progress updates
    ensureWs();
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
      status: FileStatus.Uploading,
    }));

    setFiles((prev) => [...prev, ...placeholders]);

    // Initialize progress for all placeholders
    setUploadProgress((prev) => {
      const updated = { ...prev };
      for (const p of placeholders) {
        updated[p.id] = 0;
      }
      return updated;
    });

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
    const { promise, abort } = uploadFile(file, (loaded, total) => {
      const pct = (loaded / total) * 100;
      // Update progress in separate signal - doesn't cause file row re-renders
      setUploadProgress((prev) => ({ ...prev, [placeholderId]: pct }));
    });

    // Store abort function outside of reactive state
    abortFunctions.set(placeholderId, abort);

    try {
      const uploaded = await promise;

      // Clean up abort function and progress
      abortFunctions.delete(placeholderId);
      setUploadProgress((prev) => {
        const updated = { ...prev };
        delete updated[placeholderId];
        // Keep progress for the new ID
        updated[uploaded.id] = 100;
        return updated;
      });

      // Replace placeholder with real file info
      const realFile: TranscriptionFile = {
        id: uploaded.id,
        name: uploaded.name,
        path: uploaded.path,
        size: uploaded.size,
        status: FileStatus.Waiting,
      };

      setFiles((prev) =>
        prev.map((f) => (f.id === placeholderId ? realFile : f))
      );

      // Immediately request transcription
      requestTranscription(realFile);
    } catch (err) {
      // Clean up abort function
      abortFunctions.delete(placeholderId);

      const errorMessage = err instanceof Error ? err.message : String(err);
      // Don't show error for cancelled uploads
      if (errorMessage === "Upload cancelled") {
        setFiles((prev) =>
          prev.map((f) =>
            f.id === placeholderId ? { ...f, status: FileStatus.Cancelled } : f
          )
        );
      } else {
        console.error(`Upload failed for ${file.name}:`, err);
        setFiles((prev) =>
          prev.map((f) =>
            f.id === placeholderId
              ? {
                  ...f,
                  status: FileStatus.Error,
                  error: `Upload failed: ${errorMessage}`,
                }
              : f
          )
        );
      }

      // Clean up progress
      setUploadProgress((prev) => {
        const updated = { ...prev };
        delete updated[placeholderId];
        return updated;
      });
    }
  };

  const handleCancel = (fileId: string) => {
    // Abort upload if in progress (stored outside reactive state)
    const abortFn = abortFunctions.get(fileId);
    if (abortFn) {
      abortFn();
      abortFunctions.delete(fileId);
    }

    // Cancel transcription via WebSocket
    const socket = ws();
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: "cancel", file_id: fileId }));
    }
  };

  const handleCancelAll = () => {
    // Abort all uploads in progress
    for (const [id, abortFn] of abortFunctions) {
      abortFn();
      abortFunctions.delete(id);
    }

    // Cancel all transcriptions via WebSocket
    const socket = ws();
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: "cancel_all" }));
    }
  };

  const handleDeleteAll = async () => {
    // Delete all from server
    try {
      await deleteAllTranscriptions();
    } catch (err) {
      console.error("Failed to delete transcriptions:", err);
    }

    // Remove finished items from UI (done, cancelled, error)
    const removedIds = new Set(
      files()
        .filter((f) => isTerminalStatus(f.status))
        .map((f) => f.id)
    );

    setFiles((prev) => prev.filter((f) => !removedIds.has(f.id)));
    setUploadProgress((prev) => {
      const updated = { ...prev };
      for (const id of removedIds) {
        delete updated[id];
      }
      return updated;
    });
  };

  const handleRemoveFile = async (fileId: string) => {
    // Delete from server (for completed transcriptions)
    try {
      await deleteTranscription(fileId);
    } catch (err) {
      console.error("Failed to delete transcription:", err);
    }

    setFiles((prev) => prev.filter((f) => f.id !== fileId));
    setUploadProgress((prev) => {
      const updated = { ...prev };
      delete updated[fileId];
      return updated;
    });
  };

  onCleanup(() => {
    ws()?.close();
  });

  return (
    <div class={styles.app}>
      <Header />
      <main class={styles.main}>
        <FileUpload onFilesSelected={handleFilesSelected} disabled={false} />
        <FileList
          files={files()}
          uploadProgress={uploadProgress()}
          onCancel={handleCancel}
          onCancelAll={handleCancelAll}
          onDeleteAll={handleDeleteAll}
          onRemoveFile={handleRemoveFile}
        />
      </main>
    </div>
  );
}
