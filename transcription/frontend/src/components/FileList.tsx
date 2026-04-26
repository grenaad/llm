import { For, Show } from "solid-js";
import type { TranscriptionFile } from "../lib/types";
import { formatFileSize } from "../lib/api";
import styles from "../App.module.css";

interface FileListProps {
  files: TranscriptionFile[];
  onCancel: (fileId: string) => void;
  onCancelAll: () => void;
  onClear: () => void;
  onRemoveFile: (fileId: string) => void;
}

function statusLabel(file: TranscriptionFile): string {
  switch (file.status) {
    case "uploading": {
      const pct = Math.round(file.uploadProgress);
      return `Uploading ${pct}%`;
    }
    case "uploaded":
      return "Uploaded";
    case "waiting":
      return "Waiting...";
    case "transcribing": {
      if (file.progressSeconds != null && file.totalSeconds) {
        const pct = Math.round((file.progressSeconds / file.totalSeconds) * 100);
        return `Transcribing ${pct}%`;
      }
      return "Transcribing...";
    }
    case "done":
      return "Done";
    case "error":
      return "Error";
    case "cancelled":
      return "Cancelled";
    default:
      return file.status;
  }
}

function statusClass(status: string): string {
  switch (status) {
    case "uploading":
      return styles.statusTranscribing;
    case "transcribing":
      return styles.statusTranscribing;
    case "done":
      return styles.statusDone;
    case "error":
      return styles.statusErr;
    case "cancelled":
      return styles.statusCancelled;
    default:
      return styles.statusWaiting;
  }
}

export default function FileList(props: FileListProps) {
  const hasFiles = () => props.files.length > 0;
  const isProcessing = () =>
    props.files.some(
      (f) =>
        f.status === "transcribing" ||
        f.status === "waiting" ||
        f.status === "uploading",
    );

  return (
    <Show when={hasFiles()}>
      <div class={styles.fileSection}>
        <div class={styles.fileSectionHeader}>
          <h2 class={styles.sectionTitle}>Files</h2>
          <div class={styles.fileSectionActions}>
            <Show when={isProcessing()}>
              <button class={styles.btnCancel} onClick={props.onCancelAll}>
                Cancel All
              </button>
            </Show>
            <button class={styles.btnClear} onClick={props.onClear}>
              Clear
            </button>
          </div>
        </div>

        <div class={styles.fileList}>
          <For each={props.files}>
            {(file) => (
              <div class={styles.fileItem}>
                <div class={styles.fileInfo}>
                  <span class={styles.fileName}>{file.name}</span>
                  <span class={styles.fileSize}>
                    {formatFileSize(file.size)}
                  </span>
                </div>
                <div class={styles.fileActions}>
                  <span
                    class={`${styles.fileStatus} ${statusClass(file.status)}`}
                  >
                    {statusLabel(file)}
                  </span>
                  <Show
                    when={
                      file.status === "transcribing" ||
                      file.status === "waiting"
                    }
                  >
                    <button
                      class={styles.btnCancelSmall}
                      onClick={() => props.onCancel(file.id)}
                      title="Cancel"
                    >
                      <svg
                        width="14"
                        height="14"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        stroke-width="2"
                      >
                        <line x1="18" y1="6" x2="6" y2="18" />
                        <line x1="6" y1="6" x2="18" y2="18" />
                      </svg>
                    </button>
                  </Show>
                  <Show
                    when={
                      file.status === "done" ||
                      file.status === "error" ||
                      file.status === "cancelled"
                    }
                  >
                    <button
                      class={styles.btnRemove}
                      onClick={() => props.onRemoveFile(file.id)}
                      title="Remove"
                    >
                      <svg
                        width="14"
                        height="14"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        stroke-width="2"
                      >
                        <line x1="18" y1="6" x2="6" y2="18" />
                        <line x1="6" y1="6" x2="18" y2="18" />
                      </svg>
                    </button>
                  </Show>
                </div>
              </div>
            )}
          </For>
        </div>
      </div>
    </Show>
  );
}
