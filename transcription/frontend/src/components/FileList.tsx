import { For, Show } from "solid-js";
import { FileStatus, isTerminalStatus } from "../lib/types";
import type { TranscriptionFile } from "../lib/types";
import { formatFileSize } from "../lib/api";
import styles from "../App.module.css";

interface FileListProps {
  files: TranscriptionFile[];
  uploadProgress: Record<string, number>;
  onCancel: (fileId: string) => void;
  onCancelAll: () => void;
  onDeleteAll: () => void;
  onRemoveFile: (fileId: string) => void;
}

function statusLabel(file: TranscriptionFile, uploadProgress: Record<string, number>): string {
  switch (file.status) {
    case FileStatus.Uploading: {
      const pct = Math.round(uploadProgress[file.id] ?? 0);
      return `Uploading ${pct}%`;
    }
    case FileStatus.Uploaded:
      return "Uploaded";
    case FileStatus.Waiting:
      return "Waiting...";
    case FileStatus.LoadingModel:
      return "Loading Model...";
    case FileStatus.Transcribing: {
      if (file.progressSeconds != null && file.totalSeconds) {
        const pct = Math.round((file.progressSeconds / file.totalSeconds) * 100);
        return `Transcribing ${pct}%`;
      }
      return "Transcribing...";
    }
    case FileStatus.Done:
      return "Done";
    case FileStatus.Error:
      return "Error";
    case FileStatus.Cancelled:
      return "Cancelled";
    default:
      return file.status;
  }
}

function statusClass(status: FileStatus): string {
  switch (status) {
    case FileStatus.Uploading:
      return styles.statusTranscribing;
    case FileStatus.LoadingModel:
      return styles.statusTranscribing;
    case FileStatus.Transcribing:
      return styles.statusTranscribing;
    case FileStatus.Done:
      return styles.statusDone;
    case FileStatus.Error:
      return styles.statusErr;
    case FileStatus.Cancelled:
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
        f.status === FileStatus.Transcribing ||
        f.status === FileStatus.Waiting ||
        f.status === FileStatus.Uploading ||
        f.status === FileStatus.LoadingModel,
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
            <button class={styles.btnClear} onClick={props.onDeleteAll}>
              Delete all
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
                    {statusLabel(file, props.uploadProgress)}
                  </span>
                  {/* Cancel button for uploading/waiting/loading_model/transcribing - red X */}
                  <Show
                    when={!isTerminalStatus(file.status)}
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
                  {/* Remove button for done/error/cancelled - trash icon */}
                  <Show
                    when={isTerminalStatus(file.status)}
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
                        <path d="M3 6h18" />
                        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6" />
                        <path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
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
