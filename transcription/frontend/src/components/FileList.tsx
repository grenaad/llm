import { For, Show, createSignal } from "solid-js";
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

// Toast state - shared across all rows
const [toastMessage, setToastMessage] = createSignal<string | null>(null);
let toastTimeout: number | undefined;

const showToast = (message: string) => {
  if (toastTimeout) clearTimeout(toastTimeout);
  setToastMessage(message);
  toastTimeout = window.setTimeout(() => setToastMessage(null), 3000);
};

const copyToClipboard = async (text: string) => {
  try {
    await navigator.clipboard.writeText(text);
    showToast("Copied to clipboard");
  } catch {
    const ta = document.createElement("textarea");
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand("copy");
    document.body.removeChild(ta);
    showToast("Copied to clipboard");
  }
};

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
              <Show
                when={file.status === FileStatus.Done && file.text}
                fallback={
                  <InProgressRow
                    file={file}
                    uploadProgress={props.uploadProgress}
                    onCancel={props.onCancel}
                    onRemove={props.onRemoveFile}
                  />
                }
              >
                <DoneRow file={file} onRemove={props.onRemoveFile} />
              </Show>
            )}
          </For>
        </div>
      </div>

      {/* Toast notification */}
      <Show when={toastMessage()}>
        <div class={styles.toast}>{toastMessage()}</div>
      </Show>
    </Show>
  );
}

// --- In-progress / Error / Cancelled row ---

function InProgressRow(props: {
  file: TranscriptionFile;
  uploadProgress: Record<string, number>;
  onCancel: (fileId: string) => void;
  onRemove: (fileId: string) => void;
}) {
  return (
    <div class={styles.fileItem}>
      <div class={styles.fileInfo}>
        <span class={styles.fileName}>{props.file.name}</span>
        <span class={styles.fileSize}>
          {formatFileSize(props.file.size)}
        </span>
      </div>
      <div class={styles.fileActions}>
        <span
          class={`${styles.fileStatus} ${statusClass(props.file.status)}`}
        >
          {statusLabel(props.file, props.uploadProgress)}
        </span>
        {/* Cancel button for uploading/waiting/loading_model/transcribing */}
        <Show when={!isTerminalStatus(props.file.status)}>
          <button
            class={styles.btnCancelSmall}
            onClick={() => props.onCancel(props.file.id)}
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
        {/* Remove button for error/cancelled */}
        <Show when={isTerminalStatus(props.file.status)}>
          <button
            class={styles.btnRemove}
            onClick={() => props.onRemove(props.file.id)}
            title="Remove"
          >
            <TrashIcon />
          </button>
        </Show>
      </div>
    </div>
  );
}

// --- Done row with expandable transcription ---

function DoneRow(props: {
  file: TranscriptionFile;
  onRemove: (fileId: string) => void;
}) {
  const [expanded, setExpanded] = createSignal(false);
  const [copied, setCopied] = createSignal(false);

  const handleCopy = () => {
    copyToClipboard(props.file.text ?? "");
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const text = props.file.text ?? "";
    const blob = new Blob([text], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    const baseName = props.file.name.replace(/\.[^/.]+$/, "");
    a.download = `${baseName}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div classList={{ [styles.fileItemDone]: true, [styles.fileItemDoneExpanded]: expanded() }}>
      <div class={styles.fileItemDoneHeader} onClick={() => setExpanded(!expanded())}>
        <div class={styles.fileInfo}>
          <svg
            class={styles.chevron}
            classList={{ [styles.chevronOpen]: expanded() }}
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
          >
            <polyline points="9 18 15 12 9 6" />
          </svg>
          <span class={styles.fileName}>{props.file.name}</span>
          <span class={styles.fileSize}>
            {formatFileSize(props.file.size)}
          </span>
        </div>
        <div class={styles.resultCardActions}>
          <button
            class={styles.btnCopy}
            onClick={(e) => {
              e.stopPropagation();
              handleCopy();
            }}
            title="Copy to clipboard"
          >
            {copied() ? <CheckIcon /> : <CopyIcon />}
          </button>
          <button
            class={styles.btnCopy}
            onClick={(e) => {
              e.stopPropagation();
              handleDownload();
            }}
            title="Download as .txt"
          >
            <DownloadIcon />
          </button>
          <button
            class={styles.btnCopy}
            onClick={(e) => {
              e.stopPropagation();
              props.onRemove(props.file.id);
            }}
            title="Delete"
          >
            <TrashIcon />
          </button>
        </div>
      </div>
      <Show when={expanded()}>
        <div class={styles.resultCardBody}>
          <p class={styles.resultText}>{props.file.text}</p>
        </div>
      </Show>
    </div>
  );
}

// --- Icons ---

function CopyIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
    </svg>
  );
}

function CheckIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--success)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
}

function DownloadIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="7 10 12 15 17 10" />
      <line x1="12" y1="15" x2="12" y2="3" />
    </svg>
  );
}

function TrashIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <path d="M3 6h18" />
      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6" />
      <path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
    </svg>
  );
}
