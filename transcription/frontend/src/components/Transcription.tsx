import { For, Show, createSignal } from "solid-js";
import { FileStatus } from "../lib/types";
import type { TranscriptionFile } from "../lib/types";
import styles from "../App.module.css";

interface TranscriptionProps {
  files: TranscriptionFile[];
  onDelete: (fileId: string) => void;
}

// Toast state - shared across all ResultCards
const [toastMessage, setToastMessage] = createSignal<string | null>(null);
let toastTimeout: number | undefined;

const showToast = (message: string) => {
  if (toastTimeout) clearTimeout(toastTimeout);
  setToastMessage(message);
  toastTimeout = window.setTimeout(() => setToastMessage(null), 3000);
};

export default function Transcription(props: TranscriptionProps) {
  const completedFiles = () => props.files.filter((f) => f.status === FileStatus.Done && f.text);
  const hasResults = () => completedFiles().length > 0;

  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      showToast("Copied to clipboard");
    } catch {
      // Fallback: create textarea
      const ta = document.createElement("textarea");
      ta.value = text;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
      showToast("Copied to clipboard");
    }
  };

  return (
    <Show when={hasResults()}>
      <div class={styles.resultsSection}>
        <h2 class={styles.sectionTitle}>Results</h2>

        <div class={styles.resultsList}>
          <For each={completedFiles()}>
            {(file) => <ResultCard file={file} onCopy={copyToClipboard} onDelete={props.onDelete} />}
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

function ResultCard(props: {
  file: TranscriptionFile;
  onCopy: (text: string) => void;
  onDelete: (fileId: string) => void;
}) {
  const [expanded, setExpanded] = createSignal(true);
  const [copied, setCopied] = createSignal(false);

  const handleCopy = () => {
    props.onCopy(props.file.text ?? "");
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const text = props.file.text ?? "";
    const blob = new Blob([text], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    // Strip original extension and add .txt
    const baseName = props.file.name.replace(/\.[^/.]+$/, "");
    a.download = `${baseName}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div class={styles.resultCard}>
      <div class={styles.resultCardHeader} onClick={() => setExpanded(!expanded())}>
        <div class={styles.resultCardTitle}>
          <svg
            class={styles.chevron}
            classList={{ [styles.chevronOpen]: expanded() }}
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
          >
            <polyline points="9 18 15 12 9 6" />
          </svg>
          <span>{props.file.name}</span>
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
              props.onDelete(props.file.id);
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
