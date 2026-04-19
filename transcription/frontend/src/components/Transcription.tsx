import { For, Show, createSignal } from "solid-js";
import type { TranscriptionFile } from "../lib/types";
import styles from "../App.module.css";

interface TranscriptionProps {
  files: TranscriptionFile[];
}

export default function Transcription(props: TranscriptionProps) {
  const completedFiles = () => props.files.filter((f) => f.status === "done" && f.text);
  const hasResults = () => completedFiles().length > 0;

  const allPlainText = () =>
    completedFiles()
      .map((f) => `${f.name}\n${f.text}`)
      .join("\n\n---\n\n");

  const allMarkdown = () =>
    completedFiles()
      .map((f) => `### ${f.name}\n\n${f.text}`)
      .join("\n\n---\n\n");

  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      // Fallback: create textarea
      const ta = document.createElement("textarea");
      ta.value = text;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
    }
  };

  const downloadTxt = () => {
    const text = allPlainText();
    const blob = new Blob([text], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "transcription.txt";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <Show when={hasResults()}>
      <div class={styles.resultsSection}>
        <div class={styles.resultsSectionHeader}>
          <h2 class={styles.sectionTitle}>Results</h2>
          <div class={styles.resultsActions}>
            <button
              class={styles.btnSecondary}
              onClick={() => copyToClipboard(allPlainText())}
              title="Copy as plain text"
            >
              <CopyIcon /> Text
            </button>
            <button
              class={styles.btnSecondary}
              onClick={() => copyToClipboard(allMarkdown())}
              title="Copy as markdown"
            >
              <CopyIcon /> Markdown
            </button>
            <button class={styles.btnSecondary} onClick={downloadTxt} title="Download as .txt">
              <DownloadIcon /> Download
            </button>
          </div>
        </div>

        <div class={styles.resultsList}>
          <For each={completedFiles()}>
            {(file) => <ResultCard file={file} onCopy={copyToClipboard} />}
          </For>
        </div>
      </div>
    </Show>
  );
}

function ResultCard(props: {
  file: TranscriptionFile;
  onCopy: (text: string) => void;
}) {
  const [expanded, setExpanded] = createSignal(true);
  const [copied, setCopied] = createSignal(false);

  const handleCopy = () => {
    props.onCopy(props.file.text ?? "");
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
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
