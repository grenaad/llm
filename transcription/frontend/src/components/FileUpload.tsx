import { createSignal } from "solid-js";
import styles from "../App.module.css";

interface FileUploadProps {
  onFilesSelected: (files: File[]) => void;
  disabled?: boolean;
}

const ACCEPTED_TYPES = [
  ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma",
  ".mp4", ".mkv", ".webm", ".mov", ".avi",
];

export default function FileUpload(props: FileUploadProps) {
  const [isDragging, setIsDragging] = createSignal(false);
  let inputRef!: HTMLInputElement;

  const handleDrop = (e: DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (props.disabled) return;

    const files = Array.from(e.dataTransfer?.files ?? []);
    if (files.length > 0) {
      props.onFilesSelected(files);
    }
  };

  const handleDragOver = (e: DragEvent) => {
    e.preventDefault();
    if (!props.disabled) setIsDragging(true);
  };

  const handleDragLeave = (e: DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleClick = () => {
    if (!props.disabled) inputRef.click();
  };

  const handleFileInput = (e: Event) => {
    const target = e.target as HTMLInputElement;
    const files = Array.from(target.files ?? []);
    if (files.length > 0) {
      props.onFilesSelected(files);
    }
    target.value = "";
  };

  return (
    <div
      class={styles.dropZone}
      classList={{
        [styles.dropZoneActive]: isDragging(),
        [styles.dropZoneDisabled]: props.disabled,
      }}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onClick={handleClick}
    >
      <input
        ref={inputRef}
        type="file"
        multiple
        accept={ACCEPTED_TYPES.join(",")}
        onChange={handleFileInput}
        style={{ display: "none" }}
      />
      <div class={styles.dropZoneContent}>
        <svg
          class={styles.dropZoneIcon}
          width="48"
          height="48"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="1.5"
          stroke-linecap="round"
          stroke-linejoin="round"
        >
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
          <polyline points="17 8 12 3 7 8" />
          <line x1="12" y1="3" x2="12" y2="15" />
        </svg>
        <p class={styles.dropZoneText}>
          {props.disabled ? "Processing..." : "Drag & drop files here"}
        </p>
        <p class={styles.dropZoneHint}>
          or click to browse
        </p>
        <p class={styles.dropZoneFormats}>
          MP4, MKV, WAV, MP3, FLAC, OGG, AAC, WebM, MOV, AVI
        </p>
      </div>
    </div>
  );
}
