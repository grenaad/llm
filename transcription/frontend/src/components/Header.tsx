import { createResource, createSignal, onCleanup } from "solid-js";
import { fetchStatus } from "../lib/api";
import type { GpuInfo } from "../lib/types";
import styles from "../App.module.css";

export default function Header() {
  const [status, { refetch }] = createResource<GpuInfo>(fetchStatus);
  const [polling, setPolling] = createSignal(true);

  // Poll /api/status every 3s until model is loaded, then stop
  const interval = setInterval(() => {
    if (!polling()) return;
    refetch();
    const s = status();
    if (s?.model_loaded) {
      setPolling(false);
      clearInterval(interval);
    }
  }, 3000);

  onCleanup(() => clearInterval(interval));

  const statusText = () => {
    const s = status();
    if (!s) return "";
    if (!s.model_loaded) return "Loading model...";
    return s.gpu_name ?? s.device;
  };

  return (
    <header class={styles.header}>
      <div class={styles.headerLeft}>
        <h1 class={styles.title}>Whisper Transcription</h1>
        <span class={styles.subtitle}>Local speech-to-text</span>
      </div>
      <div class={styles.headerRight}>
        {status() && (
          <div class={styles.statusBadge}>
            <span
              class={styles.statusDot}
              classList={{
                [styles.statusOk]: status()!.model_loaded,
                [styles.statusError]: !status()!.model_loaded,
              }}
            />
            <span class={styles.statusText}>{statusText()}</span>
          </div>
        )}
      </div>
    </header>
  );
}
