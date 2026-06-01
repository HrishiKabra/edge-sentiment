/**
 * EdgeSentiment — on-device sentiment instrument.
 *
 * Orchestrates the two inference backends (in-browser WebAssembly and the
 * Cloudflare Worker), a 300 ms-debounced classify-on-keystroke loop, determinate
 * model-load progress, and clean error surfacing. The visual language is a dark
 * "edge signal" instrument: sentiment reads as a live red↔green meter and
 * latency is an instrument readout.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { classify, isReady, loadModel, type ClassifyResult } from "./inference";
import { BACKEND_LABELS, type Backend } from "./types";
import { BackendToggle } from "./components/BackendToggle";
import { SentimentBar } from "./components/SentimentBar";
import { LatencyDisplay } from "./components/LatencyDisplay";

const DEFAULT_TEXT = "The movie was absolutely brilliant.";
const DEBOUNCE_MS = 300;
const WORKER_URL = import.meta.env.VITE_WORKER_URL ?? "https://edge-sentiment.workers.dev";

interface EdgeResponse {
  label?: ClassifyResult["label"];
  confidence?: number;
  latency_ms?: number;
  error?: string;
}

/** Call the Cloudflare Worker's /classify endpoint. */
async function classifyEdge(text: string): Promise<ClassifyResult> {
  const response = await fetch(`${WORKER_URL.replace(/\/$/, "")}/classify`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  const data = (await response.json()) as EdgeResponse;
  if (!response.ok || data.error || data.label == null) {
    throw new Error(data.error ?? `Edge request failed (${response.status})`);
  }
  return {
    label: data.label,
    confidence: data.confidence ?? 0,
    latency_ms: data.latency_ms ?? 0,
  };
}

export default function App() {
  const [text, setText] = useState(DEFAULT_TEXT);
  const [backend, setBackend] = useState<Backend>("wasm");
  const [result, setResult] = useState<ClassifyResult | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [modelLoading, setModelLoading] = useState(true);
  const [progress, setProgress] = useState(0);
  const [loadInfo, setLoadInfo] = useState<{ load_time_ms: number; size_mb: number } | null>(null);

  // Guards against state updates from a stale inference resolving out of order.
  const requestId = useRef(0);

  // Kick off the model download on mount (and whenever we return to the wasm
  // backend without a loaded session) to drive the progress bar.
  useEffect(() => {
    if (backend !== "wasm") return;
    if (isReady()) {
      setModelLoading(false);
      return;
    }
    let cancelled = false;
    setModelLoading(true);
    loadModel((fraction) => {
      if (!cancelled) setProgress(fraction);
    })
      .then((info) => {
        if (cancelled) return;
        setLoadInfo(info);
        setModelLoading(false);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "Failed to load model");
        setModelLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [backend]);

  const runInference = useCallback(async (value: string, target: Backend) => {
    if (!value.trim()) {
      setResult(null);
      setError(null);
      return;
    }
    const id = ++requestId.current;
    setRunning(true);
    setError(null);
    try {
      const r = target === "wasm" ? await classify(value) : await classifyEdge(value);
      if (id === requestId.current) setResult(r);
    } catch (err: unknown) {
      if (id === requestId.current) {
        setError(err instanceof Error ? err.message : "Inference failed");
      }
    } finally {
      if (id === requestId.current) setRunning(false);
    }
  }, []);

  // Debounced classify on every keystroke / backend change.
  useEffect(() => {
    const handle = window.setTimeout(() => {
      void runInference(text, backend);
    }, DEBOUNCE_MS);
    return () => window.clearTimeout(handle);
  }, [text, backend, runInference]);

  const skeleton = backend === "wasm" && modelLoading && result === null;
  const sizeLabel = loadInfo ? `${Math.round(loadInfo.size_mb)} MB` : "65 MB";
  const loadedLabel = loadInfo ? `${Math.round(loadInfo.load_time_ms)} ms` : "—";

  return (
    <div className="app">
      <div className="app__grain" aria-hidden="true" />
      <div className="app__glow" aria-hidden="true" />

      <main className="panel">
        <header className="panel__head">
          <div className="brand">
            <span className="brand__mark" aria-hidden="true" />
            <h1 className="brand__name">EdgeSentiment</h1>
          </div>
          <p className="tagline">
            Sentiment analysis running <em>on your device</em> — INT8 DistilBERT,
            compiled to WebAssembly. Nothing leaves the browser.
          </p>
        </header>

        <BackendToggle value={backend} onChange={setBackend} />

        <label className="field">
          <span className="field__label">input text</span>
          <textarea
            className="field__input"
            value={text}
            spellCheck={false}
            rows={3}
            onChange={(event) => setText(event.target.value)}
            aria-label="Text to classify"
          />
        </label>

        <SentimentBar result={result} skeleton={skeleton} />

        <div className="panel__readout">
          <LatencyDisplay
            backendLabel={BACKEND_LABELS[backend]}
            latencyMs={result?.latency_ms ?? null}
            running={running}
          />
        </div>

        {modelLoading && backend === "wasm" && (
          <div className="loader" role="status">
            <div className="loader__row">
              <span>Loading model (~65 MB)…</span>
              <span className="loader__pct">{Math.round(progress * 100)}%</span>
            </div>
            <div className="loader__track">
              <span className="loader__fill" style={{ width: `${progress * 100}%` }} />
            </div>
          </div>
        )}

        {error && (
          <div className="error" role="alert">
            <strong>error</strong> {error}
            {backend === "edge" && (
              <span className="error__hint">
                Set <code>VITE_WORKER_URL</code> to your deployed Worker.
              </span>
            )}
          </div>
        )}

        <footer className="stats">
          <span className="stats__item">
            <span className="stats__key">model</span> DistilBERT INT8
          </span>
          <span className="stats__sep" aria-hidden="true">
            ·
          </span>
          <span className="stats__item">
            <span className="stats__key">size</span> {sizeLabel}
          </span>
          <span className="stats__sep" aria-hidden="true">
            ·
          </span>
          <span className="stats__item">
            <span className="stats__key">loaded in</span> {loadedLabel}
          </span>
        </footer>
      </main>
    </div>
  );
}
