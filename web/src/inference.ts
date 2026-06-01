/**
 * Browser-side inference for EdgeSentiment.
 *
 * Runs the INT8-quantized DistilBERT entirely on-device: WordPiece tokenization
 * via @xenova/transformers (using the exact vocab the model was trained on,
 * bundled under /models/distilbert-sst2/) and the forward pass via
 * onnxruntime-web on the WebAssembly backend. No text ever leaves the browser.
 *
 * The InferenceSession and tokenizer are created once and reused. `loadModel()`
 * streams the ~65 MB model with real download-progress reporting so the UI can
 * show a determinate progress bar instead of an indefinite spinner.
 */

import * as ort from "onnxruntime-web";
import { AutoTokenizer, env, type PreTrainedTokenizer } from "@xenova/transformers";

export type SentimentLabel = "positive" | "negative";

export interface ClassifyResult {
  label: SentimentLabel;
  confidence: number; // 0..1, probability of the predicted label
  latency_ms: number; // wall-clock time around session.run() only
}

export interface LoadResult {
  load_time_ms: number;
  size_mb: number;
}

// BASE_URL is "/" in dev and the Vite `base` (e.g. "/edge-sentiment/") in a
// project-page build, so all asset paths must be prefixed with it to work both
// at a domain root and under a GitHub Pages subpath.
const BASE = import.meta.env.BASE_URL;
const MODEL_URL = `${BASE}model/distilbert-sst2-int8.onnx`;
const TOKENIZER_ID = "distilbert-sst2";
const MAX_SEQ_LENGTH = 128;

// --- onnxruntime-web environment ------------------------------------------- //
// wasmPaths points at the binaries copied by vite-plugin-static-copy (see
// vite.config.ts). numThreads is the project's `numThreads` setting: multi-
// threaded wasm requires SharedArrayBuffer, which requires cross-origin
// isolation (COOP/COEP). We adapt to whatever the host grants and fall back to
// single-threaded so the demo works on any static host (e.g. GitHub Pages).
ort.env.wasm.wasmPaths = `${BASE}ort/`;
ort.env.wasm.simd = true;
ort.env.wasm.numThreads =
  typeof self !== "undefined" && self.crossOriginIsolated
    ? Math.min(4, navigator.hardwareConcurrency || 1)
    : 1;

// @xenova/transformers: load the tokenizer from our bundled local copy only,
// never from the Hugging Face Hub. Keeps the app self-contained and private.
env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = `${BASE}models/`;

let sessionPromise: Promise<ort.InferenceSession> | null = null;
let tokenizerPromise: Promise<PreTrainedTokenizer> | null = null;
let loadResult: LoadResult | null = null;

/** Fetch a binary URL while reporting download progress (0..1). */
async function fetchWithProgress(
  url: string,
  onProgress?: (fraction: number) => void,
): Promise<ArrayBuffer> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch model (${response.status} ${response.statusText})`);
  }

  const total = Number(response.headers.get("Content-Length") ?? 0);
  if (!response.body || total === 0) {
    // No streaming / unknown length: fall back to a single read.
    return response.arrayBuffer();
  }

  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (value) {
      chunks.push(value);
      received += value.length;
      onProgress?.(Math.min(received / total, 1));
    }
  }

  const buffer = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    buffer.set(chunk, offset);
    offset += chunk.length;
  }
  return buffer.buffer;
}

/**
 * Create (once) the tokenizer and the ONNX inference session. Safe to call
 * repeatedly — subsequent calls return the cached session. `onProgress` reports
 * model-download progress in the 0..1 range.
 */
export async function loadModel(
  onProgress?: (fraction: number) => void,
): Promise<LoadResult> {
  if (loadResult && sessionPromise && tokenizerPromise) return loadResult;

  const start = performance.now();

  tokenizerPromise ??= AutoTokenizer.from_pretrained(TOKENIZER_ID);

  sessionPromise ??= (async () => {
    const modelBytes = await fetchWithProgress(MODEL_URL, onProgress);
    const session = await ort.InferenceSession.create(modelBytes, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    loadResult = {
      load_time_ms: performance.now() - start,
      size_mb: modelBytes.byteLength / (1024 * 1024),
    };
    return session;
  })();

  await Promise.all([tokenizerPromise, sessionPromise]);
  onProgress?.(1);
  // loadResult is assigned inside sessionPromise above.
  return loadResult as LoadResult;
}

function toInt64Tensor(data: ArrayLike<number | bigint>, dims: readonly number[]): ort.Tensor {
  const arr = new BigInt64Array(data.length);
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    arr[i] = typeof v === "bigint" ? v : BigInt(v as number);
  }
  return new ort.Tensor("int64", arr, dims as number[]);
}

function softmax2(a: number, b: number): [number, number] {
  const m = Math.max(a, b);
  const ea = Math.exp(a - m);
  const eb = Math.exp(b - m);
  const sum = ea + eb;
  return [ea / sum, eb / sum];
}

/**
 * Classify a single string. Lazily loads the model on first call, then reuses
 * the cached session. Latency is measured around `session.run()` only, matching
 * how the benchmark and Worker report it (excludes tokenization).
 */
export async function classify(text: string): Promise<ClassifyResult> {
  await loadModel();
  const session = await (sessionPromise as Promise<ort.InferenceSession>);
  const tokenizer = await (tokenizerPromise as Promise<PreTrainedTokenizer>);

  const encoded = await tokenizer(text, {
    add_special_tokens: true,
    truncation: true,
    max_length: MAX_SEQ_LENGTH,
  });

  const inputIds = toInt64Tensor(
    encoded.input_ids.data as ArrayLike<number | bigint>,
    encoded.input_ids.dims,
  );
  const attentionMask = toInt64Tensor(
    encoded.attention_mask.data as ArrayLike<number | bigint>,
    encoded.attention_mask.dims,
  );

  const t0 = performance.now();
  const output = await session.run({
    input_ids: inputIds,
    attention_mask: attentionMask,
  });
  const latency_ms = performance.now() - t0;

  const logitsTensor = output["logits"];
  if (!logitsTensor) {
    throw new Error("Model did not return a 'logits' output");
  }
  const logits = logitsTensor.data as Float32Array;
  const negative = logits[0] ?? 0;
  const positive = logits[1] ?? 0;
  const [pNeg, pPos] = softmax2(negative, positive);

  const isPositive = pPos >= pNeg;
  return {
    label: isPositive ? "positive" : "negative",
    confidence: isPositive ? pPos : pNeg,
    latency_ms,
  };
}

/** True once the model + tokenizer are loaded and ready for inference. */
export function isReady(): boolean {
  return loadResult !== null;
}
