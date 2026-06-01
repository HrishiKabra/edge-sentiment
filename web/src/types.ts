/** Shared UI types for the EdgeSentiment web app. */

export type Backend = "wasm" | "edge";

export const BACKEND_LABELS: Record<Backend, string> = {
  wasm: "Browser (Wasm)",
  edge: "Cloudflare Edge",
};

export type { ClassifyResult, SentimentLabel } from "./inference";
