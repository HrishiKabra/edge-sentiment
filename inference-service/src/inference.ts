/**
 * ONNX inference for the off-loaded EdgeSentiment service.
 *
 * Runs the project's own INT8-quantized DistilBERT at full speed using
 * `onnxruntime-node` — a native ONNX runtime with no WebAssembly/JIT
 * restrictions (unlike onnxruntime-web inside Cloudflare's workerd). The
 * Cloudflare Worker proxies requests here so the "edge" entry point keeps its
 * CORS/routing role while the heavy forward pass happens on a capable runtime.
 *
 * The model + vocab are resolved from the repo (or overridden via MODEL_PATH /
 * VOCAB_PATH) and loaded once; subsequent calls reuse the session.
 */

import { existsSync, readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import * as ort from "onnxruntime-node";
import { WordPieceTokenizer } from "./tokenizer.js";

export type SentimentLabel = "positive" | "negative";

export interface ClassifyResult {
  label: SentimentLabel;
  confidence: number;
  latency_ms: number;
}

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "..", "..");

/** First existing path among candidates, else throw with all candidates listed. */
function resolveAsset(envValue: string | undefined, candidates: string[]): string {
  const tried = envValue ? [envValue, ...candidates] : candidates;
  for (const c of tried) {
    if (existsSync(c)) return c;
  }
  throw new Error(`Asset not found. Tried:\n  ${tried.join("\n  ")}`);
}

const MODEL_PATH = resolveAsset(process.env["MODEL_PATH"], [
  join(repoRoot, "training", "models", "distilbert-sst2-int8.onnx"),
  join(repoRoot, "web", "public", "model", "distilbert-sst2-int8.onnx"),
]);

const VOCAB_PATH = resolveAsset(process.env["VOCAB_PATH"], [
  join(repoRoot, "web", "public", "models", "distilbert-sst2", "vocab.txt"),
  join(repoRoot, "training", "models", "distilbert-sst2-finetuned", "vocab.txt"),
]);

let sessionPromise: Promise<ort.InferenceSession> | null = null;
let tokenizer: WordPieceTokenizer | null = null;

function getTokenizer(): WordPieceTokenizer {
  tokenizer ??= new WordPieceTokenizer(readFileSync(VOCAB_PATH, "utf-8"));
  return tokenizer;
}

function getSession(): Promise<ort.InferenceSession> {
  // The `(x ??= ...)` value is non-null even though the variable type includes
  // null, so this both caches and satisfies the non-null return type.
  return (sessionPromise ??= ort.InferenceSession.create(MODEL_PATH, {
    executionProviders: ["cpu"],
    graphOptimizationLevel: "all",
  }));
}

function softmax2(a: number, b: number): [number, number] {
  const m = Math.max(a, b);
  const ea = Math.exp(a - m);
  const eb = Math.exp(b - m);
  const sum = ea + eb;
  return [ea / sum, eb / sum];
}

/** Warm the model + tokenizer at startup so the first request isn't slow. */
export async function warmup(): Promise<{ model: string; vocab: string }> {
  getTokenizer();
  await getSession();
  return { model: MODEL_PATH, vocab: VOCAB_PATH };
}

export async function classify(text: string): Promise<ClassifyResult> {
  const tok = getTokenizer();
  const session = await getSession();

  const { inputIds, attentionMask } = tok.encode(text);
  const dims = [1, inputIds.length];
  const feeds = {
    input_ids: new ort.Tensor("int64", BigInt64Array.from(inputIds), dims),
    attention_mask: new ort.Tensor("int64", BigInt64Array.from(attentionMask), dims),
  };

  const t0 = performance.now();
  const output = await session.run(feeds);
  const latency_ms = performance.now() - t0;

  const logitsTensor = output["logits"];
  if (!logitsTensor) {
    throw new Error("Model did not return a 'logits' output");
  }
  const logits = logitsTensor.data as Float32Array;
  const [pNeg, pPos] = softmax2(logits[0] ?? 0, logits[1] ?? 0);
  const isPositive = pPos >= pNeg;

  return {
    label: isPositive ? "positive" : "negative",
    confidence: isPositive ? pPos : pNeg,
    latency_ms: Math.round(latency_ms * 100) / 100,
  };
}
