/**
 * EdgeSentiment — Cloudflare Worker inference endpoint.
 *
 * Serves the same INT8 DistilBERT sentiment model as the browser app, but from a
 * Cloudflare edge POP instead of the client device. Routes:
 *   POST /classify  { text }            -> { label, confidence, latency_ms }
 *   GET  /health                        -> { status, model, version }
 *
 * Asset strategy (two real platform limits drive this):
 *   - The Worker script is capped at ~10 MB, so the model can't be bundled.
 *   - Workers static assets are capped at 25 MiB per file, and the INT8 model is
 *     ~64 MiB — so it can't live in the assets binding either. Large models
 *     belong in object storage, so the model is fetched from a configurable
 *     MODEL_URL (an R2 bucket or CDN) and cached per-isolate.
 *   - The tokenizer vocab (~230 KB) *does* fit, so it stays in the assets binding.
 *
 * RUNTIME NOTE (honest caveat): executing onnxruntime-web's WebAssembly kernels
 * inside Cloudflare's `workerd` runtime is constrained — workerd restricts
 * runtime Wasm compilation, and a full BERT forward pass (~90 ms CPU, ~64 MB
 * working set) exceeds the free-tier 10 ms CPU / 128 MB limits. This handler is
 * written in the intended production shape; running real inference reliably
 * requires a paid plan with raised limits (and may require providing the ORT
 * Wasm as a statically-bundled module). /health and tokenization work anywhere.
 */

import * as ort from "onnxruntime-web";
import { WordPieceTokenizer } from "./tokenizer";

export interface Env {
  /** Static-assets binding for the tokenizer vocab (configured in wrangler.toml). */
  ASSETS: Fetcher;
  /** URL of the INT8 ONNX model in object storage (R2 / CDN). Set as a var/secret. */
  MODEL_URL?: string;
}

type SentimentLabel = "positive" | "negative";

interface ClassifyResult {
  label: SentimentLabel;
  confidence: number;
  latency_ms: number;
}

const VOCAB_PATH = "/models/distilbert-sst2/vocab.txt";

// onnxruntime-web: single-threaded (Workers have no SharedArrayBuffer), SIMD on.
// wasmPaths points at a CDN matching the pinned version so the runtime can fetch
// the kernels where Wasm fetch/compile is permitted.
ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/";

// Per-isolate caches so the model/tokenizer load at most once per cold start.
let sessionPromise: Promise<ort.InferenceSession> | null = null;
let tokenizerPromise: Promise<WordPieceTokenizer> | null = null;

const CORS_HEADERS: Record<string, string> = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
  "Access-Control-Max-Age": "86400",
};

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", ...CORS_HEADERS },
  });
}

async function loadAsset(env: Env, path: string): Promise<Response> {
  const response = await env.ASSETS.fetch(new Request(`https://assets.local${path}`));
  if (!response.ok) {
    throw new Error(`Asset ${path} unavailable (${response.status})`);
  }
  return response;
}

function getTokenizer(env: Env): Promise<WordPieceTokenizer> {
  tokenizerPromise ??= (async () => {
    const vocabText = await (await loadAsset(env, VOCAB_PATH)).text();
    return new WordPieceTokenizer(vocabText);
  })();
  return tokenizerPromise;
}

// onnxruntime-web's create() is overloaded and its buffer types are defined
// against the DOM ArrayBuffer, which clashes with @cloudflare/workers-types'
// globals. We pin the Uint8Array overload via a single-signature view of the
// function so the model bytes type-check cleanly (no `any`).
const createSession = ort.InferenceSession.create as (
  buffer: Uint8Array,
  options?: ort.InferenceSession.SessionOptions,
) => Promise<ort.InferenceSession>;

function getSession(env: Env): Promise<ort.InferenceSession> {
  sessionPromise ??= (async () => {
    if (!env.MODEL_URL) {
      throw new Error(
        "MODEL_URL is not configured. Host distilbert-sst2-int8.onnx in R2/CDN " +
          "and set MODEL_URL (see wrangler.toml).",
      );
    }
    const response = await fetch(env.MODEL_URL);
    if (!response.ok) {
      throw new Error(`Model fetch failed (${response.status}) from ${env.MODEL_URL}`);
    }
    const bytes = (await response.arrayBuffer()) as unknown as ArrayBufferLike;
    return createSession(new Uint8Array(bytes), {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
  })();
  return sessionPromise;
}

function softmax2(a: number, b: number): [number, number] {
  const m = Math.max(a, b);
  const ea = Math.exp(a - m);
  const eb = Math.exp(b - m);
  const sum = ea + eb;
  return [ea / sum, eb / sum];
}

async function classify(text: string, env: Env): Promise<ClassifyResult> {
  const [tokenizer, session] = await Promise.all([getTokenizer(env), getSession(env)]);

  const { inputIds, attentionMask } = tokenizer.encode(text);
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

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    // CORS preflight.
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: CORS_HEADERS });
    }

    const url = new URL(request.url);

    try {
      if (request.method === "GET" && url.pathname === "/health") {
        return json({ status: "ok", model: "distilbert-int8", version: "1.0" });
      }

      if (request.method === "POST" && url.pathname === "/classify") {
        const body = (await request.json().catch(() => null)) as { text?: unknown } | null;
        const text = body?.text;
        if (typeof text !== "string" || text.trim() === "") {
          return json({ error: "Request body must be { text: <non-empty string> }" }, 400);
        }

        const result = await classify(text, env);
        // Structured log line for Workers observability (wrangler tail / dashboard).
        console.log(
          JSON.stringify({
            event: "classify",
            label: result.label,
            confidence: Number(result.confidence.toFixed(4)),
            latency_ms: result.latency_ms,
            chars: text.length,
          }),
        );
        return json(result);
      }

      return json({ error: `No route for ${request.method} ${url.pathname}` }, 404);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Internal error";
      console.error("inference_error", message);
      return json({ error: message, status: 500 }, 500);
    }
  },
} satisfies ExportedHandler<Env>;
