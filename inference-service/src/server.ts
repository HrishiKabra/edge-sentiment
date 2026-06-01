/**
 * EdgeSentiment inference service.
 *
 * A small HTTP service that runs the project's INT8 DistilBERT via
 * onnxruntime-node. The Cloudflare Worker proxies to it (see worker/src/index.ts)
 * so the edge stays the public entry point while inference runs on a runtime
 * that fully supports ONNX. Endpoints mirror the Worker's contract:
 *
 *   POST /classify  { text }   -> { label, confidence, latency_ms }
 *   GET  /health               -> { status, model, version }
 *
 * Uses only the Node standard library (no web framework) to keep the dependency
 * surface minimal.
 */

import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { classify, warmup } from "./inference.js";

const PORT = Number(process.env["PORT"] ?? 8099);
const VERSION = "1.0";

const CORS_HEADERS: Record<string, string> = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
  "Access-Control-Max-Age": "86400",
};

function send(res: ServerResponse, status: number, body: unknown): void {
  const payload = JSON.stringify(body);
  res.writeHead(status, { "Content-Type": "application/json", ...CORS_HEADERS });
  res.end(payload);
}

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    req.on("data", (chunk: Buffer) => {
      size += chunk.length;
      if (size > 1_000_000) {
        reject(new Error("Request body too large"));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf-8")));
    req.on("error", reject);
  });
}

const server = createServer((req, res) => {
  void handle(req, res).catch((err: unknown) => {
    const message = err instanceof Error ? err.message : "Internal error";
    console.error("inference_error", message);
    send(res, 500, { error: message, status: 500 });
  });
});

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  if (req.method === "OPTIONS") {
    res.writeHead(204, CORS_HEADERS);
    res.end();
    return;
  }

  const url = new URL(req.url ?? "/", `http://localhost:${PORT}`);

  if (req.method === "GET" && url.pathname === "/health") {
    send(res, 200, { status: "ok", model: "distilbert-int8", version: VERSION });
    return;
  }

  if (req.method === "POST" && url.pathname === "/classify") {
    const raw = await readBody(req);
    let parsed: { text?: unknown } | null = null;
    try {
      parsed = JSON.parse(raw) as { text?: unknown };
    } catch {
      send(res, 400, { error: "Body must be valid JSON: { text: <string> }" });
      return;
    }
    const text = parsed?.text;
    if (typeof text !== "string" || text.trim() === "") {
      send(res, 400, { error: "Request body must be { text: <non-empty string> }" });
      return;
    }

    const result = await classify(text);
    console.log(
      JSON.stringify({
        event: "classify",
        label: result.label,
        confidence: Number(result.confidence.toFixed(4)),
        latency_ms: result.latency_ms,
        chars: text.length,
      }),
    );
    send(res, 200, result);
    return;
  }

  send(res, 404, { error: `No route for ${req.method} ${url.pathname}` });
}

warmup()
  .then(({ model }) => {
    server.listen(PORT, () => {
      console.log(`EdgeSentiment inference service on :${PORT}`);
      console.log(`  model: ${model}`);
    });
  })
  .catch((err: unknown) => {
    console.error("Failed to load model:", err instanceof Error ? err.message : err);
    process.exit(1);
  });
