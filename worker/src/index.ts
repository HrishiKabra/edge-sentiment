/**
 * EdgeSentiment — Cloudflare Worker edge gateway.
 *
 * The Worker is the public edge entry point: it terminates requests at a
 * Cloudflare POP, handles CORS, validates input, logs latency for observability,
 * and proxies the actual inference to an off-loaded service (onnxruntime-node)
 * that runs the project's INT8 DistilBERT at full speed.
 *
 * Why proxy instead of inferring in-Worker: onnxruntime-web's WebAssembly
 * kernels cannot be compiled/run inside Cloudflare's `workerd` runtime (no
 * runtime Wasm compilation in a request, plus free-tier CPU/memory limits). The
 * model still runs at the edge logically — the Worker is microseconds of
 * overhead in front of a runtime that can actually execute a 64 MB transformer.
 *
 * Routes:
 *   POST /classify  { text }  -> { label, confidence, latency_ms }
 *   GET  /health              -> { status, model, version, upstream }
 */

export interface Env {
  /** Base URL of the off-loaded inference service (worker/.dev.vars or a secret). */
  INFERENCE_URL?: string;
}

type SentimentLabel = "positive" | "negative";

interface ClassifyResult {
  label: SentimentLabel;
  confidence: number;
  latency_ms: number;
}

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

function upstreamBase(env: Env): string {
  return (env.INFERENCE_URL ?? "http://localhost:8099").replace(/\/$/, "");
}

async function proxyClassify(text: string, env: Env): Promise<Response> {
  const t0 = Date.now();
  let upstream: Response;
  try {
    upstream = await fetch(`${upstreamBase(env)}/classify`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "upstream unreachable";
    console.error(JSON.stringify({ event: "upstream_error", message }));
    return json({ error: `Inference service unreachable: ${message}` }, 502);
  }

  const data = (await upstream.json().catch(() => null)) as Partial<ClassifyResult> & {
    error?: string;
  } | null;

  if (!upstream.ok || !data || data.label == null) {
    return json({ error: data?.error ?? `Inference failed (${upstream.status})` }, upstream.status || 502);
  }

  // Observability: log the edge round-trip and the upstream model latency.
  console.log(
    JSON.stringify({
      event: "classify",
      label: data.label,
      confidence: data.confidence,
      model_latency_ms: data.latency_ms,
      edge_latency_ms: Date.now() - t0,
      chars: text.length,
    }),
  );

  return json({
    label: data.label,
    confidence: data.confidence ?? 0,
    latency_ms: data.latency_ms ?? 0,
  } satisfies ClassifyResult);
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: CORS_HEADERS });
    }

    const url = new URL(request.url);

    try {
      if (request.method === "GET" && url.pathname === "/health") {
        // Report our own health plus a best-effort upstream probe.
        let upstream = "unknown";
        try {
          const r = await fetch(`${upstreamBase(env)}/health`);
          upstream = r.ok ? "ok" : `error:${r.status}`;
        } catch {
          upstream = "unreachable";
        }
        return json({ status: "ok", model: "distilbert-int8", version: "1.0", upstream });
      }

      if (request.method === "POST" && url.pathname === "/classify") {
        const body = (await request.json().catch(() => null)) as { text?: unknown } | null;
        const text = body?.text;
        if (typeof text !== "string" || text.trim() === "") {
          return json({ error: "Request body must be { text: <non-empty string> }" }, 400);
        }
        return proxyClassify(text, env);
      }

      return json({ error: `No route for ${request.method} ${url.pathname}` }, 404);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Internal error";
      console.error("gateway_error", message);
      return json({ error: message, status: 500 }, 500);
    }
  },
} satisfies ExportedHandler<Env>;
