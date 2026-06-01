---
title: EdgeSentiment Inference
emoji: 🛰️
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# EdgeSentiment Inference Service

ONNX inference backend for [EdgeSentiment](https://github.com/HrishiKabra/edge-sentiment) —
runs the INT8-quantized DistilBERT sentiment model via `onnxruntime-node`. The
project's Cloudflare Worker proxies requests here.

**Endpoints**
- `POST /classify` — body `{ "text": "..." }` → `{ label, confidence, latency_ms }`
- `GET /health` — `{ status, model, version }`

This Space is built from a Dockerfile that clones the project repo (including the
model) and starts the service. To pick up new code, trigger a Factory rebuild.
