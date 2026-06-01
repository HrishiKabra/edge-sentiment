# Hugging Face Space (Docker) for the EdgeSentiment inference service.
#
# Self-contained: clones the project at build time (which includes the committed
# 64 MB INT8 model under web/public/model/) and runs the onnxruntime-node service.
# To update after pushing new code to GitHub, trigger a Factory rebuild on the Space.

FROM node:20-slim

# git to clone the repo; libgomp1 is the OpenMP runtime onnxruntime-node needs.
RUN apt-get update \
  && apt-get install -y --no-install-recommends git libgomp1 ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ARG REPO=https://github.com/HrishiKabra/edge-sentiment
RUN git clone --depth 1 "$REPO" .

WORKDIR /app/inference-service
# --include=dev so tsx/typescript install even under NODE_ENV=production.
RUN npm ci --include=dev

# HF Spaces route to app_port (set to 7860 in README.md); the server reads PORT.
ENV PORT=7860
EXPOSE 7860

CMD ["npm", "start"]
