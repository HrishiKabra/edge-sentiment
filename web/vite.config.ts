import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { viteStaticCopy } from "vite-plugin-static-copy";

// onnxruntime-web ships its WebAssembly binaries inside node_modules. We copy
// them into the served asset tree under /ort/ so the runtime can fetch them at
// `ort.env.wasm.wasmPaths = "/ort/"` (set in src/inference.ts). Without this the
// browser would 404 on the .wasm files and inference would never initialise.
export default defineConfig({
  // Served from "/" in dev and locally. For GitHub Pages project sites the site
  // lives under /<repo>/, so the deploy workflow sets BASE_PATH=/edge-sentiment/.
  base: process.env["BASE_PATH"] ?? "/",
  plugins: [
    react(),
    viteStaticCopy({
      targets: [
        {
          src: "node_modules/onnxruntime-web/dist/*.wasm",
          dest: "ort",
        },
      ],
    }),
  ],
  // onnxruntime-web must not be pre-bundled/transformed by esbuild — it loads
  // its own wasm glue at runtime and breaks if Vite rewrites it. But
  // @xenova/transformers ships a *nested* copy whose ESM crashes in the dev
  // server unless esbuild pre-bundles it (resolving its internal module graph),
  // so we include exactly that nested copy.
  optimizeDeps: {
    exclude: ["onnxruntime-web"],
    include: ["@xenova/transformers > onnxruntime-web"],
  },
  build: {
    target: "es2021",
    // The INT8 model + wasm are large binaries served as-is; keep them as
    // assets rather than inlining, and don't warn on the expected bundle size.
    chunkSizeWarningLimit: 1500,
  },
  server: {
    port: 5173,
  },
});
