/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Cloudflare Worker base URL for the "Cloudflare Edge" backend. */
  readonly VITE_WORKER_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
