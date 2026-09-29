/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Optional FastAPI base URL, e.g. http://192.168.1.20:8000. Defaults to http://<page host>:8000. */
  readonly VITE_API_BASE_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
