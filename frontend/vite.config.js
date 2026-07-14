// Vite config — dev server + build settings for the dashboard.
//
// The proxy is the piece worth understanding: in dev, the React app runs on
// :5173 (Vite) while the API runs on :8000 (uvicorn). fetch("/v1/ask") from
// the browser would hit :5173 and 404 — the proxy forwards anything under
// /v1 or /health to the API instead. In production there is no proxy and no
// :5173 at all: `npm run build` emits static files into dist/, and FastAPI
// serves them itself from the same origin as the API (see api/main.py).
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/v1": "http://localhost:8000",
      "/health": "http://localhost:8000",
    },
  },
});
