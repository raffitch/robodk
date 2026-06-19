import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server proxies the platform API + job WebSocket to the FastAPI backend,
// so the React app and the Python backend share one origin from the browser's
// point of view (no CORS). Prod build lands in dist/ and FastAPI serves it.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/ws": { target: "ws://127.0.0.1:8000", ws: true },
    },
  },
  build: { outDir: "dist", emptyOutDir: true },
});
