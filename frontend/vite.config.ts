import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The API runs separately in development (uvicorn on :8000); proxy /api to it so
// the browser makes same-origin requests and SSE works without CORS quirks.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
  },
});
