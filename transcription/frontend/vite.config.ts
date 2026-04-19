import { defineConfig } from "vite";
import solidPlugin from "vite-plugin-solid";

export default defineConfig({
  plugins: [solidPlugin()],
  server: {
    port: 3001,
    proxy: {
      "/api": {
        target: "http://localhost:7860",
        ws: true,
      },
    },
  },
  build: {
    target: "esnext",
  },
});
