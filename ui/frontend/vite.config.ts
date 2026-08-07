import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The console is served by the Python backend in production, and proxied to it
// during development so both halves see the same API and the same event stream.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: 'dist',
    // The backend already serves the generated map bundles under /assets/, so the
    // build output goes somewhere it cannot collide with them.
    assetsDir: 'app',
    sourcemap: false,
    chunkSizeWarningLimit: 2000,
  },
  server: {
    port: 5173,
    proxy: {
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/assets/previews': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/brand': { target: 'http://127.0.0.1:8000', changeOrigin: true },
    },
  },
})
