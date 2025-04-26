import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
    plugins: [react()],
    server: {
        port: 5173, // Ensure this matches the CORS origin in Flask
        proxy: {
            '/api': {
                target: 'http://127.0.0.1:5001',
                changeOrigin: true, // Needed for cookie/session handling
            },
        },
    },
    build: {
        outDir: '../dist/www',
    },
});
