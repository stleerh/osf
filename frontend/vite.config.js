import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
    plugins: [react()],
    server: {
        port: 5173, // Ensure this matches the CORS origin in Flask
        proxy: {
            // proxy requests to Flask server
            '/chat': 'http://localhost:5001',
            '/available_models': 'http://localhost:5001',
            '/clear_session': 'http://localhost:5001',
            '/transcribe': 'http://localhost:5001',
            '/login': 'http://localhost:5001',
            '/logout': 'http://localhost:5001',
            '/submit': 'http://localhost:5001',
            '/check_login': 'http://localhost:5001',
        },
    },
});
