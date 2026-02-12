import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Default base for GitHub Pages on this repository.
const base = process.env.VITE_BASE_PATH ?? "/synthetic-huber-wiesel/";

export default defineConfig({
  plugins: [react()],
  base,
});
