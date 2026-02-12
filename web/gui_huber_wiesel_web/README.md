# gui_huber_wiesel_web

Web rewrite scaffold for the Synthetic Hubel-Wiesel GUI.

## Local dev
1. Install Node.js 20+
2. In this folder:
   - `npm install`
   - `npm run dev`

## Build
- `npm run build`
- Output goes to `dist/`

## GitHub Pages
- The repository includes a workflow at `.github/workflows/deploy-web.yml`.
- On push to `main`, it builds this app and deploys to GitHub Pages.
- Vite base path defaults to `/synthetic-huber-wiesel/`.
- Public URL: `https://antonio-lozano.github.io/synthetic-huber-wiesel/`
- If you get `404`, confirm `Settings -> Pages -> Source = GitHub Actions`.

## Current scope
- Single-neuron interactive mode with neuron ID selection.
- Synthetic V1 kernel generation with RF scaling, optional grayscale, and gray-outside masking.
- Stimulus modes: bar, grating, and CIFAR-like natural patch stream for browser-only runtime.
- Timed ON/OFF presentation mode with per-cycle auto-parameter changes.
- Real response model:
  - normalized cosine-like response + bounded `Hz` mapping
  - legacy unbounded dot-product mode
- Real-time plots:
  - firing-rate traces
  - spike-time lines
  - spike-shape buffer panel with jitter/noise.
