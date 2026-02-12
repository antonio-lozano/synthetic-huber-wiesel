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

## Scope of this scaffold
- Multi-neuron RF overlays (up to 5)
- Color-coded firing-rate and spike-time plots
- Basic stimulus controls and animated grating phase
- Designed as phase-1 web foundation; full parity with desktop GUI is next.
