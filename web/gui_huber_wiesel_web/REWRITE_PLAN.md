# Synthetic Hubel-Wiesel Web Rewrite Plan

This folder is for a browser-first rewrite that can be hosted on GitHub Pages.

## Goal
- Rebuild the current desktop GUI (`2026/gui_hubel_wiesel.py`) as a web app.
- Keep core behavior aligned with `2026/computing.py` concepts.
- Run fully client-side in browser (no server required for core demo).

## Target Stack
- UI: React + TypeScript + Vite
- Plotting/2D: Canvas + lightweight custom drawing (or Plotly for charts)
- Audio: WebAudio API
- Deployment: GitHub Pages (static site)

## Folder Shape
- `web/gui_huber_wiesel_web/`
- `web/gui_huber_wiesel_web/src/core/` (math and simulation logic)
- `web/gui_huber_wiesel_web/src/ui/` (controls and panels)
- `web/gui_huber_wiesel_web/src/plots/` (rate/spike/waveform renderers)
- `web/gui_huber_wiesel_web/src/audio/` (spike/rate sound engine)

## Feature Mapping (Desktop -> Web)
1. RF generation and resizing
- Port kernel generation, masking, grayscale, energy-match grayscale logic.
- Keep value ranges in `[-1, 1]`.

2. Multi-neuron simulation
- Keep up to 5 simultaneous neurons with independent RF positions.
- Preserve neuron color order:
  - white, blue, red, green, yellow

3. Stimulus controls
- Bar: orientation, thickness, size, color
- Grating: orientation, spatial frequency, phase, color
- CIFAR patch: index/random in timed mode
- Timed experiment ON/OFF windows

4. Response/rate model
- Default normalized mode:
  - cosine-like normalized response
  - bounded mapping `0..Max Hz`
- Keep legacy mode optional for teaching comparison.

5. Plots
- Stimulus + RF overlays (multiple RF boxes)
- Firing rate traces per neuron
- Spike-time lines per neuron
- Spike-shape overlay buffer per neuron

6. Audio
- Browser audio triggered by spikes and rate, analogous to current `sounds.py`.

## Implementation Phases
1. Phase 1: Core math parity
- Build RF + stimulus + response modules in TS.
- Unit-test against reference outputs from Python for fixed seeds.

2. Phase 2: Basic UI and plotting
- Controls panel + canvas + rate/spike charts.
- Single-neuron end-to-end loop first.

3. Phase 3: Multi-neuron + per-neuron RF positioning
- Add up to 5 neurons with independent RF coordinates.
- Color-coded traces and overlays.

4. Phase 4: Audio + polish
- Add spike/rate sound synthesis with user toggle.
- Improve performance and mobile behavior.

5. Phase 5: GitHub Pages deployment
- Vite base path config for repo pages.
- CI workflow to build and deploy to `gh-pages`.

## Definition of Done
- Fully usable in browser without backend.
- Core controls and outputs match desktop behavior conceptually.
- Stable 60 FPS interaction on typical laptop for up to 5 neurons.
- Public GitHub Pages URL with README usage instructions.

## Notes
- Keep desktop Python GUI as research/dev reference during rewrite.
- Do not depend on paper parsing or paper content in web implementation.
