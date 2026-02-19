# NeuroAI for Biomedical Engineering 2026

Class materials and scripts for NeuroAI for Biomedical Engineering (2026).

## Overview

This repository contains educational materials, Python scripts, and interactive tools for learning NeuroAI concepts in biomedical engineering.

## Structure

- `2025/`: Materials from 2025
- `2026/`: Current year scripts and notebooks
  - Includes GUI applications for Hubel-Wiesel model simulations
  - Audio engine for spike rate sonification
- `web/`: Web-based interactive tools
  - `gui_huber_wiesel_web/`: Synthetic Hubel-Wiesel GUI (React/Vite app)
- `papers/`: Related research papers

## Web Application

The Synthetic Hubel-Wiesel GUI is available online at: [https://antonio-lozano.github.io/synthetic-huber-wiesel/](https://antonio-lozano.github.io/synthetic-huber-wiesel/)

For local development:
- Navigate to `web/gui_huber_wiesel_web/`
- Run `npm install` and `npm run dev`

## Installation

Requires Python 3.11. Install dependencies with:
```bash
pip install -e .
```

Optional dependencies:
- `pip install -e .[mlp]` for machine learning packages
- `pip install -e .[gui]` for GUI components

## Usage

See individual folders and files for specific usage instructions.