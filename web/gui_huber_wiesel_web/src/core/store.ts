/**
 * Central Zustand store – single source of truth for all simulation state.
 *
 * Replaces the 40+ useState calls that previously lived in App.tsx.
 * Each slice is grouped by domain (neuron, stimulus, dynamics, history, UI).
 */

import { create } from "zustand";
import { subscribeWithSelector } from "zustand/middleware";
import { buildHistory, Hist, WaveBuffer } from "./history";
import { buildActiveNeuronIds } from "./neuronSelection";
import {
  buildActiveKernel,
  buildStimulusPatch,
  centeredKernelPreviewRgba,
  createNeurons,
  generatePseudoCifarDataset,
  makeRng,
  m11TensorToRgba,
  placePatchOnCanvas,
  poissonSpikeStep,
  responseFromFrame,
  sampleNoisyJitteredSpikeWaveform,
  generateSpikeWaveformFromRf,
  StimulusParams,
} from "./sim";
import {
  GratingAutoMode,
  ResponseMode,
  RgbTensor,
  SimNeuron,
  StimulusColor,
  StimulusKind,
} from "./types";
import { NEURON_COLOR_HEX } from "./colors";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

export const CANVAS_SIZE = 512;
export const KERNEL_VIEW_SIZE = 512;
export const HISTORY_SEC = 12;
export const MAX_NEURONS = 10;

// ---------------------------------------------------------------------------
// Preset types
// ---------------------------------------------------------------------------

export type Preset = {
  name: string;
  stimKind: StimulusKind;
  orientationDeg: number;
  barLengthPx: number;
  barThicknessPx: number;
  barSizePx: number;
  barContrast: number;
  stimColorBar: StimulusColor;
  gratingSfCpp: number;
  gratingSizePx: number;
  gratingContrast: number;
  stimColorGrating: StimulusColor;
  phaseDeg: number;
  phaseSpeedDegPerSec: number;
  timedExperiment: boolean;
  onMs: number;
  offMs: number;
  autoBarStepDeg: number;
  autoGrMode: GratingAutoMode;
  autoGrOriStepDeg: number;
  autoGrSfStepCpp: number;
  responseMode: ResponseMode;
  maxRateHz: number;
  rateGain: number;
  baselineHz: number;
  fps: number;
};

export const BUILTIN_PRESETS: Preset[] = [
  {
    name: "Bar sweep (default)",
    stimKind: "bar",
    orientationDeg: 45,
    barLengthPx: 180,
    barThicknessPx: 10,
    barSizePx: 220,
    barContrast: 1.0,
    stimColorBar: "white",
    gratingSfCpp: 0.1,
    gratingSizePx: 220,
    gratingContrast: 1.0,
    stimColorGrating: "white",
    phaseDeg: 0,
    phaseSpeedDegPerSec: 200,
    timedExperiment: true,
    onMs: 500,
    offMs: 500,
    autoBarStepDeg: 20,
    autoGrMode: "orientation",
    autoGrOriStepDeg: 15,
    autoGrSfStepCpp: 0.002,
    responseMode: "normalized",
    maxRateHz: 100,
    rateGain: 5,
    baselineHz: 0,
    fps: 30,
  },
  {
    name: "Grating orientation sweep",
    stimKind: "grating",
    orientationDeg: 0,
    barLengthPx: 180,
    barThicknessPx: 10,
    barSizePx: 220,
    barContrast: 1.0,
    stimColorBar: "white",
    gratingSfCpp: 0.12,
    gratingSizePx: 240,
    gratingContrast: 1.0,
    stimColorGrating: "white",
    phaseDeg: 0,
    phaseSpeedDegPerSec: 180,
    timedExperiment: true,
    onMs: 600,
    offMs: 400,
    autoBarStepDeg: 20,
    autoGrMode: "orientation",
    autoGrOriStepDeg: 15,
    autoGrSfStepCpp: 0.002,
    responseMode: "normalized",
    maxRateHz: 100,
    rateGain: 5,
    baselineHz: 0,
    fps: 30,
  },
  {
    name: "CIFAR random",
    stimKind: "cifar",
    orientationDeg: 45,
    barLengthPx: 180,
    barThicknessPx: 10,
    barSizePx: 220,
    barContrast: 1.0,
    stimColorBar: "white",
    gratingSfCpp: 0.1,
    gratingSizePx: 220,
    gratingContrast: 1.0,
    stimColorGrating: "white",
    phaseDeg: 0,
    phaseSpeedDegPerSec: 200,
    timedExperiment: true,
    onMs: 400,
    offMs: 600,
    autoBarStepDeg: 20,
    autoGrMode: "orientation",
    autoGrOriStepDeg: 15,
    autoGrSfStepCpp: 0.002,
    responseMode: "normalized",
    maxRateHz: 100,
    rateGain: 5,
    baselineHz: 0,
    fps: 30,
  },
  {
    name: "Slow high-gain bar",
    stimKind: "bar",
    orientationDeg: 90,
    barLengthPx: 200,
    barThicknessPx: 14,
    barSizePx: 260,
    barContrast: 1.0,
    stimColorBar: "white",
    gratingSfCpp: 0.1,
    gratingSizePx: 220,
    gratingContrast: 1.0,
    stimColorGrating: "white",
    phaseDeg: 0,
    phaseSpeedDegPerSec: 200,
    timedExperiment: true,
    onMs: 1000,
    offMs: 800,
    autoBarStepDeg: 10,
    autoGrMode: "orientation",
    autoGrOriStepDeg: 15,
    autoGrSfStepCpp: 0.002,
    responseMode: "normalized",
    maxRateHz: 200,
    rateGain: 12,
    baselineHz: 5,
    fps: 30,
  },
];

// ---------------------------------------------------------------------------
// DrawBox
// ---------------------------------------------------------------------------

export type DrawBox = {
  x: number;
  y: number;
  size: number;
  colorHex: string;
};

// ---------------------------------------------------------------------------
// Perf HUD
// ---------------------------------------------------------------------------

export type PerfStats = {
  fps: number;
  stepMs: number;
  droppedFrames: number;
};

// ---------------------------------------------------------------------------
// Store shape
// ---------------------------------------------------------------------------

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

export interface SimStore {
  // ---- Neuron + RF ----
  neurons: SimNeuron[];
  activeNeuronIndex: number;
  simNeurons: number;
  rfScalePct: number;
  maskRf: boolean;
  grayRf: boolean;
  grayEnergy: boolean;
  neuronEnabled: boolean[];          // per-neuron enable toggle (multi-neuron UX)

  // ---- Stimulus ----
  stimKind: StimulusKind;
  stimColorBar: StimulusColor;
  stimColorGrating: StimulusColor;
  centerX: number;
  centerY: number;
  orientationDeg: number;
  phaseDeg: number;
  phaseSpeedDegPerSec: number;
  barLengthPx: number;
  barThicknessPx: number;
  barSizePx: number;
  barContrast: number;
  gratingSfCpp: number;
  gratingSizePx: number;
  gratingContrast: number;
  cifarIndex: number;
  cifarSizePx: number;

  // ---- Timed experiment ----
  timedExperiment: boolean;
  onMs: number;
  offMs: number;
  autoBarStepDeg: number;
  autoGrMode: GratingAutoMode;
  autoGrOriStepDeg: number;
  autoGrSfStepCpp: number;

  // ---- Dynamics ----
  responseMode: ResponseMode;
  maxRateHz: number;
  rateGain: number;
  baselineHz: number;
  rateYMax: number;
  fps: number;
  spikeBuffer: number;
  bufferRefreshSec: number;
  spikeShapeYAbs: number;
  audioEnabled: boolean;
  paused: boolean;

  // ---- CIFAR ----
  cifarDataset: RgbTensor[];
  cifarSource: "real" | "pseudo";

  // ---- Runtime ----
  tNow: number;
  hist: Hist;
  frameRgba: Uint8ClampedArray;
  kernelRgba: Uint8ClampedArray;
  drawBoxes: DrawBox[];

  // ---- Perf HUD ----
  perf: PerfStats;

  // ---- Presets ----
  presets: Preset[];
  customPresets: Preset[];

  // ---- Derived helpers (not state but computed) ----
  activeNeuronIds: () => number[];
  activeNeurons: () => SimNeuron[];
  editNeuron: () => SimNeuron;
  editNeuronScaledSize: () => number;
  editMax: () => number;
  meanRate: () => number;

  // ---- Actions ----
  set: (partial: Partial<SimStore>) => void;
  setNeuronRfPos: (neuronIndex: number, x: number, y: number) => void;
  togglePause: () => void;
  resetTraces: () => void;
  applyPreset: (preset: Preset) => void;
  saveCustomPreset: (name: string) => void;
  deleteCustomPreset: (index: number) => void;
  setCifarDataset: (dataset: RgbTensor[], source: "real" | "pseudo") => void;
  toggleNeuronEnabled: (neuronIndex: number) => void;
  updatePerf: (stats: Partial<PerfStats>) => void;
}

export const useSimStore = create<SimStore>()(subscribeWithSelector((set, get) => ({
  // ---- Neuron + RF ----
  neurons: createNeurons(MAX_NEURONS, CANVAS_SIZE, 2026),
  activeNeuronIndex: 0,
  simNeurons: 1,
  rfScalePct: 100,
  maskRf: true,
  grayRf: false,
  grayEnergy: true,
  neuronEnabled: new Array(MAX_NEURONS).fill(true),

  // ---- Stimulus ----
  stimKind: "bar",
  stimColorBar: "white",
  stimColorGrating: "white",
  centerX: CANVAS_SIZE / 2,
  centerY: CANVAS_SIZE / 2,
  orientationDeg: 45,
  phaseDeg: 0,
  phaseSpeedDegPerSec: 200,
  barLengthPx: 180,
  barThicknessPx: 10,
  barSizePx: 220,
  barContrast: 1.0,
  gratingSfCpp: 0.1,
  gratingSizePx: 220,
  gratingContrast: 1.0,
  cifarIndex: 0,
  cifarSizePx: 140,

  // ---- Timed experiment ----
  timedExperiment: false,
  onMs: 500,
  offMs: 500,
  autoBarStepDeg: 20,
  autoGrMode: "orientation",
  autoGrOriStepDeg: 15,
  autoGrSfStepCpp: 0.002,

  // ---- Dynamics ----
  responseMode: "normalized",
  maxRateHz: 100,
  rateGain: 5,
  baselineHz: 0,
  rateYMax: 120,
  fps: 30,
  spikeBuffer: 3,
  bufferRefreshSec: 5,
  spikeShapeYAbs: 0.9,
  audioEnabled: true,
  paused: false,

  // ---- CIFAR ----
  cifarDataset: generatePseudoCifarDataset(50),
  cifarSource: "pseudo",

  // ---- Runtime ----
  tNow: 0,
  hist: buildHistory(MAX_NEURONS),
  frameRgba: new Uint8ClampedArray(CANVAS_SIZE * CANVAS_SIZE * 4),
  kernelRgba: new Uint8ClampedArray(KERNEL_VIEW_SIZE * KERNEL_VIEW_SIZE * 4),
  drawBoxes: [],

  // ---- Perf HUD ----
  perf: { fps: 0, stepMs: 0, droppedFrames: 0 },

  // ---- Presets ----
  presets: BUILTIN_PRESETS,
  customPresets: (() => {
    try {
      const raw = localStorage.getItem("hw-sim-custom-presets");
      return raw ? (JSON.parse(raw) as Preset[]) : [];
    } catch {
      return [];
    }
  })(),

  // ---- Derived helpers ----
  activeNeuronIds: () => {
    const s = get();
    const base = buildActiveNeuronIds(s.activeNeuronIndex, s.simNeurons, MAX_NEURONS);
    return base.filter((id) => s.neuronEnabled[id]);
  },
  activeNeurons: () => {
    const s = get();
    return s.activeNeuronIds().map((id) => s.neurons[id]);
  },
  editNeuron: () => {
    const s = get();
    return s.neurons[s.activeNeuronIndex];
  },
  editNeuronScaledSize: () => {
    const s = get();
    const n = s.neurons[s.activeNeuronIndex];
    return Math.max(5, Math.round((n.baseRfSize * s.rfScalePct) / 100));
  },
  editMax: () => {
    const s = get();
    return Math.max(0, CANVAS_SIZE - s.editNeuronScaledSize());
  },
  meanRate: () => {
    const s = get();
    const ids = s.activeNeuronIds();
    if (ids.length === 0) return 0;
    const sum = ids.reduce((acc, id) => acc + (s.hist.lastRate[id] ?? 0), 0);
    return sum / ids.length;
  },

  // ---- Actions ----
  set: (partial) => set(partial),

  setNeuronRfPos: (neuronIndex, x, y) =>
    set((s) => ({
      neurons: s.neurons.map((n, i) =>
        i === neuronIndex ? { ...n, rfPos: { x, y } } : n,
      ),
    })),

  togglePause: () => set((s) => ({ paused: !s.paused })),

  resetTraces: () =>
    set({
      tNow: 0,
      hist: buildHistory(MAX_NEURONS),
    }),

  applyPreset: (preset) =>
    set({
      stimKind: preset.stimKind,
      orientationDeg: preset.orientationDeg,
      barLengthPx: preset.barLengthPx,
      barThicknessPx: preset.barThicknessPx,
      barSizePx: preset.barSizePx,
      barContrast: preset.barContrast,
      stimColorBar: preset.stimColorBar,
      gratingSfCpp: preset.gratingSfCpp,
      gratingSizePx: preset.gratingSizePx,
      gratingContrast: preset.gratingContrast,
      stimColorGrating: preset.stimColorGrating,
      phaseDeg: preset.phaseDeg,
      phaseSpeedDegPerSec: preset.phaseSpeedDegPerSec,
      timedExperiment: preset.timedExperiment,
      onMs: preset.onMs,
      offMs: preset.offMs,
      autoBarStepDeg: preset.autoBarStepDeg,
      autoGrMode: preset.autoGrMode,
      autoGrOriStepDeg: preset.autoGrOriStepDeg,
      autoGrSfStepCpp: preset.autoGrSfStepCpp,
      responseMode: preset.responseMode,
      maxRateHz: preset.maxRateHz,
      rateGain: preset.rateGain,
      baselineHz: preset.baselineHz,
      fps: preset.fps,
    }),

  saveCustomPreset: (name) => {
    const s = get();
    const preset: Preset = {
      name,
      stimKind: s.stimKind,
      orientationDeg: s.orientationDeg,
      barLengthPx: s.barLengthPx,
      barThicknessPx: s.barThicknessPx,
      barSizePx: s.barSizePx,
      barContrast: s.barContrast,
      stimColorBar: s.stimColorBar,
      gratingSfCpp: s.gratingSfCpp,
      gratingSizePx: s.gratingSizePx,
      gratingContrast: s.gratingContrast,
      stimColorGrating: s.stimColorGrating,
      phaseDeg: s.phaseDeg,
      phaseSpeedDegPerSec: s.phaseSpeedDegPerSec,
      timedExperiment: s.timedExperiment,
      onMs: s.onMs,
      offMs: s.offMs,
      autoBarStepDeg: s.autoBarStepDeg,
      autoGrMode: s.autoGrMode,
      autoGrOriStepDeg: s.autoGrOriStepDeg,
      autoGrSfStepCpp: s.autoGrSfStepCpp,
      responseMode: s.responseMode,
      maxRateHz: s.maxRateHz,
      rateGain: s.rateGain,
      baselineHz: s.baselineHz,
      fps: s.fps,
    };
    const next = [...s.customPresets, preset];
    try {
      localStorage.setItem("hw-sim-custom-presets", JSON.stringify(next));
    } catch { /* quota exceeded – ignore */ }
    set({ customPresets: next });
  },

  deleteCustomPreset: (index) => {
    const s = get();
    const next = s.customPresets.filter((_, i) => i !== index);
    try {
      localStorage.setItem("hw-sim-custom-presets", JSON.stringify(next));
    } catch { /* ignore */ }
    set({ customPresets: next });
  },

  setCifarDataset: (dataset, source) =>
    set({
      cifarDataset: dataset,
      cifarSource: source,
      cifarIndex: clamp(0, 0, dataset.length - 1),
    }),

  toggleNeuronEnabled: (neuronIndex) =>
    set((s) => ({
      neuronEnabled: s.neuronEnabled.map((v, i) => (i === neuronIndex ? !v : v)),
    })),

  updatePerf: (stats) =>
    set((s) => ({ perf: { ...s.perf, ...stats } })),
})));
