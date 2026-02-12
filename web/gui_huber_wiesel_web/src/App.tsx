import { useEffect, useMemo, useRef, useState } from "react";
import { NEURON_COLOR_HEX } from "./core/colors";
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
  timedCycle,
  generateSpikeWaveformFromRf,
} from "./core/sim";
import { GratingAutoMode, ResponseMode, SimNeuron, StimulusColor, StimulusKind } from "./core/types";

const CANVAS_SIZE = 512;
const KERNEL_VIEW_SIZE = 512;
const HISTORY_SEC = 12;
const MAX_NEURONS = 10;

type WaveBuffer = {
  tMs: number[];
  amp: number[];
};

type Hist = {
  t: number[];
  rates: number[][];
  spikes: number[][];
  waves: WaveBuffer[][];
  lastRaw: number[];
  lastEff: number[];
  lastRate: number[];
};

type DrawBox = {
  x: number;
  y: number;
  size: number;
  colorHex: string;
};

type PlotMargins = {
  left: number;
  right: number;
  top: number;
  bottom: number;
};

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

function buildHistory(n: number): Hist {
  return {
    t: [],
    rates: Array.from({ length: n }, () => []),
    spikes: Array.from({ length: n }, () => []),
    waves: Array.from({ length: n }, () => []),
    lastRaw: new Array(n).fill(0),
    lastEff: new Array(n).fill(0),
    lastRate: new Array(n).fill(0),
  };
}

function polylinePoints(
  xs: number[],
  ys: number[],
  width: number,
  height: number,
  yMax: number,
  margins: PlotMargins = { left: 0, right: 0, top: 0, bottom: 0 },
  xStart?: number,
  xEnd?: number,
): string {
  if (xs.length === 0 || ys.length === 0) return "";
  const t0 = xStart ?? xs[0];
  const t1 = xEnd ?? xs[xs.length - 1];
  const dt = Math.max(1e-6, t1 - t0);
  const innerW = Math.max(1, width - margins.left - margins.right);
  const innerH = Math.max(1, height - margins.top - margins.bottom);
  return xs
    .map((t, i) => {
      const x = margins.left + ((t - t0) / dt) * innerW;
      const y = margins.top + (1 - clamp(ys[i], 0, yMax) / Math.max(1e-6, yMax)) * innerH;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
}

function spikeLines(
  spikeTimes: number[],
  tStart: number,
  tEnd: number,
  width: number,
  _height: number,
  margins: PlotMargins = { left: 0, right: 0, top: 0, bottom: 0 },
): Array<{ x: number; y0: number; y1: number }> {
  const innerW = Math.max(1, width - margins.left - margins.right);
  const dt = Math.max(1e-6, tEnd - tStart);
  return spikeTimes
    .filter((t) => t >= tStart && t <= tEnd)
    .map((t) => ({
      x: margins.left + ((t - tStart) / dt) * innerW,
      y0: margins.top,
      y1: _height - margins.bottom,
    }));
}

function waveformPolyline(
  xMs: number[],
  y: number[],
  width: number,
  height: number,
  xMinMs = -0.5,
  xMaxMs = 2.0,
  yAbs = 1.6,
): string {
  if (xMs.length === 0 || y.length === 0) return "";
  const dx = Math.max(1e-6, xMaxMs - xMinMs);
  return xMs
    .map((v, i) => {
      const x = ((v - xMinMs) / dx) * width;
      const yy = height - ((clamp(y[i], -yAbs, yAbs) + yAbs) / (2 * yAbs)) * height;
      return `${x.toFixed(2)},${yy.toFixed(2)}`;
    })
    .join(" ");
}

export default function App() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const kernelCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const rngRef = useRef(makeRng(2026));
  const timeRef = useRef(0);
  const autoRef = useRef({ lastCycleIndex: -1, cifarIndex: 0 });
  const audioCtxRef = useRef<AudioContext | null>(null);
  const humOscRef = useRef<OscillatorNode | null>(null);
  const humGainRef = useRef<GainNode | null>(null);

  const cifarDataset = useMemo(() => generatePseudoCifarDataset(300), []);

  const [neurons, setNeurons] = useState<SimNeuron[]>(() => createNeurons(MAX_NEURONS, CANVAS_SIZE, 2026));
  const [activeNeuronIndex, setActiveNeuronIndex] = useState(0);
  const [rfScalePct, setRfScalePct] = useState(100);
  const [maskRf, setMaskRf] = useState(true);
  const [grayRf, setGrayRf] = useState(false);
  const [grayEnergy, setGrayEnergy] = useState(true);

  const [stimKind, setStimKind] = useState<StimulusKind>("bar");
  const [stimColorBar, setStimColorBar] = useState<StimulusColor>("white");
  const [stimColorGrating, setStimColorGrating] = useState<StimulusColor>("white");
  const [centerX, setCenterX] = useState(CANVAS_SIZE / 2);
  const [centerY, setCenterY] = useState(CANVAS_SIZE / 2);
  const [orientationDeg, setOrientationDeg] = useState(45);
  const [phaseDeg, setPhaseDeg] = useState(0);
  const [phaseSpeedDegPerSec, setPhaseSpeedDegPerSec] = useState(200);
  const [barLengthPx, setBarLengthPx] = useState(180);
  const [barThicknessPx, setBarThicknessPx] = useState(10);
  const [barSizePx, setBarSizePx] = useState(220);
  const [barContrast, setBarContrast] = useState(1.0);
  const [gratingSfCpp, setGratingSfCpp] = useState(0.1);
  const [gratingSizePx, setGratingSizePx] = useState(220);
  const [gratingContrast, setGratingContrast] = useState(1.0);
  const [cifarIndex, setCifarIndex] = useState(0);
  const [cifarSizePx, setCifarSizePx] = useState(140);

  const [timedExperiment, setTimedExperiment] = useState(false);
  const [onMs, setOnMs] = useState(500);
  const [offMs, setOffMs] = useState(500);
  const [autoBarStepDeg, setAutoBarStepDeg] = useState(20);
  const [autoGrMode, setAutoGrMode] = useState<GratingAutoMode>("orientation");
  const [autoGrOriStepDeg, setAutoGrOriStepDeg] = useState(15);
  const [autoGrSfStepCpp, setAutoGrSfStepCpp] = useState(0.002);

  const [responseMode, setResponseMode] = useState<ResponseMode>("normalized");
  const [maxRateHz, setMaxRateHz] = useState(100);
  const [rateGain, setRateGain] = useState(5);
  const [baselineHz, setBaselineHz] = useState(0);
  const [rateYMax, setRateYMax] = useState(120);
  const [fps, setFps] = useState(30);
  const [spikeBuffer, setSpikeBuffer] = useState(10);
  const [spikeShapeYAbs, setSpikeShapeYAbs] = useState(0.9);
  const [audioEnabled, setAudioEnabled] = useState(false);
  const [paused, setPaused] = useState(false);

  const [tNow, setTNow] = useState(0);
  const [hist, setHist] = useState<Hist>(() => buildHistory(MAX_NEURONS));
  const [frameRgba, setFrameRgba] = useState<Uint8ClampedArray>(() => new Uint8ClampedArray(CANVAS_SIZE * CANVAS_SIZE * 4));
  const [kernelRgba, setKernelRgba] = useState<Uint8ClampedArray>(() => new Uint8ClampedArray(KERNEL_VIEW_SIZE * KERNEL_VIEW_SIZE * 4));
  const [drawBoxes, setDrawBoxes] = useState<DrawBox[]>([]);

  const activeNeurons = useMemo(() => [neurons[activeNeuronIndex]], [neurons, activeNeuronIndex]);
  const editNeuron = activeNeurons[0];
  const editNeuronScaledSize = Math.max(
    5,
    Math.round(((editNeuron?.baseRfSize ?? neurons[0].baseRfSize) * rfScalePct) / 100),
  );
  const editMax = Math.max(0, CANVAS_SIZE - editNeuronScaledSize);
  const meanRate = useMemo(() => {
    return Number(hist.lastRate[0] ?? 0);
  }, [hist.lastRate]);
  const observedRateMax = useMemo(() => {
    if (hist.rates[0].length === 0) return 1;
    return Math.max(1, ...hist.rates[0]);
  }, [hist.rates]);
  const ratePlotYMax = Math.max(1, Math.min(rateYMax, Math.max(5, observedRateMax * 1.25)));
  const rateSvgW = 1000;
  const rateSvgH = 220;
  const rateMargins: PlotMargins = { left: 70, right: 14, top: 12, bottom: 34 };
  const rateInnerW = rateSvgW - rateMargins.left - rateMargins.right;
  const rateInnerH = rateSvgH - rateMargins.top - rateMargins.bottom;
  const traceTStart = hist.t.length > 0 ? hist.t[0] : Math.max(0, tNow - HISTORY_SEC);
  const traceTEnd = hist.t.length > 1 ? hist.t[hist.t.length - 1] : traceTStart + 1.0 / Math.max(1, fps);

  function stopAudio(): void {
    try {
      humOscRef.current?.stop();
    } catch {
      // no-op
    }
    humOscRef.current = null;
    humGainRef.current = null;
    if (audioCtxRef.current) {
      void audioCtxRef.current.close();
    }
    audioCtxRef.current = null;
  }

  useEffect(() => {
    setNeurons((prev) =>
      prev.map((n) => {
        const s = Math.max(5, Math.round((n.baseRfSize * rfScalePct) / 100));
        return {
          ...n,
          rfPos: {
            x: clamp(n.rfPos.x, 0, Math.max(0, CANVAS_SIZE - s)),
            y: clamp(n.rfPos.y, 0, Math.max(0, CANVAS_SIZE - s)),
          },
        };
      }),
    );
  }, [rfScalePct]);

  useEffect(() => {
    if (!audioEnabled) {
      stopAudio();
      return;
    }

    if (!audioCtxRef.current) {
      const ctx = new AudioContext();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "sine";
      osc.frequency.value = 120;
      gain.gain.value = 0.0001;
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start();
      audioCtxRef.current = ctx;
      humOscRef.current = osc;
      humGainRef.current = gain;
    }

    return () => stopAudio();
  }, [audioEnabled]);

  function computeFrameAndResponse(tSec: number, sampleSpikes: boolean): {
    frame: Uint8ClampedArray;
    kernelPreview: Uint8ClampedArray;
    draw: DrawBox[];
    rates: number[];
    raw: number[];
    effective: number[];
    spikes: number[];
    waves: Array<WaveBuffer | null>;
  } {
    const cycle = timedCycle(tSec, onMs, offMs);
    if (timedExperiment && stimKind === "cifar" && cycle.visible && cycle.cycleIndex !== autoRef.current.lastCycleIndex) {
      autoRef.current.lastCycleIndex = cycle.cycleIndex;
      autoRef.current.cifarIndex = Math.floor(rngRef.current() * cifarDataset.length);
    }
    if (!timedExperiment) {
      autoRef.current.lastCycleIndex = -1;
    }

    const stim = buildStimulusPatch(
      {
        kind: stimKind,
        center: { x: centerX, y: centerY },
        orientationDeg,
        phaseDeg,
        phaseSpeedDegPerSec,
        barLengthPx,
        barThicknessPx,
        barSizePx,
        barContrast,
        barColor: stimColorBar,
        gratingSfCpp,
        gratingSizePx,
        gratingContrast,
        gratingColor: stimColorGrating,
        cifarIndex,
        cifarSizePx,
        timedExperiment,
        onMs,
        offMs,
        autoBarStepDeg,
        autoGrMode,
        autoGrOriStepDeg,
        autoGrSfStepCpp,
      },
      tSec,
      autoRef.current.cifarIndex,
      cifarDataset,
    );

    const frameM11 = placePatchOnCanvas(CANVAS_SIZE, stim.patch, { x: centerX, y: centerY }, stim.visible);
    const frame = m11TensorToRgba(frameM11);

    const rates: number[] = new Array(MAX_NEURONS).fill(0);
    const raw: number[] = new Array(MAX_NEURONS).fill(0);
    const effective: number[] = new Array(MAX_NEURONS).fill(0);
    const spikes: number[] = new Array(MAX_NEURONS).fill(0);
    const waves: Array<WaveBuffer | null> = new Array(MAX_NEURONS).fill(null);
    const draw: DrawBox[] = [];
    let firstKernelPreview = kernelRgba;

    const n = neurons[activeNeuronIndex];
    const kernel = buildActiveKernel(n.kernelBase, {
      scalePct: rfScalePct,
      useMask: maskRf,
      useGray: grayRf,
      grayMatchEnergy: grayEnergy,
    });
    const s = kernel.width;
    const x = clamp(n.rfPos.x, 0, Math.max(0, CANVAS_SIZE - s));
    const y = clamp(n.rfPos.y, 0, Math.max(0, CANVAS_SIZE - s));
    draw.push({ x, y, size: s, colorHex: NEURON_COLOR_HEX[n.color] });

    const out = responseFromFrame(frameM11, kernel, { x, y }, responseMode, rateGain, baselineHz, maxRateHz);
    raw[0] = out.raw;
    effective[0] = out.effective;
    rates[0] = out.rateHz;
    firstKernelPreview = centeredKernelPreviewRgba(kernel, KERNEL_VIEW_SIZE, 5);

    if (sampleSpikes) {
      const spike = poissonSpikeStep(out.rateHz, 1.0 / Math.max(1, fps), rngRef.current);
      spikes[0] = spike;
      if (spike === 1) {
        const base = generateSpikeWaveformFromRf(kernel, 1.5, 20000);
        const sample = sampleNoisyJitteredSpikeWaveform(base, rngRef.current, 0.04, 0.035, 0.08);
        waves[0] = {
          tMs: Array.from(sample.tMs),
          amp: Array.from(sample.amp),
        };
      }
    }

    return {
      frame,
      kernelPreview: firstKernelPreview,
      draw,
      rates,
      raw,
      effective,
      spikes,
      waves,
    };
  }

  useEffect(() => {
    const preview = computeFrameAndResponse(timeRef.current, false);
    setFrameRgba(preview.frame);
    setKernelRgba(preview.kernelPreview);
    setDrawBoxes(preview.draw);
    setHist((prev) => ({
      ...prev,
      lastRate: preview.rates.slice(),
      lastRaw: preview.raw.slice(),
      lastEff: preview.effective.slice(),
    }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    neurons,
    activeNeuronIndex,
    rfScalePct,
    maskRf,
    grayRf,
    grayEnergy,
    stimKind,
    stimColorBar,
    stimColorGrating,
    centerX,
    centerY,
    orientationDeg,
    phaseDeg,
    phaseSpeedDegPerSec,
    barLengthPx,
    barThicknessPx,
    barSizePx,
    barContrast,
    gratingSfCpp,
    gratingSizePx,
    gratingContrast,
    cifarIndex,
    cifarSizePx,
    timedExperiment,
    onMs,
    offMs,
    autoBarStepDeg,
    autoGrMode,
    autoGrOriStepDeg,
    autoGrSfStepCpp,
    responseMode,
    maxRateHz,
    rateGain,
    baselineHz,
    fps,
    rateYMax,
  ]);

  useEffect(() => {
    if (paused) return;
    const dtSec = 1.0 / Math.max(1, fps);
    const id = window.setInterval(() => {
      const nextTime = timeRef.current + dtSec;
      timeRef.current = nextTime;
      setTNow(nextTime);

      const step = computeFrameAndResponse(nextTime, true);
      setFrameRgba(step.frame);
      setKernelRgba(step.kernelPreview);
      setDrawBoxes(step.draw);

      setHist((prev) => {
        const t = [...prev.t, nextTime];
        const rates = prev.rates.map((arr) => [...arr]);
        const spikes = prev.spikes.map((arr) => [...arr]);
        const waves = prev.waves.map((arr) => [...arr]);

        for (let i = 0; i < MAX_NEURONS; i += 1) {
          const isActive = i === 0;
          const rate = isActive ? step.rates[i] : 0;
          rates[i].push(rate);
          if (isActive && step.spikes[i] === 1) {
            spikes[i].push(nextTime);
            if (step.waves[i]) {
              waves[i].push(step.waves[i] as WaveBuffer);
              if (waves[i].length > spikeBuffer) {
                waves[i] = waves[i].slice(waves[i].length - spikeBuffer);
              }
            }
          }
        }

        const tMin = nextTime - HISTORY_SEC;
        let start = 0;
        for (let i = 0; i < t.length; i += 1) {
          if (t[i] >= tMin) {
            start = i;
            break;
          }
        }
        const trimT = t.slice(start);
        const trimRates = rates.map((arr) => arr.slice(start));
        const trimSpikes = spikes.map((arr) => arr.filter((tt) => tt >= tMin));
        return {
          t: trimT,
          rates: trimRates,
          spikes: trimSpikes,
          waves,
          lastRate: step.rates.slice(),
          lastRaw: step.raw.slice(),
          lastEff: step.effective.slice(),
        };
      });

      if (audioEnabled && audioCtxRef.current && humOscRef.current && humGainRef.current) {
        const ctx = audioCtxRef.current;
        if (ctx.state === "suspended") {
          void ctx.resume();
        }
        const tAudio = ctx.currentTime;
        const r = Number(step.rates[0] ?? 0);
        humOscRef.current.frequency.setTargetAtTime(100 + r * 3.5, tAudio, 0.05);
        humGainRef.current.gain.setTargetAtTime(0.0001 + Math.min(0.02, r / 4000), tAudio, 0.08);
        if (step.spikes[0] === 1) {
          const clickOsc = ctx.createOscillator();
          const clickGain = ctx.createGain();
          clickOsc.type = "triangle";
          clickOsc.frequency.setValueAtTime(1100, tAudio);
          clickOsc.frequency.exponentialRampToValueAtTime(650, tAudio + 0.015);
          clickGain.gain.setValueAtTime(0.0001, tAudio);
          clickGain.gain.exponentialRampToValueAtTime(0.018, tAudio + 0.002);
          clickGain.gain.exponentialRampToValueAtTime(0.0001, tAudio + 0.02);
          clickOsc.connect(clickGain);
          clickGain.connect(ctx.destination);
          clickOsc.start(tAudio);
          clickOsc.stop(tAudio + 0.03);
        }
      }
    }, Math.round(1000.0 * dtSec));
    return () => window.clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    paused,
    fps,
    neurons,
    activeNeuronIndex,
    rfScalePct,
    maskRf,
    grayRf,
    grayEnergy,
    stimKind,
    stimColorBar,
    stimColorGrating,
    centerX,
    centerY,
    orientationDeg,
    phaseDeg,
    phaseSpeedDegPerSec,
    barLengthPx,
    barThicknessPx,
    barSizePx,
    barContrast,
    gratingSfCpp,
    gratingSizePx,
    gratingContrast,
    cifarIndex,
    cifarSizePx,
    timedExperiment,
    onMs,
    offMs,
    autoBarStepDeg,
    autoGrMode,
    autoGrOriStepDeg,
    autoGrSfStepCpp,
    responseMode,
    maxRateHz,
    rateGain,
    baselineHz,
    spikeBuffer,
    audioEnabled,
  ]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const img = ctx.createImageData(CANVAS_SIZE, CANVAS_SIZE);
    img.data.set(frameRgba);
    ctx.putImageData(img, 0, 0);

    drawBoxes.forEach((b) => {
      ctx.strokeStyle = b.colorHex;
      ctx.lineWidth = 2;
      ctx.strokeRect(b.x + 0.5, b.y + 0.5, b.size, b.size);
    });
  }, [frameRgba, drawBoxes]);

  useEffect(() => {
    const canvas = kernelCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const img = ctx.createImageData(KERNEL_VIEW_SIZE, KERNEL_VIEW_SIZE);
    img.data.set(kernelRgba);
    ctx.putImageData(img, 0, 0);
  }, [kernelRgba]);

  return (
    <div className="app">
      <aside className="left">
        <h1>Synthetic Hubel-Wiesel Web</h1>
        <p className="muted">Browser parity pass: real RF convolution, bounded rates, Poisson spikes, waveform buffer.</p>

        <section className="card">
          <h2>Neuron + RF</h2>
          <label>
            Neuron ID
            <select value={activeNeuronIndex} onChange={(e) => setActiveNeuronIndex(Number(e.target.value))}>
              {Array.from({ length: MAX_NEURONS }).map((_, i) => (
                <option key={i} value={i}>
                  N{i + 1}
                </option>
              ))}
            </select>
          </label>
          <label>
            RF X
            <input
              type="range"
              min={0}
              max={editMax}
              value={clamp(editNeuron?.rfPos.x ?? 0, 0, editMax)}
              onChange={(e) => {
                const v = Number(e.target.value);
                setNeurons((prev) =>
                  prev.map((n, i) => (i === activeNeuronIndex ? { ...n, rfPos: { ...n.rfPos, x: v } } : n)),
                );
              }}
            />
          </label>
          <label>
            RF Y
            <input
              type="range"
              min={0}
              max={editMax}
              value={clamp(editNeuron?.rfPos.y ?? 0, 0, editMax)}
              onChange={(e) => {
                const v = Number(e.target.value);
                setNeurons((prev) =>
                  prev.map((n, i) => (i === activeNeuronIndex ? { ...n, rfPos: { ...n.rfPos, y: v } } : n)),
                );
              }}
            />
          </label>
          <label>
            RF scale (%): {rfScalePct}
            <input type="range" min={50} max={400} value={rfScalePct} onChange={(e) => setRfScalePct(Number(e.target.value))} />
          </label>
          <label className="row-check">
            <input type="checkbox" checked={maskRf} onChange={(e) => setMaskRf(e.target.checked)} />
            Mask RF to gray outside
          </label>
          <label className="row-check">
            <input type="checkbox" checked={grayRf} onChange={(e) => setGrayRf(e.target.checked)} />
            Full RF grayscale
          </label>
          <label className="row-check">
            <input type="checkbox" checked={grayEnergy} onChange={(e) => setGrayEnergy(e.target.checked)} />
            Grayscale energy match
          </label>
        </section>

        <section className="card">
          <h2>Stimulus</h2>
          <label>
            Type
            <select value={stimKind} onChange={(e) => setStimKind(e.target.value as StimulusKind)}>
              <option value="bar">bar</option>
              <option value="grating">grating</option>
              <option value="cifar">cifar-like</option>
            </select>
          </label>
          <label>
            Stim X
            <input type="range" min={0} max={CANVAS_SIZE} value={centerX} onChange={(e) => setCenterX(Number(e.target.value))} />
          </label>
          <label>
            Stim Y
            <input type="range" min={0} max={CANVAS_SIZE} value={centerY} onChange={(e) => setCenterY(Number(e.target.value))} />
          </label>
          <label>
            Orientation (deg): {orientationDeg}
            <input type="range" min={0} max={179} value={orientationDeg} onChange={(e) => setOrientationDeg(Number(e.target.value))} />
          </label>
          <label>
            Bar color
            <select value={stimColorBar} onChange={(e) => setStimColorBar(e.target.value as StimulusColor)}>
              <option value="white">white</option>
              <option value="red">red</option>
              <option value="green">green</option>
              <option value="blue">blue</option>
              <option value="yellow">yellow</option>
              <option value="cyan">cyan</option>
              <option value="magenta">magenta</option>
            </select>
          </label>
          <label>
            Bar length: {barLengthPx}
            <input type="range" min={10} max={300} value={barLengthPx} onChange={(e) => setBarLengthPx(Number(e.target.value))} />
          </label>
          <label>
            Bar thickness: {barThicknessPx}
            <input type="range" min={1} max={60} value={barThicknessPx} onChange={(e) => setBarThicknessPx(Number(e.target.value))} />
          </label>
          <label>
            Bar patch size: {barSizePx}
            <input type="range" min={40} max={300} value={barSizePx} onChange={(e) => setBarSizePx(Number(e.target.value))} />
          </label>
          <label>
            Bar contrast: {barContrast.toFixed(2)}
            <input type="range" min={0.1} max={1} step={0.01} value={barContrast} onChange={(e) => setBarContrast(Number(e.target.value))} />
          </label>

          <label>
            Grating color
            <select value={stimColorGrating} onChange={(e) => setStimColorGrating(e.target.value as StimulusColor)}>
              <option value="white">white</option>
              <option value="red">red</option>
              <option value="green">green</option>
              <option value="blue">blue</option>
              <option value="yellow">yellow</option>
              <option value="cyan">cyan</option>
              <option value="magenta">magenta</option>
            </select>
          </label>
          <label>
            Grating sf (cycles/pixel): {gratingSfCpp.toFixed(3)}
            <input
              type="range"
              min={0.01}
              max={0.45}
              step={0.001}
              value={gratingSfCpp}
              onChange={(e) => setGratingSfCpp(Number(e.target.value))}
            />
          </label>
          <label>
            Grating phase (deg): {phaseDeg.toFixed(0)}
            <input type="range" min={0} max={360} value={phaseDeg} onChange={(e) => setPhaseDeg(Number(e.target.value))} />
          </label>
          <label>
            Phase speed (deg/s): {phaseSpeedDegPerSec}
            <input type="range" min={0} max={500} value={phaseSpeedDegPerSec} onChange={(e) => setPhaseSpeedDegPerSec(Number(e.target.value))} />
          </label>
          <label>
            Grating size: {gratingSizePx}
            <input type="range" min={40} max={300} value={gratingSizePx} onChange={(e) => setGratingSizePx(Number(e.target.value))} />
          </label>
          <label>
            Grating contrast: {gratingContrast.toFixed(2)}
            <input
              type="range"
              min={0.1}
              max={1}
              step={0.01}
              value={gratingContrast}
              onChange={(e) => setGratingContrast(Number(e.target.value))}
            />
          </label>

          <label>
            Cifar-like index: {cifarIndex}
            <input
              type="range"
              min={0}
              max={cifarDataset.length - 1}
              value={cifarIndex}
              onChange={(e) => setCifarIndex(Number(e.target.value))}
            />
          </label>
          <label>
            Cifar-like size: {cifarSizePx}
            <input type="range" min={32} max={260} value={cifarSizePx} onChange={(e) => setCifarSizePx(Number(e.target.value))} />
          </label>
        </section>

        <section className="card">
          <h2>Timed Experiment</h2>
          <label className="row-check">
            <input type="checkbox" checked={timedExperiment} onChange={(e) => setTimedExperiment(e.target.checked)} />
            Enable ON/OFF presentation
          </label>
          <label>
            ON (ms): {onMs}
            <input type="range" min={10} max={2000} step={10} value={onMs} onChange={(e) => setOnMs(Number(e.target.value))} />
          </label>
          <label>
            OFF (ms): {offMs}
            <input type="range" min={10} max={4000} step={10} value={offMs} onChange={(e) => setOffMs(Number(e.target.value))} />
          </label>
          <label>
            Auto bar step (deg/presentation): {autoBarStepDeg}
            <input type="range" min={0} max={180} value={autoBarStepDeg} onChange={(e) => setAutoBarStepDeg(Number(e.target.value))} />
          </label>
          <label>
            Grating auto mode
            <select value={autoGrMode} onChange={(e) => setAutoGrMode(e.target.value as GratingAutoMode)}>
              <option value="orientation">orientation</option>
              <option value="frequency">frequency</option>
              <option value="both">both</option>
            </select>
          </label>
          <label>
            Auto grating ori step: {autoGrOriStepDeg}
            <input type="range" min={0} max={90} value={autoGrOriStepDeg} onChange={(e) => setAutoGrOriStepDeg(Number(e.target.value))} />
          </label>
          <label>
            Auto grating sf step x1e-3: {(autoGrSfStepCpp * 1000).toFixed(1)}
            <input
              type="range"
              min={0}
              max={20}
              step={0.5}
              value={autoGrSfStepCpp * 1000}
              onChange={(e) => setAutoGrSfStepCpp(Number(e.target.value) / 1000)}
            />
          </label>
        </section>

        <section className="card">
          <h2>Dynamics</h2>
          <label>
            Response mode
            <select value={responseMode} onChange={(e) => setResponseMode(e.target.value as ResponseMode)}>
              <option value="normalized">normalized (bounded)</option>
              <option value="legacy">legacy (unbounded dot)</option>
            </select>
          </label>
          <label>
            Max rate Hz: {maxRateHz}
            <input type="range" min={20} max={300} value={maxRateHz} onChange={(e) => setMaxRateHz(Number(e.target.value))} />
          </label>
          <label>
            Rate Y max: {rateYMax}
            <input type="range" min={1} max={1000} value={rateYMax} onChange={(e) => setRateYMax(Number(e.target.value))} />
          </label>
          <label>
            Gain: {rateGain.toFixed(1)}
            <input type="range" min={1} max={120} step={1} value={rateGain} onChange={(e) => setRateGain(Number(e.target.value))} />
          </label>
          <label>
            Baseline Hz: {baselineHz}
            <input type="range" min={0} max={40} value={baselineHz} onChange={(e) => setBaselineHz(Number(e.target.value))} />
          </label>
          <label>
            FPS: {fps}
            <input type="range" min={10} max={60} value={fps} onChange={(e) => setFps(Number(e.target.value))} />
          </label>
          <label>
            Spike buffer: {spikeBuffer}
            <input type="range" min={1} max={120} value={spikeBuffer} onChange={(e) => setSpikeBuffer(Number(e.target.value))} />
          </label>
          <label>
            Spike shape Y +/-: {spikeShapeYAbs.toFixed(1)}
            <input
              type="range"
              min={0.2}
              max={5.0}
              step={0.1}
              value={spikeShapeYAbs}
              onChange={(e) => setSpikeShapeYAbs(Number(e.target.value))}
            />
          </label>
          <label className="row-check">
            <input type="checkbox" checked={audioEnabled} onChange={(e) => setAudioEnabled(e.target.checked)} />
            Tiny sounds
          </label>
          <div className="row-actions">
            <button type="button" onClick={() => setPaused((p) => !p)}>
              {paused ? "Start" : "Pause"}
            </button>
            <button
              type="button"
              onClick={() => {
                timeRef.current = 0;
                setTNow(0);
                setHist(buildHistory(MAX_NEURONS));
              }}
            >
              Reset traces
            </button>
          </div>
          <div className="readout">
            Mode={responseMode} | Mean rate={meanRate.toFixed(1)} Hz | t={tNow.toFixed(2)} s
          </div>
          <div className="legend">
            {activeNeurons.map((n, i) => (
              <span key={n.id} style={{ color: NEURON_COLOR_HEX[n.color] }}>
                N{i + 1}
              </span>
            ))}
          </div>
        </section>
      </aside>

      <main className="right">
        <section className="top-row">
          <section className="panel panel-stimulus">
            <h2>Stimulus + RF</h2>
            <div className="panel-body">
              <canvas
                ref={canvasRef}
                width={CANVAS_SIZE}
                height={CANVAS_SIZE}
                className="media-canvas"
                onMouseMove={(e) => {
                  if (e.buttons !== 1) return;
                  const rect = e.currentTarget.getBoundingClientRect();
                  setCenterX(clamp(e.clientX - rect.left, 0, CANVAS_SIZE));
                  setCenterY(clamp(e.clientY - rect.top, 0, CANVAS_SIZE));
                }}
              />
            </div>
          </section>

          <section className="panel panel-kernel">
            <h2>Selected V1 Kernel</h2>
            <div className="panel-body">
              <canvas
                ref={kernelCanvasRef}
                width={KERNEL_VIEW_SIZE}
                height={KERNEL_VIEW_SIZE}
                className="media-canvas kernel-canvas"
              />
            </div>
          </section>

          <section className="panel panel-wave">
            <h2>Detected Spike Shape Buffer</h2>
            <div className="panel-body">
              <svg viewBox="0 0 1000 1000" className="plot top-plot">
                <rect x={0} y={0} width={1000} height={1000} fill="#000" />
                <line x1={0} y1={500} x2={1000} y2={500} stroke="#ffffff" strokeWidth={1.2} opacity={0.95} />
                {activeNeurons.map((n, i) =>
                  hist.waves[i].map((w, j) => (
                    <polyline
                      key={`${n.id}-w-${j}`}
                      points={waveformPolyline(w.tMs, w.amp, 1000, 1000, -0.5, 2.0, spikeShapeYAbs)}
                      fill="none"
                      stroke={NEURON_COLOR_HEX[n.color]}
                      strokeWidth={1.4}
                      opacity={0.9}
                    />
                  )),
                )}
              </svg>
            </div>
          </section>
        </section>

        <section className="panel">
          <h2>Firing Rate (0..{ratePlotYMax.toFixed(1)} Hz)</h2>
          <svg viewBox={`0 0 ${rateSvgW} ${rateSvgH}`} className="plot">
            <rect x={0} y={0} width={rateSvgW} height={rateSvgH} fill="#000" />
            <line
              x1={rateMargins.left}
              y1={rateMargins.top + rateInnerH}
              x2={rateMargins.left + rateInnerW}
              y2={rateMargins.top + rateInnerH}
              stroke="#ffffff"
              strokeWidth={1.5}
            />
            <line
              x1={rateMargins.left}
              y1={rateMargins.top}
              x2={rateMargins.left}
              y2={rateMargins.top + rateInnerH}
              stroke="#ffffff"
              strokeWidth={1.5}
            />
            {[0, 3, 6, 9, 12].map((tv) => {
              const x = rateMargins.left + (tv / HISTORY_SEC) * rateInnerW;
              return (
                <g key={`tx-${tv}`}>
                  <line x1={x} y1={rateMargins.top + rateInnerH} x2={x} y2={rateMargins.top + rateInnerH + 6} stroke="#ffffff" strokeWidth={1} />
                  <text className="axis-tick" x={x} y={rateSvgH - 8} fill="#ffffff" textAnchor="middle">
                    {tv.toString()}
                  </text>
                </g>
              );
            })}
            {[0, 0.25, 0.5, 0.75, 1].map((frac) => {
              const val = ratePlotYMax * frac;
              const y = rateMargins.top + (1 - frac) * rateInnerH;
              return (
                <g key={`ty-${frac.toFixed(2)}`}>
                  <line x1={rateMargins.left - 6} y1={y} x2={rateMargins.left} y2={y} stroke="#ffffff" strokeWidth={1} />
                  <text className="axis-tick" x={rateMargins.left - 10} y={y + 4} fill="#ffffff" textAnchor="end">
                    {val.toFixed(0)}
                  </text>
                </g>
              );
            })}
            <text
              className="axis-label"
              x={rateMargins.left + rateInnerW / 2}
              y={rateSvgH - 2}
              fill="#ffffff"
              textAnchor="middle"
            >
              Time (s)
            </text>
            <text
              className="axis-label"
              x={18}
              y={rateMargins.top + rateInnerH / 2}
              fill="#ffffff"
              textAnchor="middle"
              transform={`rotate(-90 18 ${rateMargins.top + rateInnerH / 2})`}
            >
              Rate (Hz)
            </text>
            {activeNeurons.map((n, i) => (
              <polyline
                key={n.id}
                points={polylinePoints(
                  hist.t,
                  hist.rates[i],
                  rateSvgW,
                  rateSvgH,
                  ratePlotYMax,
                  rateMargins,
                  traceTStart,
                  traceTEnd,
                )}
                fill="none"
                stroke={NEURON_COLOR_HEX[n.color]}
                strokeWidth={2}
              />
            ))}
          </svg>
        </section>

        <section className="panel">
          <h2>Spikes Over Time</h2>
          <svg viewBox="0 0 1000 220" className="plot">
            <rect x={0} y={0} width={1000} height={220} fill="#000" />
            <line
              x1={rateMargins.left}
              y1={rateMargins.top + rateInnerH}
              x2={rateMargins.left + rateInnerW}
              y2={rateMargins.top + rateInnerH}
              stroke="#ffffff"
              strokeWidth={1.5}
            />
            <line
              x1={rateMargins.left}
              y1={rateMargins.top}
              x2={rateMargins.left}
              y2={rateMargins.top + rateInnerH}
              stroke="#ffffff"
              strokeWidth={1.5}
            />
            {[0, 3, 6, 9, 12].map((tv) => {
              const x = rateMargins.left + (tv / HISTORY_SEC) * rateInnerW;
              return (
                <g key={`sx-${tv}`}>
                  <line x1={x} y1={rateMargins.top + rateInnerH} x2={x} y2={rateMargins.top + rateInnerH + 6} stroke="#ffffff" strokeWidth={1} />
                  <text className="axis-tick" x={x} y={rateSvgH - 8} fill="#ffffff" textAnchor="middle">
                    {tv.toString()}
                  </text>
                </g>
              );
            })}
            {[0, 1].map((sv) => {
              const y = rateMargins.top + (1 - sv) * rateInnerH;
              return (
                <g key={`sy-${sv}`}>
                  <line x1={rateMargins.left - 6} y1={y} x2={rateMargins.left} y2={y} stroke="#ffffff" strokeWidth={1} />
                  <text className="axis-tick" x={rateMargins.left - 10} y={y + 4} fill="#ffffff" textAnchor="end">
                    {sv.toString()}
                  </text>
                </g>
              );
            })}
            <text
              className="axis-label"
              x={rateMargins.left + rateInnerW / 2}
              y={rateSvgH - 2}
              fill="#ffffff"
              textAnchor="middle"
            >
              Time (s)
            </text>
            <text
              className="axis-label"
              x={18}
              y={rateMargins.top + rateInnerH / 2}
              fill="#ffffff"
              textAnchor="middle"
              transform={`rotate(-90 18 ${rateMargins.top + rateInnerH / 2})`}
            >
              Spikes
            </text>
            {activeNeurons.map((n, i) =>
              spikeLines(hist.spikes[i], traceTStart, traceTEnd, 1000, 220, rateMargins).map((ln, j) => (
                <line
                  key={`${n.id}-${j}`}
                  x1={ln.x}
                  y1={ln.y0}
                  x2={ln.x}
                  y2={ln.y1}
                  stroke={NEURON_COLOR_HEX[n.color]}
                  strokeWidth={1.5}
                />
              )),
            )}
          </svg>
        </section>
      </main>
    </div>
  );
}
