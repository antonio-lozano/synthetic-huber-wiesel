import { useEffect, useMemo, useRef, useState } from "react";
import { NEURON_COLOR_HEX } from "./core/colors";
import { createNeurons, fakeRateForNeuron, updateStimulusPhase } from "./core/sim";
import { SimNeuron, StimulusKind } from "./core/types";

const CANVAS_SIZE = 512;
const HISTORY_SEC = 12;
const DT_SEC = 0.05;
const MAX_NEURONS = 5;

type Hist = {
  t: number[];
  rates: number[][];
  spikes: number[][];
};

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

function buildHistory(n: number): Hist {
  return {
    t: [],
    rates: Array.from({ length: n }, () => []),
    spikes: Array.from({ length: n }, () => []),
  };
}

function drawStimulus(
  ctx: CanvasRenderingContext2D,
  kind: StimulusKind,
  centerX: number,
  centerY: number,
  orientationDeg: number,
  phaseDeg: number,
): void {
  if (kind === "bar") {
    ctx.save();
    ctx.translate(centerX, centerY);
    ctx.rotate((orientationDeg * Math.PI) / 180);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(-100, -4, 200, 8);
    ctx.restore();
    return;
  }
  if (kind === "grating") {
    ctx.save();
    ctx.translate(centerX, centerY);
    ctx.rotate((orientationDeg * Math.PI) / 180);
    const phasePx = (phaseDeg / 360) * 20;
    for (let x = -220; x <= 220; x += 20) {
      const c = ((x + phasePx) / 20) % 2 === 0 ? 220 : 60;
      ctx.fillStyle = `rgb(${c},${c},${c})`;
      ctx.fillRect(x, -180, 12, 360);
    }
    ctx.restore();
    return;
  }
  ctx.fillStyle = "#808080";
  ctx.fillRect(centerX - 70, centerY - 70, 140, 140);
}

function drawRfBoxes(ctx: CanvasRenderingContext2D, neurons: SimNeuron[]): void {
  neurons.forEach((n) => {
    ctx.strokeStyle = NEURON_COLOR_HEX[n.color];
    ctx.lineWidth = 2;
    ctx.strokeRect(n.rfPos.x, n.rfPos.y, n.rfSize, n.rfSize);
  });
}

function polylinePoints(xs: number[], ys: number[], width: number, height: number, yMax: number): string {
  if (xs.length === 0) return "";
  const t0 = xs[0];
  const t1 = xs[xs.length - 1];
  const dt = Math.max(1e-6, t1 - t0);
  return xs
    .map((t, i) => {
      const x = ((t - t0) / dt) * width;
      const y = height - (clamp(ys[i], 0, yMax) / yMax) * height;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

function spikeLines(
  spikeTimes: number[],
  tNow: number,
  width: number,
  height: number,
): Array<{ x: number; y0: number; y1: number }> {
  const tMin = tNow - HISTORY_SEC;
  return spikeTimes
    .filter((t) => t >= tMin)
    .map((t) => ({
      x: ((t - tMin) / HISTORY_SEC) * width,
      y0: 0,
      y1: height,
    }));
}

export default function App() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [nNeurons, setNNeurons] = useState(3);
  const [slot, setSlot] = useState(0);
  const [stimKind, setStimKind] = useState<StimulusKind>("bar");
  const [stimOrientation, setStimOrientation] = useState(45);
  const [stimPhase, setStimPhase] = useState(0);
  const [phaseSpeed, setPhaseSpeed] = useState(200);
  const [centerX, setCenterX] = useState(CANVAS_SIZE / 2);
  const [centerY, setCenterY] = useState(CANVAS_SIZE / 2);
  const [rateYMax, setRateYMax] = useState(100);
  const [tNow, setTNow] = useState(0);
  const [neurons, setNeurons] = useState<SimNeuron[]>(() => createNeurons(MAX_NEURONS, CANVAS_SIZE));
  const [hist, setHist] = useState<Hist>(() => buildHistory(MAX_NEURONS));

  const activeNeurons = useMemo(() => neurons.slice(0, nNeurons), [neurons, nNeurons]);
  const activeSlot = Math.min(slot, nNeurons - 1);

  useEffect(() => {
    const id = window.setInterval(() => {
      setTNow((prev) => prev + DT_SEC);
      if (stimKind === "grating") {
        setStimPhase((prev) => updateStimulusPhase(prev, DT_SEC, phaseSpeed));
      }

      setHist((prev) => {
        const nextT = [...prev.t, (prev.t.at(-1) ?? 0) + DT_SEC];
        const rates = prev.rates.map((arr) => [...arr]);
        const spikes = prev.spikes.map((arr) => [...arr]);

        for (let i = 0; i < MAX_NEURONS; i += 1) {
          const n = neurons[i];
          const isActive = i < nNeurons;
          const r = isActive ? fakeRateForNeuron(n, stimKind, stimOrientation) : 0;
          rates[i].push(r);
          const p = clamp(r * DT_SEC, 0, 1);
          if (isActive && Math.random() < p) {
            spikes[i].push(nextT[nextT.length - 1]);
          }
        }

        const tMin = nextT[nextT.length - 1] - HISTORY_SEC;
        let start = 0;
        for (let i = 0; i < nextT.length; i += 1) {
          if (nextT[i] >= tMin) {
            start = i;
            break;
          }
        }
        const trimT = nextT.slice(start);
        const trimRates = rates.map((arr) => arr.slice(start));
        const trimSpikes = spikes.map((arr) => arr.filter((t) => t >= tMin));
        return { t: trimT, rates: trimRates, spikes: trimSpikes };
      });
    }, Math.round(DT_SEC * 1000));
    return () => window.clearInterval(id);
  }, [neurons, nNeurons, phaseSpeed, stimKind, stimOrientation]);

  useEffect(() => {
    const c = canvasRef.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    drawStimulus(ctx, stimKind, centerX, centerY, stimOrientation, stimPhase);
    drawRfBoxes(ctx, activeNeurons);
  }, [activeNeurons, centerX, centerY, stimKind, stimOrientation, stimPhase]);

  const editNeuron = activeNeurons[activeSlot] ?? activeNeurons[0];
  const canEdit = !!editNeuron;

  return (
    <div className="app">
      <aside className="left">
        <h1>Synthetic Hubel-Wiesel Web</h1>
        <p className="muted">Scaffold build for GitHub Pages. Browser-first rewrite starts here.</p>

        <section className="card">
          <h2>Neurons</h2>
          <label>
            Sim neurons (1-5)
            <input type="range" min={1} max={5} value={nNeurons} onChange={(e) => setNNeurons(Number(e.target.value))} />
          </label>
          <label>
            Edit RF slot
            <select value={activeSlot} onChange={(e) => setSlot(Number(e.target.value))}>
              {Array.from({ length: nNeurons }).map((_, i) => (
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
              max={canEdit ? CANVAS_SIZE - editNeuron.rfSize : CANVAS_SIZE}
              value={canEdit ? editNeuron.rfPos.x : 0}
              onChange={(e) => {
                const v = Number(e.target.value);
                setNeurons((prev) =>
                  prev.map((n, i) => (i === activeSlot ? { ...n, rfPos: { ...n.rfPos, x: v } } : n)),
                );
              }}
            />
          </label>
          <label>
            RF Y
            <input
              type="range"
              min={0}
              max={canEdit ? CANVAS_SIZE - editNeuron.rfSize : CANVAS_SIZE}
              value={canEdit ? editNeuron.rfPos.y : 0}
              onChange={(e) => {
                const v = Number(e.target.value);
                setNeurons((prev) =>
                  prev.map((n, i) => (i === activeSlot ? { ...n, rfPos: { ...n.rfPos, y: v } } : n)),
                );
              }}
            />
          </label>
        </section>

        <section className="card">
          <h2>Stimulus</h2>
          <label>
            Type
            <select value={stimKind} onChange={(e) => setStimKind(e.target.value as StimulusKind)}>
              <option value="bar">bar</option>
              <option value="grating">grating</option>
              <option value="cifar">cifar</option>
            </select>
          </label>
          <label>
            Orientation
            <input type="range" min={0} max={179} value={stimOrientation} onChange={(e) => setStimOrientation(Number(e.target.value))} />
          </label>
          <label>
            Phase speed (deg/s, grating)
            <input type="range" min={0} max={500} value={phaseSpeed} onChange={(e) => setPhaseSpeed(Number(e.target.value))} />
          </label>
        </section>

        <section className="card">
          <h2>Plots</h2>
          <label>
            Rate Y max
            <input type="range" min={20} max={200} value={rateYMax} onChange={(e) => setRateYMax(Number(e.target.value))} />
          </label>
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
        <section className="panel">
          <h2>Stimulus + RF</h2>
          <canvas
            ref={canvasRef}
            width={CANVAS_SIZE}
            height={CANVAS_SIZE}
            onMouseMove={(e) => {
              if (e.buttons !== 1) return;
              const rect = e.currentTarget.getBoundingClientRect();
              setCenterX(clamp(e.clientX - rect.left, 0, CANVAS_SIZE));
              setCenterY(clamp(e.clientY - rect.top, 0, CANVAS_SIZE));
            }}
          />
        </section>

        <section className="panel">
          <h2>Firing Rate (0..{rateYMax} Hz)</h2>
          <svg viewBox="0 0 1000 220" className="plot">
            <rect x={0} y={0} width={1000} height={220} fill="#000" />
            {activeNeurons.map((n, i) => {
              const pts = polylinePoints(hist.t, hist.rates[i], 1000, 220, rateYMax);
              return <polyline key={n.id} points={pts} fill="none" stroke={NEURON_COLOR_HEX[n.color]} strokeWidth={2} />;
            })}
          </svg>
        </section>

        <section className="panel">
          <h2>Spikes Over Time</h2>
          <svg viewBox="0 0 1000 160" className="plot">
            <rect x={0} y={0} width={1000} height={160} fill="#000" />
            {activeNeurons.map((n, i) =>
              spikeLines(hist.spikes[i], tNow, 1000, 160).map((ln, j) => (
                <line key={`${n.id}-${j}`} x1={ln.x} y1={ln.y0} x2={ln.x} y2={ln.y1} stroke={NEURON_COLOR_HEX[n.color]} strokeWidth={1.5} />
              )),
            )}
          </svg>
        </section>
      </main>
    </div>
  );
}
