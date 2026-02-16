import { useEffect, useMemo, useRef } from "react";
import { loadCifarDatasetFromSprite } from "./core/cifarSprite";
import {
  CANVAS_SIZE,
  KERNEL_VIEW_SIZE,
  HISTORY_SEC,
  useSimStore,
} from "./core/store";
import { NeuronPanel } from "./ui/NeuronPanel";
import { StimulusPanel, TimedExperimentPanel } from "./ui/StimulusPanel";
import { DynamicsPanel } from "./ui/DynamicsPanel";
import { PresetPanel } from "./ui/PresetPanel";
import { PerfHUD } from "./ui/PerfHUD";
import { RatePlotPanel } from "./ui/RatePlotPanel";
import { SpikePlotPanel } from "./ui/SpikePlotPanel";
import { WaveformPanel } from "./ui/WaveformPanel";
import { useSimEngine } from "./ui/useSimEngine";
import { useKeyboardShortcuts } from "./ui/useKeyboardShortcuts";
import { clamp, PlotMargins } from "./ui/plotUtils";

export default function App() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const kernelCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const s = useSimStore();

  // Hooks
  useSimEngine();
  useKeyboardShortcuts();

  // Load real CIFAR on mount
  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const dataset = await loadCifarDatasetFromSprite();
        if (!cancelled && dataset.length > 0) {
          useSimStore.getState().setCifarDataset(dataset, "real");
        }
      } catch (err) {
        console.warn("Falling back to synthetic CIFAR-like set:", err);
      }
    })();
    return () => { cancelled = true; };
  }, []);

  // Derived values for plots
  const activeNeurons = s.activeNeurons();
  const activeNeuronIds = s.activeNeuronIds();
  const hist = s.hist;

  const observedRateMax = useMemo(() => {
    const values = activeNeuronIds.flatMap((id) => hist.rates[id]);
    if (values.length === 0) return 1;
    return Math.max(1, ...values);
  }, [activeNeuronIds, hist.rates]);

  const ratePlotYMax = Math.max(1, Math.min(s.rateYMax, Math.max(5, observedRateMax * 1.25)));
  const rateMargins: PlotMargins = { left: 70, right: 14, top: 12, bottom: 34 };
  const traceTStart = hist.t.length > 0 ? hist.t[0] : Math.max(0, s.tNow - HISTORY_SEC);
  const traceTEnd = hist.t.length > 1 ? hist.t[hist.t.length - 1] : traceTStart + 1.0 / Math.max(1, s.fps);

  // Canvas rendering
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const img = ctx.createImageData(CANVAS_SIZE, CANVAS_SIZE);
    img.data.set(s.frameRgba);
    ctx.putImageData(img, 0, 0);
    s.drawBoxes.forEach((b) => {
      ctx.strokeStyle = b.colorHex;
      ctx.lineWidth = 2;
      ctx.strokeRect(b.x + 0.5, b.y + 0.5, b.size, b.size);
    });
  }, [s.frameRgba, s.drawBoxes]);

  useEffect(() => {
    const canvas = kernelCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const img = ctx.createImageData(KERNEL_VIEW_SIZE, KERNEL_VIEW_SIZE);
    img.data.set(s.kernelRgba);
    ctx.putImageData(img, 0, 0);
  }, [s.kernelRgba]);

  return (
    <div className="app" role="application" aria-label="Synthetic Hubel-Wiesel Simulation">
      <aside className="left" role="complementary" aria-label="Controls">
        <h1>Synthetic Hubel-Wiesel Web</h1>
        <p className="muted">
          Real RF convolution, bounded rates, Poisson spikes, waveform buffer.
          <br />
          <kbd>Space</kbd>=pause <kbd>R</kbd>=reset <kbd>N/P</kbd>=neuron <kbd>1-3</kbd>=stim <kbd>T</kbd>=timed
        </p>

        <PresetPanel />
        <NeuronPanel />
        <StimulusPanel />
        <TimedExperimentPanel />
        <DynamicsPanel />
      </aside>

      <main className="right" role="main" aria-label="Visualization">
        <PerfHUD />

        <section className="top-row">
          <section className="panel panel-stimulus">
            <h2>Stimulus + RF</h2>
            <div className="panel-body">
              <canvas
                ref={canvasRef}
                width={CANVAS_SIZE}
                height={CANVAS_SIZE}
                className="media-canvas"
                role="img"
                aria-label="Stimulus canvas with RF overlay"
                tabIndex={0}
                onMouseMove={(e) => {
                  if (e.buttons !== 1) return;
                  const rect = e.currentTarget.getBoundingClientRect();
                  s.set({
                    centerX: clamp(e.clientX - rect.left, 0, CANVAS_SIZE),
                    centerY: clamp(e.clientY - rect.top, 0, CANVAS_SIZE),
                  });
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
                role="img"
                aria-label="Selected neuron kernel preview"
              />
            </div>
          </section>

          <WaveformPanel activeNeurons={activeNeurons} wavesByNeuron={hist.waves} spikeShapeYAbs={s.spikeShapeYAbs} />
        </section>

        <RatePlotPanel
          activeNeurons={activeNeurons}
          traceTime={hist.t}
          ratesByNeuron={hist.rates}
          ratePlotYMax={ratePlotYMax}
          historySec={HISTORY_SEC}
          traceTStart={traceTStart}
          traceTEnd={traceTEnd}
          margins={rateMargins}
        />
        <SpikePlotPanel
          activeNeurons={activeNeurons}
          spikesByNeuron={hist.spikes}
          historySec={HISTORY_SEC}
          traceTStart={traceTStart}
          traceTEnd={traceTEnd}
          margins={rateMargins}
        />
      </main>
    </div>
  );
}
