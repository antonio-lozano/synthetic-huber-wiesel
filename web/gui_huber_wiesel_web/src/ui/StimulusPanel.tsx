/**
 * StimulusPanel – stimulus type, bar/grating/CIFAR params, timed experiment.
 */
import { CANVAS_SIZE, useSimStore } from "../core/store";
import { GratingAutoMode, StimulusColor, StimulusKind } from "../core/types";

const COLOR_OPTIONS: StimulusColor[] = ["white", "red", "green", "blue", "yellow", "cyan", "magenta"];

export function StimulusPanel() {
  const s = useSimStore();

  const isBar = s.stimKind === "bar";
  const isGrating = s.stimKind === "grating";
  const isCifar = s.stimKind === "cifar";

  return (
    <section className="card" role="region" aria-label="Stimulus controls">
      <h2>Stimulus</h2>

      <label>
        Type
        <select value={s.stimKind} onChange={(e) => s.set({ stimKind: e.target.value as StimulusKind })} aria-label="Stimulus type">
          <option value="bar">bar</option>
          <option value="grating">grating</option>
          <option value="cifar">cifar-10 preload</option>
        </select>
      </label>

      <label>
        Stim X
        <input type="range" min={0} max={CANVAS_SIZE} value={s.centerX} onChange={(e) => s.set({ centerX: Number(e.target.value) })} aria-label="Stimulus X position" />
      </label>
      <label>
        Stim Y
        <input type="range" min={0} max={CANVAS_SIZE} value={s.centerY} onChange={(e) => s.set({ centerY: Number(e.target.value) })} aria-label="Stimulus Y position" />
      </label>

      {(isBar || isGrating) && (
        <label>
          Orientation (deg): {s.orientationDeg}
          <input type="range" min={0} max={179} value={s.orientationDeg} onChange={(e) => s.set({ orientationDeg: Number(e.target.value) })} aria-label="Orientation degrees" />
        </label>
      )}

      {isBar && (
        <>
          <label>
            Bar color
            <select value={s.stimColorBar} onChange={(e) => s.set({ stimColorBar: e.target.value as StimulusColor })} aria-label="Bar color">
              {COLOR_OPTIONS.map((c) => (<option key={c} value={c}>{c}</option>))}
            </select>
          </label>
          <label>Bar length: {s.barLengthPx}<input type="range" min={10} max={300} value={s.barLengthPx} onChange={(e) => s.set({ barLengthPx: Number(e.target.value) })} /></label>
          <label>Bar thickness: {s.barThicknessPx}<input type="range" min={1} max={60} value={s.barThicknessPx} onChange={(e) => s.set({ barThicknessPx: Number(e.target.value) })} /></label>
          <label>Bar patch size: {s.barSizePx}<input type="range" min={40} max={300} value={s.barSizePx} onChange={(e) => s.set({ barSizePx: Number(e.target.value) })} /></label>
          <label>Bar contrast: {s.barContrast.toFixed(2)}<input type="range" min={0.1} max={1} step={0.01} value={s.barContrast} onChange={(e) => s.set({ barContrast: Number(e.target.value) })} /></label>
        </>
      )}

      {isGrating && (
        <>
          <label>
            Grating color
            <select value={s.stimColorGrating} onChange={(e) => s.set({ stimColorGrating: e.target.value as StimulusColor })} aria-label="Grating color">
              {COLOR_OPTIONS.map((c) => (<option key={c} value={c}>{c}</option>))}
            </select>
          </label>
          <label>Grating sf (cycles/pixel): {s.gratingSfCpp.toFixed(3)}<input type="range" min={0.01} max={0.45} step={0.001} value={s.gratingSfCpp} onChange={(e) => s.set({ gratingSfCpp: Number(e.target.value) })} /></label>
          <label>Grating phase (deg): {s.phaseDeg.toFixed(0)}<input type="range" min={0} max={360} value={s.phaseDeg} onChange={(e) => s.set({ phaseDeg: Number(e.target.value) })} /></label>
          <label>Phase speed (deg/s): {s.phaseSpeedDegPerSec}<input type="range" min={0} max={500} value={s.phaseSpeedDegPerSec} onChange={(e) => s.set({ phaseSpeedDegPerSec: Number(e.target.value) })} /></label>
          <label>Grating size: {s.gratingSizePx}<input type="range" min={40} max={300} value={s.gratingSizePx} onChange={(e) => s.set({ gratingSizePx: Number(e.target.value) })} /></label>
          <label>Grating contrast: {s.gratingContrast.toFixed(2)}<input type="range" min={0.1} max={1} step={0.01} value={s.gratingContrast} onChange={(e) => s.set({ gratingContrast: Number(e.target.value) })} /></label>
        </>
      )}

      {isCifar && (
        <>
          <label>CIFAR index: {s.cifarIndex}<input type="range" min={0} max={s.cifarDataset.length - 1} value={s.cifarIndex} onChange={(e) => s.set({ cifarIndex: Number(e.target.value) })} /></label>
          <label>CIFAR size: {s.cifarSizePx}<input type="range" min={32} max={260} value={s.cifarSizePx} onChange={(e) => s.set({ cifarSizePx: Number(e.target.value) })} /></label>
          <div className="readout">Dataset source: {s.cifarSource === "real" ? "real CIFAR-10 (50 preload)" : "synthetic fallback"}</div>
        </>
      )}
    </section>
  );
}

export function TimedExperimentPanel() {
  const s = useSimStore();
  const isBar = s.stimKind === "bar";
  const isGrating = s.stimKind === "grating";

  return (
    <section className="card" role="region" aria-label="Timed experiment controls">
      <h2>Timed Experiment</h2>
      <label className="row-check">
        <input type="checkbox" checked={s.timedExperiment} onChange={(e) => s.set({ timedExperiment: e.target.checked })} />
        Enable ON/OFF presentation
      </label>
      <label>ON (ms): {s.onMs}<input type="range" min={10} max={2000} step={10} value={s.onMs} onChange={(e) => s.set({ onMs: Number(e.target.value) })} /></label>
      <label>OFF (ms): {s.offMs}<input type="range" min={10} max={4000} step={10} value={s.offMs} onChange={(e) => s.set({ offMs: Number(e.target.value) })} /></label>

      {isBar && (
        <label>Auto bar step (deg/presentation): {s.autoBarStepDeg}<input type="range" min={0} max={180} value={s.autoBarStepDeg} onChange={(e) => s.set({ autoBarStepDeg: Number(e.target.value) })} /></label>
      )}

      {isGrating && (
        <>
          <label>
            Grating auto mode
            <select value={s.autoGrMode} onChange={(e) => s.set({ autoGrMode: e.target.value as GratingAutoMode })} aria-label="Grating auto mode">
              <option value="orientation">orientation</option>
              <option value="frequency">frequency</option>
              <option value="both">both</option>
            </select>
          </label>
          <label>Auto grating ori step: {s.autoGrOriStepDeg}<input type="range" min={0} max={90} value={s.autoGrOriStepDeg} onChange={(e) => s.set({ autoGrOriStepDeg: Number(e.target.value) })} /></label>
          <label>Auto grating sf step x1e-3: {(s.autoGrSfStepCpp * 1000).toFixed(1)}<input type="range" min={0} max={20} step={0.5} value={s.autoGrSfStepCpp * 1000} onChange={(e) => s.set({ autoGrSfStepCpp: Number(e.target.value) / 1000 })} /></label>
        </>
      )}
    </section>
  );
}
