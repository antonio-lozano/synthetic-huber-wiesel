/**
 * DynamicsPanel – response mode, rate params, FPS, audio, pause/reset, neuron legend.
 */
import { NEURON_COLOR_HEX } from "../core/colors";
import { useSimStore } from "../core/store";
import { ResponseMode } from "../core/types";

export function DynamicsPanel() {
  const s = useSimStore();
  const activeNeurons = s.activeNeurons();
  const meanRate = s.meanRate();

  return (
    <section className="card" role="region" aria-label="Dynamics controls">
      <h2>Dynamics</h2>

      <label>
        Response mode
        <select value={s.responseMode} onChange={(e) => s.set({ responseMode: e.target.value as ResponseMode })} aria-label="Response mode">
          <option value="normalized">normalized (bounded)</option>
          <option value="legacy">legacy (unbounded dot)</option>
        </select>
      </label>

      <label>Max rate Hz: {s.maxRateHz}<input type="range" min={20} max={300} value={s.maxRateHz} onChange={(e) => s.set({ maxRateHz: Number(e.target.value) })} /></label>
      <label>Rate Y max: {s.rateYMax}<input type="range" min={1} max={1000} value={s.rateYMax} onChange={(e) => s.set({ rateYMax: Number(e.target.value) })} /></label>
      <label>Gain: {s.rateGain.toFixed(1)}<input type="range" min={1} max={120} step={1} value={s.rateGain} onChange={(e) => s.set({ rateGain: Number(e.target.value) })} /></label>
      <label>Baseline Hz: {s.baselineHz}<input type="range" min={0} max={40} value={s.baselineHz} onChange={(e) => s.set({ baselineHz: Number(e.target.value) })} /></label>
      <label>FPS: {s.fps}<input type="range" min={10} max={60} value={s.fps} onChange={(e) => s.set({ fps: Number(e.target.value) })} /></label>
      <label>Spike buffer: {s.spikeBuffer}<input type="range" min={1} max={120} value={s.spikeBuffer} onChange={(e) => s.set({ spikeBuffer: Number(e.target.value) })} /></label>
      <label>Buffer refresh (s): {s.bufferRefreshSec.toFixed(1)}<input type="range" min={1} max={30} step={0.5} value={s.bufferRefreshSec} onChange={(e) => s.set({ bufferRefreshSec: Number(e.target.value) })} /></label>
      <label>Spike shape Y +/-: {s.spikeShapeYAbs.toFixed(1)}<input type="range" min={0.2} max={5.0} step={0.1} value={s.spikeShapeYAbs} onChange={(e) => s.set({ spikeShapeYAbs: Number(e.target.value) })} /></label>

      <label className="row-check">
        <input type="checkbox" checked={s.audioEnabled} onChange={(e) => s.set({ audioEnabled: e.target.checked })} />
        Tiny sounds
      </label>

      <div className="row-actions">
        <button type="button" onClick={s.togglePause} aria-label={s.paused ? "Resume simulation" : "Pause simulation"}>
          {s.paused ? "Start" : "Pause"}
        </button>
        <button type="button" onClick={s.resetTraces} aria-label="Reset all traces">
          Reset traces
        </button>
      </div>

      <div className="readout" aria-live="polite">
        Mode={s.responseMode} | Mean rate={meanRate.toFixed(1)} Hz | t={s.tNow.toFixed(2)} s
      </div>

      <div className="legend" role="group" aria-label="Active neuron legend">
        {activeNeurons.map((n) => (
          <span key={n.id} style={{ color: NEURON_COLOR_HEX[n.color] }}>
            N{n.id + 1}
          </span>
        ))}
      </div>
    </section>
  );
}
