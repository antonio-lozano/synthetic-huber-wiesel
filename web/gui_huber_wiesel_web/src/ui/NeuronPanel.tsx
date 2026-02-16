/**
 * NeuronPanel – neuron selection, RF position, scale, mask/gray controls.
 */
import { NEURON_COLOR_HEX } from "../core/colors";
import { CANVAS_SIZE, MAX_NEURONS, useSimStore } from "../core/store";
import { clamp } from "./plotUtils";

export function NeuronPanel() {
  const s = useSimStore();
  const editSize = s.editNeuronScaledSize();
  const editMax = Math.max(0, CANVAS_SIZE - editSize);

  return (
    <section className="card" role="region" aria-label="Neuron & RF controls">
      <h2>Neuron + RF</h2>

      <label>
        Neuron ID
        <select
          value={s.activeNeuronIndex}
          onChange={(e) => s.set({ activeNeuronIndex: Number(e.target.value) })}
          aria-label="Select active neuron"
        >
          {Array.from({ length: MAX_NEURONS }).map((_, i) => (
            <option key={i} value={i}>
              N{i + 1}
            </option>
          ))}
        </select>
      </label>

      {/* Per-neuron enable toggles */}
      <div className="neuron-toggles" role="group" aria-label="Neuron visibility toggles">
        {s.activeNeuronIds().map((id) => (
          <label key={id} className="row-check neuron-toggle" style={{ color: NEURON_COLOR_HEX[s.neurons[id].color] }}>
            <input
              type="checkbox"
              checked={s.neuronEnabled[id]}
              onChange={() => s.toggleNeuronEnabled(id)}
              aria-label={`Toggle neuron N${id + 1}`}
            />
            N{id + 1}
          </label>
        ))}
      </div>

      <label>
        Sim neurons: {s.simNeurons}
        <input
          type="range"
          min={1}
          max={5}
          value={s.simNeurons}
          onChange={(e) => s.set({ simNeurons: Number(e.target.value) })}
          aria-label="Number of simultaneous neurons"
        />
      </label>

      <label>
        RF X
        <input
          type="range"
          min={0}
          max={editMax}
          value={clamp(s.editNeuron().rfPos.x, 0, editMax)}
          onChange={(e) => s.setNeuronRfPos(s.activeNeuronIndex, Number(e.target.value), s.editNeuron().rfPos.y)}
          aria-label="RF horizontal position"
        />
      </label>

      <label>
        RF Y
        <input
          type="range"
          min={0}
          max={editMax}
          value={clamp(s.editNeuron().rfPos.y, 0, editMax)}
          onChange={(e) => s.setNeuronRfPos(s.activeNeuronIndex, s.editNeuron().rfPos.x, Number(e.target.value))}
          aria-label="RF vertical position"
        />
      </label>

      <label>
        RF scale (%): {s.rfScalePct}
        <input
          type="range"
          min={50}
          max={400}
          value={s.rfScalePct}
          onChange={(e) => s.set({ rfScalePct: Number(e.target.value) })}
          aria-label="RF scale percentage"
        />
      </label>

      <label className="row-check">
        <input type="checkbox" checked={s.maskRf} onChange={(e) => s.set({ maskRf: e.target.checked })} />
        Mask RF to gray outside
      </label>
      <label className="row-check">
        <input type="checkbox" checked={s.grayRf} onChange={(e) => s.set({ grayRf: e.target.checked })} />
        Full RF grayscale
      </label>
      <label className="row-check">
        <input type="checkbox" checked={s.grayEnergy} onChange={(e) => s.set({ grayEnergy: e.target.checked })} />
        Grayscale energy match
      </label>
    </section>
  );
}
