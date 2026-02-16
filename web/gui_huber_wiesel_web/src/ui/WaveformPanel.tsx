import { NEURON_COLOR_HEX } from "../core/colors";
import { WaveBuffer } from "../core/history";
import { SimNeuron } from "../core/types";
import { waveformPolyline } from "./plotUtils";

type WaveformPanelProps = {
  activeNeurons: SimNeuron[];
  wavesByNeuron: WaveBuffer[][];
  spikeShapeYAbs: number;
};

export function WaveformPanel({ activeNeurons, wavesByNeuron, spikeShapeYAbs }: WaveformPanelProps) {
  return (
    <section className="panel panel-wave">
      <h2>Detected Spike Shape Buffer</h2>
      <div className="panel-body">
        <svg viewBox="0 0 1000 1000" className="plot top-plot">
          <rect x={0} y={0} width={1000} height={1000} fill="#000" />
          <line x1={0} y1={500} x2={1000} y2={500} stroke="#ffffff" strokeWidth={1.2} opacity={0.95} />
          {activeNeurons.map((n) =>
            wavesByNeuron[n.id].map((w, j) => (
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
  );
}
