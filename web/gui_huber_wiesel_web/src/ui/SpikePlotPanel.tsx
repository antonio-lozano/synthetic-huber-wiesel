import { NEURON_COLOR_HEX } from "../core/colors";
import { SimNeuron } from "../core/types";
import { PlotMargins, spikeLines } from "./plotUtils";

type SpikePlotPanelProps = {
  activeNeurons: SimNeuron[];
  spikesByNeuron: number[][];
  historySec: number;
  traceTStart: number;
  traceTEnd: number;
  margins: PlotMargins;
};

export function SpikePlotPanel({
  activeNeurons,
  spikesByNeuron,
  historySec,
  traceTStart,
  traceTEnd,
  margins,
}: SpikePlotPanelProps) {
  const svgW = 1000;
  const svgH = 220;
  const innerW = svgW - margins.left - margins.right;
  const innerH = svgH - margins.top - margins.bottom;

  return (
    <section className="panel">
      <h2>Spikes Over Time</h2>
      <svg viewBox={`0 0 ${svgW} ${svgH}`} className="plot">
        <rect x={0} y={0} width={svgW} height={svgH} fill="#000" />
        <line
          x1={margins.left}
          y1={margins.top + innerH}
          x2={margins.left + innerW}
          y2={margins.top + innerH}
          stroke="#ffffff"
          strokeWidth={1.5}
        />
        <line
          x1={margins.left}
          y1={margins.top}
          x2={margins.left}
          y2={margins.top + innerH}
          stroke="#ffffff"
          strokeWidth={1.5}
        />
        {[0, 3, 6, 9, 12].map((tv) => {
          const x = margins.left + (tv / historySec) * innerW;
          return (
            <g key={`sx-${tv}`}>
              <line x1={x} y1={margins.top + innerH} x2={x} y2={margins.top + innerH + 6} stroke="#ffffff" strokeWidth={1} />
              <text className="axis-tick" x={x} y={svgH - 8} fill="#ffffff" textAnchor="middle">
                {tv.toString()}
              </text>
            </g>
          );
        })}
        {[0, 1].map((sv) => {
          const y = margins.top + (1 - sv) * innerH;
          return (
            <g key={`sy-${sv}`}>
              <line x1={margins.left - 6} y1={y} x2={margins.left} y2={y} stroke="#ffffff" strokeWidth={1} />
              <text className="axis-tick" x={margins.left - 10} y={y + 4} fill="#ffffff" textAnchor="end">
                {sv.toString()}
              </text>
            </g>
          );
        })}
        <text className="axis-label" x={margins.left + innerW / 2} y={svgH - 2} fill="#ffffff" textAnchor="middle">
          Time (s)
        </text>
        <text
          className="axis-label"
          x={18}
          y={margins.top + innerH / 2}
          fill="#ffffff"
          textAnchor="middle"
          transform={`rotate(-90 18 ${margins.top + innerH / 2})`}
        >
          Spikes
        </text>
        {activeNeurons.map((n) =>
          spikeLines(spikesByNeuron[n.id], traceTStart, traceTEnd, svgW, svgH, margins).map((ln, j) => (
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
  );
}
