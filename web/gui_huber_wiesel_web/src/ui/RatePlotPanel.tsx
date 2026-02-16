import { NEURON_COLOR_HEX } from "../core/colors";
import { SimNeuron } from "../core/types";
import { PlotMargins, polylinePoints } from "./plotUtils";

type RatePlotPanelProps = {
  activeNeurons: SimNeuron[];
  traceTime: number[];
  ratesByNeuron: number[][];
  ratePlotYMax: number;
  historySec: number;
  traceTStart: number;
  traceTEnd: number;
  margins: PlotMargins;
};

export function RatePlotPanel({
  activeNeurons,
  traceTime,
  ratesByNeuron,
  ratePlotYMax,
  historySec,
  traceTStart,
  traceTEnd,
  margins,
}: RatePlotPanelProps) {
  const rateSvgW = 1000;
  const rateSvgH = 220;
  const rateInnerW = rateSvgW - margins.left - margins.right;
  const rateInnerH = rateSvgH - margins.top - margins.bottom;

  return (
    <section className="panel">
      <h2>Firing Rate (0..{ratePlotYMax.toFixed(1)} Hz)</h2>
      <svg viewBox={`0 0 ${rateSvgW} ${rateSvgH}`} className="plot">
        <rect x={0} y={0} width={rateSvgW} height={rateSvgH} fill="#000" />
        <line
          x1={margins.left}
          y1={margins.top + rateInnerH}
          x2={margins.left + rateInnerW}
          y2={margins.top + rateInnerH}
          stroke="#ffffff"
          strokeWidth={1.5}
        />
        <line
          x1={margins.left}
          y1={margins.top}
          x2={margins.left}
          y2={margins.top + rateInnerH}
          stroke="#ffffff"
          strokeWidth={1.5}
        />
        {[0, 3, 6, 9, 12].map((tv) => {
          const x = margins.left + (tv / historySec) * rateInnerW;
          return (
            <g key={`tx-${tv}`}>
              <line x1={x} y1={margins.top + rateInnerH} x2={x} y2={margins.top + rateInnerH + 6} stroke="#ffffff" strokeWidth={1} />
              <text className="axis-tick" x={x} y={rateSvgH - 8} fill="#ffffff" textAnchor="middle">
                {tv.toString()}
              </text>
            </g>
          );
        })}
        {[0, 0.25, 0.5, 0.75, 1].map((frac) => {
          const val = ratePlotYMax * frac;
          const y = margins.top + (1 - frac) * rateInnerH;
          return (
            <g key={`ty-${frac.toFixed(2)}`}>
              <line x1={margins.left - 6} y1={y} x2={margins.left} y2={y} stroke="#ffffff" strokeWidth={1} />
              <text className="axis-tick" x={margins.left - 10} y={y + 4} fill="#ffffff" textAnchor="end">
                {val.toFixed(0)}
              </text>
            </g>
          );
        })}
        <text className="axis-label" x={margins.left + rateInnerW / 2} y={rateSvgH - 2} fill="#ffffff" textAnchor="middle">
          Time (s)
        </text>
        <text
          className="axis-label"
          x={18}
          y={margins.top + rateInnerH / 2}
          fill="#ffffff"
          textAnchor="middle"
          transform={`rotate(-90 18 ${margins.top + rateInnerH / 2})`}
        >
          Rate (Hz)
        </text>
        {activeNeurons.map((n) => (
          <polyline
            key={n.id}
            points={polylinePoints(
              traceTime,
              ratesByNeuron[n.id],
              rateSvgW,
              rateSvgH,
              ratePlotYMax,
              margins,
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
  );
}
