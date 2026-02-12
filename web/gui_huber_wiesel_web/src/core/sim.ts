import { NEURON_COLOR_ORDER } from "./colors";
import { SimNeuron, StimulusKind } from "./types";

export function createNeurons(n: number, canvasSize: number): SimNeuron[] {
  const out: SimNeuron[] = [];
  for (let i = 0; i < n; i += 1) {
    const rfSize = 24 + i * 4;
    out.push({
      id: i,
      color: NEURON_COLOR_ORDER[i % NEURON_COLOR_ORDER.length],
      rfPos: {
        x: Math.round(canvasSize * 0.2 + i * 20),
        y: Math.round(canvasSize * 0.2 + i * 16),
      },
      rfSize,
      orientationDeg: (i * 30) % 180,
    });
  }
  return out;
}

export function updateStimulusPhase(phaseDeg: number, dtSec: number, speedDegPerSec: number): number {
  const next = phaseDeg + speedDegPerSec * dtSec;
  return ((next % 360) + 360) % 360;
}

export function fakeRateForNeuron(
  neuron: SimNeuron,
  stimKind: StimulusKind,
  stimOrientationDeg: number,
): number {
  const d = Math.abs((((stimOrientationDeg - neuron.orientationDeg) % 180) + 180) % 180);
  const dMin = Math.min(d, 180 - d);
  let match = Math.max(0, 1 - dMin / 90);
  if (stimKind === "cifar") {
    match *= 0.7;
  }
  return 100 * match;
}
