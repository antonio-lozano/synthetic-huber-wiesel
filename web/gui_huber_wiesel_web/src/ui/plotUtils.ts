export type PlotMargins = {
  left: number;
  right: number;
  top: number;
  bottom: number;
};

export type SpikeLine = {
  x: number;
  y0: number;
  y1: number;
};

export function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

export function polylinePoints(
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

export function spikeLines(
  spikeTimes: number[],
  tStart: number,
  tEnd: number,
  width: number,
  height: number,
  margins: PlotMargins = { left: 0, right: 0, top: 0, bottom: 0 },
): SpikeLine[] {
  const innerW = Math.max(1, width - margins.left - margins.right);
  const dt = Math.max(1e-6, tEnd - tStart);
  return spikeTimes
    .filter((t) => t >= tStart && t <= tEnd)
    .map((t) => ({
      x: margins.left + ((t - tStart) / dt) * innerW,
      y0: margins.top,
      y1: height - margins.bottom,
    }));
}

export function waveformPolyline(
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
