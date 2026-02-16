export type WaveBuffer = {
  tMs: number[];
  amp: number[];
};

export type Hist = {
  t: number[];
  rates: number[][];
  spikes: number[][];
  waves: WaveBuffer[][];
  lastRaw: number[];
  lastEff: number[];
  lastRate: number[];
};

export function buildHistory(n: number): Hist {
  return {
    t: [],
    rates: Array.from({ length: n }, () => []),
    spikes: Array.from({ length: n }, () => []),
    waves: Array.from({ length: n }, () => []),
    lastRaw: new Array(n).fill(0),
    lastEff: new Array(n).fill(0),
    lastRate: new Array(n).fill(0),
  };
}
