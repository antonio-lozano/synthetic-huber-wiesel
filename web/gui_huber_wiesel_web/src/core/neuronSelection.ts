export function buildActiveNeuronIds(startIndex: number, count: number, maxNeurons: number): number[] {
  const maxN = Math.max(1, Math.floor(maxNeurons));
  const n = Math.max(1, Math.min(maxN, Math.floor(count)));
  const start = ((Math.floor(startIndex) % maxN) + maxN) % maxN;
  return Array.from({ length: n }, (_, i) => (start + i) % maxN);
}
