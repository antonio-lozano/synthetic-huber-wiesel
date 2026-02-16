import { describe, expect, it } from "vitest";
import { buildActiveNeuronIds } from "./neuronSelection";
import { responseFromFrame, timedCycle } from "./sim";
import { RgbTensor } from "./types";

function tensor1x1(rgb: [number, number, number]): RgbTensor {
  return {
    width: 1,
    height: 1,
    data: new Float32Array(rgb),
  };
}

describe("buildActiveNeuronIds", () => {
  it("wraps indices and respects count", () => {
    expect(buildActiveNeuronIds(8, 4, 10)).toEqual([8, 9, 0, 1]);
  });

  it("clamps invalid count and start values", () => {
    expect(buildActiveNeuronIds(-1, 0, 5)).toEqual([4]);
    expect(buildActiveNeuronIds(2, 99, 5)).toEqual([2, 3, 4, 0, 1]);
  });
});

describe("timedCycle", () => {
  it("reports visibility and cycle index", () => {
    const c0 = timedCycle(0.15, 200, 300);
    const c1 = timedCycle(0.35, 200, 300);
    const c2 = timedCycle(1.05, 200, 300);

    expect(c0.visible).toBe(true);
    expect(c1.visible).toBe(false);
    expect(c2.cycleIndex).toBe(2);
  });
});

describe("responseFromFrame", () => {
  it("produces bounded normalized rates", () => {
    const frame = tensor1x1([1, -1, -1]);
    const kernel = tensor1x1([1, -1, -1]);
    const out = responseFromFrame(frame, kernel, { x: 0, y: 0 }, "normalized", 5, 0, 100);
    expect(out.rateHz).toBeGreaterThan(10);
    expect(out.rateHz).toBeLessThanOrEqual(100);
  });

  it("keeps legacy rate unbounded by maxRate", () => {
    const frame = tensor1x1([1, -1, -1]);
    const kernel = tensor1x1([1, -1, -1]);
    const out = responseFromFrame(frame, kernel, { x: 0, y: 0 }, "legacy", 2, 0, 1);
    expect(out.rateHz).toBeGreaterThan(1);
  });
});
