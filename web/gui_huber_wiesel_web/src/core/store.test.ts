/**
 * Tests for the Zustand store: actions, presets, neuron toggling, derived state.
 */
import { afterEach, describe, expect, it } from "vitest";
import {
  useSimStore,
  BUILTIN_PRESETS,
  MAX_NEURONS,
  CANVAS_SIZE,
  Preset,
} from "./store";

/** Reset store to defaults between tests. */
function resetStore() {
  useSimStore.setState(useSimStore.getInitialState());
}

afterEach(resetStore);

// ---------------------------------------------------------------------------
// applyPreset
// ---------------------------------------------------------------------------
describe("applyPreset", () => {
  it("applies all preset fields to the store", () => {
    const preset = BUILTIN_PRESETS[1]; // Grating orientation sweep
    useSimStore.getState().applyPreset(preset);

    const s = useSimStore.getState();
    expect(s.stimKind).toBe(preset.stimKind);
    expect(s.orientationDeg).toBe(preset.orientationDeg);
    expect(s.timedExperiment).toBe(preset.timedExperiment);
    expect(s.onMs).toBe(preset.onMs);
    expect(s.offMs).toBe(preset.offMs);
    expect(s.responseMode).toBe(preset.responseMode);
    expect(s.maxRateHz).toBe(preset.maxRateHz);
    expect(s.rateGain).toBe(preset.rateGain);
    expect(s.baselineHz).toBe(preset.baselineHz);
    expect(s.fps).toBe(preset.fps);
    expect(s.gratingSfCpp).toBe(preset.gratingSfCpp);
    expect(s.gratingSizePx).toBe(preset.gratingSizePx);
  });

  it("applies each builtin preset without throwing", () => {
    for (const preset of BUILTIN_PRESETS) {
      expect(() => useSimStore.getState().applyPreset(preset)).not.toThrow();
    }
  });
});

// ---------------------------------------------------------------------------
// saveCustomPreset / deleteCustomPreset
// ---------------------------------------------------------------------------
describe("custom presets", () => {
  it("saves and retrieves a custom preset", () => {
    useSimStore.getState().saveCustomPreset("My preset");
    const s = useSimStore.getState();
    expect(s.customPresets).toHaveLength(1);
    expect(s.customPresets[0].name).toBe("My preset");
    expect(s.customPresets[0].stimKind).toBe(s.stimKind);
  });

  it("deletes a custom preset by index", () => {
    useSimStore.getState().saveCustomPreset("A");
    useSimStore.getState().saveCustomPreset("B");
    expect(useSimStore.getState().customPresets).toHaveLength(2);

    useSimStore.getState().deleteCustomPreset(0);
    const s = useSimStore.getState();
    expect(s.customPresets).toHaveLength(1);
    expect(s.customPresets[0].name).toBe("B");
  });

  it("produces a preset with all expected fields", () => {
    useSimStore.getState().saveCustomPreset("test");
    const p = useSimStore.getState().customPresets[0];
    const requiredKeys: (keyof Preset)[] = [
      "name", "stimKind", "orientationDeg", "barLengthPx", "barThicknessPx",
      "barSizePx", "barContrast", "stimColorBar", "gratingSfCpp", "gratingSizePx",
      "gratingContrast", "stimColorGrating", "phaseDeg", "phaseSpeedDegPerSec",
      "timedExperiment", "onMs", "offMs", "autoBarStepDeg", "autoGrMode",
      "autoGrOriStepDeg", "autoGrSfStepCpp", "responseMode", "maxRateHz",
      "rateGain", "baselineHz", "fps",
    ];
    for (const key of requiredKeys) {
      expect(p).toHaveProperty(key);
    }
  });
});

// ---------------------------------------------------------------------------
// toggleNeuronEnabled
// ---------------------------------------------------------------------------
describe("toggleNeuronEnabled", () => {
  it("toggles a specific neuron", () => {
    const before = useSimStore.getState().neuronEnabled[2];
    useSimStore.getState().toggleNeuronEnabled(2);
    const after = useSimStore.getState().neuronEnabled[2];
    expect(after).toBe(!before);
  });

  it("does not affect other neurons", () => {
    const before = [...useSimStore.getState().neuronEnabled];
    useSimStore.getState().toggleNeuronEnabled(3);
    const after = useSimStore.getState().neuronEnabled;
    for (let i = 0; i < MAX_NEURONS; i++) {
      if (i === 3) continue;
      expect(after[i]).toBe(before[i]);
    }
  });
});

// ---------------------------------------------------------------------------
// activeNeuronIds (derived)
// ---------------------------------------------------------------------------
describe("activeNeuronIds", () => {
  it("returns active neurons filtered by neuronEnabled", () => {
    useSimStore.setState({ simNeurons: 3, activeNeuronIndex: 0 });
    // All enabled by default → [0,1,2]
    expect(useSimStore.getState().activeNeuronIds()).toEqual([0, 1, 2]);

    // Disable neuron 1
    useSimStore.getState().toggleNeuronEnabled(1);
    const ids = useSimStore.getState().activeNeuronIds();
    expect(ids).not.toContain(1);
    expect(ids).toContain(0);
    expect(ids).toContain(2);
  });
});

// ---------------------------------------------------------------------------
// togglePause / resetTraces
// ---------------------------------------------------------------------------
describe("togglePause", () => {
  it("flips the paused state", () => {
    expect(useSimStore.getState().paused).toBe(false);
    useSimStore.getState().togglePause();
    expect(useSimStore.getState().paused).toBe(true);
    useSimStore.getState().togglePause();
    expect(useSimStore.getState().paused).toBe(false);
  });
});

describe("resetTraces", () => {
  it("resets history and tNow to zero", () => {
    useSimStore.setState({ tNow: 42 });
    useSimStore.getState().resetTraces();
    expect(useSimStore.getState().tNow).toBe(0);
    expect(useSimStore.getState().hist.t).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// setNeuronRfPos
// ---------------------------------------------------------------------------
describe("setNeuronRfPos", () => {
  it("updates the RF position of a specific neuron", () => {
    useSimStore.getState().setNeuronRfPos(0, 100, 200);
    const pos = useSimStore.getState().neurons[0].rfPos;
    expect(pos.x).toBe(100);
    expect(pos.y).toBe(200);
  });

  it("does not affect other neurons", () => {
    const before = useSimStore.getState().neurons[1].rfPos;
    useSimStore.getState().setNeuronRfPos(0, 50, 50);
    const after = useSimStore.getState().neurons[1].rfPos;
    expect(after.x).toBe(before.x);
    expect(after.y).toBe(before.y);
  });
});

// ---------------------------------------------------------------------------
// updatePerf
// ---------------------------------------------------------------------------
describe("updatePerf", () => {
  it("merges partial perf stats", () => {
    useSimStore.getState().updatePerf({ fps: 59.5, stepMs: 2.1 });
    const p = useSimStore.getState().perf;
    expect(p.fps).toBe(59.5);
    expect(p.stepMs).toBe(2.1);
  });

  it("preserves unaffected perf fields", () => {
    useSimStore.getState().updatePerf({ fps: 60 });
    useSimStore.getState().updatePerf({ stepMs: 3 });
    const p = useSimStore.getState().perf;
    expect(p.fps).toBe(60);
    expect(p.stepMs).toBe(3);
  });
});

// ---------------------------------------------------------------------------
// set (generic setter)
// ---------------------------------------------------------------------------
describe("set (generic partial state)", () => {
  it("updates arbitrary fields", () => {
    useSimStore.getState().set({ orientationDeg: 90, barContrast: 0.5 });
    const s = useSimStore.getState();
    expect(s.orientationDeg).toBe(90);
    expect(s.barContrast).toBe(0.5);
  });
});
