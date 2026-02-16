/**
 * useSimEngine – rAF-based simulation loop with perf tracking.
 *
 * Replaces the old setInterval-based loop with requestAnimationFrame for
 * smoother rendering. Accumulates sim time in fixed dt steps while rendering
 * at display refresh rate.
 */
import { useEffect, useRef } from "react";
import { NEURON_COLOR_HEX } from "../core/colors";
import {
  CANVAS_SIZE,
  KERNEL_VIEW_SIZE,
  HISTORY_SEC,
  MAX_NEURONS,
  useSimStore,
  DrawBox,
} from "../core/store";
import { buildHistory, WaveBuffer } from "../core/history";
import { buildActiveNeuronIds } from "../core/neuronSelection";
import {
  buildActiveKernel,
  buildStimulusPatch,
  centeredKernelPreviewRgba,
  m11TensorToRgba,
  placePatchOnCanvas,
  poissonSpikeStep,
  responseFromFrame,
  sampleNoisyJitteredSpikeWaveform,
  generateSpikeWaveformFromRf,
  makeRng,
  StimulusParams,
} from "../core/sim";
import { clamp } from "./plotUtils";

export function useSimEngine() {
  const rngRef = useRef(makeRng(2026));
  const timeRef = useRef(0);
  const autoRef = useRef({ lastCycleIndex: -1, cifarIndex: 0 });
  const audioCtxRef = useRef<AudioContext | null>(null);
  const humOscRef = useRef<OscillatorNode | null>(null);
  const humGainRef = useRef<GainNode | null>(null);
  const rafRef = useRef(0);
  const lastFrameTimeRef = useRef(0);
  const frameCountRef = useRef(0);
  const fpsAccumRef = useRef(0);
  const droppedRef = useRef(0);

  // ---- Audio setup/teardown ----
  useEffect(() => {
    const unsub = useSimStore.subscribe(
      (s) => s.audioEnabled,
      (audioEnabled) => {
        if (!audioEnabled) {
          stopAudio();
        } else if (!audioCtxRef.current) {
          startAudio();
        }
      },
    );

    // Initial
    if (useSimStore.getState().audioEnabled) {
      startAudio();
    }

    return () => {
      unsub();
      stopAudio();
    };
  }, []);

  function startAudio() {
    if (audioCtxRef.current) return;
    const ctx = new AudioContext();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = "sine";
    osc.frequency.value = 120;
    gain.gain.value = 0.0001;
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    audioCtxRef.current = ctx;
    humOscRef.current = osc;
    humGainRef.current = gain;
  }

  function stopAudio() {
    try {
      humOscRef.current?.stop();
    } catch { /* no-op */ }
    humOscRef.current = null;
    humGainRef.current = null;
    if (audioCtxRef.current) void audioCtxRef.current.close();
    audioCtxRef.current = null;
  }

  // ---- Waveform buffer refresh ----
  useEffect(() => {
    const unsub = useSimStore.subscribe(
      (s) => s.bufferRefreshSec,
      (bufferRefreshSec) => {
        // Handled via periodic timer
      },
    );

    let bufferId: number;
    function startBufferRefresh() {
      const ms = Math.max(250, Math.round(useSimStore.getState().bufferRefreshSec * 1000));
      bufferId = window.setInterval(() => {
        useSimStore.setState((prev) => ({
          hist: { ...prev.hist, waves: prev.hist.waves.map(() => []) },
        }));
      }, ms);
    }
    startBufferRefresh();

    return () => {
      unsub();
      window.clearInterval(bufferId);
    };
  }, []);

  // ---- RF scale sync ----
  useEffect(() => {
    const unsub = useSimStore.subscribe(
      (s) => s.rfScalePct,
      () => {
        useSimStore.setState((s) => ({
          neurons: s.neurons.map((n) => {
            const sz = Math.max(5, Math.round((n.baseRfSize * s.rfScalePct) / 100));
            return {
              ...n,
              rfPos: {
                x: clamp(n.rfPos.x, 0, Math.max(0, CANVAS_SIZE - sz)),
                y: clamp(n.rfPos.y, 0, Math.max(0, CANVAS_SIZE - sz)),
              },
            };
          }),
        }));
      },
    );
    return unsub;
  }, []);

  // ---- Main rAF simulation loop ----
  useEffect(() => {
    let running = true;

    function tick(timestamp: number) {
      if (!running) return;

      const s = useSimStore.getState();

      // Perf tracking
      const frameDelta = timestamp - lastFrameTimeRef.current;
      lastFrameTimeRef.current = timestamp;
      fpsAccumRef.current += frameDelta;
      frameCountRef.current += 1;
      if (fpsAccumRef.current >= 1000) {
        const measuredFps = (frameCountRef.current * 1000) / fpsAccumRef.current;
        useSimStore.getState().updatePerf({ fps: measuredFps });
        frameCountRef.current = 0;
        fpsAccumRef.current = 0;
      }

      const dtSec = 1.0 / Math.max(1, s.fps);

      if (!s.paused) {
        // Throttle: only step if enough wall time elapsed
        const expectedMs = dtSec * 1000;
        if (frameDelta < expectedMs * 0.5 && timeRef.current > 0) {
          // Too early – skip step but still rAF
          rafRef.current = requestAnimationFrame(tick);
          return;
        }

        const stepStart = performance.now();
        const nextTime = timeRef.current + dtSec;
        timeRef.current = nextTime;

        const step = computeStep(s, nextTime, true);
        const stepMs = performance.now() - stepStart;

        useSimStore.setState((prev) => {
          const t = [...prev.hist.t, nextTime];
          const rates = prev.hist.rates.map((arr) => [...arr]);
          const spikes = prev.hist.spikes.map((arr) => [...arr]);
          const waves = prev.hist.waves.map((arr) => [...arr]);
          const activeIds = step.activeIds;

          for (let i = 0; i < MAX_NEURONS; i += 1) {
            const isActive = activeIds.includes(i);
            rates[i].push(isActive ? step.rates[i] : 0);
            if (isActive && step.spikes[i] === 1) {
              spikes[i].push(nextTime);
              if (step.waves[i]) {
                waves[i].push(step.waves[i] as WaveBuffer);
                if (waves[i].length > s.spikeBuffer) {
                  waves[i] = waves[i].slice(waves[i].length - s.spikeBuffer);
                }
              }
            }
          }

          const tMin = nextTime - HISTORY_SEC;
          let start = 0;
          for (let i = 0; i < t.length; i += 1) {
            if (t[i] >= tMin) { start = i; break; }
          }

          return {
            tNow: nextTime,
            frameRgba: step.frame,
            kernelRgba: step.kernelPreview,
            drawBoxes: step.draw,
            hist: {
              t: t.slice(start),
              rates: rates.map((arr) => arr.slice(start)),
              spikes: spikes.map((arr) => arr.filter((tt) => tt >= tMin)),
              waves,
              lastRate: step.rates.slice(),
              lastRaw: step.raw.slice(),
              lastEff: step.effective.slice(),
            },
            perf: { ...prev.perf, stepMs },
          };
        });

        // Audio
        handleAudio(s, step);
      }

      rafRef.current = requestAnimationFrame(tick);
    }

    rafRef.current = requestAnimationFrame(tick);

    return () => {
      running = false;
      cancelAnimationFrame(rafRef.current);
    };
  }, []);

  // ---- Compute a single simulation step ----
  function computeStep(
    s: ReturnType<typeof useSimStore.getState>,
    tSec: number,
    sampleSpikes: boolean,
  ) {
    const { timedExperiment, stimKind, onMs, offMs } = s;
    const activeIds = s.activeNeuronIds();

    // Auto-cycle for CIFAR timed mode
    const cycle = timedCycleHelper(tSec, onMs, offMs);
    if (timedExperiment && stimKind === "cifar" && cycle.visible && cycle.cycleIndex !== autoRef.current.lastCycleIndex) {
      autoRef.current.lastCycleIndex = cycle.cycleIndex;
      autoRef.current.cifarIndex = Math.floor(rngRef.current() * s.cifarDataset.length);
    }
    if (!timedExperiment) {
      autoRef.current.lastCycleIndex = -1;
    }

    const stimParams: StimulusParams = {
      kind: s.stimKind,
      center: { x: s.centerX, y: s.centerY },
      orientationDeg: s.orientationDeg,
      phaseDeg: s.phaseDeg,
      phaseSpeedDegPerSec: s.phaseSpeedDegPerSec,
      barLengthPx: s.barLengthPx,
      barThicknessPx: s.barThicknessPx,
      barSizePx: s.barSizePx,
      barContrast: s.barContrast,
      barColor: s.stimColorBar,
      gratingSfCpp: s.gratingSfCpp,
      gratingSizePx: s.gratingSizePx,
      gratingContrast: s.gratingContrast,
      gratingColor: s.stimColorGrating,
      cifarIndex: s.cifarIndex,
      cifarSizePx: s.cifarSizePx,
      timedExperiment: s.timedExperiment,
      onMs: s.onMs,
      offMs: s.offMs,
      autoBarStepDeg: s.autoBarStepDeg,
      autoGrMode: s.autoGrMode,
      autoGrOriStepDeg: s.autoGrOriStepDeg,
      autoGrSfStepCpp: s.autoGrSfStepCpp,
    };

    const stim = buildStimulusPatch(stimParams, tSec, autoRef.current.cifarIndex, s.cifarDataset);
    const frameM11 = placePatchOnCanvas(CANVAS_SIZE, stim.patch, { x: s.centerX, y: s.centerY }, stim.visible);
    const frame = m11TensorToRgba(frameM11);

    const rates = new Array(MAX_NEURONS).fill(0);
    const raw = new Array(MAX_NEURONS).fill(0);
    const effective = new Array(MAX_NEURONS).fill(0);
    const spikes = new Array(MAX_NEURONS).fill(0);
    const waves: Array<WaveBuffer | null> = new Array(MAX_NEURONS).fill(null);
    const draw: DrawBox[] = [];
    let selectedKernelPreview = s.kernelRgba;

    for (const neuronId of activeIds) {
      const neuron = s.neurons[neuronId];
      const kernel = buildActiveKernel(neuron.kernelBase, {
        scalePct: s.rfScalePct,
        useMask: s.maskRf,
        useGray: s.grayRf,
        grayMatchEnergy: s.grayEnergy,
      });
      const sz = kernel.width;
      const x = clamp(neuron.rfPos.x, 0, Math.max(0, CANVAS_SIZE - sz));
      const y = clamp(neuron.rfPos.y, 0, Math.max(0, CANVAS_SIZE - sz));
      draw.push({ x, y, size: sz, colorHex: NEURON_COLOR_HEX[neuron.color] });

      const out = responseFromFrame(frameM11, kernel, { x, y }, s.responseMode, s.rateGain, s.baselineHz, s.maxRateHz);
      raw[neuronId] = out.raw;
      effective[neuronId] = out.effective;
      rates[neuronId] = out.rateHz;

      if (neuronId === s.activeNeuronIndex) {
        selectedKernelPreview = centeredKernelPreviewRgba(kernel, KERNEL_VIEW_SIZE, 5);
      }

      if (sampleSpikes) {
        const spike = poissonSpikeStep(out.rateHz, 1.0 / Math.max(1, s.fps), rngRef.current);
        spikes[neuronId] = spike;
        if (spike === 1) {
          const base = generateSpikeWaveformFromRf(kernel, 1.5, 20000);
          const sample = sampleNoisyJitteredSpikeWaveform(base, rngRef.current, 0.04, 0.035, 0.08);
          waves[neuronId] = { tMs: Array.from(sample.tMs), amp: Array.from(sample.amp) };
        }
      }
    }

    return { frame, kernelPreview: selectedKernelPreview, draw, rates, raw, effective, spikes, waves, activeIds };
  }

  // ---- Audio handling ----
  function handleAudio(
    s: ReturnType<typeof useSimStore.getState>,
    step: ReturnType<typeof computeStep>,
  ) {
    if (!s.audioEnabled || !audioCtxRef.current || !humOscRef.current || !humGainRef.current) return;

    const ctx = audioCtxRef.current;
    if (ctx.state === "suspended") void ctx.resume();

    const tAudio = ctx.currentTime;
    const activeRates = step.activeIds.map((id) => Number(step.rates[id] ?? 0));
    const r = activeRates.length > 0 ? activeRates.reduce((a, v) => a + v, 0) / activeRates.length : 0;

    humOscRef.current.frequency.setTargetAtTime(100 + r * 3.5, tAudio, 0.05);
    humGainRef.current.gain.setTargetAtTime(0.0001 + Math.min(0.02, r / 4000), tAudio, 0.08);

    if (step.activeIds.some((id) => step.spikes[id] === 1)) {
      const clickOsc = ctx.createOscillator();
      const clickGain = ctx.createGain();
      clickOsc.type = "triangle";
      clickOsc.frequency.setValueAtTime(1100, tAudio);
      clickOsc.frequency.exponentialRampToValueAtTime(650, tAudio + 0.015);
      clickGain.gain.setValueAtTime(0.0001, tAudio);
      clickGain.gain.exponentialRampToValueAtTime(0.018, tAudio + 0.002);
      clickGain.gain.exponentialRampToValueAtTime(0.0001, tAudio + 0.02);
      clickOsc.connect(clickGain);
      clickGain.connect(ctx.destination);
      clickOsc.start(tAudio);
      clickOsc.stop(tAudio + 0.03);
    }
  }
}

function timedCycleHelper(tSec: number, onMs: number, offMs: number) {
  const onS = Math.max(0.001, onMs / 1000);
  const offS = Math.max(0.001, offMs / 1000);
  const period = onS + offS;
  const cycleIndex = Math.floor(tSec / period);
  const phase = tSec - cycleIndex * period;
  const visible = phase < onS;
  return { cycleIndex, visible };
}
