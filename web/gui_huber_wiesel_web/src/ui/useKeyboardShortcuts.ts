/**
 * useKeyboardShortcuts – global keyboard shortcuts for the simulation.
 *
 * Space  = pause/resume
 * R      = reset traces
 * N      = next neuron
 * P      = previous neuron
 * 1-3    = switch stimulus type (bar/grating/cifar)
 * T      = toggle timed experiment
 */
import { useEffect } from "react";
import { useSimStore, MAX_NEURONS } from "../core/store";

export function useKeyboardShortcuts() {
  useEffect(() => {
    function handler(e: KeyboardEvent) {
      // Don't capture when typing in an input
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;

      const store = useSimStore.getState();

      switch (e.key) {
        case " ":
          e.preventDefault();
          store.togglePause();
          break;
        case "r":
        case "R":
          store.resetTraces();
          break;
        case "n":
        case "N":
          store.set({
            activeNeuronIndex: (store.activeNeuronIndex + 1) % MAX_NEURONS,
          });
          break;
        case "p":
        case "P":
          store.set({
            activeNeuronIndex:
              (store.activeNeuronIndex - 1 + MAX_NEURONS) % MAX_NEURONS,
          });
          break;
        case "1":
          store.set({ stimKind: "bar" });
          break;
        case "2":
          store.set({ stimKind: "grating" });
          break;
        case "3":
          store.set({ stimKind: "cifar" });
          break;
        case "t":
        case "T":
          store.set({ timedExperiment: !store.timedExperiment });
          break;
      }
    }

    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);
}
