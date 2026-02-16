/**
 * PerfHUD – small overlay showing FPS, step time, dropped frames.
 */
import { useSimStore } from "../core/store";

export function PerfHUD() {
  const perf = useSimStore((s) => s.perf);

  return (
    <div className="perf-hud" role="status" aria-label="Performance statistics">
      <span>FPS: {perf.fps.toFixed(0)}</span>
      <span>Step: {perf.stepMs.toFixed(1)}ms</span>
      <span>Dropped: {perf.droppedFrames}</span>
    </div>
  );
}
