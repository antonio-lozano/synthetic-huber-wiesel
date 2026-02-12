import { NEURON_COLOR_ORDER } from "./colors";
import {
  GratingAutoMode,
  ResponseMode,
  RgbTensor,
  SimNeuron,
  SpikeWaveform,
  StimulusColor,
  StimulusKind,
  Vec2,
} from "./types";

type Rng = () => number;

export type StimulusParams = {
  kind: StimulusKind;
  center: Vec2;
  orientationDeg: number;
  phaseDeg: number;
  phaseSpeedDegPerSec: number;
  barLengthPx: number;
  barThicknessPx: number;
  barSizePx: number;
  barContrast: number;
  barColor: StimulusColor;
  gratingSfCpp: number;
  gratingSizePx: number;
  gratingContrast: number;
  gratingColor: StimulusColor;
  cifarIndex: number;
  cifarSizePx: number;
  timedExperiment: boolean;
  onMs: number;
  offMs: number;
  autoBarStepDeg: number;
  autoGrMode: GratingAutoMode;
  autoGrOriStepDeg: number;
  autoGrSfStepCpp: number;
};

export type KernelPostprocessParams = {
  scalePct: number;
  useMask: boolean;
  useGray: boolean;
  grayMatchEnergy: boolean;
  maskRadiusRatio?: number;
  maskSoftEdgePx?: number;
};

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

function wrap360(v: number): number {
  return ((v % 360) + 360) % 360;
}

function wrap180(v: number): number {
  return ((v % 180) + 180) % 180;
}

function rgbIndex(width: number, x: number, y: number, c: number): number {
  return (y * width + x) * 3 + c;
}

function mulberry32(seed: number): Rng {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let x = Math.imul(t ^ (t >>> 15), 1 | t);
    x ^= x + Math.imul(x ^ (x >>> 7), 61 | x);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

function randUniform(rng: Rng, lo: number, hi: number): number {
  return lo + (hi - lo) * rng();
}

function randNormal(rng: Rng): number {
  const u1 = Math.max(1e-12, rng());
  const u2 = rng();
  const mag = Math.sqrt(-2.0 * Math.log(u1));
  return mag * Math.cos(2.0 * Math.PI * u2);
}

function imageMean(img: RgbTensor): number {
  let s = 0;
  for (let i = 0; i < img.data.length; i += 1) s += img.data[i];
  return s / Math.max(1, img.data.length);
}

function imageStd(img: RgbTensor, mean?: number): number {
  const m = mean ?? imageMean(img);
  let ss = 0;
  for (let i = 0; i < img.data.length; i += 1) {
    const d = img.data[i] - m;
    ss += d * d;
  }
  return Math.sqrt(ss / Math.max(1, img.data.length));
}

function imageMin(img: RgbTensor): number {
  let m = Number.POSITIVE_INFINITY;
  for (let i = 0; i < img.data.length; i += 1) {
    if (img.data[i] < m) m = img.data[i];
  }
  return Number.isFinite(m) ? m : 0;
}

function imageMax(img: RgbTensor): number {
  let m = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < img.data.length; i += 1) {
    if (img.data[i] > m) m = img.data[i];
  }
  return Number.isFinite(m) ? m : 1;
}

function createRgbTensor(width: number, height: number, fillValue = -1): RgbTensor {
  return {
    width,
    height,
    data: new Float32Array(width * height * 3).fill(fillValue),
  };
}

function cloneRgbTensor(src: RgbTensor): RgbTensor {
  return {
    width: src.width,
    height: src.height,
    data: src.data.slice(),
  };
}

function normalizeToMinus1Plus1(x: Float32Array): Float32Array {
  let minV = Number.POSITIVE_INFINITY;
  let maxV = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < x.length; i += 1) {
    if (x[i] < minV) minV = x[i];
    if (x[i] > maxV) maxV = x[i];
  }
  const out = new Float32Array(x.length);
  const den = Math.max(1e-8, maxV - minV);
  for (let i = 0; i < x.length; i += 1) {
    out[i] = ((x[i] - minV) / den) * 2.0 - 1.0;
  }
  return out;
}

function gaborKernel(size: number, frequency: number, thetaRad: number): Float32Array {
  const out = new Float32Array(size * size);
  const cx = (size - 1) * 0.5;
  const cy = (size - 1) * 0.5;
  const sigma = size / 4.0;
  const inv2sigma2 = 1.0 / Math.max(1e-8, 2.0 * sigma * sigma);
  const ct = Math.cos(thetaRad);
  const st = Math.sin(thetaRad);

  let i = 0;
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const xx = x - cx;
      const yy = y - cy;
      const xr = xx * ct + yy * st;
      const yr = -xx * st + yy * ct;
      const env = Math.exp(-(xr * xr + yr * yr) * inv2sigma2);
      const carr = Math.cos(2.0 * Math.PI * frequency * xr);
      out[i] = env * carr;
      i += 1;
    }
  }
  return normalizeToMinus1Plus1(out);
}

function bilinearSample(img: RgbTensor, x: number, y: number, c: number): number {
  const x0 = Math.floor(clamp(x, 0, img.width - 1));
  const y0 = Math.floor(clamp(y, 0, img.height - 1));
  const x1 = Math.min(img.width - 1, x0 + 1);
  const y1 = Math.min(img.height - 1, y0 + 1);
  const dx = x - x0;
  const dy = y - y0;

  const v00 = img.data[rgbIndex(img.width, x0, y0, c)];
  const v10 = img.data[rgbIndex(img.width, x1, y0, c)];
  const v01 = img.data[rgbIndex(img.width, x0, y1, c)];
  const v11 = img.data[rgbIndex(img.width, x1, y1, c)];

  const vx0 = v00 * (1.0 - dx) + v10 * dx;
  const vx1 = v01 * (1.0 - dx) + v11 * dx;
  return vx0 * (1.0 - dy) + vx1 * dy;
}

export function resizeKernelRgb(kernel: RgbTensor, newSize: number): RgbTensor {
  const size = Math.max(3, Math.round(newSize));
  if (size === kernel.width && size === kernel.height) {
    return cloneRgbTensor(kernel);
  }
  const out = createRgbTensor(size, size, 0);
  const sx = (kernel.width - 1) / Math.max(1, size - 1);
  const sy = (kernel.height - 1) / Math.max(1, size - 1);
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const xx = x * sx;
      const yy = y * sy;
      for (let c = 0; c < 3; c += 1) {
        out.data[rgbIndex(size, x, y, c)] = bilinearSample(kernel, xx, yy, c);
      }
    }
  }

  const m0 = imageMean(kernel);
  const s0 = imageStd(kernel, m0);
  const m1 = imageMean(out);
  const s1 = imageStd(out, m1);
  if (s1 > 1e-8 && s0 > 0) {
    const scale = s0 / s1;
    for (let i = 0; i < out.data.length; i += 1) {
      out.data[i] = (out.data[i] - m1) * scale + m0;
    }
  } else {
    for (let i = 0; i < out.data.length; i += 1) {
      out.data[i] = out.data[i] - m1 + m0;
    }
  }
  return out;
}

export function toGrayscaleKernel(kernel: RgbTensor, matchEnergy: boolean): RgbTensor {
  const out = createRgbTensor(kernel.width, kernel.height, 0);
  for (let y = 0; y < kernel.height; y += 1) {
    for (let x = 0; x < kernel.width; x += 1) {
      let g = 0;
      for (let c = 0; c < 3; c += 1) {
        g += kernel.data[rgbIndex(kernel.width, x, y, c)];
      }
      g /= 3.0;
      for (let c = 0; c < 3; c += 1) {
        out.data[rgbIndex(out.width, x, y, c)] = g;
      }
    }
  }
  if (!matchEnergy) return out;

  const km = imageMean(kernel);
  const gm = imageMean(out);
  let kn = 0;
  let gn = 0;
  for (let i = 0; i < out.data.length; i += 1) {
    const kd = kernel.data[i] - km;
    const gd = out.data[i] - gm;
    kn += kd * kd;
    gn += gd * gd;
  }
  const scale = gn > 1e-8 ? Math.sqrt(kn / gn) : 1.0;
  for (let i = 0; i < out.data.length; i += 1) {
    out.data[i] = (out.data[i] - gm) * scale + km;
  }
  return out;
}

export function makeCircularMask(size: number, radiusRatio = 0.45, softEdgePx = 1.5): Float32Array {
  const out = new Float32Array(size * size);
  const cx = (size - 1) * 0.5;
  const cy = (size - 1) * 0.5;
  const radius = radiusRatio * size;
  let i = 0;
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const dx = x - cx;
      const dy = y - cy;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (softEdgePx <= 0) {
        out[i] = dist <= radius ? 1.0 : 0.0;
      } else if (dist <= radius) {
        out[i] = 1.0;
      } else if (dist >= radius + softEdgePx) {
        out[i] = 0.0;
      } else {
        const t = clamp((dist - radius) / softEdgePx, 0, 1);
        out[i] = 0.5 * (1.0 + Math.cos(Math.PI * t));
      }
      i += 1;
    }
  }
  return out;
}

export function applyRfGrayOutside(
  kernel: RgbTensor,
  grayValue = 0.0,
  radiusRatio = 0.45,
  softEdgePx = 1.5,
): RgbTensor {
  const out = cloneRgbTensor(kernel);
  const mask = makeCircularMask(kernel.width, radiusRatio, softEdgePx);
  for (let y = 0; y < kernel.height; y += 1) {
    for (let x = 0; x < kernel.width; x += 1) {
      const m = mask[y * kernel.width + x];
      for (let c = 0; c < 3; c += 1) {
        const i = rgbIndex(kernel.width, x, y, c);
        out.data[i] = kernel.data[i] * m + grayValue * (1.0 - m);
      }
    }
  }
  return out;
}

export function buildActiveKernel(base: RgbTensor, params: KernelPostprocessParams): RgbTensor {
  const targetSize = Math.max(5, Math.round((base.width * params.scalePct) / 100));
  let out = resizeKernelRgb(base, targetSize);
  if (params.useGray) {
    out = toGrayscaleKernel(out, params.grayMatchEnergy);
  }
  if (params.useMask) {
    out = applyRfGrayOutside(
      out,
      0.0,
      params.maskRadiusRatio ?? 0.45,
      params.maskSoftEdgePx ?? 1.5,
    );
  }
  return out;
}

export function createNeurons(n: number, canvasSize: number, seed = 2026): SimNeuron[] {
  const rng = mulberry32(seed);
  const neurons: SimNeuron[] = [];
  const baseSizes = [19, 23, 27, 31, 35];
  for (let i = 0; i < n; i += 1) {
    const size = baseSizes[i % baseSizes.length];
    const theta = randUniform(rng, 0, Math.PI);
    const frequency = randUniform(rng, 0.08, 0.28);
    const g = gaborKernel(size, frequency, theta);
    const w0 = randNormal(rng);
    const w1 = randNormal(rng);
    const w2 = randNormal(rng);
    const kernel = createRgbTensor(size, size, 0);
    for (let y = 0; y < size; y += 1) {
      for (let x = 0; x < size; x += 1) {
        const v = g[y * size + x];
        kernel.data[rgbIndex(size, x, y, 0)] = v * w0;
        kernel.data[rgbIndex(size, x, y, 1)] = v * w1;
        kernel.data[rgbIndex(size, x, y, 2)] = v * w2;
      }
    }
    const minV = imageMin(kernel);
    const maxV = imageMax(kernel);
    const den = Math.max(1e-8, maxV - minV);
    for (let k = 0; k < kernel.data.length; k += 1) {
      kernel.data[k] = ((kernel.data[k] - minV) / den) * 2.0 - 1.0;
    }

    const targetMean = randUniform(rng, 0.1, 0.5);
    const m = imageMean(kernel);
    for (let k = 0; k < kernel.data.length; k += 1) {
      kernel.data[k] += targetMean - m;
    }

    const x = Math.round(canvasSize * 0.2 + i * 24);
    const y = Math.round(canvasSize * 0.2 + i * 18);
    const maxRf = Math.max(0, canvasSize - size);
    neurons.push({
      id: i,
      color: NEURON_COLOR_ORDER[i % NEURON_COLOR_ORDER.length],
      rfPos: { x: clamp(x, 0, maxRf), y: clamp(y, 0, maxRf) },
      rfSize: size,
      baseRfSize: size,
      orientationDeg: wrap180((theta * 180) / Math.PI),
      frequency,
      kernelBase: kernel,
    });
  }
  return neurons;
}

export function colorNameToM11(name: StimulusColor): [number, number, number] {
  switch (name) {
    case "red":
      return [1, -1, -1];
    case "green":
      return [-1, 1, -1];
    case "blue":
      return [-1, -1, 1];
    case "yellow":
      return [1, 1, -1];
    case "cyan":
      return [-1, 1, 1];
    case "magenta":
      return [1, -1, 1];
    case "white":
    default:
      return [1, 1, 1];
  }
}

export function generateBarStimulus(
  size: number,
  orientationDeg: number,
  lengthPx: number,
  thicknessPx: number,
  contrast: number,
  color: StimulusColor,
): RgbTensor {
  const out = createRgbTensor(size, size, -1);
  const center = (size - 1) * 0.5;
  const theta = (orientationDeg * Math.PI) / 180.0;
  const dx = Math.cos(theta);
  const dy = Math.sin(theta);
  const fg = colorNameToM11(color);
  const c = clamp(contrast, 0, 1);
  const lengthHalf = lengthPx * 0.5;
  const thickHalf = thicknessPx * 0.5;

  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const xx = x - center;
      const yy = y - center;
      const proj = xx * dx + yy * dy;
      const perp = -xx * dy + yy * dx;
      if (Math.abs(proj) <= lengthHalf && Math.abs(perp) <= thickHalf) {
        for (let ch = 0; ch < 3; ch += 1) {
          const i = rgbIndex(size, x, y, ch);
          out.data[i] = -1.0 + c * (fg[ch] + 1.0);
        }
      }
    }
  }
  return out;
}

export function generateGratingStimulus(
  size: number,
  orientationDeg: number,
  spatialFrequencyCpp: number,
  phaseDeg: number,
  contrast: number,
  color: StimulusColor,
): RgbTensor {
  const out = createRgbTensor(size, size, -1);
  const center = (size - 1) * 0.5;
  const theta = (orientationDeg * Math.PI) / 180.0;
  const phase = (phaseDeg * Math.PI) / 180.0;
  const ct = Math.cos(theta);
  const st = Math.sin(theta);
  const fg = colorNameToM11(color);
  const c = clamp(contrast, 0, 1);

  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const xx = x - center;
      const yy = y - center;
      const xRot = xx * ct + yy * st;
      const grat = Math.cos(2.0 * Math.PI * spatialFrequencyCpp * xRot + phase);
      const g = clamp(grat * c, -1, 1);
      const w = (g + 1.0) * 0.5;
      for (let ch = 0; ch < 3; ch += 1) {
        const i = rgbIndex(size, x, y, ch);
        out.data[i] = -1.0 + w * (fg[ch] + 1.0);
      }
    }
  }
  return out;
}

function pseudoCifarImage(index: number, size = 32): RgbTensor {
  const rng = mulberry32(1931 + index * 17);
  const out = createRgbTensor(size, size, 0);
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const xx = (x - size / 2) / size;
      const yy = (y - size / 2) / size;
      const v0 = 0.35 * Math.sin(10.0 * xx + 3.0 * yy + randUniform(rng, 0, 6.28));
      const v1 = 0.35 * Math.cos(8.0 * yy - 2.2 * xx + randUniform(rng, 0, 6.28));
      const noise = randUniform(rng, -0.25, 0.25);
      const base = clamp(v0 + v1 + noise, -1, 1);
      const tint = [randUniform(rng, -0.6, 0.6), randUniform(rng, -0.6, 0.6), randUniform(rng, -0.6, 0.6)];
      for (let c = 0; c < 3; c += 1) {
        out.data[rgbIndex(size, x, y, c)] = clamp(base + 0.35 * tint[c], -1, 1);
      }
    }
  }
  return out;
}

export function generatePseudoCifarDataset(count: number): RgbTensor[] {
  const out: RgbTensor[] = [];
  for (let i = 0; i < count; i += 1) {
    out.push(pseudoCifarImage(i, 32));
  }
  return out;
}

export function generateCifarPatchStimulus(
  dataset: RgbTensor[],
  imageIndex: number,
  size: number,
): RgbTensor {
  const idx = clamp(Math.round(imageIndex), 0, dataset.length - 1);
  const src = dataset[idx];
  const resized = resizeKernelRgb(src, size);
  // Keep vertical orientation aligned with desktop GUI display.
  const out = createRgbTensor(size, size, 0);
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const yy = size - 1 - y;
      for (let c = 0; c < 3; c += 1) {
        out.data[rgbIndex(size, x, y, c)] = resized.data[rgbIndex(size, x, yy, c)];
      }
    }
  }
  return out;
}

export function placePatchOnCanvas(
  canvasSize: number,
  patch: RgbTensor,
  center: Vec2,
  visible: boolean,
): RgbTensor {
  const out = createRgbTensor(canvasSize, canvasSize, -1);
  if (!visible) return out;

  const x0 = Math.round(center.x - patch.width / 2);
  const y0 = Math.round(center.y - patch.height / 2);
  for (let y = 0; y < patch.height; y += 1) {
    const oy = y0 + y;
    if (oy < 0 || oy >= canvasSize) continue;
    for (let x = 0; x < patch.width; x += 1) {
      const ox = x0 + x;
      if (ox < 0 || ox >= canvasSize) continue;
      for (let c = 0; c < 3; c += 1) {
        out.data[rgbIndex(canvasSize, ox, oy, c)] = patch.data[rgbIndex(patch.width, x, y, c)];
      }
    }
  }
  return out;
}

function dotCentered(patch: Float32Array, kernel: Float32Array): number {
  let pm = 0;
  let km = 0;
  for (let i = 0; i < patch.length; i += 1) {
    pm += patch[i];
    km += kernel[i];
  }
  pm /= Math.max(1, patch.length);
  km /= Math.max(1, kernel.length);
  let num = 0;
  let pn = 0;
  let kn = 0;
  for (let i = 0; i < patch.length; i += 1) {
    const p = patch[i] - pm;
    const k = kernel[i] - km;
    num += p * k;
    pn += p * p;
    kn += k * k;
  }
  const den = Math.sqrt(Math.max(1e-8, pn * kn));
  return den > 1e-8 ? num / den : 0;
}

export function computeSingleFrameResponse(
  frame: RgbTensor,
  kernel: RgbTensor,
  rfPos: Vec2,
): number {
  const s = kernel.width;
  const x0 = clamp(Math.round(rfPos.x), 0, Math.max(0, frame.width - s));
  const y0 = clamp(Math.round(rfPos.y), 0, Math.max(0, frame.height - s));
  let sum = 0;
  for (let y = 0; y < s; y += 1) {
    for (let x = 0; x < s; x += 1) {
      for (let c = 0; c < 3; c += 1) {
        const f = frame.data[rgbIndex(frame.width, x0 + x, y0 + y, c)];
        const k = kernel.data[rgbIndex(s, x, y, c)];
        sum += f * k;
      }
    }
  }
  return sum;
}

export function computeSingleFrameResponseNormalized(
  frame: RgbTensor,
  kernel: RgbTensor,
  rfPos: Vec2,
): number {
  const s = kernel.width;
  const x0 = clamp(Math.round(rfPos.x), 0, Math.max(0, frame.width - s));
  const y0 = clamp(Math.round(rfPos.y), 0, Math.max(0, frame.height - s));
  const n = s * s * 3;
  const patch = new Float32Array(n);
  let i = 0;
  for (let y = 0; y < s; y += 1) {
    for (let x = 0; x < s; x += 1) {
      for (let c = 0; c < 3; c += 1) {
        patch[i] = frame.data[rgbIndex(frame.width, x0 + x, y0 + y, c)];
        i += 1;
      }
    }
  }
  return dotCentered(patch, kernel.data);
}

export function responseToRateHz(response: number, gain = 2.0, baselineHz = 0): number {
  return Math.max(0, baselineHz + gain * response);
}

export function responseToRateHzBounded(response: number, maxRateHz = 100, baselineHz = 0): number {
  const maxR = Math.max(1e-6, maxRateHz);
  const baseline = clamp(baselineHz, 0, maxR);
  const pos = Math.max(0, clamp(response, -1, 1));
  return clamp(baseline + (maxR - baseline) * pos, 0, maxR);
}

export function poissonSpikeStep(rateHz: number, dtSec: number, rng: Rng): number {
  const p = clamp(rateHz * dtSec, 0, 1);
  return rng() < p ? 1 : 0;
}

export function generateSpikeWaveformFromRf(
  kernel: RgbTensor,
  durationMs = 1.5,
  sampleRateHz = 20000,
): SpikeWaveform {
  const n = Math.max(8, Math.round((sampleRateHz * durationMs) / 1000));
  const t = new Float32Array(n);
  const w = new Float32Array(n);
  const m = imageMean(kernel);
  const s = imageStd(kernel, m);
  let energy = 0;
  let skewLike = 0;
  for (let i = 0; i < kernel.data.length; i += 1) {
    const d = kernel.data[i] - m;
    energy += Math.abs(kernel.data[i]);
    skewLike += d * d * d;
  }
  energy /= Math.max(1, kernel.data.length);
  skewLike /= Math.max(1, kernel.data.length);

  const widthScale = clamp(0.75 + 1.25 * s, 0.6, 1.8);
  const asym = clamp(Math.tanh(2.0 * skewLike), -0.35, 0.35);
  const negCenter = durationMs * (0.38 + 0.08 * asym);
  const posCenter = durationMs * (0.78 + 0.06 * asym);
  const negSigma = durationMs * 0.09 * widthScale;
  const posSigma = durationMs * 0.12 * widthScale;
  const posAmp = 0.55 + 0.35 * energy;

  let maxAbs = 1e-8;
  for (let i = 0; i < n; i += 1) {
    const tm = (i / Math.max(1, n - 1)) * durationMs;
    t[i] = tm;
    const neg = -Math.exp(-0.5 * ((tm - negCenter) / Math.max(1e-6, negSigma)) ** 2);
    const pos = posAmp * Math.exp(-0.5 * ((tm - posCenter) / Math.max(1e-6, posSigma)) ** 2);
    w[i] = neg + pos;
    maxAbs = Math.max(maxAbs, Math.abs(w[i]));
  }
  for (let i = 0; i < n; i += 1) {
    w[i] /= maxAbs;
  }
  return { tMs: t, amp: w };
}

export function sampleNoisyJitteredSpikeWaveform(
  base: SpikeWaveform,
  rng: Rng,
  jitterStdMs = 0.04,
  noiseStd = 0.035,
  ampJitterFrac = 0.08,
): SpikeWaveform {
  const t = base.tMs.slice();
  const w = base.amp.slice();
  const jitter = randNormal(rng) * jitterStdMs;
  const ampScale = 1.0 + randNormal(rng) * ampJitterFrac;
  for (let i = 0; i < t.length; i += 1) {
    t[i] += jitter;
    w[i] = ampScale * w[i] + randNormal(rng) * noiseStd;
  }
  return { tMs: t, amp: w };
}

export function m11TensorToRgba(img: RgbTensor): Uint8ClampedArray {
  const out = new Uint8ClampedArray(img.width * img.height * 4);
  let j = 0;
  for (let i = 0; i < img.data.length; i += 3) {
    out[j] = clamp(Math.round(((clamp(img.data[i], -1, 1) + 1.0) * 0.5) * 255), 0, 255);
    out[j + 1] = clamp(Math.round(((clamp(img.data[i + 1], -1, 1) + 1.0) * 0.5) * 255), 0, 255);
    out[j + 2] = clamp(Math.round(((clamp(img.data[i + 2], -1, 1) + 1.0) * 0.5) * 255), 0, 255);
    out[j + 3] = 255;
    j += 4;
  }
  return out;
}

export function centeredKernelPreviewRgba(
  kernel: RgbTensor,
  canvasSize: number,
  scale: number,
): Uint8ClampedArray {
  const scaled = resizeKernelRgb(
    kernel,
    Math.max(8, Math.round(Math.min(canvasSize - 8, kernel.width * Math.max(1, scale)))),
  );
  const canvas = createRgbTensor(canvasSize, canvasSize, -1);
  const x0 = Math.floor((canvasSize - scaled.width) / 2);
  const y0 = Math.floor((canvasSize - scaled.height) / 2);
  for (let y = 0; y < scaled.height; y += 1) {
    for (let x = 0; x < scaled.width; x += 1) {
      const ox = x0 + x;
      const oy = y0 + y;
      if (ox < 0 || oy < 0 || ox >= canvasSize || oy >= canvasSize) continue;
      for (let c = 0; c < 3; c += 1) {
        canvas.data[rgbIndex(canvasSize, ox, oy, c)] = scaled.data[rgbIndex(scaled.width, x, y, c)];
      }
    }
  }
  return m11TensorToRgba(canvas);
}

export function updateStimulusPhase(phaseDeg: number, dtSec: number, speedDegPerSec: number): number {
  return wrap360(phaseDeg + speedDegPerSec * dtSec);
}

export function timedCycle(tNowSec: number, onMs: number, offMs: number): {
  cycleIndex: number;
  visible: boolean;
  onPhase: number;
} {
  const onS = Math.max(0.001, onMs / 1000);
  const offS = Math.max(0.001, offMs / 1000);
  const period = onS + offS;
  const cycleIndex = Math.floor(tNowSec / period);
  const phase = tNowSec - cycleIndex * period;
  const visible = phase < onS;
  const onPhase = visible ? clamp(phase / onS, 0, 1) : 0;
  return { cycleIndex, visible, onPhase };
}

export function buildStimulusPatch(
  p: StimulusParams,
  tNowSec: number,
  autoCifarIndex: number,
  dataset: RgbTensor[],
): { patch: RgbTensor; visible: boolean; effectiveCifarIndex: number } {
  const cycle = timedCycle(tNowSec, p.onMs, p.offMs);
  const visible = !p.timedExperiment || cycle.visible;
  let cifarIdx = autoCifarIndex;
  if (!p.timedExperiment) {
    cifarIdx = p.cifarIndex;
  }

  if (p.kind === "bar") {
    const ori = p.timedExperiment ? p.orientationDeg + cycle.cycleIndex * p.autoBarStepDeg : p.orientationDeg;
    return {
      patch: generateBarStimulus(
        p.barSizePx,
        wrap180(ori),
        p.barLengthPx,
        p.barThicknessPx,
        p.barContrast,
        p.barColor,
      ),
      visible,
      effectiveCifarIndex: cifarIdx,
    };
  }

  if (p.kind === "grating") {
    let ori = p.orientationDeg;
    let sf = p.gratingSfCpp;
    if (p.timedExperiment) {
      if (p.autoGrMode === "orientation" || p.autoGrMode === "both") {
        ori += cycle.cycleIndex * p.autoGrOriStepDeg;
      }
      if (p.autoGrMode === "frequency" || p.autoGrMode === "both") {
        sf += cycle.cycleIndex * p.autoGrSfStepCpp;
      }
    }
    const phase = p.timedExperiment
      ? wrap360(p.phaseDeg + 360 * cycle.onPhase)
      : wrap360(p.phaseDeg + p.phaseSpeedDegPerSec * tNowSec);
    return {
      patch: generateGratingStimulus(
        p.gratingSizePx,
        wrap180(ori),
        clamp(sf, 0.01, 0.45),
        phase,
        p.gratingContrast,
        p.gratingColor,
      ),
      visible,
      effectiveCifarIndex: cifarIdx,
    };
  }

  return {
    patch: generateCifarPatchStimulus(dataset, cifarIdx, p.cifarSizePx),
    visible,
    effectiveCifarIndex: cifarIdx,
  };
}

export function responseFromFrame(
  frame: RgbTensor,
  kernel: RgbTensor,
  rfPos: Vec2,
  mode: ResponseMode,
  gain: number,
  baselineHz: number,
  maxRateHz: number,
): { raw: number; effective: number; rateHz: number } {
  if (mode === "normalized") {
    const raw = computeSingleFrameResponseNormalized(frame, kernel, rfPos);
    const sensitivity = gain / 5.0;
    const effective = Math.tanh(sensitivity * raw);
    const rateHz = responseToRateHzBounded(effective, maxRateHz, baselineHz);
    return { raw, effective, rateHz };
  }
  const raw = computeSingleFrameResponse(frame, kernel, rfPos);
  const rateHz = responseToRateHz(raw, gain, baselineHz);
  return { raw, effective: raw, rateHz };
}

export function makeRng(seed: number): () => number {
  return mulberry32(seed);
}
