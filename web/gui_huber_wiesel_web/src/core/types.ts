export type Vec2 = { x: number; y: number };

export type NeuronColor = "white" | "blue" | "red" | "green" | "yellow";

export type StimulusColor = "white" | "red" | "green" | "blue" | "yellow" | "cyan" | "magenta";

export type GratingAutoMode = "orientation" | "frequency" | "both";

export type ResponseMode = "normalized" | "legacy";

export type RgbTensor = {
  width: number;
  height: number;
  data: Float32Array;
};

export type SimNeuron = {
  id: number;
  color: NeuronColor;
  rfPos: Vec2;
  rfSize: number;
  baseRfSize: number;
  orientationDeg: number;
  frequency: number;
  kernelBase: RgbTensor;
};

export type StimulusKind = "bar" | "grating" | "cifar";

export type SpikeWaveform = {
  tMs: Float32Array;
  amp: Float32Array;
};
