export type Vec2 = { x: number; y: number };

export type NeuronColor = "white" | "blue" | "red" | "green" | "yellow";

export type SimNeuron = {
  id: number;
  color: NeuronColor;
  rfPos: Vec2;
  rfSize: number;
  orientationDeg: number;
};

export type StimulusKind = "bar" | "grating" | "cifar";
