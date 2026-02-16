import { RgbTensor } from "./types";

type CifarSpriteMeta = {
  tile_size: number;
  cols: number;
  rows: number;
  count: number;
};

export async function loadCifarDatasetFromSprite(
  metaPath = "assets/cifar50.json",
  pngPath = "assets/cifar50.png",
): Promise<RgbTensor[]> {
  const metaResp = await fetch(metaPath, { cache: "force-cache" });
  if (!metaResp.ok) {
    throw new Error(`Failed to load CIFAR metadata: ${metaResp.status}`);
  }
  const meta = (await metaResp.json()) as CifarSpriteMeta;

  const image = new Image();
  image.decoding = "async";
  const imageLoaded = new Promise<void>((resolve, reject) => {
    image.onload = () => resolve();
    image.onerror = () => reject(new Error("Failed to load CIFAR sprite image"));
  });
  image.src = pngPath;
  await imageLoaded;

  const spriteCanvas = document.createElement("canvas");
  spriteCanvas.width = image.width;
  spriteCanvas.height = image.height;
  const ctx = spriteCanvas.getContext("2d");
  if (!ctx) throw new Error("Could not allocate canvas context for CIFAR sprite decoding");
  ctx.drawImage(image, 0, 0);
  const spriteRgba = ctx.getImageData(0, 0, spriteCanvas.width, spriteCanvas.height).data;

  const tile = Math.max(1, Math.round(meta.tile_size));
  const cols = Math.max(1, Math.round(meta.cols));
  const rows = Math.max(1, Math.round(meta.rows));
  const count = Math.max(1, Math.min(Math.round(meta.count), cols * rows));
  const out: RgbTensor[] = [];

  for (let idx = 0; idx < count; idx += 1) {
    const row = Math.floor(idx / cols);
    const col = idx % cols;
    if (row >= rows) break;

    const data = new Float32Array(tile * tile * 3);
    let dst = 0;
    for (let y = 0; y < tile; y += 1) {
      const sy = row * tile + y;
      for (let x = 0; x < tile; x += 1) {
        const sx = col * tile + x;
        const src = (sy * spriteCanvas.width + sx) * 4;
        data[dst] = (spriteRgba[src] / 255.0) * 2.0 - 1.0;
        data[dst + 1] = (spriteRgba[src + 1] / 255.0) * 2.0 - 1.0;
        data[dst + 2] = (spriteRgba[src + 2] / 255.0) * 2.0 - 1.0;
        dst += 3;
      }
    }
    out.push({ width: tile, height: tile, data });
  }
  return out;
}
