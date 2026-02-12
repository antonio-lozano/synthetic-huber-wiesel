import matplotlib.pyplot as plt
import numpy as np


def norm_img(img: np.ndarray, per_channel: bool = False) -> np.ndarray:
    arr = img.astype(np.float32)
    eps = 1e-10
    if per_channel and arr.ndim == 3 and arr.shape[-1] == 3:
        out = np.empty_like(arr, dtype=np.float32)
        for c in range(3):
            channel = arr[..., c]
            out[..., c] = (channel - channel.min()) / (channel.max() - channel.min() + eps)
        return out
    return (arr - arr.min()) / (arr.max() - arr.min() + eps)


def kernel_to_rgba(kernel_rgb: np.ndarray, threshold: float = 0.08) -> np.ndarray:
    kernel_norm = norm_img(kernel_rgb, per_channel=True)
    intensity = np.mean(kernel_norm, axis=-1)
    alpha = np.where(np.abs(intensity - 0.5) < threshold, 0.0, 1.0).astype(np.float32)
    return np.dstack([kernel_norm, alpha])


def plot_random_grid(images_01: np.ndarray, nrows: int = 4, ncols: int = 4, seed: int = 0) -> None:
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 10))
    local_rng = np.random.default_rng(seed)
    for ax in axes.flat:
        idx = int(local_rng.integers(0, images_01.shape[0]))
        ax.imshow(images_01[idx])
        ax.axis("off")
    plt.suptitle("Random natural images", fontsize=14)
    plt.tight_layout()
    plt.show()
