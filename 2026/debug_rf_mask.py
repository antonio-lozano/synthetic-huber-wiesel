#%% Debug | RF gray-outside mask experiments
"""
Standalone debugging script for receptive-field masking.

Goal:
- Keep RF center active.
- Force outside region to gray (0.0 in [-1, 1]).
- Compare hard/soft aperture settings visually and numerically.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from computing import generate_rgb_gabor_kernel
#%% Helpers | Aperture masks
def make_circular_mask(size: int, radius_ratio: float = 0.45, soft_edge_px: float = 1.5) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    cy = (size - 1) / 2.0
    cx = (size - 1) / 2.0
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    radius = radius_ratio * size
    if soft_edge_px <= 0:
        return (dist <= radius).astype(np.float32)

    t = np.clip((dist - radius) / max(1e-6, soft_edge_px), 0.0, 1.0)
    edge = 0.5 * (1.0 + np.cos(np.pi * t))
    edge[dist <= radius] = 1.0
    edge[dist >= radius + soft_edge_px] = 0.0
    return edge.astype(np.float32)


def apply_rf_gray_outside(
    kernel_rgb: np.ndarray,
    gray_value: float = 0.0,
    radius_ratio: float = 0.45,
    soft_edge_px: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    size = kernel_rgb.shape[0]
    mask = make_circular_mask(size=size, radius_ratio=radius_ratio, soft_edge_px=soft_edge_px)
    out = kernel_rgb * mask[..., None] + gray_value * (1.0 - mask[..., None])
    return out.astype(np.float32), mask


def m11_to_01(img_m11: np.ndarray) -> np.ndarray:
    """Display helper that preserves the true gray level from [-1, 1] -> [0, 1]."""
    return (np.clip(img_m11, -1.0, 1.0) + 1.0) / 2.0


def summarize_mask_effect(original: np.ndarray, masked: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    inside = mask > 0.5
    outside = mask < 0.05

    inside_energy_orig = float(np.mean(np.abs(original[inside])))
    inside_energy_mask = float(np.mean(np.abs(masked[inside])))
    outside_energy_orig = float(np.mean(np.abs(original[outside]))) if np.any(outside) else 0.0
    outside_energy_mask = float(np.mean(np.abs(masked[outside]))) if np.any(outside) else 0.0
    return {
        "inside_abs_mean_orig": inside_energy_orig,
        "inside_abs_mean_masked": inside_energy_mask,
        "outside_abs_mean_orig": outside_energy_orig,
        "outside_abs_mean_masked": outside_energy_mask,
    }


#%% Generate one baseline RF kernel
seed = 2026
rng = np.random.default_rng(seed)
kernel_rgb, freq, theta, size, _coords = generate_rgb_gabor_kernel(
    size=21,
    frequency_range=(0.1, 0.3),
    img_shape=(32, 32, 3),
    local_rng=rng,
)

print(f"Kernel created | size={size}, freq={freq:.3f}, theta_deg={np.degrees(theta):.1f}")


#%% Compare mask configurations
configs = [
    ("Hard 0.45", 0.45, 0.0),
    ("Soft 0.45 / 1.5px", 0.45, 1.5),
    ("Soft 0.50 / 2.0px", 0.50, 2.0),
]

fig, axes = plt.subplots(len(configs), 4, figsize=(12, 3.6 * len(configs)), dpi=150)
if len(configs) == 1:
    axes = np.array([axes])

for r, (name, radius_ratio, soft_edge_px) in enumerate(configs):
    # Important: apply mask AFTER kernel normalization (kernel is already in [-1, 1]).
    # Do not renormalize after masking if you want outside gray to remain true gray (0.0).
    masked, mask = apply_rf_gray_outside(
        kernel_rgb=kernel_rgb,
        gray_value=0.0,
        radius_ratio=radius_ratio,
        soft_edge_px=soft_edge_px,
    )
    stats = summarize_mask_effect(kernel_rgb, masked, mask)
    print(
        f"{name} | inside_abs {stats['inside_abs_mean_orig']:.4f}->{stats['inside_abs_mean_masked']:.4f} | "
        f"outside_abs {stats['outside_abs_mean_orig']:.4f}->{stats['outside_abs_mean_masked']:.4f}"
    )

    axes[r, 0].imshow(m11_to_01(kernel_rgb))
    axes[r, 0].set_title("Original")
    axes[r, 0].axis("off")

    axes[r, 1].imshow(mask, cmap="gray", vmin=0.0, vmax=1.0)
    axes[r, 1].set_title(f"Mask\n{name}")
    axes[r, 1].axis("off")

    axes[r, 2].imshow(m11_to_01(masked))
    axes[r, 2].set_title("Masked RF")
    axes[r, 2].axis("off")

    diff = np.mean(np.abs(masked - kernel_rgb), axis=2)
    axes[r, 3].imshow(diff, cmap="magma")
    axes[r, 3].set_title("|Masked - Original|")
    axes[r, 3].axis("off")

plt.suptitle("RF Gray-Outside Mask Debug", fontsize=16)
plt.tight_layout()
plt.show()


#%% Quick recommendation print
print("\nRecommended starter postprocess:")
print("radius_ratio=0.45, soft_edge_px=1.5, gray_value=0.0")
print("Then tune radius_ratio in [0.40, 0.55] depending on desired RF support.")


#%% Population view | Many neurons at once
"""
Show many kernels to quickly inspect whether gray-outside masking behaves well
across diverse RFs.
"""

N_SHOW = 48
KERNEL_SIZE = 21
MASK_RADIUS_RATIO = 0.45
MASK_SOFT_EDGE_PX = 1.5

kernels = []
masked_kernels = []
outside_abs_before = []
outside_abs_after = []

for _ in range(N_SHOW):
    k, _f, _th, _s, _coords = generate_rgb_gabor_kernel(
        size=KERNEL_SIZE,
        frequency_range=(0.1, 0.3),
        img_shape=(32, 32, 3),
        local_rng=rng,
    )
    km, m = apply_rf_gray_outside(
        kernel_rgb=k,
        gray_value=0.0,
        radius_ratio=MASK_RADIUS_RATIO,
        soft_edge_px=MASK_SOFT_EDGE_PX,
    )
    kernels.append(k)
    masked_kernels.append(km)
    outside = m < 0.05
    if np.any(outside):
        outside_abs_before.append(float(np.mean(np.abs(k[outside]))))
        outside_abs_after.append(float(np.mean(np.abs(km[outside]))))

print(f"\nPopulation debug: N_SHOW={N_SHOW}")
if outside_abs_before:
    print(
        "Mean outside |abs| "
        f"before={np.mean(outside_abs_before):.4f} "
        f"after={np.mean(outside_abs_after):.4f}"
    )

cols = 12
n_blocks = int(np.ceil(N_SHOW / cols))
rows = 2 * n_blocks
fig, axes = plt.subplots(rows, cols, figsize=(1.55 * cols, 1.6 * rows), dpi=150)
axes = np.atleast_2d(axes)

for b in range(n_blocks):
    row_orig = 2 * b
    row_mask = row_orig + 1
    start = b * cols
    end = min(start + cols, N_SHOW)
    for c in range(cols):
        idx = start + c
        ax_o = axes[row_orig, c]
        ax_m = axes[row_mask, c]
        if idx < end:
            ax_m.imshow((np.clip(masked_kernels[idx], -1.0, 1.0) + 1.0) / 2.0)
            ax_o.imshow((np.clip(kernels[idx], -1.0, 1.0) + 1.0) / 2.0)
        ax_m.axis("off")
        ax_o.axis("off")

    axes[row_orig, 0].text(
        -0.18,
        0.5,
        "Unmasked",
        transform=axes[row_orig, 0].transAxes,
        rotation=90,
        va="center",
        fontsize=10,
    )
    axes[row_mask, 0].text(
        -0.18,
        0.5,
        "Masked",
        transform=axes[row_mask, 0].transAxes,
        rotation=90,
        va="center",
        fontsize=10,
    )

plt.suptitle(
    f"Many RFs | radius={MASK_RADIUS_RATIO}, soft_edge={MASK_SOFT_EDGE_PX}px",
    fontsize=14,
)
plt.tight_layout()
plt.show()
