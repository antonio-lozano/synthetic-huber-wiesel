from dataclasses import dataclass
from typing import Optional

import numpy as np
from skimage.filters import gabor_kernel
from skimage.transform import resize


@dataclass(frozen=True)
class ClassConfig:
    seed: int
    num_images: int
    n_v1_cells: int
    sizes: tuple[int, ...]
    frequency_range: tuple[float, float]
    trial_time_ms: int
    intertrial_time_ms: int
    bin_size_ms: int
    num_trials_sta: int
    num_runs_raster: int
    selected_neurons: tuple[int, ...]
    enable_mlp: bool
    mlp_epochs: int
    mlp_batch_size: int


@dataclass
class V1Cell:
    kernel_rgb: np.ndarray
    frequency: float
    theta: float
    size: int
    coords: tuple[int, int]


@dataclass
class TuningResult:
    preferred_orientation_deg: float
    osi: float
    preferred_frequency: float
    curve_orientation: np.ndarray
    curve_frequency: np.ndarray
    orientation_axis_deg: np.ndarray
    frequency_axis: np.ndarray


def make_config(profile: str = "class", enable_mlp: bool = False) -> ClassConfig:
    profile = profile.lower().strip()
    presets: dict[str, dict[str, object]] = {
        "quick": {
            "seed": 2026,
            "num_images": 300,
            "n_v1_cells": 16,
            "sizes": (5, 8, 11, 15),
            "frequency_range": (0.1, 0.3),
            "trial_time_ms": 200,
            "intertrial_time_ms": 500,
            "bin_size_ms": 10,
            "num_trials_sta": 180,
            "num_runs_raster": 25,
            "selected_neurons": (0, 3, 7, 10),
            "mlp_epochs": 4,
            "mlp_batch_size": 64,
        },
        "class": {
            "seed": 2026,
            "num_images": 1200,
            "n_v1_cells": 20,
            "sizes": (5, 8, 11, 15),
            "frequency_range": (0.1, 0.3),
            "trial_time_ms": 200,
            "intertrial_time_ms": 500,
            "bin_size_ms": 10,
            "num_trials_sta": 400,
            "num_runs_raster": 50,
            "selected_neurons": (0, 3, 5, 10, 15),
            "mlp_epochs": 8,
            "mlp_batch_size": 64,
        },
        "deep": {
            "seed": 2026,
            "num_images": 2500,
            "n_v1_cells": 24,
            "sizes": (5, 8, 11, 15),
            "frequency_range": (0.1, 0.3),
            "trial_time_ms": 200,
            "intertrial_time_ms": 500,
            "bin_size_ms": 10,
            "num_trials_sta": 900,
            "num_runs_raster": 75,
            "selected_neurons": (0, 3, 5, 10, 15, 18, 20),
            "mlp_epochs": 12,
            "mlp_batch_size": 128,
        },
    }
    if profile not in presets:
        valid = ", ".join(sorted(presets.keys()))
        raise ValueError(f"Unknown profile '{profile}'. Valid options: {valid}")
    p = presets[profile]
    return ClassConfig(
        seed=int(p["seed"]),
        num_images=int(p["num_images"]),
        n_v1_cells=int(p["n_v1_cells"]),
        sizes=tuple(p["sizes"]),
        frequency_range=tuple(p["frequency_range"]),
        trial_time_ms=int(p["trial_time_ms"]),
        intertrial_time_ms=int(p["intertrial_time_ms"]),
        bin_size_ms=int(p["bin_size_ms"]),
        num_trials_sta=int(p["num_trials_sta"]),
        num_runs_raster=int(p["num_runs_raster"]),
        selected_neurons=tuple(p["selected_neurons"]),
        enable_mlp=enable_mlp,
        mlp_epochs=int(p["mlp_epochs"]),
        mlp_batch_size=int(p["mlp_batch_size"]),
    )


def set_seed(seed: int) -> np.random.Generator:
    np.random.seed(seed)
    return np.random.default_rng(seed)


def to_0_1(images_uint8: np.ndarray) -> np.ndarray:
    return images_uint8.astype(np.float32) / 255.0


def to_minus1_1(images_uint8: np.ndarray) -> np.ndarray:
    return to_0_1(images_uint8) * 2.0 - 1.0


def load_cifar_subset(cfg: ClassConfig) -> np.ndarray:
    import importlib.util
    import os

    has_tf = importlib.util.find_spec("tensorflow") is not None
    has_torch = importlib.util.find_spec("torch") is not None
    if not has_tf and has_torch:
        os.environ.setdefault("KERAS_BACKEND", "torch")

    cifar10_module: Optional[object] = None
    images: Optional[np.ndarray] = None
    try:
        from keras.datasets import cifar10 as c10

        cifar10_module = c10
    except Exception:
        try:
            from tensorflow.keras.datasets import cifar10 as c10

            cifar10_module = c10
        except Exception:
            cifar10_module = None

    if cifar10_module is not None:
        (x_train, _), (x_test, _) = cifar10_module.load_data()
        images = np.concatenate([x_train, x_test], axis=0)

    if images is None:
        import pickle
        import tarfile
        from pathlib import Path
        from urllib.request import urlretrieve

        url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        cache_dir = Path.home() / ".cache" / "neuroai_2026"
        cache_dir.mkdir(parents=True, exist_ok=True)
        tar_path = cache_dir / "cifar-10-python.tar.gz"
        extract_dir = cache_dir / "cifar-10-batches-py"
        if not tar_path.exists():
            print("Downloading CIFAR-10 from Toronto website...")
            urlretrieve(url, tar_path)
        if not extract_dir.exists():
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(cache_dir)

        data_batches = []
        for batch_idx in range(1, 6):
            batch_path = extract_dir / f"data_batch_{batch_idx}"
            with open(batch_path, "rb") as f:
                batch = pickle.load(f, encoding="bytes")
            data_batches.append(batch[b"data"])
        test_path = extract_dir / "test_batch"
        with open(test_path, "rb") as f:
            test_batch = pickle.load(f, encoding="bytes")
        data_all = np.concatenate(data_batches + [test_batch[b"data"]], axis=0)
        images = data_all.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.uint8)

    if cfg.num_images > images.shape[0]:
        raise ValueError("cfg.num_images exceeds CIFAR-10 available samples.")
    local_rng = np.random.default_rng(cfg.seed)
    subset_idx = local_rng.choice(images.shape[0], size=cfg.num_images, replace=False)
    return images[subset_idx]


def assert_timing_is_valid(cfg: ClassConfig) -> None:
    if cfg.trial_time_ms % cfg.bin_size_ms != 0:
        raise ValueError("trial_time_ms must be divisible by bin_size_ms.")
    if cfg.intertrial_time_ms % cfg.bin_size_ms != 0:
        raise ValueError("intertrial_time_ms must be divisible by bin_size_ms.")


def generate_rgb_gabor_kernel(
    size: int,
    frequency_range: tuple[float, float],
    img_shape: tuple[int, int, int],
    local_rng: np.random.Generator,
    mean_range: tuple[float, float] = (0.1, 0.5),
) -> tuple[np.ndarray, float, float, int, tuple[int, int]]:
    frequency = float(local_rng.uniform(*frequency_range))
    theta = float(local_rng.uniform(0.0, np.pi))
    sigma = size / 4.0
    kernel = np.real(gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma))
    kernel_resized = resize(kernel, (size, size), anti_aliasing=True)
    eps = 1e-10
    kernel_norm = 2.0 * (kernel_resized - kernel_resized.min()) / (kernel_resized.max() - kernel_resized.min() + eps) - 1.0
    color_weights = local_rng.normal(0.0, 1.0, size=3)
    kernel_rgb = np.stack([kernel_norm * w for w in color_weights], axis=-1)
    kmin, kmax = float(kernel_rgb.min()), float(kernel_rgb.max())
    kernel_rgb = 2.0 * (kernel_rgb - kmin) / (kmax - kmin + eps) - 1.0
    target_mean = float(local_rng.uniform(*mean_range))
    kernel_rgb = kernel_rgb - float(np.mean(kernel_rgb)) + target_mean
    img_h, img_w = img_shape[:2]
    max_x = img_w - size
    max_y = img_h - size
    x_coord = int(local_rng.integers(0, max_x + 1))
    y_coord = int(local_rng.integers(0, max_y + 1))
    return kernel_rgb.astype(np.float32), frequency, theta, size, (x_coord, y_coord)


def generate_v1_cells(cfg: ClassConfig, image_shape: tuple[int, int, int]) -> list[V1Cell]:
    cells: list[V1Cell] = []
    local_rng = np.random.default_rng(cfg.seed + 11)
    n_sizes = len(cfg.sizes)
    cells_per_size = cfg.n_v1_cells // n_sizes
    remaining = cfg.n_v1_cells % n_sizes
    counts = [cells_per_size] * n_sizes
    for i in range(remaining):
        counts[i] += 1
    for size, n_cells in zip(cfg.sizes, counts):
        for _ in range(n_cells):
            k, f, th, s, coords = generate_rgb_gabor_kernel(size, cfg.frequency_range, image_shape, local_rng)
            cells.append(V1Cell(kernel_rgb=k, frequency=f, theta=th, size=s, coords=coords))
    return cells


def build_temporal_inputs(images_m11: np.ndarray, cfg: ClassConfig) -> np.ndarray:
    if cfg.trial_time_ms % cfg.bin_size_ms != 0:
        raise ValueError("trial_time_ms must be divisible by bin_size_ms")
    if cfg.intertrial_time_ms % cfg.bin_size_ms != 0:
        raise ValueError("intertrial_time_ms must be divisible by bin_size_ms")
    bins_image = cfg.trial_time_ms // cfg.bin_size_ms
    bins_intertrial = cfg.intertrial_time_ms // cfg.bin_size_ms
    h, w, c = images_m11.shape[1:]
    gray = np.zeros((h, w, c), dtype=np.float32)
    frames = []
    for _ in range(bins_intertrial):
        frames.append(gray.copy())
    for img in images_m11:
        for _ in range(bins_image):
            frames.append(img.copy())
        for _ in range(bins_intertrial):
            frames.append(gray.copy())
    return np.array(frames, dtype=np.float32)


def build_trial_windows(cfg: ClassConfig, n_trials: int) -> list[tuple[int, int]]:
    bins_image = cfg.trial_time_ms // cfg.bin_size_ms
    bins_intertrial = cfg.intertrial_time_ms // cfg.bin_size_ms
    windows = []
    start = bins_intertrial
    for _ in range(n_trials):
        end = start + bins_image
        windows.append((start, end))
        start = end + bins_intertrial
    return windows


def simulate_firing_rates(temporal_inputs: np.ndarray, v1_cells: list[V1Cell]) -> np.ndarray:
    t = temporal_inputs.shape[0]
    n_neurons = len(v1_cells)
    firing = np.zeros((t, n_neurons), dtype=np.float32)
    for i, cell in enumerate(v1_cells):
        x_rf, y_rf = cell.coords
        s = cell.size
        patch = temporal_inputs[:, y_rf : y_rf + s, x_rf : x_rf + s, :]
        firing[:, i] = np.sum(patch * cell.kernel_rgb, axis=(1, 2, 3))
    return firing


def normalize_firing_rates(firing_raw: np.ndarray, target_max: float = 5.0) -> np.ndarray:
    rectified = np.maximum(firing_raw, 0.0)
    max_vals = np.max(rectified, axis=0, keepdims=True)
    return np.where(max_vals > 0, target_max * rectified / max_vals, rectified).astype(np.float32)


def compute_sta(stimulus: np.ndarray, firing_rates: np.ndarray, neuron_ids: list[int]) -> dict[int, np.ndarray]:
    if stimulus.shape[0] != firing_rates.shape[0]:
        raise ValueError("Stimulus and firing rates must have same time dimension")
    sta_results: dict[int, np.ndarray] = {}
    for neuron in neuron_ids:
        rate = np.maximum(firing_rates[:, neuron], 0.0)
        total_weight = float(np.sum(rate))
        if total_weight > 0:
            sta = np.sum(rate[:, None, None, None] * stimulus, axis=0) / total_weight
        else:
            sta = np.zeros(stimulus.shape[1:], dtype=np.float32)
        sta_results[neuron] = sta.astype(np.float32)
    return sta_results


def _calculate_osi(curve: np.ndarray, angles_deg: np.ndarray) -> float:
    angles_rad = np.deg2rad(angles_deg)
    complex_sum = np.sum(curve * np.exp(2j * angles_rad))
    return float(np.abs(complex_sum) / (np.sum(curve) + 1e-10))


def analyze_tuning(kernel_rgb: np.ndarray) -> TuningResult:
    gray = np.mean(kernel_rgb, axis=2) if kernel_rgb.ndim == 3 else kernel_rgb.copy()
    gray = gray.astype(np.float32)
    gray = gray - float(np.mean(gray))
    dy, dx = np.gradient(gray)
    mag = np.hypot(dx, dy)
    theta = np.rad2deg(np.arctan2(dy, dx)) % 180.0
    ori_bins = np.linspace(0.0, 180.0, 37)
    ori_curve, edges = np.histogram(theta, bins=ori_bins, weights=mag)
    ori_axis = (edges[:-1] + edges[1:]) / 2.0
    ori_curve = ori_curve.astype(np.float32)
    ori_curve = ori_curve / (float(np.max(ori_curve)) + 1e-10)
    pref_ori = float(ori_axis[int(np.argmax(ori_curve))])
    osi = _calculate_osi(ori_curve, ori_axis)

    fft = np.fft.fftshift(np.fft.fft2(gray))
    power = np.abs(fft) ** 2
    ny, nx = gray.shape
    y_grid, x_grid = np.indices((ny, nx))
    cy, cx = ny // 2, nx // 2
    radius = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
    radius = radius / (float(np.max(radius)) + 1e-10)
    freq_bins = np.linspace(0.0, 1.0, 21)
    freq_curve = np.zeros(len(freq_bins) - 1, dtype=np.float32)
    for i in range(len(freq_bins) - 1):
        mask = (radius >= freq_bins[i]) & (radius < freq_bins[i + 1])
        freq_curve[i] = float(np.mean(power[mask])) if np.any(mask) else 0.0
    freq_axis = (freq_bins[:-1] + freq_bins[1:]) / 2.0
    freq_curve = freq_curve / (float(np.max(freq_curve)) + 1e-10)
    pref_freq = float(freq_axis[int(np.argmax(freq_curve))])
    return TuningResult(
        preferred_orientation_deg=pref_ori,
        osi=osi,
        preferred_frequency=pref_freq,
        curve_orientation=ori_curve,
        curve_frequency=freq_curve,
        orientation_axis_deg=ori_axis,
        frequency_axis=freq_axis,
    )


def train_optional_mlp(stimulus: np.ndarray, targets: np.ndarray, cfg: ClassConfig) -> Optional[dict[int, np.ndarray]]:
    if not cfg.enable_mlp:
        print("MLP module disabled. Set ENABLE_MLP=True to run optional MLP cells.")
        return None

    import importlib.util
    import os

    has_tf = importlib.util.find_spec("tensorflow") is not None
    has_torch = importlib.util.find_spec("torch") is not None

    # Prefer torch backend when TensorFlow is unavailable (typical Windows + RTX 50 setup).
    if not has_tf and has_torch:
        os.environ.setdefault("KERAS_BACKEND", "torch")

    try:
        import keras
    except Exception as exc:
        print("Keras is not available. Skipping MLP module.")
        print(f"Reason: {exc}")
        return None

    keras.utils.set_random_seed(cfg.seed)
    x = stimulus.reshape(stimulus.shape[0], -1)
    y = targets
    max_train_bins = 15000
    if x.shape[0] > max_train_bins:
        idx = np.linspace(0, x.shape[0] - 1, max_train_bins).astype(int)
        x = x[idx]
        y = y[idx]
        print(f"MLP training downsampled to {max_train_bins} bins for runtime control.")

    reg = keras.regularizers.l1_l2(l1=1e-4, l2=1e-2)
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(x.shape[1],)),
            keras.layers.Dense(y.shape[1], activation="relu", kernel_regularizer=reg),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(), loss="mse")
    model.fit(x, y, epochs=cfg.mlp_epochs, batch_size=cfg.mlp_batch_size, verbose=0)

    weights = model.layers[0].get_weights()[0]
    h, w, c = stimulus.shape[1:]
    return {n: weights[:, n].reshape(h, w, c).astype(np.float32) for n in range(y.shape[1])}


def minimal_circular_diff_deg(angle1: float, angle2: float) -> float:
    diff = abs(angle1 - angle2)
    return float(min(diff, 180.0 - diff))


def compute_color_profile(kernel_rgb: np.ndarray) -> np.ndarray:
    energy = np.mean(np.abs(kernel_rgb), axis=(0, 1))
    return energy / (np.sum(energy) + 1e-10)


def draw_bar(
    image: np.ndarray,
    center: tuple[float, float],
    angle_rad: float,
    length: float,
    thickness: float,
    color_rgb: tuple[int, int, int],
) -> np.ndarray:
    h, w, _ = image.shape
    y_grid, x_grid = np.ogrid[:h, :w]
    x0, y0 = center
    dx = np.cos(angle_rad)
    dy = np.sin(angle_rad)
    proj = (x_grid - x0) * dx + (y_grid - y0) * dy
    perp = -(x_grid - x0) * dy + (y_grid - y0) * dx
    mask = (np.abs(proj) <= length / 2.0) & (np.abs(perp) <= thickness / 2.0)
    for c in range(3):
        image[..., c][mask] = color_rgb[c]
    return image


def generate_bar_stimulus(
    size: int,
    orientation_deg: float,
    length: float,
    thickness: float,
    contrast: float = 1.0,
    background_level: float = -1.0,
    color_rgb_m11: Optional[tuple[float, float, float]] = None,
) -> np.ndarray:
    """
    Generate a square RGB bar stimulus in [-1, 1].
    """
    img = np.ones((size, size, 3), dtype=np.float32) * background_level
    center = (size / 2.0, size / 2.0)
    angle_rad = np.deg2rad(orientation_deg)
    if color_rgb_m11 is None:
        fg_val = np.clip(background_level + 2.0 * contrast, -1.0, 1.0)
        color_u8 = tuple(int(((fg_val + 1.0) / 2.0) * 255.0) for _ in range(3))
    else:
        fg_vals = []
        for c in color_rgb_m11:
            v = background_level + float(contrast) * (float(c) - background_level)
            fg_vals.append(float(np.clip(v, -1.0, 1.0)))
        color_u8 = tuple(int(((v + 1.0) / 2.0) * 255.0) for v in fg_vals)

    img_u8 = (((img + 1.0) / 2.0) * 255.0).astype(np.uint8)
    img_u8 = draw_bar(
        image=img_u8,
        center=center,
        angle_rad=angle_rad,
        length=length,
        thickness=thickness,
        color_rgb=color_u8,
    )
    return (img_u8.astype(np.float32) / 255.0) * 2.0 - 1.0


def generate_grating_stimulus(
    size: int,
    orientation_deg: float,
    spatial_frequency: float,
    phase_deg: float = 0.0,
    contrast: float = 1.0,
    background_level: float = -1.0,
    color_rgb_m11: Optional[tuple[float, float, float]] = None,
) -> np.ndarray:
    """
    Generate a square sinusoidal grating in [-1, 1].

    spatial_frequency is in cycles per pixel.
    """
    y_grid, x_grid = np.mgrid[0:size, 0:size]
    x = x_grid - size / 2.0
    y = y_grid - size / 2.0
    theta = np.deg2rad(orientation_deg)
    phase = np.deg2rad(phase_deg)

    x_rot = x * np.cos(theta) + y * np.sin(theta)
    grat = np.cos(2.0 * np.pi * spatial_frequency * x_rot + phase)
    grat = np.clip(grat * contrast, -1.0, 1.0)

    weight = (grat + 1.0) / 2.0
    if color_rgb_m11 is None:
        # Blend between background and white modulation in [-1, 1].
        stim = background_level + weight * (1.0 - background_level)
        stim = np.clip(stim, -1.0, 1.0).astype(np.float32)
        return np.repeat(stim[..., None], 3, axis=2)

    out = np.empty((size, size, 3), dtype=np.float32)
    for c in range(3):
        fg_c = float(np.clip(color_rgb_m11[c], -1.0, 1.0))
        out[..., c] = background_level + weight * (fg_c - background_level)
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def generate_cifar_patch_stimulus(
    cifar_images_uint8: np.ndarray,
    image_index: int,
    size: int,
) -> np.ndarray:
    """
    Generate a resized CIFAR patch stimulus in [-1, 1].
    """
    idx = int(np.clip(image_index, 0, cifar_images_uint8.shape[0] - 1))
    img = cifar_images_uint8[idx].astype(np.float32) / 255.0
    patch = resize(img, (size, size), anti_aliasing=True, preserve_range=True).astype(np.float32)
    patch = np.flipud(patch)
    return patch * 2.0 - 1.0


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
    desaturate: bool = False,
) -> np.ndarray:
    """
    Postprocess an already-normalized RF kernel in [-1, 1] so outside region is gray.
    """
    size = kernel_rgb.shape[0]
    src = kernel_rgb.astype(np.float32)
    if desaturate:
        gray = np.mean(src, axis=2, keepdims=True)
        src = np.repeat(gray, 3, axis=2)
    mask = make_circular_mask(size=size, radius_ratio=radius_ratio, soft_edge_px=soft_edge_px)
    out = src * mask[..., None] + gray_value * (1.0 - mask[..., None])
    return out.astype(np.float32)


def to_grayscale_kernel(
    kernel_rgb: np.ndarray,
    match_energy: bool = False,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Convert RF kernel to grayscale.

    If match_energy=True, scale grayscale kernel to match original centered L2 norm.
    """
    k = kernel_rgb.astype(np.float32)
    gray = np.mean(k, axis=2, keepdims=True)
    g3 = np.repeat(gray, 3, axis=2).astype(np.float32)
    if not match_energy:
        return g3

    k_mean = float(np.mean(k))
    g_mean = float(np.mean(g3))
    kc = k - k_mean
    gc = g3 - g_mean
    norm_k = float(np.linalg.norm(kc))
    norm_g = float(np.linalg.norm(gc))
    if norm_g <= eps:
        return g3
    scale = norm_k / norm_g
    return (gc * scale + k_mean).astype(np.float32)


def resize_kernel_rgb(kernel_rgb: np.ndarray, new_size: int) -> np.ndarray:
    """
    Resize an RGB kernel while preserving approximate mean/std energy.
    """
    size = max(3, int(new_size))
    resized = resize(
        kernel_rgb,
        (size, size, kernel_rgb.shape[2]),
        order=3,
        anti_aliasing=True,
        preserve_range=True,
    ).astype(np.float32)

    orig_mean = float(np.mean(kernel_rgb))
    orig_std = float(np.std(kernel_rgb))
    new_mean = float(np.mean(resized))
    new_std = float(np.std(resized))
    if new_std > 1e-8 and orig_std > 0:
        resized = (resized - new_mean) * (orig_std / new_std) + orig_mean
    else:
        resized = resized - new_mean + orig_mean
    return resized.astype(np.float32)


def place_patch_on_canvas(
    canvas_m11: np.ndarray,
    patch_m11: np.ndarray,
    center_xy: tuple[float, float],
) -> np.ndarray:
    """
    Place a patch on a canvas, clipped at borders.
    """
    out = canvas_m11.copy()
    h, w, _ = out.shape
    ph, pw, _ = patch_m11.shape

    cx, cy = center_xy
    x0 = int(round(cx - pw / 2.0))
    y0 = int(round(cy - ph / 2.0))
    x1 = x0 + pw
    y1 = y0 + ph

    ox0 = max(0, x0)
    oy0 = max(0, y0)
    ox1 = min(w, x1)
    oy1 = min(h, y1)
    if ox0 >= ox1 or oy0 >= oy1:
        return out

    px0 = ox0 - x0
    py0 = oy0 - y0
    px1 = px0 + (ox1 - ox0)
    py1 = py0 + (oy1 - oy0)

    out[oy0:oy1, ox0:ox1, :] = patch_m11[py0:py1, px0:px1, :]
    return out


def compute_single_frame_response(
    cell: V1Cell,
    frame_m11: np.ndarray,
    rf_coords: tuple[int, int],
) -> float:
    """
    Compute one instantaneous linear response for a cell on a frame.
    """
    x_rf, y_rf = rf_coords
    s = cell.size
    h, w = frame_m11.shape[:2]
    x_rf = int(np.clip(x_rf, 0, max(0, w - s)))
    y_rf = int(np.clip(y_rf, 0, max(0, h - s)))
    patch = frame_m11[y_rf : y_rf + s, x_rf : x_rf + s, :]
    return float(np.sum(patch * cell.kernel_rgb))


def compute_single_frame_response_normalized(
    cell: V1Cell,
    frame_m11: np.ndarray,
    rf_coords: tuple[int, int],
    eps: float = 1e-8,
) -> float:
    """
    Cosine-similarity response in [-1, 1] for interpretability and stable scaling.

    Both RF kernel and stimulus patch are mean-centered before similarity.
    """
    x_rf, y_rf = rf_coords
    s = cell.size
    h, w = frame_m11.shape[:2]
    x_rf = int(np.clip(x_rf, 0, max(0, w - s)))
    y_rf = int(np.clip(y_rf, 0, max(0, h - s)))
    patch = frame_m11[y_rf : y_rf + s, x_rf : x_rf + s, :].astype(np.float32)
    kernel = cell.kernel_rgb.astype(np.float32)

    patch = patch - float(np.mean(patch))
    kernel = kernel - float(np.mean(kernel))

    p = patch.reshape(-1)
    k = kernel.reshape(-1)
    denom = float(np.linalg.norm(p) * np.linalg.norm(k) + eps)
    if denom <= eps:
        return 0.0
    return float(np.dot(p, k) / denom)


def response_to_rate_hz(response: float, gain: float = 2.0, baseline_hz: float = 0.0) -> float:
    """
    Convert linear response to non-negative firing rate.
    """
    return max(0.0, baseline_hz + gain * response)


def response_to_rate_hz_bounded(
    response: float,
    max_rate_hz: float = 100.0,
    baseline_hz: float = 0.0,
) -> float:
    """
    Map normalized response to a bounded rate range [0, max_rate_hz].
    """
    max_rate = max(1e-6, float(max_rate_hz))
    baseline = float(np.clip(baseline_hz, 0.0, max_rate))
    r = float(np.clip(response, -1.0, 1.0))
    pos = max(0.0, r)
    rate = baseline + (max_rate - baseline) * pos
    return float(np.clip(rate, 0.0, max_rate))


def generate_spike_waveform_from_rf(
    kernel_rgb: np.ndarray,
    duration_ms: float = 1.5,
    sample_rate_hz: int = 20000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a biphasic spike waveform from RF statistics.

    Returns:
    - time axis in ms
    - waveform amplitude in arbitrary units
    """
    n = max(8, int(round(sample_rate_hz * duration_ms / 1000.0)))
    t_ms = np.linspace(0.0, duration_ms, n, dtype=np.float32)

    # RF-derived parameters for consistent-but-distinct waveforms.
    k = kernel_rgb.astype(np.float32)
    k_energy = float(np.mean(np.abs(k)))
    k_std = float(np.std(k))
    k_skew_like = float(np.mean((k - np.mean(k)) ** 3))

    # Width and phase shape vary smoothly with kernel stats.
    width_scale = float(np.clip(0.75 + 1.25 * k_std, 0.6, 1.8))
    asym = float(np.clip(np.tanh(2.0 * k_skew_like), -0.35, 0.35))

    neg_center = duration_ms * (0.38 + 0.08 * asym)
    pos_center = duration_ms * (0.78 + 0.06 * asym)
    neg_sigma = duration_ms * 0.09 * width_scale
    pos_sigma = duration_ms * 0.12 * width_scale

    neg = -np.exp(-0.5 * ((t_ms - neg_center) / max(1e-6, neg_sigma)) ** 2)
    pos = (0.55 + 0.35 * k_energy) * np.exp(-0.5 * ((t_ms - pos_center) / max(1e-6, pos_sigma)) ** 2)
    wave = neg + pos

    # Normalize peak magnitude near 1 for display consistency.
    wave = wave / (float(np.max(np.abs(wave))) + 1e-8)
    return t_ms.astype(np.float32), wave.astype(np.float32)


def sample_noisy_jittered_spike_waveform(
    t_ms: np.ndarray,
    waveform: np.ndarray,
    rng: np.random.Generator,
    jitter_std_ms: float = 0.04,
    noise_std: float = 0.04,
    amp_jitter_frac: float = 0.08,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample one spike waveform instance with slight temporal jitter and amplitude noise.
    """
    jitter = float(rng.normal(0.0, jitter_std_ms))
    amp_scale = float(1.0 + rng.normal(0.0, amp_jitter_frac))
    noisy = amp_scale * waveform + rng.normal(0.0, noise_std, size=waveform.shape).astype(np.float32)
    return (t_ms + jitter).astype(np.float32), noisy.astype(np.float32)


def poisson_spike_step(rate_hz: float, dt_s: float, rng: np.random.Generator) -> int:
    """
    Bernoulli approximation for one timestep of Poisson spiking.
    """
    p = float(np.clip(rate_hz * dt_s, 0.0, 1.0))
    return int(rng.random() < p)
