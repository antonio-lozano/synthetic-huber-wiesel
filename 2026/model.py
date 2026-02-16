"""
Model layer for the Hubel-Wiesel Synthetic V1 Explorer.

Contains:
- ``SimState`` dataclass holding ALL simulation state (no Qt references).
- Pure functions for stimulus generation, response computation, history trimming,
  timed-cycle bookkeeping, kernel helpers, etc.

Every function is free of Qt/PyQtGraph imports so it can be tested in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.random import Generator

from computing import (
    ClassConfig,
    V1Cell,
    apply_rf_gray_outside,
    compute_single_frame_response,
    compute_single_frame_response_normalized,
    generate_bar_stimulus,
    generate_cifar_patch_stimulus,
    generate_grating_stimulus,
    generate_spike_waveform_from_rf,
    generate_v1_cells,
    load_cifar_subset,
    make_config,
    place_patch_on_canvas,
    poisson_spike_step,
    resize_kernel_rgb,
    response_to_rate_hz,
    response_to_rate_hz_bounded,
    sample_noisy_jittered_spike_waveform,
    set_seed,
    to_grayscale_kernel,
)

# ── Constants ──────────────────────────────────────────────────────────────────
NEURON_COLORS = ["w", "b", "r", "g", "y"]
NEURON_RGB = {
    "w": (255, 255, 255),
    "b": (0, 120, 255),
    "r": (255, 0, 0),
    "g": (0, 220, 0),
    "y": (255, 220, 0),
}
MAX_SIM_NEURONS = 5

COLOR_LUT: dict[str, tuple[float, float, float]] = {
    "white":   (1.0,  1.0,  1.0),
    "red":     (1.0, -1.0, -1.0),
    "green":   (-1.0, 1.0, -1.0),
    "blue":    (-1.0, -1.0, 1.0),
    "yellow":  (1.0,  1.0, -1.0),
    "cyan":    (-1.0, 1.0,  1.0),
    "magenta": (1.0, -1.0,  1.0),
}

StimulusKind = Literal["Bar", "Grating", "CIFAR patch"]
PresentationMode = Literal["Manual", "Timed experiment"]
GratingAutoMode = Literal["Orientation", "Frequency", "Both"]
ResponseMode = Literal["normalized", "legacy"]


# ── Simulation state ─────────────────────────────────────────────────────────
@dataclass
class SimState:
    """All mutable simulation state, decoupled from the UI."""

    # Core data (initialised by ``init_sim_state``)
    cfg: ClassConfig = field(default_factory=lambda: make_config("quick", False))
    rng: Generator = field(default_factory=lambda: set_seed(2026))
    v1_cells: list[V1Cell] = field(default_factory=list)
    cifar_images_uint8: np.ndarray = field(default_factory=lambda: np.empty(0))

    # Canvas / sizing
    canvas_size: int = 256
    kernel_view_size: int = 192
    window_sec: float = 12.0
    fps: int = 30

    # Neuron selection
    active_neuron_index: int = 0
    sim_neurons: int = 1
    rf_scale_pct: int = 100
    mask_rf: bool = False
    gray_rf: bool = False
    gray_energy: bool = True
    edit_slot_index: int = 0
    rf_pos_by_neuron: dict[int, tuple[int, int]] = field(default_factory=dict)

    # Stimulus
    stim_kind: StimulusKind = "Bar"
    stim_center_x: int = 128
    stim_center_y: int = 128

    bar_orientation_deg: int = 45
    bar_length_px: int = 80
    bar_thickness_px: int = 6
    bar_size_px: int = 120
    bar_contrast_pct: int = 100
    bar_color: str = "White"

    gr_orientation_deg: int = 45
    gr_sf_x100: int = 10
    gr_phase_deg: int = 0
    gr_size_px: int = 128
    gr_contrast_pct: int = 100
    gr_color: str = "White"

    cifar_index: int = 0
    cifar_size_px: int = 96

    presentation_mode: PresentationMode = "Manual"
    on_ms: int = 100
    off_ms: int = 500
    auto_bar_step_deg: int = 20
    auto_gr_mode: GratingAutoMode = "Orientation"
    auto_gr_ori_step_deg: int = 15
    auto_gr_sf_step_x1000: int = 2

    # Dynamics
    response_mode: ResponseMode = "normalized"
    max_rate_hz: int = 100
    rate_ymax: int = 120
    rate_gain_raw: int = 25
    baseline_hz_raw: int = 0
    spike_buffer: int = 10

    # Running state
    paused: bool = False
    current_time: float = 0.0
    times: list[float] = field(default_factory=list)
    rates_by_neuron: list[list[float]] = field(default_factory=list)
    spike_times_by_neuron: list[list[float]] = field(default_factory=list)
    spike_waveforms_xy_by_neuron: list[list[tuple[np.ndarray, np.ndarray]]] = field(default_factory=list)
    _last_on_cycle_idx: int = -1
    _auto_cifar_index: int = 0

    @property
    def dt_s(self) -> float:
        return 1.0 / max(1, self.fps)

    @property
    def gain_hz(self) -> float:
        return self.rate_gain_raw / 5.0

    @property
    def baseline_hz(self) -> float:
        return float(self.baseline_hz_raw)


def init_sim_state(seed: int = 2026, canvas_size: int = 256) -> SimState:
    """Create a fully-initialised ``SimState`` with cells + CIFAR loaded."""
    cfg = make_config(profile="quick", enable_mlp=False)
    rng = set_seed(cfg.seed)
    v1_cells = generate_v1_cells(cfg, image_shape=(32, 32, 3))
    cifar = load_cifar_subset(cfg)
    return SimState(
        cfg=cfg,
        rng=rng,
        v1_cells=v1_cells,
        cifar_images_uint8=cifar,
        canvas_size=canvas_size,
        stim_center_x=canvas_size // 2,
        stim_center_y=canvas_size // 2,
    )


# ── Pure helpers ──────────────────────────────────────────────────────────────

def color_m11_from_name(name: str) -> tuple[float, float, float]:
    return COLOR_LUT.get(name.lower().strip(), (1.0, 1.0, 1.0))


def active_neuron_ids(state: SimState) -> list[int]:
    """Wrapping neuron-id list from current index + sim_neurons count."""
    n_total = len(state.v1_cells)
    n_sim = min(state.sim_neurons, n_total)
    start = state.active_neuron_index % n_total if n_total > 0 else 0
    return [(start + i) % n_total for i in range(n_sim)]


def edited_neuron_id(state: SimState) -> int:
    ids = active_neuron_ids(state)
    if not ids:
        return 0
    idx = min(state.edit_slot_index, len(ids) - 1)
    return ids[idx]


def scaled_rf_size(cell: V1Cell, scale_pct: int) -> int:
    return max(3, int(round(cell.size * scale_pct / 100.0)))


def ensure_rf_positions(state: SimState) -> None:
    """Ensure that every active neuron has an RF position entry."""
    for nid in active_neuron_ids(state):
        if nid not in state.rf_pos_by_neuron:
            sz = scaled_rf_size(state.v1_cells[nid], state.rf_scale_pct)
            m = max(0, state.canvas_size - sz) // 2
            state.rf_pos_by_neuron[nid] = (m, m)


def clamped_rf_coords(state: SimState, neuron_id: int) -> tuple[int, int]:
    """Return RF (x, y) for *neuron_id*, clamped to valid canvas range."""
    ensure_rf_positions(state)
    cell = state.v1_cells[neuron_id]
    sz = scaled_rf_size(cell, state.rf_scale_pct)
    max_rf = max(0, state.canvas_size - sz)
    x, y = state.rf_pos_by_neuron.get(neuron_id, (max_rf // 2, max_rf // 2))
    return int(np.clip(x, 0, max_rf)), int(np.clip(y, 0, max_rf))


def build_kernel(cell: V1Cell, state: SimState) -> np.ndarray:
    """Build the kernel array for *cell*, applying scale, grayscale, and mask."""
    kernel = resize_kernel_rgb(cell.kernel_rgb, scaled_rf_size(cell, state.rf_scale_pct))
    if state.gray_rf:
        kernel = to_grayscale_kernel(kernel, match_energy=state.gray_energy)
    if state.mask_rf:
        kernel = apply_rf_gray_outside(kernel, gray_value=0.0, radius_ratio=0.45, soft_edge_px=1.5)
    return kernel


def timed_cycle(state: SimState) -> tuple[int, bool, float]:
    """Return (cycle_index, visible, on_phase) for the current time."""
    on_s = state.on_ms / 1000.0
    off_s = state.off_ms / 1000.0
    period = max(1e-6, on_s + off_s)
    cycle_idx = int(state.current_time // period)
    phase = state.current_time - cycle_idx * period
    visible = phase < on_s
    on_phase = float(np.clip(phase / max(on_s, 1e-6), 0.0, 1.0)) if visible else 0.0
    return cycle_idx, visible, on_phase


def stimulus_visible(state: SimState) -> bool:
    if state.presentation_mode == "Manual":
        return True
    _, visible, _ = timed_cycle(state)
    return visible


def build_stimulus_patch(state: SimState) -> np.ndarray:
    """Generate the current stimulus RGB patch (float32, values ∈ [-1, 1])."""
    kind = state.stim_kind
    cycle_idx, _, on_phase = timed_cycle(state)
    timed = state.presentation_mode == "Timed experiment"

    # Auto-advance CIFAR index on each new cycle
    if timed and cycle_idx != state._last_on_cycle_idx and stimulus_visible(state):
        state._last_on_cycle_idx = cycle_idx
        if kind == "CIFAR patch":
            state._auto_cifar_index = int(state.rng.integers(0, state.cifar_images_uint8.shape[0]))

    if kind == "Bar":
        angle = float(state.bar_orientation_deg)
        if timed:
            angle += cycle_idx * state.auto_bar_step_deg
        return generate_bar_stimulus(
            size=state.bar_size_px,
            orientation_deg=float(angle % 180.0),
            length=float(state.bar_length_px),
            thickness=float(state.bar_thickness_px),
            contrast=state.bar_contrast_pct / 100.0,
            background_level=-1.0,
            color_rgb_m11=color_m11_from_name(state.bar_color),
        )

    if kind == "Grating":
        ori = float(state.gr_orientation_deg)
        sf = state.gr_sf_x100 / 100.0
        phase = float(state.gr_phase_deg)
        if timed:
            mode = state.auto_gr_mode
            if mode in ("Orientation", "Both"):
                ori += cycle_idx * state.auto_gr_ori_step_deg
            if mode in ("Frequency", "Both"):
                sf += cycle_idx * (state.auto_gr_sf_step_x1000 / 1000.0)
            phase += 360.0 * on_phase
        sf = float(np.clip(sf, 0.01, 0.45))
        return generate_grating_stimulus(
            size=state.gr_size_px,
            orientation_deg=float(ori % 180.0),
            spatial_frequency=sf,
            phase_deg=float(phase % 360.0),
            contrast=state.gr_contrast_pct / 100.0,
            background_level=-1.0,
            color_rgb_m11=color_m11_from_name(state.gr_color),
        )

    # CIFAR patch
    cifar_idx = int(state.cifar_index)
    if timed:
        cifar_idx = int(state._auto_cifar_index)
    return generate_cifar_patch_stimulus(
        cifar_images_uint8=state.cifar_images_uint8,
        image_index=cifar_idx,
        size=state.cifar_size_px,
    )


def compose_canvas_m11(state: SimState) -> np.ndarray:
    """Place stimulus on canvas; returns float32 (H, W, 3) in [-1, 1]."""
    base = np.ones((state.canvas_size, state.canvas_size, 3), dtype=np.float32) * -1.0
    if not stimulus_visible(state):
        return base
    patch = build_stimulus_patch(state)
    center = (float(state.stim_center_x), float(state.stim_center_y))
    return place_patch_on_canvas(base, patch, center)


def compute_responses(
    state: SimState,
    frame_m11: np.ndarray,
) -> tuple[list[float], list[float], list[V1Cell], list[np.ndarray]]:
    """
    Compute raw/rate responses for each active neuron.

    Returns:
        raw_values, rates_hz, cell_contexts, kernels
    """
    ids = active_neuron_ids(state)
    raws: list[float] = []
    rates: list[float] = []
    contexts: list[V1Cell] = []
    kernels: list[np.ndarray] = []

    for nid in ids:
        cell = state.v1_cells[nid]
        kernel = build_kernel(cell, state)
        rx, ry = clamped_rf_coords(state, nid)
        ctx = V1Cell(
            kernel_rgb=kernel,
            frequency=cell.frequency,
            theta=cell.theta,
            size=kernel.shape[0],
            coords=(rx, ry),
        )
        contexts.append(ctx)
        kernels.append(kernel)

        if state.response_mode == "normalized":
            raw = compute_single_frame_response_normalized(ctx, frame_m11, ctx.coords)
            sens = state.gain_hz / 5.0
            eff = float(np.tanh(sens * raw))
            rate = response_to_rate_hz_bounded(eff, max_rate_hz=float(state.max_rate_hz), baseline_hz=state.baseline_hz)
        else:
            raw = compute_single_frame_response(ctx, frame_m11, ctx.coords)
            rate = response_to_rate_hz(raw, gain=state.gain_hz, baseline_hz=state.baseline_hz)
        raws.append(raw)
        rates.append(rate)

    return raws, rates, contexts, kernels


def step_simulation(state: SimState) -> tuple[list[int], float]:
    """
    Advance simulation by one dt step.

    Updates ``state`` in-place (times, rates, spikes, waveforms).
    Returns (spike_flags, mean_rate).
    """
    ensure_rf_positions(state)
    frame_m11 = compose_canvas_m11(state)
    _raws, rates, contexts, kernels = compute_responses(state, frame_m11)

    ensure_trace_buffers(state)

    spikes: list[int] = []
    for i, ((_ctx, kernel), rate_hz) in enumerate(zip(zip(contexts, kernels), rates)):
        spike = poisson_spike_step(rate_hz, state.dt_s, state.rng)
        spikes.append(spike)

        state.rates_by_neuron[i].append(rate_hz)
        if spike == 1:
            state.spike_times_by_neuron[i].append(state.current_time + state.dt_s)
            t_ms, w0 = generate_spike_waveform_from_rf(kernel, duration_ms=1.5, sample_rate_hz=20000)
            t_ms, w = sample_noisy_jittered_spike_waveform(t_ms, w0, state.rng, 0.04, 0.035, 0.08)
            state.spike_waveforms_xy_by_neuron[i].append((t_ms, w))
            if len(state.spike_waveforms_xy_by_neuron[i]) > state.spike_buffer:
                state.spike_waveforms_xy_by_neuron[i] = state.spike_waveforms_xy_by_neuron[i][-state.spike_buffer:]

    mean_rate = float(np.mean(rates)) if rates else 0.0
    state.current_time += state.dt_s
    state.times.append(state.current_time)
    trim_history(state)

    return spikes, mean_rate


def ensure_trace_buffers(state: SimState) -> None:
    n = max(1, len(active_neuron_ids(state)))
    if len(state.rates_by_neuron) != n:
        state.rates_by_neuron = [[] for _ in range(n)]
        state.spike_times_by_neuron = [[] for _ in range(n)]
        state.spike_waveforms_xy_by_neuron = [[] for _ in range(n)]


def trim_history(state: SimState) -> None:
    if not state.times:
        return
    t_min = state.times[-1] - state.window_sec
    keep = 0
    for i, t in enumerate(state.times):
        if t >= t_min:
            keep = i
            break
    state.times = state.times[keep:]
    for i in range(len(state.rates_by_neuron)):
        state.rates_by_neuron[i] = state.rates_by_neuron[i][keep:]
        state.spike_times_by_neuron[i] = [t for t in state.spike_times_by_neuron[i] if t >= t_min]


def reset_buffers(state: SimState) -> None:
    """Clear all trace history and reset time to zero."""
    ids = active_neuron_ids(state)
    n = max(1, len(ids))
    state.current_time = 0.0
    state.times = []
    state.rates_by_neuron = [[] for _ in range(n)]
    state.spike_times_by_neuron = [[] for _ in range(n)]
    state.spike_waveforms_xy_by_neuron = [[] for _ in range(n)]
