#%% 00 | Course roadmap and learning goals
"""
NeuroAI for Biomedical Engineering (2026)

Why this matters:
This script is organized as notebook-like cells so students can run, inspect,
and modify each step of a full visual-neuroscience pipeline.
"""

print("NeuroAI 2026 class flow:")
print("1) Load natural images (CIFAR-10)")
print("2) Build synthetic V1-like receptive fields")
print("3) Simulate temporal responses and spikes")
print("4) Recover features with STA")
print("5) Compare tuning metrics")
print("6) Optional: nonlinear MLP comparison")

#%% 01 | Imports and reproducibility
"""
Why this matters:
All imports and reproducibility controls live in one place to keep execution
predictable across student laptops.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

_THIS_DIR = Path(__file__).resolve().parent if "__file__" in globals() else (Path.cwd() / "2026" if (Path.cwd() / "2026").exists() else Path.cwd())
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from computing import (
    ClassConfig,
    V1Cell,
    analyze_tuning,
    assert_timing_is_valid,
    build_temporal_inputs,
    build_trial_windows,
    compute_color_profile,
    compute_sta,
    draw_bar,
    generate_grating_stimulus,
    generate_v1_cells,
    load_cifar_subset,
    make_config,
    minimal_circular_diff_deg,
    normalize_firing_rates,
    set_seed,
    simulate_firing_rates,
    to_0_1,
    to_minus1_1,
    train_optional_mlp,
)
from plotting import kernel_to_rgba, norm_img, plot_random_grid

#%% 02 | Runtime profiles and config
"""
Why this matters:
Students can switch between quick/class/deep presets with one variable, then
customize any field they want.
"""

# Students can edit these two lines first.
PROFILE_NAME = "class"
ENABLE_MLP = True
# Set to a tuple like (0, 3, 5) to control neurons used in STA/MLP analysis cells.
ANALYSIS_NEURONS: tuple[int, ...] | None = None

cfg: ClassConfig = make_config(profile=PROFILE_NAME, enable_mlp=ENABLE_MLP)
print(cfg)

#%% 03 | Helper setup and validations
"""
Why this matters:
Central setup and checks make the remaining cells deterministic and safe.
"""

rng = set_seed(cfg.seed)
assert_timing_is_valid(cfg)
print("Cell 03 setup loaded.")

#%% 04 | Load and preview CIFAR-10 subset
"""
Why this matters:
Natural images are the input distribution for the entire pipeline.
"""

subset_images_uint8 = load_cifar_subset(cfg)
subset_images_01 = to_0_1(subset_images_uint8)
subset_images_m11 = to_minus1_1(subset_images_uint8)

print(f"subset_images_uint8 shape: {subset_images_uint8.shape}")
plot_random_grid(subset_images_01, nrows=4, ncols=4, seed=cfg.seed)

#%% 05 | Generate V1-like cell population
"""
Why this matters:
Each synthetic neuron has a kernel, preference parameters, and RF coordinates.
"""

v1_cells = generate_v1_cells(cfg, image_shape=subset_images_uint8.shape[1:])
print(f"Generated {len(v1_cells)} V1 cells")

img_h, img_w = subset_images_uint8.shape[1:3]
for i, cell in enumerate(v1_cells):
    x_rf, y_rf = cell.coords
    assert 0 <= x_rf <= img_w - cell.size, f"x RF out of bounds for neuron {i}"
    assert 0 <= y_rf <= img_h - cell.size, f"y RF out of bounds for neuron {i}"
print("RF bounds check passed")

v1_cells_again = generate_v1_cells(cfg, image_shape=subset_images_uint8.shape[1:])
meta_a = [(round(c.frequency, 6), round(c.theta, 6), c.size, c.coords) for c in v1_cells]
meta_b = [(round(c.frequency, 6), round(c.theta, 6), c.size, c.coords) for c in v1_cells_again]
assert meta_a == meta_b, "Seed reproducibility check failed"
print("Seed reproducibility check passed")

cols = 5
rows = int(np.ceil(len(v1_cells) / cols))
fig, axes = plt.subplots(rows, cols, figsize=(17, 4.0 * rows), dpi=170)
axes = np.atleast_1d(axes).flatten()

for ax, cell in zip(axes, v1_cells):
    cycles = cell.frequency * cell.size
    ax.imshow(norm_img(cell.kernel_rgb, per_channel=True))
    ax.set_title(f"S:{cell.size} f:{cell.frequency:.2f}\ncyc:{cycles:.2f} RF:{cell.coords}", fontsize=12, pad=8)
    ax.axis("off")
for ax in axes[len(v1_cells) :]:
    ax.axis("off")
plt.suptitle("Synthetic V1 kernels", fontsize=20, y=0.995)
fig.subplots_adjust(left=0.04, right=0.98, bottom=0.03, top=0.90, hspace=0.90, wspace=0.35)
plt.show()

#%% 06 | Receptive fields on images
"""
Why this matters:
RF overlays make it explicit where each neuron samples the visual field.
"""

selected_neuron = int(cfg.selected_neurons[0])
cell = v1_cells[selected_neuron]
x_rf, y_rf = cell.coords
kernel_rgba = kernel_to_rgba(cell.kernel_rgb, threshold=0.06)

images_5x5 = subset_images_01[:25]
fig, axes = plt.subplots(5, 5, figsize=(10, 10))
for ax, img in zip(axes.flat, images_5x5):
    ax.imshow(img, origin="upper", alpha=0.85)
    ax.imshow(
        kernel_rgba,
        extent=[x_rf, x_rf + cell.size, y_rf, y_rf + cell.size],
        origin="upper",
        interpolation="none",
    )
    ax.set_xlim(0, 32)
    ax.set_ylim(32, 0)
    ax.axis("off")
plt.suptitle(f"Single-neuron RF overlay (neuron {selected_neuron})", fontsize=14)
plt.tight_layout()
plt.show()

multi_neurons = [n for n in cfg.selected_neurons if n < len(v1_cells)][:6]
fig, axes = plt.subplots(5, 5, figsize=(11, 11))
for ax, img in zip(axes.flat, images_5x5):
    ax.imshow(img, origin="upper", vmin=0, vmax=1)
    for n in multi_neurons:
        c = v1_cells[n]
        xr, yr = c.coords
        ax.imshow(
            norm_img(c.kernel_rgb, per_channel=True),
            extent=[xr, xr + c.size, yr, yr + c.size],
            origin="upper",
            interpolation="none",
            alpha=0.55,
        )
    ax.set_xlim(0, 32)
    ax.set_ylim(32, 0)
    ax.axis("off")
plt.suptitle("Multi-neuron RF overlays", fontsize=14)
plt.tight_layout()
plt.show()

#%% 07 | Build timeline and temporal dataset
"""
Why this matters:
The timing structure (image vs intertrial) defines the experiment.
"""

h, w, c = subset_images_01.shape[1:]
n_timeline_trials = min(4, subset_images_01.shape[0])
trial_width_px = w
gray_width_px = int(round((cfg.intertrial_time_ms / cfg.trial_time_ms) * trial_width_px))
gray_patch = np.ones((h, gray_width_px, c), dtype=np.float32) * 0.5

segments = [gray_patch]
trial_positions_px: list[tuple[int, int]] = []
intertrial_positions_px: list[tuple[int, int]] = []
current_x = gray_width_px

for img in subset_images_01[:n_timeline_trials]:
    segments.append(img)
    trial_positions_px.append((current_x, current_x + trial_width_px))
    current_x += trial_width_px
    segments.append(gray_patch)
    intertrial_positions_px.append((current_x, current_x + gray_width_px))
    current_x += gray_width_px

timeline_image = np.concatenate(segments, axis=1)
seconds_per_pixel = (cfg.trial_time_ms / trial_width_px) / 1000.0

fig, ax = plt.subplots(figsize=(16, 4), dpi=150)
ax.imshow(timeline_image, vmin=0, vmax=1)
ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Pixels")

total_pixels = timeline_image.shape[1]
xticks = np.linspace(0, total_pixels, 6)
ax.set_xticks(xticks)
ax.set_xticklabels([f"{x * seconds_per_pixel:.1f}" for x in xticks])

for start, end in intertrial_positions_px:
    ax.axvspan(start, end, ymin=0.0, ymax=0.08, color="black", alpha=0.7)
for start, end in trial_positions_px:
    ax.axvspan(start, end, ymin=0.0, ymax=0.08, color="blue", alpha=0.6)

ax.set_title("Timeline (black = intertrial, blue = image)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.show()

temporal_inputs = build_temporal_inputs(subset_images_m11, cfg)
print(f"temporal_inputs shape: {temporal_inputs.shape}")

bins_image = cfg.trial_time_ms // cfg.bin_size_ms
bins_intertrial = cfg.intertrial_time_ms // cfg.bin_size_ms
expected_bins = bins_intertrial + cfg.num_images * (bins_image + bins_intertrial)
assert temporal_inputs.shape[0] == expected_bins, "Temporal length mismatch"
print("Temporal length check passed")

#%% 08 | Simulate firing rates
"""
Why this matters:
Kernel dot products produce a simplified V1 firing-rate model.
"""

firing_raw = simulate_firing_rates(temporal_inputs, v1_cells)
firing_rates = normalize_firing_rates(firing_raw, target_max=5.0)

print(f"firing_raw shape: {firing_raw.shape}")
print(f"firing_rates shape: {firing_rates.shape}")
assert firing_rates.shape == (temporal_inputs.shape[0], len(v1_cells)), "Firing shape mismatch"
assert np.all(firing_rates >= 0), "Firing should be non-negative after rectification"
print("Firing-rate sanity checks passed")

mean_rates = np.mean(firing_rates, axis=0)
plt.figure(figsize=(10, 4))
plt.bar(np.arange(len(mean_rates)), mean_rates, color="black")
plt.xlabel("Neuron index")
plt.ylabel("Mean normalized firing")
plt.title("Mean firing per neuron")
plt.tight_layout()
plt.show()

#%% 09 | Plot rates, raster, PSTH
"""
Why this matters:
Raster and PSTH are canonical views in systems neuroscience.
"""

n_seconds_to_plot = 10
bins_to_plot = min(int((n_seconds_to_plot * 1000) / cfg.bin_size_ms), firing_rates.shape[0])
time_axis = np.arange(bins_to_plot) * cfg.bin_size_ms / 1000.0

trial_windows_bins = build_trial_windows(cfg, n_trials=cfg.num_images)
trial_windows_sec = [
    (s * cfg.bin_size_ms / 1000.0, e * cfg.bin_size_ms / 1000.0)
    for s, e in trial_windows_bins
]

example_neuron = int(cfg.selected_neurons[0])
plt.figure(figsize=(10, 4), dpi=150)
plt.plot(time_axis, firing_rates[:bins_to_plot, example_neuron], color="black", lw=2)
for s_sec, e_sec in trial_windows_sec:
    if s_sec >= n_seconds_to_plot:
        break
    plt.axvspan(s_sec, min(e_sec, n_seconds_to_plot), color="blue", alpha=0.2)
plt.xlabel("Time (s)")
plt.ylabel("Normalized firing")
plt.title(f"Neuron {example_neuron}: firing rate")
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.show()

if ANALYSIS_NEURONS is None:
    selected_for_raster = [n for n in cfg.selected_neurons if n < len(v1_cells)][:3]
else:
    selected_for_raster = [n for n in ANALYSIS_NEURONS if 0 <= n < len(v1_cells)]
if len(selected_for_raster) == 0:
    selected_for_raster = [0]
total_bins_10ms = min(1000, firing_rates.shape[0])
total_time_ms = total_bins_10ms * cfg.bin_size_ms

for neuron in selected_for_raster:
    binned_rate = firing_rates[:total_bins_10ms, neuron]
    rate_1ms = np.repeat(binned_rate, cfg.bin_size_ms)
    prob_1ms = np.clip(rate_1ms / 10.0, 0.0, 1.0)

    experiment_runs = np.zeros((cfg.num_runs_raster, total_time_ms), dtype=np.int8)
    for run in range(cfg.num_runs_raster):
        experiment_runs[run] = (rng.random(total_time_ms) < prob_1ms).astype(np.int8)

    rebinned = experiment_runs.reshape(cfg.num_runs_raster, total_bins_10ms, cfg.bin_size_ms).sum(axis=2)
    psth = rebinned.mean(axis=0)
    time_axis_psth = (np.arange(total_bins_10ms) * cfg.bin_size_ms + cfg.bin_size_ms / 2) / 1000.0

    fig = plt.figure(figsize=(15, 7), dpi=130)
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1])
    ax_raster = fig.add_subplot(gs[0, 0])
    ax_psth = fig.add_subplot(gs[1, 0])
    ax_kernel = fig.add_subplot(gs[:, 1])

    for run in range(cfg.num_runs_raster):
        spike_idx = np.where(experiment_runs[run] == 1)[0]
        spike_times = spike_idx / 1000.0
        ax_raster.vlines(spike_times, run + 0.6, run + 1.4, color="black", lw=0.5)

    for s_sec, e_sec in trial_windows_sec:
        if s_sec > total_time_ms / 1000.0:
            break
        ax_raster.axvspan(s_sec, min(e_sec, total_time_ms / 1000.0), color="blue", alpha=0.15)
        ax_psth.axvspan(s_sec, min(e_sec, total_time_ms / 1000.0), color="blue", alpha=0.15)

    ax_raster.set_xlim(0, total_time_ms / 1000.0)
    ax_raster.set_ylim(0.5, cfg.num_runs_raster + 0.5)
    ax_raster.set_xlabel("Time (s)")
    ax_raster.set_ylabel("Run")
    ax_raster.set_title(f"Raster | neuron {neuron}")
    ax_raster.invert_yaxis()

    ax_psth.plot(time_axis_psth, psth, color="black", lw=2.5)
    ax_psth.set_xlim(0, total_time_ms / 1000.0)
    ax_psth.set_xlabel("Time (s)")
    ax_psth.set_ylabel("PSTH (spikes/bin)")
    ax_psth.set_title("PSTH")

    ax_kernel.imshow(norm_img(v1_cells[neuron].kernel_rgb, per_channel=True))
    ax_kernel.set_title("Neuron kernel")
    ax_kernel.axis("off")

    plt.suptitle(f"Spike simulation for neuron {neuron}", fontsize=14)
    plt.tight_layout()
    plt.show()

#%% 10 | Compute STA
"""
Why this matters:
STA recovers stimulus features associated with strong neuronal responses.
"""

max_trials_sta = min(cfg.num_trials_sta, cfg.num_images)
bins_total_sta = bins_intertrial + max_trials_sta * (bins_image + bins_intertrial)

stimulus_sta = temporal_inputs[:bins_total_sta]
rates_sta = firing_rates[:bins_total_sta]
selected_neurons_sta = list(range(len(v1_cells)))
sta_results = compute_sta(stimulus_sta, rates_sta, selected_neurons_sta)

assert sta_results[0].shape == stimulus_sta.shape[1:], "STA result shape mismatch"
print("STA shape check passed")

for neuron in selected_for_raster:
    cell = v1_cells[neuron]
    x_rf, y_rf = cell.coords
    s = cell.size
    sta_full = sta_results[neuron]
    cropped_sta = sta_full[y_rf : y_rf + s, x_rf : x_rf + s, :]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150)
    axes[0].imshow(norm_img(sta_full, per_channel=True))
    axes[0].add_patch(plt.Rectangle((x_rf, y_rf), s, s, edgecolor="black", facecolor="none", lw=2))
    axes[0].set_title(f"Full STA | neuron {neuron}")
    axes[0].axis("off")

    axes[1].imshow(norm_img(cropped_sta, per_channel=True))
    axes[1].set_title("Cropped STA (RF)")
    axes[1].axis("off")

    axes[2].imshow(norm_img(cell.kernel_rgb, per_channel=True))
    axes[2].set_title("Original kernel")
    axes[2].axis("off")
    plt.tight_layout()
    plt.show()

#%% 11 | Didactic stimuli for orientation and spatial-frequency tuning
"""
Why this matters:
These are the canonical stimulus families used to probe orientation and
spatial-frequency selectivity in V1-like neurons.
"""

ori_examples = [0, 30, 60, 90, 120, 150]
freq_examples = [0.04, 0.08, 0.14, 0.20, 0.28, 0.34]
stim_size = 128

fig, axes = plt.subplots(2, 6, figsize=(16, 5), dpi=160)
for i, ori in enumerate(ori_examples):
    stim = generate_grating_stimulus(
        size=stim_size,
        orientation_deg=float(ori),
        spatial_frequency=0.12,
        phase_deg=0.0,
        contrast=1.0,
        background_level=-1.0,
    )
    axes[0, i].imshow(norm_img(stim, per_channel=True))
    axes[0, i].set_title(f"{ori} deg", fontsize=11)
    axes[0, i].axis("off")

for i, sf in enumerate(freq_examples):
    stim = generate_grating_stimulus(
        size=stim_size,
        orientation_deg=60.0,
        spatial_frequency=float(sf),
        phase_deg=0.0,
        contrast=1.0,
        background_level=-1.0,
    )
    axes[1, i].imshow(norm_img(stim, per_channel=True))
    axes[1, i].set_title(f"sf={sf:.2f}", fontsize=11)
    axes[1, i].axis("off")

axes[0, 0].text(
    -0.55,
    0.5,
    "Orientation sweep",
    transform=axes[0, 0].transAxes,
    va="center",
    rotation=90,
    fontsize=11,
)
axes[1, 0].text(
    -0.55,
    0.5,
    "Frequency sweep",
    transform=axes[1, 0].transAxes,
    va="center",
    rotation=90,
    fontsize=11,
)
plt.suptitle("Didactic tuning stimuli: gratings", fontsize=16)
plt.tight_layout()
plt.show()

#%% 12 | Core tuning analysis (Original vs STA)
"""
Why this matters:
This is the core quantitative evaluation of RF recovery quality.
"""

core_tuning_rows = []

for neuron in selected_for_raster:
    cell = v1_cells[neuron]
    x_rf, y_rf = cell.coords
    s = cell.size
    sta_full = sta_results[neuron]
    cropped_sta = sta_full[y_rf : y_rf + s, x_rf : x_rf + s, :]

    tune_orig = analyze_tuning(cell.kernel_rgb)
    tune_sta = analyze_tuning(cropped_sta)

    ori_err = minimal_circular_diff_deg(tune_orig.preferred_orientation_deg, tune_sta.preferred_orientation_deg)
    freq_err = abs(tune_orig.preferred_frequency - tune_sta.preferred_frequency)
    core_tuning_rows.append((neuron, ori_err, freq_err, tune_orig, tune_sta))

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=140)
    axes[0, 0].imshow(norm_img(cell.kernel_rgb, per_channel=True))
    axes[0, 0].set_title("Original kernel")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(norm_img(cropped_sta, per_channel=True))
    axes[0, 1].set_title("Cropped STA")
    axes[0, 1].axis("off")

    axes[1, 0].plot(tune_orig.orientation_axis_deg, tune_orig.curve_orientation, color="black", lw=2, label="Original")
    axes[1, 0].plot(tune_sta.orientation_axis_deg, tune_sta.curve_orientation, color="red", lw=2, label="STA")
    axes[1, 0].axvline(tune_orig.preferred_orientation_deg, color="black", ls="--", alpha=0.6)
    axes[1, 0].axvline(tune_sta.preferred_orientation_deg, color="red", ls="--", alpha=0.6)
    axes[1, 0].set_xlim(0, 180)
    axes[1, 0].set_xlabel("Orientation (deg)")
    axes[1, 0].set_ylabel("Normalized response")
    axes[1, 0].set_title(f"Orientation tuning | error {ori_err:.1f} deg")
    axes[1, 0].legend()

    axes[1, 1].plot(tune_orig.frequency_axis, tune_orig.curve_frequency, color="black", lw=2, label="Original")
    axes[1, 1].plot(tune_sta.frequency_axis, tune_sta.curve_frequency, color="red", lw=2, label="STA")
    axes[1, 1].axvline(tune_orig.preferred_frequency, color="black", ls="--", alpha=0.6)
    axes[1, 1].axvline(tune_sta.preferred_frequency, color="red", ls="--", alpha=0.6)
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_xlabel("Normalized spatial frequency")
    axes[1, 1].set_ylabel("Normalized power")
    axes[1, 1].set_title(f"Frequency tuning | error {freq_err:.3f}")
    axes[1, 1].legend()

    plt.suptitle(f"Neuron {neuron} core tuning", fontsize=14)
    plt.tight_layout()
    plt.show()

summary_ids = [r[0] for r in core_tuning_rows]
summary_ori = [r[1] for r in core_tuning_rows]
summary_freq = [r[2] for r in core_tuning_rows]

fig, axes = plt.subplots(2, 1, figsize=(10, 7), dpi=140)
axes[0].bar(summary_ids, summary_ori, color="firebrick", alpha=0.8)
axes[0].set_ylabel("Orientation error (deg)")
axes[0].set_title("STA orientation recovery error")
axes[0].set_xticks(summary_ids)
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

axes[1].bar(summary_ids, summary_freq, color="black", alpha=0.8)
axes[1].set_ylabel("Frequency error")
axes[1].set_xlabel("Neuron")
axes[1].set_title("STA frequency recovery error")
axes[1].set_xticks(summary_ids)
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)
plt.tight_layout()
plt.show()

#%% 13 | Optional MLP module
"""
Why this matters:
Adds a nonlinear model baseline while keeping the core path independent.
"""

targets_mlp = np.maximum(firing_raw[:bins_total_sta], 0.0)
mlp_weights = train_optional_mlp(stimulus_sta, targets_mlp, cfg)

if mlp_weights is not None:
    n_example = selected_for_raster[0]
    plt.figure(figsize=(4, 4), dpi=140)
    plt.imshow(norm_img(mlp_weights[n_example], per_channel=True))
    plt.title(f"MLP weight map | neuron {n_example}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

#%% 14 | Optional 3-way tuning (Original/STA/MLP)
"""
Why this matters:
Compares original kernels to both linear (STA) and nonlinear (MLP) estimates.
"""

if mlp_weights is None:
    print("Skipping 3-way tuning: no MLP weights available.")
else:
    neuron_ids = []
    ori_err_sta = []
    ori_err_mlp = []
    freq_err_sta = []
    freq_err_mlp = []

    for neuron in selected_for_raster:
        cell = v1_cells[neuron]
        x_rf, y_rf = cell.coords
        s = cell.size

        sta_full = sta_results[neuron]
        mlp_full = mlp_weights[neuron]
        cropped_sta = sta_full[y_rf : y_rf + s, x_rf : x_rf + s, :]
        cropped_mlp = mlp_full[y_rf : y_rf + s, x_rf : x_rf + s, :]

        t_orig = analyze_tuning(cell.kernel_rgb)
        t_sta = analyze_tuning(cropped_sta)
        t_mlp = analyze_tuning(cropped_mlp)

        neuron_ids.append(neuron)
        ori_err_sta.append(minimal_circular_diff_deg(t_orig.preferred_orientation_deg, t_sta.preferred_orientation_deg))
        ori_err_mlp.append(minimal_circular_diff_deg(t_orig.preferred_orientation_deg, t_mlp.preferred_orientation_deg))
        freq_err_sta.append(abs(t_orig.preferred_frequency - t_sta.preferred_frequency))
        freq_err_mlp.append(abs(t_orig.preferred_frequency - t_mlp.preferred_frequency))

    x = np.arange(len(neuron_ids))
    width = 0.35
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), dpi=140)
    axes[0].bar(x - width / 2, ori_err_sta, width, label="Original vs STA", color="red", alpha=0.8)
    axes[0].bar(x + width / 2, ori_err_mlp, width, label="Original vs MLP", color="blue", alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(neuron_ids)
    axes[0].set_ylabel("Orientation error (deg)")
    axes[0].set_title("3-way orientation error")
    axes[0].legend()

    axes[1].bar(x - width / 2, freq_err_sta, width, label="Original vs STA", color="red", alpha=0.8)
    axes[1].bar(x + width / 2, freq_err_mlp, width, label="Original vs MLP", color="blue", alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(neuron_ids)
    axes[1].set_ylabel("Frequency error")
    axes[1].set_xlabel("Neuron")
    axes[1].set_title("3-way frequency error")
    axes[1].legend()
    plt.tight_layout()
    plt.show()

#%% 15 | Didactic stimuli for color tuning
"""
Why this matters:
Color tuning is probed with controlled chromatic stimuli to compare channel
sensitivity patterns.
"""

color_names = ["Red", "Green", "Blue", "Yellow", "Cyan", "Magenta"]
color_values = [
    (1.0, -1.0, -1.0),
    (-1.0, 1.0, -1.0),
    (-1.0, -1.0, 1.0),
    (1.0, 1.0, -1.0),
    (-1.0, 1.0, 1.0),
    (1.0, -1.0, 1.0),
]

fig, axes = plt.subplots(1, len(color_values), figsize=(14, 3), dpi=160)
for ax, name, rgb_m11 in zip(axes, color_names, color_values):
    patch = np.zeros((64, 64, 3), dtype=np.float32)
    patch[..., 0] = rgb_m11[0]
    patch[..., 1] = rgb_m11[1]
    patch[..., 2] = rgb_m11[2]
    ax.imshow((np.clip(patch, -1.0, 1.0) + 1.0) / 2.0)
    ax.set_title(name, fontsize=11)
    ax.axis("off")
plt.suptitle("Didactic color stimuli", fontsize=16)
plt.tight_layout()
plt.show()

#%% 16 | Optional color tuning + synthetic bar experiment
"""
Why this matters:
Adds enrichment on channel selectivity and classic oriented-bar stimuli.
"""

for neuron in selected_for_raster:
    cell = v1_cells[neuron]
    x_rf, y_rf = cell.coords
    s = cell.size
    sta_full = sta_results[neuron]
    cropped_sta = sta_full[y_rf : y_rf + s, x_rf : x_rf + s, :]

    prof_orig = compute_color_profile(cell.kernel_rgb)
    prof_sta = compute_color_profile(cropped_sta)

    labels = ["R", "G", "B"]
    x = np.arange(3)
    width = 0.35
    plt.figure(figsize=(7, 4), dpi=130)
    plt.bar(x - width / 2, prof_orig, width, color=["red", "green", "blue"], alpha=0.8, label="Original")
    plt.bar(x + width / 2, prof_sta, width, color=["salmon", "lightgreen", "lightskyblue"], alpha=0.9, label="STA")
    plt.xticks(x, labels)
    plt.ylabel("Normalized channel energy")
    plt.title(f"Neuron {neuron} color profile")
    plt.legend()
    plt.tight_layout()
    plt.show()

n_orientations = 8
frames_per_orientation = 30
bar_length = 24
bar_thickness = 2
background_value = 128

synthetic_stimuli = []
orientation_labels_rad = []
for i in range(n_orientations):
    angle = i * np.pi / 4.0
    margin = int(np.ceil(bar_length / 2.0)) + 1
    x_start, x_end = (margin, 32 - margin) if np.cos(angle) >= 0 else (32 - margin, margin)
    y_start, y_end = (margin, 32 - margin) if np.sin(angle) >= 0 else (32 - margin, margin)

    xs = np.linspace(x_start, x_end, frames_per_orientation)
    ys = np.linspace(y_start, y_end, frames_per_orientation)
    for j in range(frames_per_orientation):
        img = np.ones((32, 32, 3), dtype=np.uint8) * background_value
        img = draw_bar(
            img,
            center=(float(xs[j]), float(ys[j])),
            angle_rad=angle,
            length=bar_length,
            thickness=bar_thickness,
            color_rgb=(255, 255, 255),
        )
        synthetic_stimuli.append(img)
        orientation_labels_rad.append(angle)

synthetic_stimuli = np.array(synthetic_stimuli, dtype=np.uint8)
orientation_labels_rad = np.array(orientation_labels_rad, dtype=np.float32)
synthetic_stimuli_m11 = to_minus1_1(synthetic_stimuli)

fig, axes = plt.subplots(n_orientations, 4, figsize=(10, 2.2 * n_orientations))
for i in range(n_orientations):
    for j in range(4):
        idx = i * frames_per_orientation + j * max(1, frames_per_orientation // 4)
        axes[i, j].imshow(synthetic_stimuli[idx])
        axes[i, j].set_title(f"{np.degrees(orientation_labels_rad[idx]):.0f} deg", fontsize=8)
        axes[i, j].axis("off")
plt.suptitle("Synthetic oriented bars")
plt.tight_layout()
plt.show()

sim_neuron = selected_for_raster[0]
sim_cell = v1_cells[sim_neuron]
x_rf, y_rf = sim_cell.coords
s = sim_cell.size
sim_sta = sta_results[sim_neuron][y_rf : y_rf + s, x_rf : x_rf + s, :]

resp_orig = []
resp_sta = []
resp_mlp = []
for img in synthetic_stimuli_m11:
    patch = img[y_rf : y_rf + s, x_rf : x_rf + s, :]
    resp_orig.append(max(0.0, float(np.sum(patch * sim_cell.kernel_rgb))))
    resp_sta.append(max(0.0, float(np.sum(patch * sim_sta))))
    if mlp_weights is not None:
        resp_mlp.append(max(0.0, float(np.sum(img * mlp_weights[sim_neuron]))))

resp_orig = np.array(resp_orig)
resp_sta = np.array(resp_sta)
if np.max(resp_orig) > 0:
    resp_orig = 10.0 * resp_orig / np.max(resp_orig)
if np.max(resp_sta) > 0:
    resp_sta = 10.0 * resp_sta / np.max(resp_sta)

if mlp_weights is not None:
    resp_mlp = np.array(resp_mlp)
    if np.max(resp_mlp) > 0:
        resp_mlp = 10.0 * resp_mlp / np.max(resp_mlp)

fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=True)
for i in range(n_orientations):
    ax = axes[i // 4, i % 4]
    start = i * frames_per_orientation
    end = start + frames_per_orientation
    x = np.arange(frames_per_orientation)
    ax.plot(x, resp_orig[start:end], color="black", lw=2, label="Original")
    ax.plot(x, resp_sta[start:end], color="red", lw=2, label="STA")
    if mlp_weights is not None:
        ax.plot(x, resp_mlp[start:end], color="blue", lw=2, label="MLP")
    ax.set_title(f"{np.degrees(orientation_labels_rad[start]):.0f} deg")
    ax.set_xlabel("Frame")
    if i % 4 == 0:
        ax.set_ylabel("Norm response")

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right")
fig.suptitle(f"Synthetic bar responses | neuron {sim_neuron}", fontsize=14)
plt.tight_layout()
plt.show()

#%% 17 | Multi-neuron recovery panel (Original vs STA vs MLP)
"""
Why this matters:
This compact panel compares RF recovery quality neuron-by-neuron in one view.
"""

# Edit this list to control which neurons are compared.
comparison_neurons = [n for n in selected_for_raster if 0 <= n < len(v1_cells)]
if len(comparison_neurons) == 0:
    comparison_neurons = [0]

sta_compare = compute_sta(stimulus_sta, rates_sta, comparison_neurons)
mlp_compare = train_optional_mlp(stimulus_sta, targets_mlp, cfg) if cfg.enable_mlp else None

n_cols = len(comparison_neurons)
fig, axes = plt.subplots(3, n_cols, figsize=(3.8 * n_cols, 9.0), dpi=160)
axes = np.atleast_2d(axes)
if axes.shape[1] != n_cols:
    axes = axes.T

for col, neuron in enumerate(comparison_neurons):
    cell = v1_cells[neuron]
    x_rf, y_rf = cell.coords
    s = cell.size

    orig = cell.kernel_rgb
    sta_crop = sta_compare[neuron][y_rf : y_rf + s, x_rf : x_rf + s, :]
    mlp_crop = None
    if mlp_compare is not None and neuron in mlp_compare:
        mlp_crop = mlp_compare[neuron][y_rf : y_rf + s, x_rf : x_rf + s, :]

    axes[0, col].imshow(norm_img(orig, per_channel=True))
    axes[0, col].set_title(f"Neuron {neuron}", fontsize=11)
    axes[0, col].axis("off")

    axes[1, col].imshow(norm_img(sta_crop, per_channel=True))
    axes[1, col].axis("off")

    if mlp_crop is not None:
        axes[2, col].imshow(norm_img(mlp_crop, per_channel=True))
    else:
        axes[2, col].text(0.5, 0.5, "MLP unavailable", ha="center", va="center", fontsize=11)
        axes[2, col].set_xlim(0, 1)
        axes[2, col].set_ylim(0, 1)
    axes[2, col].axis("off")

axes[0, 0].set_ylabel("Original", fontsize=12)
axes[1, 0].set_ylabel("STA", fontsize=12)
axes[2, 0].set_ylabel("MLP", fontsize=12)
plt.suptitle("Recovered receptive fields for selected neurons", fontsize=16)
plt.tight_layout()
plt.show()

#%% 18 | Wrap-up and student exercises
"""
Why this matters:
Provides clear next steps to turn this notebook-script into active learning.
"""

print("Wrap-up:")
print("- Core pipeline complete: images -> V1 -> rates -> spikes -> STA -> tuning")
print("- Optional modules are enabled by setting ENABLE_MLP=True")
print("")
print("Suggested exercises:")
print("1) Switch PROFILE_NAME between quick/class/deep and compare tuning stability")
print("2) Change cfg.sizes and cfg.frequency_range to probe RF recovery behavior")
print("3) Increase cfg.n_v1_cells and inspect population diversity")
print("4) Enable MLP and compare 3-way orientation/frequency errors")
print("5) Modify synthetic bar length/thickness/speed and inspect response changes")
