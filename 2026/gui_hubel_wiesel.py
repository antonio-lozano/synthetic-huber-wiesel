import sys
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from computing import (
    apply_rf_gray_outside,
    ClassConfig,
    V1Cell,
    compute_single_frame_response,
    compute_single_frame_response_normalized,
    generate_spike_waveform_from_rf,
    generate_bar_stimulus,
    generate_cifar_patch_stimulus,
    generate_grating_stimulus,
    generate_v1_cells,
    load_cifar_subset,
    make_config,
    place_patch_on_canvas,
    poisson_spike_step,
    response_to_rate_hz_bounded,
    sample_noisy_jittered_spike_waveform,
    to_grayscale_kernel,
    resize_kernel_rgb,
    response_to_rate_hz,
    set_seed,
)
from plotting_GUI import (
    centered_kernel_preview_uint8,
    draw_rf_box_uint8,
    m11_to_uint8_rgb,
    overlay_waveforms_to_lines,
    style_image_plot,
    style_plot_widget,
)
from sounds import SpikeRateAudioEngine


pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)
NEURON_COLORS = ["w", "b", "r", "g", "y"]


class StimulusCanvasPlot(pg.PlotWidget):
    moved = Signal(float, float)

    def __init__(self) -> None:
        super().__init__()
        self._drag = False
        self.setMouseEnabled(x=False, y=False)
        self.setMenuEnabled(False)

    def _emit_scene_pos(self, pos) -> None:
        scene_pos = self.mapToScene(pos.toPoint())
        view_pos = self.getPlotItem().vb.mapSceneToView(scene_pos)
        self.moved.emit(float(view_pos.x()), float(view_pos.y()))

    def mousePressEvent(self, ev) -> None:
        if ev.button() == Qt.LeftButton:
            self._drag = True
            self._emit_scene_pos(ev.position())
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev) -> None:
        if self._drag:
            self._emit_scene_pos(ev.position())
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev) -> None:
        if ev.button() == Qt.LeftButton:
            self._drag = False
        super().mouseReleaseEvent(ev)


class HubelWieselGUI(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Hubel-Wiesel Synthetic V1 Explorer (2026)")
        self.resize(1700, 980)

        self.cfg: ClassConfig = make_config(profile="quick", enable_mlp=False)
        self.rng = set_seed(self.cfg.seed)
        self.v1_cells = generate_v1_cells(self.cfg, image_shape=(32, 32, 3))
        self.cifar_images_uint8 = load_cifar_subset(self.cfg)

        self.canvas_size = 256
        self.window_sec = 12.0
        self.current_time = 0.0
        self.fps = 30
        self.dt_s = 1.0 / self.fps
        self.kernel_view_size = 192

        self.times: list[float] = []
        self.rates_by_neuron: list[list[float]] = []
        self.spike_times_by_neuron: list[list[float]] = []
        self.spike_waveforms_xy_by_neuron: list[list[tuple[np.ndarray, np.ndarray]]] = []
        self.rf_pos_by_neuron: dict[int, tuple[int, int]] = {}
        self._last_on_cycle_idx = -1
        self._auto_cifar_index = 0
        self.audio = SpikeRateAudioEngine(seed=self.cfg.seed)

        self._build_ui()
        self._setup_timer()
        self._sync_neuron_and_rf_limits(reset_to_center=True)
        self._refresh_kernel_preview()
        self._render_frame_and_plots(step_sim=False)

    def _build_ui(self) -> None:
        self.setStyleSheet(
            """
            QWidget { background-color: black; color: white; font-size: 16px; }
            QGroupBox { border: 1px solid white; margin-top: 10px; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }
            QPushButton { border: 1px solid white; padding: 6px; }
            QSlider::groove:horizontal { border: 1px solid white; height: 6px; background: #111; }
            QSlider::handle:horizontal { background: white; border: 1px solid white; width: 14px; margin: -5px 0; }
            """
        )

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        controls = QVBoxLayout()
        controls.setSpacing(12)
        root.addLayout(controls, 0)

        visuals = QVBoxLayout()
        root.addLayout(visuals, 1)

        # Controls: neuron + RF
        grp_neuron = QGroupBox("Neuron and RF")
        f_neuron = QFormLayout(grp_neuron)

        self.neuron_combo = QComboBox()
        self.neuron_combo.addItems([f"Neuron {i}" for i in range(len(self.v1_cells))])
        self.neuron_combo.currentIndexChanged.connect(self._on_neuron_changed)
        f_neuron.addRow("Select neuron", self.neuron_combo)

        self.neuron_spin = QSpinBox()
        self.neuron_spin.setRange(0, len(self.v1_cells) - 1)
        self.neuron_spin.setValue(0)
        self.neuron_spin.valueChanged.connect(self._on_neuron_spin_changed)
        f_neuron.addRow("Neuron index", self.neuron_spin)

        self.sim_neurons_spin = QSpinBox()
        self.sim_neurons_spin.setRange(1, 5)
        self.sim_neurons_spin.setValue(1)
        self.sim_neurons_spin.valueChanged.connect(self._on_sim_neurons_changed)
        f_neuron.addRow("Sim neurons", self.sim_neurons_spin)

        self.rf_x_slider, self.rf_x_label = self._make_slider(0, 100, 0, self._on_rf_slider_changed)
        self.rf_y_slider, self.rf_y_label = self._make_slider(0, 100, 0, self._on_rf_slider_changed)
        self.rf_x_spin = QSpinBox()
        self.rf_y_spin = QSpinBox()
        self.rf_x_spin.valueChanged.connect(self._on_rf_spin_changed)
        self.rf_y_spin.valueChanged.connect(self._on_rf_spin_changed)
        self.rf_slot_combo = QComboBox()
        self.rf_slot_combo.addItems(["N1", "N2", "N3", "N4", "N5"])
        self.rf_slot_combo.currentIndexChanged.connect(self._on_rf_slot_changed)
        f_neuron.addRow("Edit RF for", self.rf_slot_combo)
        f_neuron.addRow("RF X", self._stack_slider_spin_row(self.rf_x_slider, self.rf_x_label, self.rf_x_spin))
        f_neuron.addRow("RF Y", self._stack_slider_spin_row(self.rf_y_slider, self.rf_y_label, self.rf_y_spin))

        self.rf_scale_slider, self.rf_scale_label = self._make_slider(50, 400, 100, self._on_rf_scale_changed)
        self.rf_scale_spin = QSpinBox()
        self.rf_scale_spin.setRange(50, 400)
        self.rf_scale_spin.setValue(100)
        self.rf_scale_spin.valueChanged.connect(self._on_rf_scale_spin_changed)
        f_neuron.addRow("RF scale (%)", self._stack_slider_spin_row(self.rf_scale_slider, self.rf_scale_label, self.rf_scale_spin))

        self.mask_rf_btn = QPushButton("Mask RF: OFF")
        self.mask_rf_btn.setCheckable(True)
        self.mask_rf_btn.clicked.connect(self._on_mask_rf_toggled)
        f_neuron.addRow("RF masking", self.mask_rf_btn)

        self.gray_rf_btn = QPushButton("RF grayscale: OFF")
        self.gray_rf_btn.setCheckable(True)
        self.gray_rf_btn.clicked.connect(self._on_gray_rf_toggled)
        f_neuron.addRow("RF grayscale", self.gray_rf_btn)

        self.gray_rf_energy_btn = QPushButton("Gray energy match: ON")
        self.gray_rf_energy_btn.setCheckable(True)
        self.gray_rf_energy_btn.setChecked(True)
        self.gray_rf_energy_btn.clicked.connect(self._on_gray_energy_toggled)
        f_neuron.addRow("Gray scaling", self.gray_rf_energy_btn)
        controls.addWidget(grp_neuron)

        # Controls: stimulus
        grp_stim = QGroupBox("Stimulus")
        f_stim = QFormLayout(grp_stim)
        self.stim_combo = QComboBox()
        self.stim_combo.addItems(["Bar", "Grating", "CIFAR patch"])
        self.stim_combo.currentIndexChanged.connect(self._on_stim_type_changed)
        f_stim.addRow("Type", self.stim_combo)

        self.stim_x_slider, self.stim_x_label = self._make_slider(0, self.canvas_size - 1, self.canvas_size // 2, self._on_stim_slider_changed)
        self.stim_y_slider, self.stim_y_label = self._make_slider(0, self.canvas_size - 1, self.canvas_size // 2, self._on_stim_slider_changed)
        f_stim.addRow("Stim X", self._stack_slider_row(self.stim_x_slider, self.stim_x_label))
        f_stim.addRow("Stim Y", self._stack_slider_row(self.stim_y_slider, self.stim_y_label))

        self.bar_orientation, self.bar_orientation_label = self._make_slider(0, 179, 45, self._on_any_param_changed)
        self.bar_length, self.bar_length_label = self._make_slider(6, 140, 80, self._on_any_param_changed)
        self.bar_thickness, self.bar_thickness_label = self._make_slider(1, 30, 6, self._on_any_param_changed)
        self.bar_size, self.bar_size_label = self._make_slider(16, 180, 120, self._on_any_param_changed)
        self.bar_contrast, self.bar_contrast_label = self._make_slider(10, 100, 100, self._on_any_param_changed)
        self.bar_color = QComboBox()
        self.bar_color.addItems(["White", "Red", "Green", "Blue", "Yellow", "Cyan", "Magenta"])
        self.bar_color.currentIndexChanged.connect(self._on_any_param_changed)
        f_stim.addRow("Bar angle", self._stack_slider_row(self.bar_orientation, self.bar_orientation_label))
        f_stim.addRow("Bar length", self._stack_slider_row(self.bar_length, self.bar_length_label))
        f_stim.addRow("Bar thick", self._stack_slider_row(self.bar_thickness, self.bar_thickness_label))
        f_stim.addRow("Bar size", self._stack_slider_row(self.bar_size, self.bar_size_label))
        f_stim.addRow("Bar contrast", self._stack_slider_row(self.bar_contrast, self.bar_contrast_label))
        f_stim.addRow("Bar color", self.bar_color)

        self.gr_orientation, self.gr_orientation_label = self._make_slider(0, 179, 45, self._on_any_param_changed)
        self.gr_sf, self.gr_sf_label = self._make_slider(2, 35, 10, self._on_any_param_changed)
        self.gr_phase, self.gr_phase_label = self._make_slider(0, 360, 0, self._on_any_param_changed)
        self.gr_size, self.gr_size_label = self._make_slider(16, 180, 128, self._on_any_param_changed)
        self.gr_contrast, self.gr_contrast_label = self._make_slider(10, 100, 100, self._on_any_param_changed)
        self.gr_color = QComboBox()
        self.gr_color.addItems(["White", "Red", "Green", "Blue", "Yellow", "Cyan", "Magenta"])
        self.gr_color.currentIndexChanged.connect(self._on_any_param_changed)
        f_stim.addRow("Grating angle", self._stack_slider_row(self.gr_orientation, self.gr_orientation_label))
        f_stim.addRow("Spatial freq", self._stack_slider_row(self.gr_sf, self.gr_sf_label))
        f_stim.addRow("Phase", self._stack_slider_row(self.gr_phase, self.gr_phase_label))
        f_stim.addRow("Grating size", self._stack_slider_row(self.gr_size, self.gr_size_label))
        f_stim.addRow("Grating contrast", self._stack_slider_row(self.gr_contrast, self.gr_contrast_label))
        f_stim.addRow("Grating color", self.gr_color)

        self.cifar_index = QSpinBox()
        self.cifar_index.setRange(0, self.cifar_images_uint8.shape[0] - 1)
        self.cifar_index.setValue(0)
        self.cifar_index.valueChanged.connect(self._on_any_param_changed)
        self.cifar_size, self.cifar_size_label = self._make_slider(16, 180, 96, self._on_any_param_changed)
        f_stim.addRow("CIFAR index", self.cifar_index)
        f_stim.addRow("CIFAR size", self._stack_slider_row(self.cifar_size, self.cifar_size_label))

        self.presentation_mode = QComboBox()
        self.presentation_mode.addItems(["Manual", "Timed experiment"])
        self.presentation_mode.currentIndexChanged.connect(self._on_any_param_changed)
        self.on_ms = QSpinBox()
        self.on_ms.setRange(10, 5000)
        self.on_ms.setSingleStep(10)
        self.on_ms.setValue(100)
        self.on_ms.valueChanged.connect(self._on_any_param_changed)
        self.off_ms = QSpinBox()
        self.off_ms.setRange(10, 5000)
        self.off_ms.setSingleStep(10)
        self.off_ms.setValue(500)
        self.off_ms.valueChanged.connect(self._on_any_param_changed)
        self.auto_bar_step = QSpinBox()
        self.auto_bar_step.setRange(0, 180)
        self.auto_bar_step.setValue(20)
        self.auto_bar_step.valueChanged.connect(self._on_any_param_changed)
        self.auto_gr_mode = QComboBox()
        self.auto_gr_mode.addItems(["Orientation", "Frequency", "Both"])
        self.auto_gr_mode.currentIndexChanged.connect(self._on_any_param_changed)
        self.auto_gr_ori_step = QSpinBox()
        self.auto_gr_ori_step.setRange(0, 180)
        self.auto_gr_ori_step.setValue(15)
        self.auto_gr_ori_step.valueChanged.connect(self._on_any_param_changed)
        self.auto_gr_sf_step = QSpinBox()
        self.auto_gr_sf_step.setRange(0, 100)
        self.auto_gr_sf_step.setValue(2)
        self.auto_gr_sf_step.valueChanged.connect(self._on_any_param_changed)
        f_stim.addRow("Presentation", self.presentation_mode)
        f_stim.addRow("Stim ON (ms)", self.on_ms)
        f_stim.addRow("Stim OFF (ms)", self.off_ms)
        f_stim.addRow("Bar step (deg/presentation)", self.auto_bar_step)
        f_stim.addRow("Grating auto", self.auto_gr_mode)
        f_stim.addRow("Grating ori step", self.auto_gr_ori_step)
        f_stim.addRow("Grating sf step x1e-3", self.auto_gr_sf_step)

        controls.addWidget(grp_stim)

        grp_dyn = QGroupBox("Dynamics")
        f_dyn = QFormLayout(grp_dyn)
        self.response_mode = QComboBox()
        self.response_mode.addItems(["Normalized (0..Max Hz)", "Legacy (unbounded dot)"])
        self.response_mode.setCurrentIndex(0)
        self.response_mode.currentIndexChanged.connect(self._on_any_param_changed)
        self.max_rate_slider, self.max_rate_label = self._make_slider(20, 300, 100, self._on_any_param_changed)
        self.rate_ymax_slider, self.rate_ymax_label = self._make_slider(20, 1000, 120, self._on_any_param_changed)
        self.rate_gain, self.rate_gain_label = self._make_slider(1, 120, 25, self._on_any_param_changed)
        self.rate_base, self.rate_base_label = self._make_slider(0, 40, 0, self._on_any_param_changed)
        self.fps_slider, self.fps_label = self._make_slider(10, 60, 30, self._on_fps_changed)
        self.spike_buffer_spin = QSpinBox()
        self.spike_buffer_spin.setRange(1, 200)
        self.spike_buffer_spin.setValue(10)
        self.spike_buffer_spin.valueChanged.connect(self._on_spike_buffer_changed)
        f_dyn.addRow("Response mode", self.response_mode)
        f_dyn.addRow("Max Hz", self._stack_slider_row(self.max_rate_slider, self.max_rate_label))
        f_dyn.addRow("Rate Y max", self._stack_slider_row(self.rate_ymax_slider, self.rate_ymax_label))
        f_dyn.addRow("Gain", self._stack_slider_row(self.rate_gain, self.rate_gain_label))
        f_dyn.addRow("Baseline Hz", self._stack_slider_row(self.rate_base, self.rate_base_label))
        f_dyn.addRow("FPS", self._stack_slider_row(self.fps_slider, self.fps_label))
        f_dyn.addRow("Spike buffer", self.spike_buffer_spin)
        self.response_info = QLabel("Mode=Normalized | Raw=0.000 Eff=0.000 | Rate=0.0 Hz")
        self.response_info.setMinimumWidth(500)
        f_dyn.addRow("Current", self.response_info)

        self.btn_start = QPushButton("Pause")
        self.btn_start.clicked.connect(self._toggle_timer)
        self.btn_reset = QPushButton("Reset traces")
        self.btn_reset.clicked.connect(self._reset_buffers)
        f_dyn.addRow(self.btn_start, self.btn_reset)
        controls.addWidget(grp_dyn)

        controls.addStretch(1)

        # Visuals
        top_row = QHBoxLayout()
        visuals.addLayout(top_row, 2)

        self.canvas_plot = StimulusCanvasPlot()
        style_image_plot(self.canvas_plot, "Stimulus + RF", self.canvas_size)
        self.canvas_plot.moved.connect(self._on_canvas_dragged)
        self.canvas_item = pg.ImageItem()
        self.canvas_plot.addItem(self.canvas_item)
        top_row.addWidget(self.canvas_plot, 3)

        self.kernel_plot = pg.PlotWidget()
        style_image_plot(self.kernel_plot, "Selected V1 Kernel", self.kernel_view_size)
        self.kernel_plot.setMouseEnabled(x=False, y=False)
        self.kernel_plot.setMenuEnabled(False)
        self.kernel_item = pg.ImageItem()
        self.kernel_plot.addItem(self.kernel_item)
        top_row.addWidget(self.kernel_plot, 2)

        self.spike_shape_plot = pg.PlotWidget()
        style_plot_widget(self.spike_shape_plot, "Detected Spike Shape Buffer", "Amplitude + offset", x_label="Time (ms)")
        self.spike_shape_curves = [
            self.spike_shape_plot.plot(pen=pg.mkPen(c, width=1.2)) for c in NEURON_COLORS
        ]
        self.spike_shape_plot.setYRange(-1.6, 1.6, padding=0.0)
        self.spike_shape_plot.setXRange(0.0, 1.5, padding=0.0)
        top_row.addWidget(self.spike_shape_plot, 2)

        self.rate_plot = pg.PlotWidget()
        style_plot_widget(self.rate_plot, "Firing Rate (Real-time)", "Rate (Hz)")
        self.rate_curves = [self.rate_plot.plot(pen=pg.mkPen(c, width=2.0)) for c in NEURON_COLORS]
        visuals.addWidget(self.rate_plot, 1)

        self.spike_plot = pg.PlotWidget()
        style_plot_widget(self.spike_plot, "Spikes (Real-time)", "Spikes", x_label="Time (s)")
        self.spike_plot.setYRange(0, 1.4)
        self.spike_lines = [self.spike_plot.plot(pen=pg.mkPen(c, width=1.2)) for c in NEURON_COLORS]
        visuals.addWidget(self.spike_plot, 1)

        self._on_stim_type_changed()

    def _setup_timer(self) -> None:
        self.timer = QTimer(self)
        self.timer.setInterval(int(round(1000.0 / self.fps)))
        self.timer.timeout.connect(self._on_timer_tick)
        self.timer.start()

    def _make_slider(self, min_v: int, max_v: int, start_v: int, callback):
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_v, max_v)
        slider.setValue(start_v)
        label = QLabel(str(start_v))
        label.setFixedWidth(56)
        slider.valueChanged.connect(lambda v, lab=label: lab.setText(str(v)))
        slider.valueChanged.connect(callback)
        return slider, label

    def _stack_slider_row(self, slider: QSlider, label: QLabel) -> QWidget:
        w = QWidget()
        l = QHBoxLayout(w)
        l.setContentsMargins(0, 0, 0, 0)
        l.addWidget(slider, 1)
        l.addWidget(label, 0)
        return w

    def _stack_slider_spin_row(self, slider: QSlider, label: QLabel, spin: QSpinBox) -> QWidget:
        w = QWidget()
        l = QHBoxLayout(w)
        l.setContentsMargins(0, 0, 0, 0)
        l.addWidget(slider, 1)
        l.addWidget(label, 0)
        l.addWidget(spin, 0)
        return w

    def _selected_cell(self):
        return self.v1_cells[self.neuron_combo.currentIndex()]

    def _stim_center(self) -> tuple[float, float]:
        return float(self.stim_x_slider.value()), float(self.stim_y_slider.value())

    def _gain_hz(self) -> float:
        return self.rate_gain.value() / 5.0

    def _baseline_hz(self) -> float:
        return float(self.rate_base.value())

    def _max_rate_hz(self) -> float:
        return float(self.max_rate_slider.value())

    def _use_normalized_response(self) -> bool:
        return self.response_mode.currentIndex() == 0

    def _active_neuron_ids(self) -> list[int]:
        n_total = len(self.v1_cells)
        n_sim = int(self.sim_neurons_spin.value())
        start = int(self.neuron_combo.currentIndex())
        ids = []
        for i in range(min(n_sim, n_total)):
            ids.append((start + i) % n_total)
        return ids

    def _edited_slot_index(self) -> int:
        return int(np.clip(self.rf_slot_combo.currentIndex(), 0, 4))

    def _edited_neuron_id(self) -> int:
        ids = self._active_neuron_ids()
        if len(ids) == 0:
            return 0
        idx = min(self._edited_slot_index(), len(ids) - 1)
        return ids[idx]

    def _ensure_rf_positions_for_active(self) -> None:
        ids = self._active_neuron_ids()
        for nid in ids:
            if nid not in self.rf_pos_by_neuron:
                size = self._scaled_rf_size_for_cell(self.v1_cells[nid])
                max_rf = max(0, self.canvas_size - size)
                self.rf_pos_by_neuron[nid] = (max_rf // 2, max_rf // 2)

    def _rf_coords_for_neuron(self, neuron_id: int) -> tuple[int, int]:
        self._ensure_rf_positions_for_active()
        if neuron_id not in self.rf_pos_by_neuron:
            size = self._scaled_rf_size_for_cell(self.v1_cells[neuron_id])
            max_rf = max(0, self.canvas_size - size)
            self.rf_pos_by_neuron[neuron_id] = (max_rf // 2, max_rf // 2)
        x, y = self.rf_pos_by_neuron[neuron_id]
        size = self._scaled_rf_size_for_cell(self.v1_cells[neuron_id])
        max_rf = max(0, self.canvas_size - size)
        return int(np.clip(x, 0, max_rf)), int(np.clip(y, 0, max_rf))

    def _scaled_rf_size_for_cell(self, cell: V1Cell) -> int:
        scale = self.rf_scale_slider.value() / 100.0
        return max(3, int(round(cell.size * scale)))

    def _color_m11_from_name(self, name: str) -> tuple[float, float, float]:
        lut = {
            "white": (1.0, 1.0, 1.0),
            "red": (1.0, -1.0, -1.0),
            "green": (-1.0, 1.0, -1.0),
            "blue": (-1.0, -1.0, 1.0),
            "yellow": (1.0, 1.0, -1.0),
            "cyan": (-1.0, 1.0, 1.0),
            "magenta": (1.0, -1.0, 1.0),
        }
        return lut.get(name.lower().strip(), (1.0, 1.0, 1.0))

    def _kernel_for_cell(self, cell: V1Cell) -> np.ndarray:
        kernel = resize_kernel_rgb(cell.kernel_rgb, self._scaled_rf_size_for_cell(cell))
        if self.gray_rf_btn.isChecked():
            kernel = to_grayscale_kernel(
                kernel,
                match_energy=self.gray_rf_energy_btn.isChecked(),
            )
        if self.mask_rf_btn.isChecked():
            kernel = apply_rf_gray_outside(
                kernel_rgb=kernel,
                gray_value=0.0,
                radius_ratio=0.45,
                soft_edge_px=1.5,
            )
        return kernel

    def _current_kernel(self) -> np.ndarray:
        return self._kernel_for_cell(self._selected_cell())

    def _sync_neuron_and_rf_limits(self, reset_to_center: bool = False) -> None:
        self._ensure_rf_positions_for_active()
        ids = self._active_neuron_ids()
        if len(ids) == 0:
            return
        self.rf_slot_combo.setEnabled(len(ids) > 1)
        if self._edited_slot_index() >= len(ids):
            self.rf_slot_combo.blockSignals(True)
            self.rf_slot_combo.setCurrentIndex(len(ids) - 1)
            self.rf_slot_combo.blockSignals(False)

        edit_id = self._edited_neuron_id()
        edit_size = self._scaled_rf_size_for_cell(self.v1_cells[edit_id])
        max_rf = max(0, self.canvas_size - edit_size)
        prev_xy = self.rf_pos_by_neuron.get(edit_id, (max_rf // 2, max_rf // 2))
        prev_x = int(np.clip(prev_xy[0], 0, max_rf))
        prev_y = int(np.clip(prev_xy[1], 0, max_rf))
        target_x = max_rf // 2 if reset_to_center else prev_x
        target_y = max_rf // 2 if reset_to_center else prev_y
        self.rf_pos_by_neuron[edit_id] = (target_x, target_y)

        self.rf_x_slider.setRange(0, max_rf)
        self.rf_y_slider.setRange(0, max_rf)
        self.rf_x_spin.setRange(0, max_rf)
        self.rf_y_spin.setRange(0, max_rf)
        self.rf_x_slider.blockSignals(True)
        self.rf_y_slider.blockSignals(True)
        self.rf_x_spin.blockSignals(True)
        self.rf_y_spin.blockSignals(True)
        self.rf_x_slider.setValue(target_x)
        self.rf_y_slider.setValue(target_y)
        self.rf_x_spin.setValue(target_x)
        self.rf_y_spin.setValue(target_y)
        self.rf_x_slider.blockSignals(False)
        self.rf_y_slider.blockSignals(False)
        self.rf_x_spin.blockSignals(False)
        self.rf_y_spin.blockSignals(False)

    def _refresh_kernel_preview(self) -> None:
        ids = self._active_neuron_ids()
        preview_id = ids[0] if len(ids) > 0 else self.neuron_combo.currentIndex()
        kernel = self._kernel_for_cell(self.v1_cells[preview_id])
        self.kernel_item.setImage(
            centered_kernel_preview_uint8(kernel, canvas_size=self.kernel_view_size, scale=5),
            autoLevels=False,
        )
        t_ms, _ = generate_spike_waveform_from_rf(kernel_rgb=kernel, duration_ms=1.5, sample_rate_hz=20000)
        self.spike_shape_plot.setXRange(0.0, float(t_ms[-1]), padding=0.0)
        self._refresh_spike_shape_panel()

    def _ensure_trace_buffers(self) -> None:
        n = len(self._active_neuron_ids())
        if n <= 0:
            n = 1
        if len(self.rates_by_neuron) != n:
            self.rates_by_neuron = [[] for _ in range(n)]
            self.spike_times_by_neuron = [[] for _ in range(n)]
            self.spike_waveforms_xy_by_neuron = [[] for _ in range(n)]

    def _refresh_spike_shape_panel(self) -> None:
        self._ensure_trace_buffers()
        for i, curve in enumerate(self.spike_shape_curves):
            if i < len(self.spike_waveforms_xy_by_neuron):
                x, y = overlay_waveforms_to_lines(self.spike_waveforms_xy_by_neuron[i])
                curve.setData(x, y)
            else:
                curve.setData([], [])
        self.spike_shape_plot.setYRange(-1.6, 1.6, padding=0.0)

    def _timed_cycle_state(self) -> tuple[int, bool, float]:
        on_s = self.on_ms.value() / 1000.0
        off_s = self.off_ms.value() / 1000.0
        period = max(1e-6, on_s + off_s)
        cycle_idx = int(self.current_time // period)
        phase = self.current_time - cycle_idx * period
        visible = phase < on_s
        on_phase = float(np.clip(phase / max(on_s, 1e-6), 0.0, 1.0)) if visible else 0.0
        return cycle_idx, visible, on_phase

    def _current_stimulus_patch(self) -> np.ndarray:
        kind = self.stim_combo.currentText()
        cycle_idx, _, on_phase = self._timed_cycle_state()
        timed_mode = self.presentation_mode.currentText() == "Timed experiment"

        if timed_mode and cycle_idx != self._last_on_cycle_idx and self._stimulus_visible():
            self._last_on_cycle_idx = cycle_idx
            if kind == "CIFAR patch":
                self._auto_cifar_index = int(self.rng.integers(0, self.cifar_images_uint8.shape[0]))

        if kind == "Bar":
            angle = float(self.bar_orientation.value())
            if timed_mode:
                angle = angle + cycle_idx * self.auto_bar_step.value()
            return generate_bar_stimulus(
                size=self.bar_size.value(),
                orientation_deg=float(angle % 180.0),
                length=float(self.bar_length.value()),
                thickness=float(self.bar_thickness.value()),
                contrast=self.bar_contrast.value() / 100.0,
                background_level=-1.0,
                color_rgb_m11=self._color_m11_from_name(self.bar_color.currentText()),
            )
        if kind == "Grating":
            ori = float(self.gr_orientation.value())
            sf = self.gr_sf.value() / 100.0
            phase = float(self.gr_phase.value())
            if timed_mode:
                mode = self.auto_gr_mode.currentText()
                if mode in ("Orientation", "Both"):
                    ori = ori + cycle_idx * self.auto_gr_ori_step.value()
                if mode in ("Frequency", "Both"):
                    sf = sf + cycle_idx * (self.auto_gr_sf_step.value() / 1000.0)
                # Keep orientation/sf fixed during each ON presentation,
                # but sweep phase continuously across a full cycle.
                phase = phase + 360.0 * on_phase
            sf = float(np.clip(sf, 0.01, 0.45))
            return generate_grating_stimulus(
                size=self.gr_size.value(),
                orientation_deg=float(ori % 180.0),
                spatial_frequency=sf,
                phase_deg=float(phase % 360.0),
                contrast=self.gr_contrast.value() / 100.0,
                background_level=-1.0,
                color_rgb_m11=self._color_m11_from_name(self.gr_color.currentText()),
            )
        cifar_idx = int(self.cifar_index.value())
        if timed_mode:
            cifar_idx = int(self._auto_cifar_index)
        return generate_cifar_patch_stimulus(
            cifar_images_uint8=self.cifar_images_uint8,
            image_index=cifar_idx,
            size=self.cifar_size.value(),
        )

    def _compose_canvas_m11(self) -> np.ndarray:
        base = np.ones((self.canvas_size, self.canvas_size, 3), dtype=np.float32) * -1.0
        if not self._stimulus_visible():
            return base
        stim_patch = self._current_stimulus_patch()
        return place_patch_on_canvas(base, stim_patch, self._stim_center())

    def _stimulus_visible(self) -> bool:
        if self.presentation_mode.currentText() == "Manual":
            return True
        _, visible, _ = self._timed_cycle_state()
        return visible

    def _render_frame_and_plots(self, step_sim: bool = True) -> None:
        self._ensure_rf_positions_for_active()
        frame_m11 = self._compose_canvas_m11()
        neuron_ids = self._active_neuron_ids()
        contexts: list[tuple[V1Cell, np.ndarray]] = []
        for nid in neuron_ids:
            base_cell = self.v1_cells[nid]
            kernel = self._kernel_for_cell(base_cell)
            rf_x, rf_y = self._rf_coords_for_neuron(nid)
            cell_ctx = V1Cell(
                kernel_rgb=kernel,
                frequency=base_cell.frequency,
                theta=base_cell.theta,
                size=kernel.shape[0],
                coords=(rf_x, rf_y),
            )
            contexts.append((cell_ctx, kernel))

        self._ensure_trace_buffers()

        responses: list[float] = []
        rates_now: list[float] = []
        for cell_ctx, _kernel in contexts:
            if self._use_normalized_response():
                raw = compute_single_frame_response_normalized(cell_ctx, frame_m11, cell_ctx.coords)
                sensitivity = self._gain_hz() / 5.0
                eff = float(np.tanh(sensitivity * raw))
                rate = response_to_rate_hz_bounded(
                    eff,
                    max_rate_hz=self._max_rate_hz(),
                    baseline_hz=self._baseline_hz(),
                )
                responses.append(raw)
                rates_now.append(rate)
            else:
                raw = compute_single_frame_response(cell_ctx, frame_m11, cell_ctx.coords)
                rate = response_to_rate_hz(raw, gain=self._gain_hz(), baseline_hz=self._baseline_hz())
                responses.append(raw)
                rates_now.append(rate)

        if len(rates_now) > 0:
            if self._use_normalized_response():
                self.response_info.setText(
                    f"Mode=Normalized | N={len(rates_now)} | Mean rate={float(np.mean(rates_now)):.1f} Hz"
                )
            else:
                self.response_info.setText(
                    f"Mode=Legacy | N={len(rates_now)} | Mean rate={float(np.mean(rates_now)):.1f} Hz"
                )

        if step_sim:
            spikes_now: list[int] = []
            for i, ((cell_ctx, kernel), rate_hz) in enumerate(zip(contexts, rates_now)):
                spike = poisson_spike_step(rate_hz, self.dt_s, self.rng)
                spikes_now.append(spike)
                self.rates_by_neuron[i].append(rate_hz)
                if spike == 1:
                    self.spike_times_by_neuron[i].append(self.current_time + self.dt_s)
                    t_ms, w0 = generate_spike_waveform_from_rf(
                        kernel_rgb=kernel,
                        duration_ms=1.5,
                        sample_rate_hz=20000,
                    )
                    t_ms, w = sample_noisy_jittered_spike_waveform(
                        t_ms,
                        w0,
                        self.rng,
                        jitter_std_ms=0.04,
                        noise_std=0.035,
                        amp_jitter_frac=0.08,
                    )
                    self.spike_waveforms_xy_by_neuron[i].append((t_ms, w))
                    max_buf = int(self.spike_buffer_spin.value())
                    if len(self.spike_waveforms_xy_by_neuron[i]) > max_buf:
                        self.spike_waveforms_xy_by_neuron[i] = self.spike_waveforms_xy_by_neuron[i][-max_buf:]

            mean_rate = float(np.mean(rates_now)) if len(rates_now) > 0 else 0.0
            spike_any = 1 if np.any(np.array(spikes_now) == 1) else 0
            self.audio.update(mean_rate, spike_any, self.dt_s)
            self.current_time += self.dt_s
            self.times.append(self.current_time)
            self._refresh_spike_shape_panel()
            self._trim_history()

        rgb = m11_to_uint8_rgb(frame_m11)
        for i, (cell_ctx, _kernel) in enumerate(contexts):
            col = NEURON_COLORS[i % len(NEURON_COLORS)]
            if col == "w":
                rgb_col = (255, 255, 255)
            elif col == "b":
                rgb_col = (0, 120, 255)
            elif col == "r":
                rgb_col = (255, 0, 0)
            elif col == "g":
                rgb_col = (0, 220, 0)
            else:
                rgb_col = (255, 220, 0)
            rgb = draw_rf_box_uint8(rgb, cell_ctx.coords[0], cell_ctx.coords[1], cell_ctx.size, color=rgb_col, thickness=2)
        self.canvas_item.setImage(rgb, autoLevels=False)

        if len(self.times) == 0:
            for c in self.rate_curves:
                c.setData([], [])
            for c in self.spike_lines:
                c.setData([], [])
            return

        t0 = self.times[-1] - self.window_sec
        rel_t = np.array(self.times) - t0
        for i, c in enumerate(self.rate_curves):
            if i < len(self.rates_by_neuron):
                rates = np.array(self.rates_by_neuron[i], dtype=np.float32)
                c.setData(rel_t, rates)
            else:
                c.setData([], [])
        self.rate_plot.setXRange(max(0.0, rel_t[-1] - self.window_sec), rel_t[-1] + 0.001, padding=0.0)
        self.rate_plot.setYRange(0.0, float(self.rate_ymax_slider.value()), padding=0.0)

        for i, c in enumerate(self.spike_lines):
            if i < len(self.spike_times_by_neuron):
                rel_spikes = np.array([t - t0 for t in self.spike_times_by_neuron[i] if t >= t0], dtype=np.float32)
                if rel_spikes.size > 0:
                    x_lines = np.repeat(rel_spikes, 3)
                    y_lines = np.tile(np.array([0.0, 1.0, np.nan], dtype=np.float32), rel_spikes.size)
                    x_lines[2::3] = np.nan
                    c.setData(x_lines, y_lines)
                else:
                    c.setData([], [])
            else:
                c.setData([], [])
        self.spike_plot.setXRange(max(0.0, rel_t[-1] - self.window_sec), rel_t[-1] + 0.001, padding=0.0)

    def _trim_history(self) -> None:
        if len(self.times) == 0:
            return
        t_min = self.times[-1] - self.window_sec
        keep_idx = 0
        for i, t in enumerate(self.times):
            if t >= t_min:
                keep_idx = i
                break
        self.times = self.times[keep_idx:]
        for i in range(len(self.rates_by_neuron)):
            self.rates_by_neuron[i] = self.rates_by_neuron[i][keep_idx:]
            self.spike_times_by_neuron[i] = [t for t in self.spike_times_by_neuron[i] if t >= t_min]

    def _set_visibility_for_stimulus(self, is_bar: bool, is_grating: bool, is_cifar: bool) -> None:
        for widget in [
            self.bar_orientation, self.bar_orientation_label, self.bar_length, self.bar_length_label,
            self.bar_thickness, self.bar_thickness_label, self.bar_size, self.bar_size_label,
            self.bar_contrast, self.bar_contrast_label, self.bar_color,
        ]:
            widget.setVisible(is_bar)
        for widget in [
            self.gr_orientation, self.gr_orientation_label, self.gr_sf, self.gr_sf_label,
            self.gr_phase, self.gr_phase_label, self.gr_size, self.gr_size_label,
            self.gr_contrast, self.gr_contrast_label, self.gr_color,
        ]:
            widget.setVisible(is_grating)
        self.cifar_index.setVisible(is_cifar)
        self.cifar_size.setVisible(is_cifar)
        self.cifar_size_label.setVisible(is_cifar)

    def _on_neuron_changed(self) -> None:
        idx = self.neuron_combo.currentIndex()
        self.neuron_spin.blockSignals(True)
        self.neuron_spin.setValue(idx)
        self.neuron_spin.blockSignals(False)
        self._sync_neuron_and_rf_limits(reset_to_center=True)
        self._refresh_kernel_preview()
        self._reset_buffers()

    def _on_neuron_spin_changed(self, value: int) -> None:
        if value == self.neuron_combo.currentIndex():
            return
        self.neuron_combo.blockSignals(True)
        self.neuron_combo.setCurrentIndex(value)
        self.neuron_combo.blockSignals(False)
        self._on_neuron_changed()

    def _on_sim_neurons_changed(self) -> None:
        self._sync_neuron_and_rf_limits(reset_to_center=False)
        self._refresh_kernel_preview()
        self._reset_buffers()

    def _on_rf_slot_changed(self) -> None:
        self._sync_neuron_and_rf_limits(reset_to_center=False)
        self._render_frame_and_plots(step_sim=False)

    def _on_rf_slider_changed(self) -> None:
        edit_id = self._edited_neuron_id()
        self.rf_pos_by_neuron[edit_id] = (int(self.rf_x_slider.value()), int(self.rf_y_slider.value()))
        self.rf_x_spin.blockSignals(True)
        self.rf_y_spin.blockSignals(True)
        self.rf_x_spin.setValue(self.rf_x_slider.value())
        self.rf_y_spin.setValue(self.rf_y_slider.value())
        self.rf_x_spin.blockSignals(False)
        self.rf_y_spin.blockSignals(False)
        self._render_frame_and_plots(step_sim=False)

    def _on_rf_spin_changed(self) -> None:
        edit_id = self._edited_neuron_id()
        self.rf_pos_by_neuron[edit_id] = (int(self.rf_x_spin.value()), int(self.rf_y_spin.value()))
        self.rf_x_slider.blockSignals(True)
        self.rf_y_slider.blockSignals(True)
        self.rf_x_slider.setValue(self.rf_x_spin.value())
        self.rf_y_slider.setValue(self.rf_y_spin.value())
        self.rf_x_slider.blockSignals(False)
        self.rf_y_slider.blockSignals(False)
        self._render_frame_and_plots(step_sim=False)

    def _on_rf_scale_changed(self) -> None:
        self.rf_scale_spin.blockSignals(True)
        self.rf_scale_spin.setValue(self.rf_scale_slider.value())
        self.rf_scale_spin.blockSignals(False)
        self._sync_neuron_and_rf_limits(reset_to_center=False)
        self._refresh_kernel_preview()
        self._render_frame_and_plots(step_sim=False)

    def _on_rf_scale_spin_changed(self) -> None:
        self.rf_scale_slider.blockSignals(True)
        self.rf_scale_slider.setValue(self.rf_scale_spin.value())
        self.rf_scale_slider.blockSignals(False)
        self._sync_neuron_and_rf_limits(reset_to_center=False)
        self._refresh_kernel_preview()
        self._render_frame_and_plots(step_sim=False)

    def _on_mask_rf_toggled(self) -> None:
        self.mask_rf_btn.setText("Mask RF: ON" if self.mask_rf_btn.isChecked() else "Mask RF: OFF")
        self._refresh_kernel_preview()
        self._render_frame_and_plots(step_sim=False)

    def _on_gray_rf_toggled(self) -> None:
        self.gray_rf_btn.setText("RF grayscale: ON" if self.gray_rf_btn.isChecked() else "RF grayscale: OFF")
        self._refresh_kernel_preview()
        self._render_frame_and_plots(step_sim=False)

    def _on_gray_energy_toggled(self) -> None:
        self.gray_rf_energy_btn.setText(
            "Gray energy match: ON" if self.gray_rf_energy_btn.isChecked() else "Gray energy match: OFF"
        )
        if self.gray_rf_btn.isChecked():
            self._refresh_kernel_preview()
            self._render_frame_and_plots(step_sim=False)

    def _on_stim_slider_changed(self) -> None:
        self._render_frame_and_plots(step_sim=False)

    def _on_any_param_changed(self) -> None:
        self._render_frame_and_plots(step_sim=False)

    def _on_spike_buffer_changed(self) -> None:
        max_buf = int(self.spike_buffer_spin.value())
        for i in range(len(self.spike_waveforms_xy_by_neuron)):
            if len(self.spike_waveforms_xy_by_neuron[i]) > max_buf:
                self.spike_waveforms_xy_by_neuron[i] = self.spike_waveforms_xy_by_neuron[i][-max_buf:]
        self._refresh_spike_shape_panel()

    def _on_stim_type_changed(self) -> None:
        kind = self.stim_combo.currentText()
        self._set_visibility_for_stimulus(
            is_bar=(kind == "Bar"),
            is_grating=(kind == "Grating"),
            is_cifar=(kind == "CIFAR patch"),
        )
        self._render_frame_and_plots(step_sim=False)

    def _on_canvas_dragged(self, x: float, y: float) -> None:
        sx = int(np.clip(round(x), 0, self.canvas_size - 1))
        sy = int(np.clip(round(y), 0, self.canvas_size - 1))
        self.stim_x_slider.blockSignals(True)
        self.stim_y_slider.blockSignals(True)
        self.stim_x_slider.setValue(sx)
        self.stim_y_slider.setValue(sy)
        self.stim_x_slider.blockSignals(False)
        self.stim_y_slider.blockSignals(False)
        self._render_frame_and_plots(step_sim=False)

    def _on_fps_changed(self) -> None:
        self.fps = int(self.fps_slider.value())
        self.dt_s = 1.0 / max(1.0, float(self.fps))
        self.timer.setInterval(int(round(1000.0 / self.fps)))

    def _toggle_timer(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
            self.btn_start.setText("Start")
        else:
            self.timer.start()
            self.btn_start.setText("Pause")

    def _reset_buffers(self) -> None:
        self.current_time = 0.0
        self.times = []
        self.rates_by_neuron = [[] for _ in self._active_neuron_ids()]
        self.spike_times_by_neuron = [[] for _ in self._active_neuron_ids()]
        self.spike_waveforms_xy_by_neuron = [[] for _ in self._active_neuron_ids()]
        self._refresh_spike_shape_panel()
        self.audio.reset()
        self._render_frame_and_plots(step_sim=False)

    def _on_timer_tick(self) -> None:
        self._render_frame_and_plots(step_sim=True)

    def closeEvent(self, event: QCloseEvent) -> None:
        self.audio.stop()
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    window = HubelWieselGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
