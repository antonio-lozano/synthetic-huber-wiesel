"""
View layer for the Hubel-Wiesel Synthetic V1 Explorer.

Contains:
- ``SimView`` class: all Qt widget creation, layout, styling.
- ``StimulusCanvasPlot``: custom PlotWidget with click-drag signal.

No simulation logic lives here – the view exposes its widgets so the
controller can wire up signals.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
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

from model import NEURON_COLORS, MAX_SIM_NEURONS
from plotting_GUI import style_image_plot, style_plot_widget

pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)

_STYLESHEET = """
QWidget { background-color: black; color: white; font-size: 16px; }
QGroupBox { border: 1px solid white; margin-top: 10px; font-weight: bold; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }
QPushButton { border: 1px solid white; padding: 6px; }
QSlider::groove:horizontal { border: 1px solid white; height: 6px; background: #111; }
QSlider::handle:horizontal { background: white; border: 1px solid white; width: 14px; margin: -5px 0; }
"""


# ── Draggable stimulus canvas ─────────────────────────────────────────────────

class StimulusCanvasPlot(pg.PlotWidget):
    """PlotWidget that emits ``moved(x, y)`` on left-button drag."""

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


# ── Main view ──────────────────────────────────────────────────────────────────

class SimView(QMainWindow):
    """
    Pure UI shell – creates all widgets and lays them out.

    Widgets are exposed as public attributes so the controller can read
    values and connect signals.
    """

    def __init__(
        self,
        n_cells: int,
        n_cifar: int,
        canvas_size: int = 256,
        kernel_view_size: int = 192,
    ) -> None:
        super().__init__()
        self.setWindowTitle("Hubel-Wiesel Synthetic V1 Explorer (2026)")
        self.resize(1700, 980)
        self.setStyleSheet(_STYLESHEET)
        self._canvas_size = canvas_size
        self._kernel_view_size = kernel_view_size

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        controls = QVBoxLayout()
        controls.setSpacing(12)
        root.addLayout(controls, 0)

        visuals = QVBoxLayout()
        root.addLayout(visuals, 1)

        # ── Neuron & RF group ──────────────────────────────────────
        grp_neuron = QGroupBox("Neuron and RF")
        f_neuron = QFormLayout(grp_neuron)

        self.neuron_combo = QComboBox()
        self.neuron_combo.addItems([f"Neuron {i}" for i in range(n_cells)])
        f_neuron.addRow("Select neuron", self.neuron_combo)

        self.neuron_spin = QSpinBox()
        self.neuron_spin.setRange(0, n_cells - 1)
        self.neuron_spin.setValue(0)
        f_neuron.addRow("Neuron index", self.neuron_spin)

        self.sim_neurons_spin = QSpinBox()
        self.sim_neurons_spin.setRange(1, MAX_SIM_NEURONS)
        self.sim_neurons_spin.setValue(1)
        f_neuron.addRow("Sim neurons", self.sim_neurons_spin)

        self.rf_x_slider, self.rf_x_label = self._make_slider(0, 100, 0)
        self.rf_y_slider, self.rf_y_label = self._make_slider(0, 100, 0)
        self.rf_x_spin = QSpinBox()
        self.rf_y_spin = QSpinBox()

        self.rf_slot_combo = QComboBox()
        self.rf_slot_combo.addItems([f"N{i + 1}" for i in range(MAX_SIM_NEURONS)])
        f_neuron.addRow("Edit RF for", self.rf_slot_combo)
        f_neuron.addRow("RF X", self._slider_spin_row(self.rf_x_slider, self.rf_x_label, self.rf_x_spin))
        f_neuron.addRow("RF Y", self._slider_spin_row(self.rf_y_slider, self.rf_y_label, self.rf_y_spin))

        self.rf_scale_slider, self.rf_scale_label = self._make_slider(50, 400, 100)
        self.rf_scale_spin = QSpinBox()
        self.rf_scale_spin.setRange(50, 400)
        self.rf_scale_spin.setValue(100)
        f_neuron.addRow("RF scale (%)", self._slider_spin_row(self.rf_scale_slider, self.rf_scale_label, self.rf_scale_spin))

        self.mask_rf_btn = QPushButton("Mask RF: OFF")
        self.mask_rf_btn.setCheckable(True)
        f_neuron.addRow("RF masking", self.mask_rf_btn)

        self.gray_rf_btn = QPushButton("RF grayscale: OFF")
        self.gray_rf_btn.setCheckable(True)
        f_neuron.addRow("RF grayscale", self.gray_rf_btn)

        self.gray_rf_energy_btn = QPushButton("Gray energy match: ON")
        self.gray_rf_energy_btn.setCheckable(True)
        self.gray_rf_energy_btn.setChecked(True)
        f_neuron.addRow("Gray scaling", self.gray_rf_energy_btn)

        controls.addWidget(grp_neuron)

        # ── Stimulus group ─────────────────────────────────────────
        grp_stim = QGroupBox("Stimulus")
        f_stim = QFormLayout(grp_stim)

        self.stim_combo = QComboBox()
        self.stim_combo.addItems(["Bar", "Grating", "CIFAR patch"])
        f_stim.addRow("Type", self.stim_combo)

        self.stim_x_slider, self.stim_x_label = self._make_slider(0, canvas_size - 1, canvas_size // 2)
        self.stim_y_slider, self.stim_y_label = self._make_slider(0, canvas_size - 1, canvas_size // 2)
        f_stim.addRow("Stim X", self._slider_row(self.stim_x_slider, self.stim_x_label))
        f_stim.addRow("Stim Y", self._slider_row(self.stim_y_slider, self.stim_y_label))

        # Bar params
        self.bar_orientation, self.bar_orientation_label = self._make_slider(0, 179, 45)
        self.bar_length, self.bar_length_label = self._make_slider(6, 140, 80)
        self.bar_thickness, self.bar_thickness_label = self._make_slider(1, 30, 6)
        self.bar_size, self.bar_size_label = self._make_slider(16, 180, 120)
        self.bar_contrast, self.bar_contrast_label = self._make_slider(10, 100, 100)
        self.bar_color = QComboBox()
        self.bar_color.addItems(["White", "Red", "Green", "Blue", "Yellow", "Cyan", "Magenta"])
        for label_text, slider, label in [
            ("Bar angle", self.bar_orientation, self.bar_orientation_label),
            ("Bar length", self.bar_length, self.bar_length_label),
            ("Bar thick", self.bar_thickness, self.bar_thickness_label),
            ("Bar size", self.bar_size, self.bar_size_label),
            ("Bar contrast", self.bar_contrast, self.bar_contrast_label),
        ]:
            f_stim.addRow(label_text, self._slider_row(slider, label))
        f_stim.addRow("Bar color", self.bar_color)

        # Grating params
        self.gr_orientation, self.gr_orientation_label = self._make_slider(0, 179, 45)
        self.gr_sf, self.gr_sf_label = self._make_slider(2, 35, 10)
        self.gr_phase, self.gr_phase_label = self._make_slider(0, 360, 0)
        self.gr_size, self.gr_size_label = self._make_slider(16, 180, 128)
        self.gr_contrast, self.gr_contrast_label = self._make_slider(10, 100, 100)
        self.gr_color = QComboBox()
        self.gr_color.addItems(["White", "Red", "Green", "Blue", "Yellow", "Cyan", "Magenta"])
        for label_text, slider, label in [
            ("Grating angle", self.gr_orientation, self.gr_orientation_label),
            ("Spatial freq", self.gr_sf, self.gr_sf_label),
            ("Phase", self.gr_phase, self.gr_phase_label),
            ("Grating size", self.gr_size, self.gr_size_label),
            ("Grating contrast", self.gr_contrast, self.gr_contrast_label),
        ]:
            f_stim.addRow(label_text, self._slider_row(slider, label))
        f_stim.addRow("Grating color", self.gr_color)

        # CIFAR params
        self.cifar_index = QSpinBox()
        self.cifar_index.setRange(0, max(0, n_cifar - 1))
        self.cifar_index.setValue(0)
        self.cifar_size, self.cifar_size_label = self._make_slider(16, 180, 96)
        f_stim.addRow("CIFAR index", self.cifar_index)
        f_stim.addRow("CIFAR size", self._slider_row(self.cifar_size, self.cifar_size_label))

        # Presentation / timed experiment
        self.presentation_mode = QComboBox()
        self.presentation_mode.addItems(["Manual", "Timed experiment"])
        self.on_ms = QSpinBox(); self.on_ms.setRange(10, 5000); self.on_ms.setSingleStep(10); self.on_ms.setValue(100)
        self.off_ms = QSpinBox(); self.off_ms.setRange(10, 5000); self.off_ms.setSingleStep(10); self.off_ms.setValue(500)
        self.auto_bar_step = QSpinBox(); self.auto_bar_step.setRange(0, 180); self.auto_bar_step.setValue(20)
        self.auto_gr_mode = QComboBox(); self.auto_gr_mode.addItems(["Orientation", "Frequency", "Both"])
        self.auto_gr_ori_step = QSpinBox(); self.auto_gr_ori_step.setRange(0, 180); self.auto_gr_ori_step.setValue(15)
        self.auto_gr_sf_step = QSpinBox(); self.auto_gr_sf_step.setRange(0, 100); self.auto_gr_sf_step.setValue(2)
        f_stim.addRow("Presentation", self.presentation_mode)
        f_stim.addRow("Stim ON (ms)", self.on_ms)
        f_stim.addRow("Stim OFF (ms)", self.off_ms)
        f_stim.addRow("Bar step (deg/presentation)", self.auto_bar_step)
        f_stim.addRow("Grating auto", self.auto_gr_mode)
        f_stim.addRow("Grating ori step", self.auto_gr_ori_step)
        f_stim.addRow("Grating sf step x1e-3", self.auto_gr_sf_step)

        controls.addWidget(grp_stim)

        # ── Dynamics group ─────────────────────────────────────────
        grp_dyn = QGroupBox("Dynamics")
        f_dyn = QFormLayout(grp_dyn)

        self.response_mode = QComboBox()
        self.response_mode.addItems(["Normalized (0..Max Hz)", "Legacy (unbounded dot)"])
        self.response_mode.setCurrentIndex(0)
        self.max_rate_slider, self.max_rate_label = self._make_slider(20, 300, 100)
        self.rate_ymax_slider, self.rate_ymax_label = self._make_slider(20, 1000, 120)
        self.rate_gain, self.rate_gain_label = self._make_slider(1, 120, 25)
        self.rate_base, self.rate_base_label = self._make_slider(0, 40, 0)
        self.fps_slider, self.fps_label = self._make_slider(10, 60, 30)
        self.spike_buffer_spin = QSpinBox()
        self.spike_buffer_spin.setRange(1, 200)
        self.spike_buffer_spin.setValue(10)

        f_dyn.addRow("Response mode", self.response_mode)
        f_dyn.addRow("Max Hz", self._slider_row(self.max_rate_slider, self.max_rate_label))
        f_dyn.addRow("Rate Y max", self._slider_row(self.rate_ymax_slider, self.rate_ymax_label))
        f_dyn.addRow("Gain", self._slider_row(self.rate_gain, self.rate_gain_label))
        f_dyn.addRow("Baseline Hz", self._slider_row(self.rate_base, self.rate_base_label))
        f_dyn.addRow("FPS", self._slider_row(self.fps_slider, self.fps_label))
        f_dyn.addRow("Spike buffer", self.spike_buffer_spin)

        self.response_info = QLabel("Mode=Normalized | Raw=0.000 Eff=0.000 | Rate=0.0 Hz")
        self.response_info.setMinimumWidth(500)
        f_dyn.addRow("Current", self.response_info)

        self.btn_start = QPushButton("Pause")
        self.btn_reset = QPushButton("Reset traces")
        f_dyn.addRow(self.btn_start, self.btn_reset)
        controls.addWidget(grp_dyn)

        controls.addStretch(1)

        # ── Visual panels ──────────────────────────────────────────
        top_row = QHBoxLayout()
        visuals.addLayout(top_row, 2)

        self.canvas_plot = StimulusCanvasPlot()
        style_image_plot(self.canvas_plot, "Stimulus + RF", canvas_size)
        self.canvas_item = pg.ImageItem()
        self.canvas_plot.addItem(self.canvas_item)
        top_row.addWidget(self.canvas_plot, 3)

        self.kernel_plot = pg.PlotWidget()
        style_image_plot(self.kernel_plot, "Selected V1 Kernel", kernel_view_size)
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

        # Collect bar/grating/cifar widgets for visibility toggling
        self._bar_widgets = [
            self.bar_orientation, self.bar_orientation_label,
            self.bar_length, self.bar_length_label,
            self.bar_thickness, self.bar_thickness_label,
            self.bar_size, self.bar_size_label,
            self.bar_contrast, self.bar_contrast_label,
            self.bar_color,
        ]
        self._grating_widgets = [
            self.gr_orientation, self.gr_orientation_label,
            self.gr_sf, self.gr_sf_label,
            self.gr_phase, self.gr_phase_label,
            self.gr_size, self.gr_size_label,
            self.gr_contrast, self.gr_contrast_label,
            self.gr_color,
        ]
        self._cifar_widgets = [
            self.cifar_index, self.cifar_size, self.cifar_size_label,
        ]

    # ── Widget-factory helpers ─────────────────────────────────────────────

    @staticmethod
    def _make_slider(min_v: int, max_v: int, start_v: int) -> tuple[QSlider, QLabel]:
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_v, max_v)
        slider.setValue(start_v)
        label = QLabel(str(start_v))
        label.setFixedWidth(56)
        slider.valueChanged.connect(lambda v, lab=label: lab.setText(str(v)))
        return slider, label

    @staticmethod
    def _slider_row(slider: QSlider, label: QLabel) -> QWidget:
        w = QWidget()
        layout = QHBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(slider, 1)
        layout.addWidget(label, 0)
        return w

    @staticmethod
    def _slider_spin_row(slider: QSlider, label: QLabel, spin: QSpinBox) -> QWidget:
        w = QWidget()
        layout = QHBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(slider, 1)
        layout.addWidget(label, 0)
        layout.addWidget(spin, 0)
        return w

    # ── Public helpers for the controller ──────────────────────────────────

    def set_stim_visibility(self, is_bar: bool, is_grating: bool, is_cifar: bool) -> None:
        for w in self._bar_widgets:
            w.setVisible(is_bar)
        for w in self._grating_widgets:
            w.setVisible(is_grating)
        for w in self._cifar_widgets:
            w.setVisible(is_cifar)
