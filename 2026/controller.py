"""
Controller layer for the Hubel-Wiesel Synthetic V1 Explorer.

Wires together ``SimView`` (widget signals) and ``SimState`` (pure model).
Handles:
- Signal → model update → view repaint cycle
- QTimer for the simulation loop
- Audio engine management
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtGui import QCloseEvent

from computing import V1Cell, generate_spike_waveform_from_rf
from model import (
    NEURON_COLORS,
    NEURON_RGB,
    SimState,
    active_neuron_ids,
    build_kernel,
    clamped_rf_coords,
    compose_canvas_m11,
    compute_responses,
    edited_neuron_id,
    ensure_rf_positions,
    init_sim_state,
    reset_buffers,
    scaled_rf_size,
    step_simulation,
)
from plotting_GUI import (
    centered_kernel_preview_uint8,
    draw_rf_box_uint8,
    m11_to_uint8_rgb,
    overlay_waveforms_to_lines,
)
from sounds import SpikeRateAudioEngine
from view import SimView


class SimController:
    """
    Connects signals from ``SimView`` to pure-model operations, then
    pushes updated data back into the view widgets / plots.
    """

    def __init__(self) -> None:
        # ── Model ──────────────────────────────────────────────
        self.state = init_sim_state()
        self.audio = SpikeRateAudioEngine(seed=self.state.cfg.seed)

        # ── View ───────────────────────────────────────────────
        self.view = SimView(
            n_cells=len(self.state.v1_cells),
            n_cifar=self.state.cifar_images_uint8.shape[0],
            canvas_size=self.state.canvas_size,
            kernel_view_size=self.state.kernel_view_size,
        )

        self._connect_signals()
        self._setup_timer()

        # Initial render
        self._sync_neuron_and_rf_limits(reset_to_center=True)
        self._refresh_kernel_preview()
        self._render_frame_and_plots(step_sim=False)
        self._on_stim_type_changed()

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _connect_signals(self) -> None:
        v = self.view

        # Neuron / RF
        v.neuron_combo.currentIndexChanged.connect(self._on_neuron_changed)
        v.neuron_spin.valueChanged.connect(self._on_neuron_spin_changed)
        v.sim_neurons_spin.valueChanged.connect(self._on_sim_neurons_changed)
        v.rf_slot_combo.currentIndexChanged.connect(self._on_rf_slot_changed)
        v.rf_x_slider.valueChanged.connect(self._on_rf_slider_changed)
        v.rf_y_slider.valueChanged.connect(self._on_rf_slider_changed)
        v.rf_x_spin.valueChanged.connect(self._on_rf_spin_changed)
        v.rf_y_spin.valueChanged.connect(self._on_rf_spin_changed)
        v.rf_scale_slider.valueChanged.connect(self._on_rf_scale_changed)
        v.rf_scale_spin.valueChanged.connect(self._on_rf_scale_spin_changed)
        v.mask_rf_btn.clicked.connect(self._on_mask_rf_toggled)
        v.gray_rf_btn.clicked.connect(self._on_gray_rf_toggled)
        v.gray_rf_energy_btn.clicked.connect(self._on_gray_energy_toggled)

        # Stimulus
        v.stim_combo.currentIndexChanged.connect(self._on_stim_type_changed)
        v.stim_x_slider.valueChanged.connect(self._on_stim_slider_changed)
        v.stim_y_slider.valueChanged.connect(self._on_stim_slider_changed)
        for w in [
            v.bar_orientation, v.bar_length, v.bar_thickness,
            v.bar_size, v.bar_contrast,
            v.gr_orientation, v.gr_sf, v.gr_phase,
            v.gr_size, v.gr_contrast,
            v.cifar_size,
        ]:
            w.valueChanged.connect(self._on_any_param_changed)
        for w in [v.bar_color, v.gr_color, v.presentation_mode, v.auto_gr_mode]:
            w.currentIndexChanged.connect(self._on_any_param_changed)
        for w in [v.cifar_index, v.on_ms, v.off_ms, v.auto_bar_step,
                   v.auto_gr_ori_step, v.auto_gr_sf_step]:
            w.valueChanged.connect(self._on_any_param_changed)

        # Dynamics
        v.response_mode.currentIndexChanged.connect(self._on_any_param_changed)
        v.max_rate_slider.valueChanged.connect(self._on_any_param_changed)
        v.rate_ymax_slider.valueChanged.connect(self._on_any_param_changed)
        v.rate_gain.valueChanged.connect(self._on_any_param_changed)
        v.rate_base.valueChanged.connect(self._on_any_param_changed)
        v.fps_slider.valueChanged.connect(self._on_fps_changed)
        v.spike_buffer_spin.valueChanged.connect(self._on_spike_buffer_changed)
        v.btn_start.clicked.connect(self._toggle_timer)
        v.btn_reset.clicked.connect(self._reset_buffers)

        # Canvas drag
        v.canvas_plot.moved.connect(self._on_canvas_dragged)

    # ------------------------------------------------------------------
    # Timer
    # ------------------------------------------------------------------

    def _setup_timer(self) -> None:
        self.timer = QTimer(self.view)
        self.timer.setInterval(int(round(1000.0 / self.state.fps)))
        self.timer.timeout.connect(self._on_timer_tick)
        self.timer.start()

    # ------------------------------------------------------------------
    # Model ↔ View sync helpers
    # ------------------------------------------------------------------

    def _read_ui_into_state(self) -> None:
        """Pull widget values into the SimState before computing frame."""
        v = self.view
        s = self.state

        s.active_neuron_index = v.neuron_combo.currentIndex()
        s.sim_neurons = v.sim_neurons_spin.value()
        s.rf_scale_pct = v.rf_scale_slider.value()
        s.mask_rf = v.mask_rf_btn.isChecked()
        s.gray_rf = v.gray_rf_btn.isChecked()
        s.gray_energy = v.gray_rf_energy_btn.isChecked()
        s.edit_slot_index = int(np.clip(v.rf_slot_combo.currentIndex(), 0, 4))

        s.stim_kind = v.stim_combo.currentText()  # type: ignore[assignment]
        s.stim_center_x = v.stim_x_slider.value()
        s.stim_center_y = v.stim_y_slider.value()

        s.bar_orientation_deg = v.bar_orientation.value()
        s.bar_length_px = v.bar_length.value()
        s.bar_thickness_px = v.bar_thickness.value()
        s.bar_size_px = v.bar_size.value()
        s.bar_contrast_pct = v.bar_contrast.value()
        s.bar_color = v.bar_color.currentText()

        s.gr_orientation_deg = v.gr_orientation.value()
        s.gr_sf_x100 = v.gr_sf.value()
        s.gr_phase_deg = v.gr_phase.value()
        s.gr_size_px = v.gr_size.value()
        s.gr_contrast_pct = v.gr_contrast.value()
        s.gr_color = v.gr_color.currentText()

        s.cifar_index = v.cifar_index.value()
        s.cifar_size_px = v.cifar_size.value()

        s.presentation_mode = "Timed experiment" if v.presentation_mode.currentIndex() == 1 else "Manual"  # type: ignore[assignment]
        s.on_ms = v.on_ms.value()
        s.off_ms = v.off_ms.value()
        s.auto_bar_step_deg = v.auto_bar_step.value()
        s.auto_gr_mode = v.auto_gr_mode.currentText()  # type: ignore[assignment]
        s.auto_gr_ori_step_deg = v.auto_gr_ori_step.value()
        s.auto_gr_sf_step_x1000 = v.auto_gr_sf_step.value()

        s.response_mode = "normalized" if v.response_mode.currentIndex() == 0 else "legacy"  # type: ignore[assignment]
        s.max_rate_hz = v.max_rate_slider.value()
        s.rate_ymax = v.rate_ymax_slider.value()
        s.rate_gain_raw = v.rate_gain.value()
        s.baseline_hz_raw = v.rate_base.value()
        s.spike_buffer = v.spike_buffer_spin.value()

    def _sync_neuron_and_rf_limits(self, reset_to_center: bool = False) -> None:
        """Update RF slider ranges and positions for the currently edited neuron."""
        s = self.state
        v = self.view
        ensure_rf_positions(s)
        ids = active_neuron_ids(s)
        if not ids:
            return

        v.rf_slot_combo.setEnabled(len(ids) > 1)
        if s.edit_slot_index >= len(ids):
            v.rf_slot_combo.blockSignals(True)
            v.rf_slot_combo.setCurrentIndex(len(ids) - 1)
            v.rf_slot_combo.blockSignals(False)
            s.edit_slot_index = len(ids) - 1

        edit_id = edited_neuron_id(s)
        edit_sz = scaled_rf_size(s.v1_cells[edit_id], s.rf_scale_pct)
        max_rf = max(0, s.canvas_size - edit_sz)
        prev = s.rf_pos_by_neuron.get(edit_id, (max_rf // 2, max_rf // 2))
        px = int(np.clip(prev[0], 0, max_rf))
        py = int(np.clip(prev[1], 0, max_rf))
        tx = max_rf // 2 if reset_to_center else px
        ty = max_rf // 2 if reset_to_center else py
        s.rf_pos_by_neuron[edit_id] = (tx, ty)

        for sl, sp in [(v.rf_x_slider, v.rf_x_spin), (v.rf_y_slider, v.rf_y_spin)]:
            sl.setRange(0, max_rf)
            sp.setRange(0, max_rf)
        for w in [v.rf_x_slider, v.rf_y_slider, v.rf_x_spin, v.rf_y_spin]:
            w.blockSignals(True)
        v.rf_x_slider.setValue(tx); v.rf_y_slider.setValue(ty)
        v.rf_x_spin.setValue(tx); v.rf_y_spin.setValue(ty)
        for w in [v.rf_x_slider, v.rf_y_slider, v.rf_x_spin, v.rf_y_spin]:
            w.blockSignals(False)

    def _refresh_kernel_preview(self) -> None:
        s = self.state
        v = self.view
        ids = active_neuron_ids(s)
        preview_id = ids[0] if ids else s.active_neuron_index
        kernel = build_kernel(s.v1_cells[preview_id], s)
        v.kernel_item.setImage(
            centered_kernel_preview_uint8(kernel, canvas_size=s.kernel_view_size, scale=5),
            autoLevels=False,
        )
        t_ms, _ = generate_spike_waveform_from_rf(kernel, duration_ms=1.5, sample_rate_hz=20000)
        v.spike_shape_plot.setXRange(0.0, float(t_ms[-1]), padding=0.0)
        self._refresh_spike_shape_panel()

    def _refresh_spike_shape_panel(self) -> None:
        s = self.state
        from model import ensure_trace_buffers
        ensure_trace_buffers(s)
        for i, curve in enumerate(self.view.spike_shape_curves):
            if i < len(s.spike_waveforms_xy_by_neuron):
                x, y = overlay_waveforms_to_lines(s.spike_waveforms_xy_by_neuron[i])
                curve.setData(x, y)
            else:
                curve.setData([], [])
        self.view.spike_shape_plot.setYRange(-1.6, 1.6, padding=0.0)

    def _render_frame_and_plots(self, step_sim: bool = True) -> None:
        """Core render: update model, composite frame, update all plots."""
        s = self.state
        v = self.view
        self._read_ui_into_state()
        ensure_rf_positions(s)

        if step_sim:
            spikes, mean_rate = step_simulation(s)
            spike_any = 1 if any(sp == 1 for sp in spikes) else 0
            self.audio.update(mean_rate, spike_any, s.dt_s)
            self._refresh_spike_shape_panel()

        # Compose canvas frame for display
        frame_m11 = compose_canvas_m11(s)
        _raws, rates, contexts, _kernels = compute_responses(s, frame_m11)

        # Status label
        if rates:
            mode_str = "Normalized" if s.response_mode == "normalized" else "Legacy"
            v.response_info.setText(f"Mode={mode_str} | N={len(rates)} | Mean rate={float(np.mean(rates)):.1f} Hz")

        # Canvas image
        rgb = m11_to_uint8_rgb(frame_m11)
        ids = active_neuron_ids(s)
        for i, (ctx, _k) in enumerate(zip(contexts, _kernels)):
            col_key = NEURON_COLORS[i % len(NEURON_COLORS)]
            rgb = draw_rf_box_uint8(rgb, ctx.coords[0], ctx.coords[1], ctx.size,
                                    color=NEURON_RGB.get(col_key, (255, 255, 255)), thickness=2)
        v.canvas_item.setImage(rgb, autoLevels=False)

        # Rate & spike plots
        if not s.times:
            for c in v.rate_curves:
                c.setData([], [])
            for c in v.spike_lines:
                c.setData([], [])
            return

        t0 = s.times[-1] - s.window_sec
        rel_t = np.array(s.times) - t0

        for i, c in enumerate(v.rate_curves):
            if i < len(s.rates_by_neuron):
                c.setData(rel_t, np.array(s.rates_by_neuron[i], dtype=np.float32))
            else:
                c.setData([], [])
        v.rate_plot.setXRange(max(0.0, rel_t[-1] - s.window_sec), rel_t[-1] + 0.001, padding=0.0)
        v.rate_plot.setYRange(0.0, float(s.rate_ymax), padding=0.0)

        for i, c in enumerate(v.spike_lines):
            if i < len(s.spike_times_by_neuron):
                rel_sp = np.array([t - t0 for t in s.spike_times_by_neuron[i] if t >= t0], dtype=np.float32)
                if rel_sp.size > 0:
                    x_l = np.repeat(rel_sp, 3)
                    y_l = np.tile(np.array([0.0, 1.0, np.nan], dtype=np.float32), rel_sp.size)
                    x_l[2::3] = np.nan
                    c.setData(x_l, y_l)
                else:
                    c.setData([], [])
            else:
                c.setData([], [])
        v.spike_plot.setXRange(max(0.0, rel_t[-1] - s.window_sec), rel_t[-1] + 0.001, padding=0.0)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_neuron_changed(self) -> None:
        idx = self.view.neuron_combo.currentIndex()
        self.view.neuron_spin.blockSignals(True)
        self.view.neuron_spin.setValue(idx)
        self.view.neuron_spin.blockSignals(False)
        self.state.active_neuron_index = idx
        self._sync_neuron_and_rf_limits(reset_to_center=True)
        self._refresh_kernel_preview()
        self._reset_buffers()

    def _on_neuron_spin_changed(self, value: int) -> None:
        if value == self.view.neuron_combo.currentIndex():
            return
        self.view.neuron_combo.blockSignals(True)
        self.view.neuron_combo.setCurrentIndex(value)
        self.view.neuron_combo.blockSignals(False)
        self._on_neuron_changed()

    def _on_sim_neurons_changed(self) -> None:
        self.state.sim_neurons = self.view.sim_neurons_spin.value()
        self._sync_neuron_and_rf_limits(reset_to_center=False)
        self._refresh_kernel_preview()
        self._reset_buffers()

    def _on_rf_slot_changed(self) -> None:
        self.state.edit_slot_index = self.view.rf_slot_combo.currentIndex()
        self._sync_neuron_and_rf_limits(reset_to_center=False)
        self._render_frame_and_plots(step_sim=False)

    def _on_rf_slider_changed(self) -> None:
        eid = edited_neuron_id(self.state)
        self.state.rf_pos_by_neuron[eid] = (
            int(self.view.rf_x_slider.value()),
            int(self.view.rf_y_slider.value()),
        )
        self.view.rf_x_spin.blockSignals(True)
        self.view.rf_y_spin.blockSignals(True)
        self.view.rf_x_spin.setValue(self.view.rf_x_slider.value())
        self.view.rf_y_spin.setValue(self.view.rf_y_slider.value())
        self.view.rf_x_spin.blockSignals(False)
        self.view.rf_y_spin.blockSignals(False)
        self._render_frame_and_plots(step_sim=False)

    def _on_rf_spin_changed(self) -> None:
        eid = edited_neuron_id(self.state)
        self.state.rf_pos_by_neuron[eid] = (
            int(self.view.rf_x_spin.value()),
            int(self.view.rf_y_spin.value()),
        )
        self.view.rf_x_slider.blockSignals(True)
        self.view.rf_y_slider.blockSignals(True)
        self.view.rf_x_slider.setValue(self.view.rf_x_spin.value())
        self.view.rf_y_slider.setValue(self.view.rf_y_spin.value())
        self.view.rf_x_slider.blockSignals(False)
        self.view.rf_y_slider.blockSignals(False)
        self._render_frame_and_plots(step_sim=False)

    def _on_rf_scale_changed(self) -> None:
        self.view.rf_scale_spin.blockSignals(True)
        self.view.rf_scale_spin.setValue(self.view.rf_scale_slider.value())
        self.view.rf_scale_spin.blockSignals(False)
        self.state.rf_scale_pct = self.view.rf_scale_slider.value()
        self._sync_neuron_and_rf_limits(reset_to_center=False)
        self._refresh_kernel_preview()
        self._render_frame_and_plots(step_sim=False)

    def _on_rf_scale_spin_changed(self) -> None:
        self.view.rf_scale_slider.blockSignals(True)
        self.view.rf_scale_slider.setValue(self.view.rf_scale_spin.value())
        self.view.rf_scale_slider.blockSignals(False)
        self.state.rf_scale_pct = self.view.rf_scale_spin.value()
        self._sync_neuron_and_rf_limits(reset_to_center=False)
        self._refresh_kernel_preview()
        self._render_frame_and_plots(step_sim=False)

    def _on_mask_rf_toggled(self) -> None:
        checked = self.view.mask_rf_btn.isChecked()
        self.view.mask_rf_btn.setText(f"Mask RF: {'ON' if checked else 'OFF'}")
        self.state.mask_rf = checked
        self._refresh_kernel_preview()
        self._render_frame_and_plots(step_sim=False)

    def _on_gray_rf_toggled(self) -> None:
        checked = self.view.gray_rf_btn.isChecked()
        self.view.gray_rf_btn.setText(f"RF grayscale: {'ON' if checked else 'OFF'}")
        self.state.gray_rf = checked
        self._refresh_kernel_preview()
        self._render_frame_and_plots(step_sim=False)

    def _on_gray_energy_toggled(self) -> None:
        checked = self.view.gray_rf_energy_btn.isChecked()
        self.view.gray_rf_energy_btn.setText(f"Gray energy match: {'ON' if checked else 'OFF'}")
        self.state.gray_energy = checked
        if self.state.gray_rf:
            self._refresh_kernel_preview()
            self._render_frame_and_plots(step_sim=False)

    def _on_stim_slider_changed(self) -> None:
        self._render_frame_and_plots(step_sim=False)

    def _on_any_param_changed(self) -> None:
        self._render_frame_and_plots(step_sim=False)

    def _on_stim_type_changed(self) -> None:
        kind = self.view.stim_combo.currentText()
        self.view.set_stim_visibility(
            is_bar=(kind == "Bar"),
            is_grating=(kind == "Grating"),
            is_cifar=(kind == "CIFAR patch"),
        )
        self._render_frame_and_plots(step_sim=False)

    def _on_canvas_dragged(self, x: float, y: float) -> None:
        cs = self.state.canvas_size
        sx = int(np.clip(round(x), 0, cs - 1))
        sy = int(np.clip(round(y), 0, cs - 1))
        self.view.stim_x_slider.blockSignals(True)
        self.view.stim_y_slider.blockSignals(True)
        self.view.stim_x_slider.setValue(sx)
        self.view.stim_y_slider.setValue(sy)
        self.view.stim_x_slider.blockSignals(False)
        self.view.stim_y_slider.blockSignals(False)
        self._render_frame_and_plots(step_sim=False)

    def _on_fps_changed(self) -> None:
        self.state.fps = int(self.view.fps_slider.value())
        self.timer.setInterval(int(round(1000.0 / self.state.fps)))

    def _on_spike_buffer_changed(self) -> None:
        self.state.spike_buffer = int(self.view.spike_buffer_spin.value())
        for i in range(len(self.state.spike_waveforms_xy_by_neuron)):
            buf = self.state.spike_waveforms_xy_by_neuron[i]
            if len(buf) > self.state.spike_buffer:
                self.state.spike_waveforms_xy_by_neuron[i] = buf[-self.state.spike_buffer:]
        self._refresh_spike_shape_panel()

    def _toggle_timer(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
            self.view.btn_start.setText("Start")
            self.state.paused = True
        else:
            self.timer.start()
            self.view.btn_start.setText("Pause")
            self.state.paused = False

    def _reset_buffers(self) -> None:
        reset_buffers(self.state)
        self._refresh_spike_shape_panel()
        self.audio.reset()
        self._render_frame_and_plots(step_sim=False)

    def _on_timer_tick(self) -> None:
        self._render_frame_and_plots(step_sim=True)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def show(self) -> None:
        self.view.show()

    def close_event_hook(self, event: QCloseEvent) -> None:
        """Call from the view's closeEvent override."""
        self.audio.stop()
