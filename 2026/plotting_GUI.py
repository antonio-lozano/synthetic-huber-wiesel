import numpy as np
import pyqtgraph as pg
from PySide6.QtGui import QFont

def m11_to_uint8_rgb(img_m11: np.ndarray) -> np.ndarray:
    clipped = np.clip(img_m11, -1.0, 1.0)
    return (((clipped + 1.0) / 2.0) * 255.0).astype(np.uint8)


def draw_rf_box_uint8(
    rgb: np.ndarray,
    x_rf: int,
    y_rf: int,
    size: int,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    out = rgb.copy()
    h, w = out.shape[:2]
    x0 = int(np.clip(x_rf, 0, w - 1))
    y0 = int(np.clip(y_rf, 0, h - 1))
    x1 = int(np.clip(x0 + size, 0, w))
    y1 = int(np.clip(y0 + size, 0, h))
    if x1 <= x0 or y1 <= y0:
        return out

    t = max(1, int(thickness))
    out[y0 : min(y0 + t, y1), x0:x1, :] = color
    out[max(y1 - t, y0) : y1, x0:x1, :] = color
    out[y0:y1, x0 : min(x0 + t, x1), :] = color
    out[y0:y1, max(x1 - t, x0) : x1, :] = color
    return out


def kernel_preview_uint8(kernel_rgb: np.ndarray) -> np.ndarray:
    return m11_to_uint8_rgb(kernel_rgb)


def centered_kernel_preview_uint8(
    kernel_rgb: np.ndarray,
    canvas_size: int = 192,
    scale: int = 4,
) -> np.ndarray:
    preview = kernel_preview_uint8(kernel_rgb)
    scale = max(1, int(scale))
    enlarged = np.repeat(np.repeat(preview, scale, axis=0), scale, axis=1)
    h, w = enlarged.shape[:2]

    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    if h > canvas_size or w > canvas_size:
        y0 = max(0, (h - canvas_size) // 2)
        x0 = max(0, (w - canvas_size) // 2)
        enlarged = enlarged[y0 : y0 + canvas_size, x0 : x0 + canvas_size]
        h, w = enlarged.shape[:2]

    cy = (canvas_size - h) // 2
    cx = (canvas_size - w) // 2
    canvas[cy : cy + h, cx : cx + w, :] = enlarged
    return canvas


def stacked_waveforms_to_lines(
    waveforms: list[np.ndarray],
    t_ms: np.ndarray,
    spacing: float = 2.2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert buffered waveforms into NaN-separated line data for pyqtgraph.
    """
    if len(waveforms) == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for i, w in enumerate(waveforms):
        offset = float(i) * spacing
        x_parts.append(t_ms.astype(np.float32))
        y_parts.append((w + offset).astype(np.float32))
        x_parts.append(np.array([np.nan], dtype=np.float32))
        y_parts.append(np.array([np.nan], dtype=np.float32))
    return np.concatenate(x_parts), np.concatenate(y_parts)


def overlay_waveforms_to_lines(
    waveforms_xy: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a list of (t_ms, waveform) into NaN-separated line data, all overlaid.
    """
    if len(waveforms_xy) == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for t_ms, wave in waveforms_xy:
        x_parts.append(t_ms.astype(np.float32))
        y_parts.append(wave.astype(np.float32))
        x_parts.append(np.array([np.nan], dtype=np.float32))
        y_parts.append(np.array([np.nan], dtype=np.float32))
    return np.concatenate(x_parts), np.concatenate(y_parts)


def style_plot_widget(plot: pg.PlotWidget, title: str, y_label: str, x_label: str = "Time (s)") -> None:
    plot.setBackground("k")
    plot.showGrid(x=True, y=True, alpha=0.2)
    plot.setTitle(title, color="w", size="16pt")
    plot.setLabel("left", y_label, color="w", **{"font-size": "14pt"})
    plot.setLabel("bottom", x_label, color="w", **{"font-size": "14pt"})
    plot.getAxis("left").setPen(pg.mkPen("w", width=1.5))
    plot.getAxis("bottom").setPen(pg.mkPen("w", width=1.5))
    tick_font = QFont("Arial", 12)
    plot.getAxis("left").setTickFont(tick_font)
    plot.getAxis("bottom").setTickFont(tick_font)


def style_image_plot(plot: pg.PlotWidget, title: str, canvas_size: int) -> None:
    plot.setBackground("k")
    plot.setTitle(title, color="w", size="16pt")
    plot.setLabel("left", "Y", color="w", **{"font-size": "14pt"})
    plot.setLabel("bottom", "X", color="w", **{"font-size": "14pt"})
    plot.getAxis("left").setPen(pg.mkPen("w", width=1.5))
    plot.getAxis("bottom").setPen(pg.mkPen("w", width=1.5))
    tick_font = QFont("Arial", 12)
    plot.getAxis("left").setTickFont(tick_font)
    plot.getAxis("bottom").setTickFont(tick_font)
    plot.setXRange(0, canvas_size, padding=0.0)
    plot.setYRange(0, canvas_size, padding=0.0)
    plot.setAspectLocked(True)
