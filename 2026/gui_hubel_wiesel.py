"""
Hubel-Wiesel Synthetic V1 Explorer – entry point.

Architecture:
    model.py      – ``SimState`` dataclass + pure simulation functions
    view.py       – ``SimView`` (all Qt widgets, layout, styling)
    controller.py – ``SimController`` (signal wiring, timer, audio)

The original monolithic version is preserved in ``gui_hubel_wiesel_original.py``.
"""

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QCloseEvent

from controller import SimController


def main() -> None:
    app = QApplication(sys.argv)
    ctrl = SimController()

    # Hook closeEvent so the controller can clean up audio
    original_close = ctrl.view.closeEvent

    def _patched_close(event: QCloseEvent) -> None:
        ctrl.close_event_hook(event)
        original_close(event)

    ctrl.view.closeEvent = _patched_close  # type: ignore[assignment]

    ctrl.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
