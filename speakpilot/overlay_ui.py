"""Module 4: Desktop overlay UI for SpeakPilot corrections."""

from __future__ import annotations

from typing import Any, Dict

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget


class FloatingWindow(QWidget):
    """Borderless always-on-top overlay window for correction display."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SpeakPilot Overlay")
        self._build_ui()
        self._apply_window_flags()
        self._position_bottom_center()

        self.hide_timer = QTimer(self)
        self.hide_timer.setSingleShot(True)
        self.hide_timer.timeout.connect(self.hide)

    def _build_ui(self) -> None:
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        self.container = QWidget(self)
        self.container.setStyleSheet(
            """
            QWidget {
                background-color: rgba(30, 30, 30, 217);
                border: 1px solid rgba(255, 255, 255, 28);
                border-radius: 14px;
            }
            """
        )

        self.original_label = QLabel("")
        self.corrected_label = QLabel("")
        self.explanation_label = QLabel("")

        self.original_label.setWordWrap(True)
        self.corrected_label.setWordWrap(True)
        self.explanation_label.setWordWrap(True)

        self.original_label.setStyleSheet("color: #ff8a8a;")
        self.corrected_label.setStyleSheet("color: #6bff8a;")
        self.explanation_label.setStyleSheet("color: #a0a0a0;")

        self.original_label.setFont(QFont("Arial", 14, QFont.Weight.Medium))
        self.corrected_label.setFont(QFont("Arial", 15, QFont.Weight.Bold))
        self.explanation_label.setFont(QFont("Arial", 11, QFont.Weight.Normal))

        box_layout = QVBoxLayout(self.container)
        box_layout.setContentsMargins(14, 12, 14, 12)
        box_layout.setSpacing(6)
        box_layout.addWidget(self.original_label)
        box_layout.addWidget(self.corrected_label)
        box_layout.addWidget(self.explanation_label)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.addWidget(self.container)

        self.setFixedSize(680, 160)

    def _apply_window_flags(self) -> None:
        flags = (
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )

        if hasattr(Qt.WindowType, "WindowTransparentForInput"):
            flags |= Qt.WindowType.WindowTransparentForInput

        self.setWindowFlags(flags)

    def _position_bottom_center(self) -> None:
        screen = QApplication.primaryScreen()
        if not screen:
            return
        geo = screen.availableGeometry()
        x = geo.x() + (geo.width() - self.width()) // 2
        y = geo.y() + geo.height() - self.height() - 50
        self.move(x, y)

    def update_from_result(self, result: Dict[str, Any]) -> None:
        """Render correction result; ignore non-error payloads."""
        if not result or not result.get("is_error", False):
            return

        original = str(result.get("original", "")).strip()
        corrected = str(result.get("corrected", "")).strip()
        explanation = str(result.get("explanation", "")).strip()

        self.original_label.setText(f"Original: {original}")
        self.corrected_label.setText(f"Correction: {corrected}")
        self.explanation_label.setText(f"Why: {explanation}")

        self._position_bottom_center()
        self.show()
        self.raise_()
        self.hide_timer.start(7000)
