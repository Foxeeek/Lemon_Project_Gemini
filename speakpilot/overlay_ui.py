"""Module 4: Desktop overlay UI for SpeakPilot corrections."""

from __future__ import annotations

from typing import Any, Dict

from PyQt6.QtCore import QEasingCurve, QPropertyAnimation, Qt, QTimer
from PyQt6.QtWidgets import QApplication, QGraphicsOpacityEffect, QLabel, QVBoxLayout, QWidget

from speakpilot.config import INTERVIEW_MODE, OVERLAY_AUTO_HIDE_MS, OVERLAY_FADE_IN_MS


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

        self.opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity_effect)
        self.fade_anim = QPropertyAnimation(self.opacity_effect, b"opacity", self)
        self.fade_anim.setDuration(OVERLAY_FADE_IN_MS)
        self.fade_anim.setEasingCurve(QEasingCurve.Type.InOutQuad)

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

        self.badge_label = QLabel("Interview Mode")
        self.badge_label.setStyleSheet("color: #f1c40f; font-size: 11px; font-weight: 700;")
        self.badge_label.setVisible(INTERVIEW_MODE)

        self.original_label = QLabel("")
        self.corrected_label = QLabel("")
        self.explanation_label = QLabel("")

        for lbl in (self.original_label, self.corrected_label, self.explanation_label):
            lbl.setWordWrap(True)

        self.original_label.setStyleSheet("color: #ff8a8a;")
        self.corrected_label.setStyleSheet("color: #6bff8a;")
        self.explanation_label.setStyleSheet("color: #a0a0a0; font-size: 11px;")

        box_layout = QVBoxLayout(self.container)
        box_layout.setContentsMargins(14, 10, 14, 10)
        box_layout.setSpacing(4)
        box_layout.addWidget(self.badge_label)
        box_layout.addWidget(self.original_label)
        box_layout.addWidget(self.corrected_label)
        box_layout.addWidget(self.explanation_label)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.addWidget(self.container)

        self.setFixedSize(700, 170)
        self.setMaximumHeight(170)

    def _apply_window_flags(self) -> None:
        flags = Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool
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

    def _animate_in(self) -> None:
        self.fade_anim.stop()
        self.opacity_effect.setOpacity(0.0)
        self.fade_anim.setStartValue(0.0)
        self.fade_anim.setEndValue(1.0)
        self.fade_anim.start()

    def update_from_result(self, result: Dict[str, Any]) -> None:
        if not result or not result.get("is_error", False):
            return

        self.original_label.setText(f"Original: {str(result.get('original', '')).strip()}")
        self.corrected_label.setText(f"Correction: {str(result.get('corrected', '')).strip()}")
        self.explanation_label.setText(f"Why: {str(result.get('explanation', '')).strip()}")

        self._position_bottom_center()
        self.show()
        self.raise_()
        self._animate_in()
        self.hide_timer.start(OVERLAY_AUTO_HIDE_MS)

    def show_summary(self, summary_text: str, common_error: str) -> None:
        self.original_label.setText(summary_text)
        self.corrected_label.setText(f"Most common error: {common_error}")
        self.explanation_label.setText("Session summary")
        self._position_bottom_center()
        self.show()
        self.raise_()
        self._animate_in()
        self.hide_timer.start(OVERLAY_AUTO_HIDE_MS)
