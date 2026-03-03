"""RF Circuit Design Tool - Smith Chart Impedance Analyzer"""

import sys
import uuid
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from collections import deque

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter,
    QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit,
    QLabel, QProgressBar, QInputDialog, QMessageBox, QSizePolicy,
    QDialog, QComboBox
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QPointF, QRectF
from PyQt5.QtGui import (
    QPainter, QPen, QBrush, QColor, QPainterPath, QFont
)

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class Component:
    id: str
    type: str            # 'R', 'L', 'C'
    x: int
    y: int
    rotation: int        # 0 or 90
    value: Optional[float]
    selected: bool = False
    error_highlight: bool = False


@dataclass
class Wire:
    id: str
    start_comp_id: str
    start_pin: str
    end_comp_id: str
    end_pin: str


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GRID = 20
PORT_X = 60
COLOR_DEFAULT = QColor('#212121')
COLOR_SELECTED = QColor('#1565C0')
COLOR_ERROR = QColor('#D32F2F')
COLOR_ERROR_BG = QColor('#FFEBEE')
COLOR_WIRE = QColor('#F44336')
COLOR_PORT = QColor('#4CAF50')
COLOR_GRID = QColor('#BDBDBD')

ERRORS = {
    'freq_empty':   "주파수 입력 오류\n시작/끝 주파수를 MHz 단위 숫자로 입력하세요.",
    'freq_order':   "주파수 범위 오류\n시작 주파수는 끝 주파수보다 작아야 합니다.",
    'no_component': "회로 없음\nPort에 연결된 소자가 없습니다.\n소자를 배치하고 Port와 연결하세요.",
    'no_value':     "값 미입력 소자가 있습니다.\n해당 소자(빨간 테두리)를 더블클릭하여 값을 입력하세요.",
    'isolated':     "고립된 소자가 있습니다.\nPort 신호선(+)에서 GND까지 경로에 포함되지 않는 소자(빨간 테두리)를 확인하세요.",
}


def snap(v: int) -> int:
    return round(v / GRID) * GRID


def new_id() -> str:
    return uuid.uuid4().hex[:8]


def pin_pos(comp: 'Component', pin: str) -> Tuple[int, int]:
    """Return absolute (x, y) for a pin of a component."""
    x, y = comp.x, comp.y
    if comp.rotation == 0:
        if pin == 'left':
            return (x - 30, y)
        elif pin == 'right':
            return (x + 30, y)
    else:  # 90
        if pin == 'top':
            return (x, y - 30)
        elif pin == 'bottom':
            return (x, y + 30)
    return (x, y)


def port_pin_pos(canvas_height: int, pin: str) -> Tuple[int, int]:
    cy = canvas_height // 2
    if pin == 'plus':
        return (PORT_X, cy - 40)
    else:  # gnd
        return (PORT_X, cy + 40)


def get_pin_pos_universal(comp_id: str, pin: str,
                           components: List[Component],
                           canvas_height: int) -> Optional[Tuple[int, int]]:
    """Get pin position for any comp_id including PORT."""
    if comp_id == 'PORT':
        return port_pin_pos(canvas_height, pin)
    for c in components:
        if c.id == comp_id:
            return pin_pos(c, pin)
    return None


def calc_wire_path(sx, sy, ex, ey) -> List[Tuple[int, int]]:
    """L-shaped wire path: horizontal first, then vertical."""
    if sy == ey:
        return [(sx, sy), (ex, ey)]
    return [(sx, sy), (ex, sy), (ex, ey)]


# ---------------------------------------------------------------------------
# CircuitCanvas
# ---------------------------------------------------------------------------

class CircuitCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.components: List[Component] = []
        self.wires: List[Wire] = []

        self.placement_mode: Optional[str] = None
        self.wiring_start: Optional[Tuple[str, str]] = None  # (comp_id, pin)
        self.move_mode: bool = False

        self._click_timer = QTimer()
        self._click_timer.setSingleShot(True)
        self._click_timer.setInterval(150)
        self._pending_click_pos = None
        self._click_timer.timeout.connect(self._process_single_click)

        # 고스트 프리뷰 및 드래그 상태
        self._ghost_pos: Optional[Tuple[int, int]] = None   # 고스트 스냅 위치
        self._drag_start_comp: Optional['Component'] = None  # 드래그 후보 소자
        self._drag_start_mouse: Optional[Tuple[int, int]] = None
        self._dragging_comp: Optional['Component'] = None    # 실제 드래그 중 소자
        _DRAG_THRESHOLD = 8  # px, 드래그로 간주하는 최소 이동거리
        self._DRAG_THRESHOLD = _DRAG_THRESHOLD

        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)

    # ------------------------------------------------------------------
    # Paint
    # ------------------------------------------------------------------

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        self._draw_grid(painter)
        self._draw_port(painter)
        for wire in self.wires:
            self._draw_wire(painter, wire)
        for comp in self.components:
            self._draw_component(painter, comp)
        # 와이어링 시작 핀 강조
        if self.wiring_start:
            cid, pin = self.wiring_start
            pos = get_pin_pos_universal(cid, pin, self.components, self.height())
            if pos:
                painter.setPen(QPen(COLOR_WIRE, 2))
                painter.setBrush(QBrush(COLOR_WIRE))
                painter.drawEllipse(pos[0] - 5, pos[1] - 5, 10, 10)
        # 고스트 프리뷰: 새 소자 배치 모드 또는 기존 소자 드래그 중
        if self._ghost_pos:
            gx, gy = self._ghost_pos
            # 드래그 중이면 고스트 소자의 연결선 점선 미리보기
            if self._dragging_comp:
                self._draw_ghost_wire_preview(painter, self._dragging_comp, gx, gy)
            # 고스트 소자 그리기 (50% 불투명도)
            ghost_type = (self._dragging_comp.type if self._dragging_comp
                          else self.placement_mode)
            ghost_rot = self._dragging_comp.rotation if self._dragging_comp else 0
            if ghost_type:
                painter.setOpacity(0.5)
                ghost = Component(id='__ghost__', type=ghost_type,
                                  x=gx, y=gy, rotation=ghost_rot, value=None)
                self._draw_component(painter, ghost)
                painter.setOpacity(1.0)

    def _draw_grid(self, painter: QPainter):
        pen = QPen(COLOR_GRID, 1, Qt.DotLine)
        painter.setPen(pen)
        w, h = self.width(), self.height()
        for x in range(0, w, GRID):
            painter.drawLine(x, 0, x, h)
        for y in range(0, h, GRID):
            painter.drawLine(0, y, w, y)

    def _draw_port(self, painter: QPainter):
        h = self.height()
        py_plus = h // 2 - 40
        py_gnd = h // 2 + 40
        x = PORT_X

        pen = QPen(COLOR_DEFAULT, 2)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawLine(x, py_plus, x, py_gnd)

        painter.setBrush(QBrush(COLOR_PORT))
        painter.setPen(QPen(COLOR_PORT, 1))
        painter.drawEllipse(x - 5, py_plus - 5, 10, 10)
        painter.drawEllipse(x - 5, py_gnd - 5, 10, 10)

        painter.setPen(QPen(COLOR_DEFAULT, 1))
        font = QFont('Arial', 9)
        painter.setFont(font)
        painter.drawText(x - 40, h // 2 - 5, 'Port')
        painter.drawText(x + 8, py_plus + 4, '+')
        painter.drawText(x + 8, py_gnd + 4, 'GND')

    def _draw_wire(self, painter: QPainter, wire: Wire):
        p1 = get_pin_pos_universal(wire.start_comp_id, wire.start_pin,
                                   self.components, self.height())
        p2 = get_pin_pos_universal(wire.end_comp_id, wire.end_pin,
                                   self.components, self.height())
        if not p1 or not p2:
            return
        pts = calc_wire_path(p1[0], p1[1], p2[0], p2[1])
        pen = QPen(COLOR_WIRE, 2)
        painter.setPen(pen)
        for i in range(len(pts) - 1):
            painter.drawLine(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1])

    def _draw_component(self, painter: QPainter, comp: Component):
        painter.save()
        painter.translate(comp.x, comp.y)
        if comp.rotation == 90:
            painter.rotate(90)

        # Background for error
        if comp.error_highlight:
            painter.fillRect(-35, -15, 70, 30, QColor('#FFEBEE'))

        # Choose pen
        if comp.error_highlight:
            pen = QPen(COLOR_ERROR, 3)
        elif comp.selected:
            pen = QPen(COLOR_SELECTED, 3)
        else:
            pen = QPen(COLOR_DEFAULT, 2)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)

        if comp.type == 'R':
            self._draw_resistor(painter)
        elif comp.type == 'L':
            self._draw_inductor(painter)
        elif comp.type == 'C':
            self._draw_capacitor(painter)

        painter.restore()

        # Value label
        font = QFont('Arial', 9)
        painter.setFont(font)
        painter.setPen(QPen(QColor('#9E9E9E') if comp.value is None else COLOR_DEFAULT, 1))
        label = self._value_label(comp)
        if comp.rotation == 0:
            painter.drawText(comp.x - 20, comp.y + 25, label)
        else:
            painter.drawText(comp.x + 14, comp.y + 5, label)

    def _draw_resistor(self, painter: QPainter):
        path = QPainterPath()
        path.moveTo(-30, 0)
        path.lineTo(-20, 0)
        path.lineTo(-15, -8)
        path.lineTo(-5, 8)
        path.lineTo(5, -8)
        path.lineTo(15, 8)
        path.lineTo(20, 0)
        path.lineTo(30, 0)
        painter.drawPath(path)

    def _draw_inductor(self, painter: QPainter):
        path = QPainterPath()
        path.moveTo(-30, 0)
        path.lineTo(-20, 0)
        # 4 arcs each 5px radius, upper bump
        for i in range(4):
            cx = -20 + i * 10 + 5
            path.arcTo(cx - 5, -5, 10, 10, 180, -180)
        path.lineTo(30, 0)
        painter.drawPath(path)

    def _draw_capacitor(self, painter: QPainter):
        # Left lead
        painter.drawLine(-30, 0, -4, 0)
        # Left plate
        pen = painter.pen()
        thick_pen = QPen(pen.color(), 3)
        painter.setPen(thick_pen)
        painter.drawLine(-4, -12, -4, 12)
        painter.drawLine(4, -12, 4, 12)
        painter.setPen(pen)
        # Right lead
        painter.drawLine(4, 0, 30, 0)

    def _value_label(self, comp: Component) -> str:
        if comp.value is None:
            return '?'
        unit = {'R': 'Ω', 'L': 'nH', 'C': 'pF'}[comp.type]
        v = comp.value
        if v == int(v):
            return f"{int(v)}{unit}"
        return f"{v:g}{unit}"

    # ------------------------------------------------------------------
    # Mouse Events
    # ------------------------------------------------------------------

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        x, y = event.x(), event.y()

        # ESC handled in keyPress; here handle placement
        if self.placement_mode:
            sx, sy = snap(x), snap(y)
            # Prevent placement near port
            if abs(sx - PORT_X) < 40:
                return
            comp = Component(
                id=new_id(), type=self.placement_mode,
                x=sx, y=sy, rotation=0, value=None
            )
            self.components.append(comp)
            self.placement_mode = None
            self.update()
            return

        # Move mode
        if self.move_mode:
            sel = self._selected_component()
            if sel:
                sx, sy = snap(x), snap(y)
                sel.x, sel.y = sx, sy
                # Remove connected wires
                self.wires = [w for w in self.wires
                              if w.start_comp_id != sel.id and w.end_comp_id != sel.id]
                self.move_mode = False
                self.update()
            return

        # Check pin click first (radius 8px)
        pin_hit = self._find_pin(x, y, radius=8)
        if pin_hit:
            if self.wiring_start is None:
                self.wiring_start = pin_hit
            else:
                if pin_hit == self.wiring_start:
                    self.wiring_start = None
                else:
                    # Check no duplicate wire
                    sc, sp = self.wiring_start
                    ec, ep = pin_hit
                    w = Wire(id=new_id(),
                             start_comp_id=sc, start_pin=sp,
                             end_comp_id=ec, end_pin=ep)
                    self.wires.append(w)
                    self.wiring_start = None
            self.update()
            return

        # Cancel wiring on empty space
        if self.wiring_start:
            self.wiring_start = None
            self.update()
            return

        # 소자 위에 있으면 드래그 후보로 등록, 빈 공간이면 선택 해제 타이머
        comp = self._find_component(x, y, radius=30)
        if comp:
            # 드래그 시작 후보: mouseMoveEvent에서 임계값 초과 시 드래그 활성화
            self._drag_start_comp = comp
            self._drag_start_mouse = (x, y)
        else:
            self._pending_click_pos = (x, y)
            self._click_timer.start()

    def mouseDoubleClickEvent(self, event):
        self._click_timer.stop()
        self._pending_click_pos = None
        # 드래그 후보 취소 (더블클릭은 드래그가 아님)
        self._drag_start_comp = None
        self._drag_start_mouse = None
        x, y = event.x(), event.y()
        comp = self._find_component(x, y, radius=30)
        if comp:
            self._edit_value(comp)

    def _process_single_click(self):
        if self._pending_click_pos is None:
            return
        x, y = self._pending_click_pos
        self._pending_click_pos = None
        # Deselect all
        for c in self.components:
            c.selected = False
        comp = self._find_component(x, y, radius=30)
        if comp:
            comp.selected = True
        self.update()

    def _edit_value(self, comp: Component):
        label = {'R': 'Ω', 'L': 'nH', 'C': 'pF'}[comp.type]
        hint = {'R': '예: 50', 'L': '예: 10', 'C': '예: 3.3'}[comp.type]
        val, ok = QInputDialog.getDouble(
            self, f'{comp.type} 값 입력',
            f'값을 입력하세요 (단위: {label})\n{hint}',
            value=comp.value or 0.0,
            min=1e-9, max=1e9, decimals=4
        )
        if ok:
            comp.value = val
            comp.error_highlight = False
            self.update()

    # ------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Escape:
            self.wiring_start = None
            self.placement_mode = None
            self.move_mode = False
            # 드래그/고스트 취소
            self._dragging_comp = None
            self._drag_start_comp = None
            self._drag_start_mouse = None
            self._ghost_pos = None
            self.update()
            return

        sel = self._selected_component()
        if sel is None:
            return

        if key == Qt.Key_R:
            sel.rotation = 90 if sel.rotation == 0 else 0
            self.update()
        elif key == Qt.Key_Delete:
            self.wires = [w for w in self.wires
                         if w.start_comp_id != sel.id and w.end_comp_id != sel.id]
            self.components.remove(sel)
            self.update()
        elif key == Qt.Key_M:
            self.move_mode = True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_component(self, x, y, radius=30) -> Optional[Component]:
        best, best_d = None, radius + 1
        for c in self.components:
            d = ((c.x - x)**2 + (c.y - y)**2) ** 0.5
            if d < best_d:
                best_d = d
                best = c
        return best

    def _find_pin(self, x, y, radius=8) -> Optional[Tuple[str, str]]:
        # Check port pins
        for pin in ('plus', 'gnd'):
            px, py = port_pin_pos(self.height(), pin)
            if ((px - x)**2 + (py - y)**2) ** 0.5 <= radius:
                return ('PORT', pin)
        # Check component pins
        for comp in self.components:
            pins = (['left', 'right'] if comp.rotation == 0
                    else ['top', 'bottom'])
            for pin in pins:
                px, py = pin_pos(comp, pin)
                if ((px - x)**2 + (py - y)**2) ** 0.5 <= radius:
                    return (comp.id, pin)
        return None

    def _selected_component(self) -> Optional[Component]:
        for c in self.components:
            if c.selected:
                return c
        return None

    def _draw_ghost_wire_preview(self, painter: QPainter,
                                  comp: 'Component', gx: int, gy: int):
        """드래그 중 소자의 연결선 위치 변경을 점선으로 미리 표시."""
        pen = QPen(QColor('#AAAAAA'), 1, Qt.DashLine)
        painter.setPen(pen)
        pins = ['left', 'right'] if comp.rotation == 0 else ['top', 'bottom']
        # 고스트 위치의 핀 좌표 계산 (오프셋 적용)
        dx, dy = gx - comp.x, gy - comp.y
        for wire in self.wires:
            for side, pin_name in [(wire.start_comp_id, wire.start_pin),
                                   (wire.end_comp_id, wire.end_pin)]:
                if side != comp.id:
                    continue
                # 이 와이어에서 상대방 핀 좌표
                other_id = wire.end_comp_id if side == wire.start_comp_id else wire.start_comp_id
                other_pin = wire.end_pin if side == wire.start_comp_id else wire.start_pin
                p_other = get_pin_pos_universal(other_id, other_pin,
                                                self.components, self.height())
                if not p_other:
                    continue
                # 고스트 핀 좌표
                orig_pin_pos = pin_pos(comp, pin_name)
                ghost_pin_x = orig_pin_pos[0] + dx
                ghost_pin_y = orig_pin_pos[1] + dy
                pts = calc_wire_path(ghost_pin_x, ghost_pin_y,
                                     p_other[0], p_other[1])
                for i in range(len(pts) - 1):
                    painter.drawLine(pts[i][0], pts[i][1],
                                     pts[i+1][0], pts[i+1][1])

    # ------------------------------------------------------------------
    # Mouse Move / Release (고스트 프리뷰 & 드래그 확정)
    # ------------------------------------------------------------------

    def mouseMoveEvent(self, event):
        x, y = event.x(), event.y()
        # 새 소자 배치 모드: 고스트 위치 업데이트
        if self.placement_mode:
            self._ghost_pos = (snap(x), snap(y))
            self.update()
            return
        # 드래그 후보 → 임계값 초과 시 실제 드래그 시작
        if self._drag_start_comp is not None:
            mx0, my0 = self._drag_start_mouse
            if ((x - mx0)**2 + (y - my0)**2) ** 0.5 > self._DRAG_THRESHOLD:
                self._dragging_comp = self._drag_start_comp
                self._drag_start_comp = None
                self._drag_start_mouse = None
                self._click_timer.stop()
                self._pending_click_pos = None
        # 드래그 중: 고스트 위치 업데이트
        if self._dragging_comp is not None:
            self._ghost_pos = (snap(x), snap(y))
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        x, y = event.x(), event.y()
        if self._dragging_comp is not None:
            # 드래그 확정: 고스트 위치로 소자 이동
            gx, gy = self._ghost_pos or (self._dragging_comp.x, self._dragging_comp.y)
            self._dragging_comp.x = gx
            self._dragging_comp.y = gy
            # 연결된 와이어 모두 제거
            self.wires = [w for w in self.wires
                          if w.start_comp_id != self._dragging_comp.id
                          and w.end_comp_id != self._dragging_comp.id]
            self._dragging_comp = None
            self._ghost_pos = None
            self.update()
        elif self._drag_start_comp is not None:
            # 짧은 클릭 → 선택으로 처리 (더블클릭 구분용 타이머 사용)
            self._drag_start_comp = None
            self._drag_start_mouse = None
            self._pending_click_pos = (x, y)
            self._click_timer.start()


# ---------------------------------------------------------------------------
# Smith Chart Canvas
# ---------------------------------------------------------------------------

class SmithChartCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        super().__init__(fig)
        self.fig = fig
        self.ax = ax
        self.Z0 = 50.0
        self._result_xs = []
        self._result_ys = []
        self._result_Z = []
        self._result_freqs = []
        self._click_marker = None
        self._annot = None
        self.draw_background()
        self.mpl_connect('button_press_event', self._on_click)

    def draw_background(self):
        ax = self.ax
        ax.clear()
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect('equal')
        ax.axis('off')

        # Unit circle
        circle = plt.Circle((0, 0), 1, color='black', fill=False, linewidth=1.2)
        ax.add_patch(circle)
        # Real axis
        ax.axhline(0, color='black', linewidth=0.8)

        # Constant resistance circles: r / (1+r), 1/(1+r)
        for r in [0, 0.2, 0.5, 1, 2, 5]:
            cx = r / (r + 1)
            rad = 1 / (r + 1)
            c = plt.Circle((cx, 0), rad, color='#9E9E9E',
                           fill=False, linewidth=0.5, linestyle='--')
            ax.add_patch(c)
            ax.text(cx + rad + 0.02, 0.02, str(r), fontsize=6, color='#9E9E9E')

        # Constant reactance arcs
        for x_val in [0.2, 0.5, 1, 2, 5, -0.2, -0.5, -1, -2, -5]:
            cy = 1.0 / x_val
            rad = abs(cy)
            c = plt.Circle((1, cy), rad, color='#9E9E9E',
                           fill=False, linewidth=0.5, linestyle=':',
                           clip_on=True)
            ax.add_patch(c)

        # Labels
        ax.text(-1.0, 0.05, '0', fontsize=7, color='#616161')
        ax.text(0.97, 0.05, '∞', fontsize=7, color='#616161')
        ax.text(0.02, 0.05, '1', fontsize=7, color='#616161')
        self.draw()

    def plot_results(self, freqs_MHz: list, Z_list: list):
        self._result_freqs = freqs_MHz
        self._result_Z = Z_list
        Z0 = self.Z0
        gammas = [(Z - Z0) / (Z + Z0) for Z in Z_list]
        xs = [g.real for g in gammas]
        ys = [g.imag for g in gammas]
        self._result_xs = xs
        self._result_ys = ys

        self.draw_background()
        ax = self.ax

        ax.plot(xs, ys, color='#1565C0', linewidth=2, zorder=5)

        # Start marker
        ax.plot(xs[0], ys[0], marker='>', color='#1565C0', markersize=8, zorder=6)
        ax.annotate(f"{freqs_MHz[0]:.0f}MHz", (xs[0], ys[0]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)

        # End marker
        ax.plot(xs[-1], ys[-1], marker='s', color='#1565C0', markersize=8, zorder=6)
        ax.annotate(f"{freqs_MHz[-1]:.0f}MHz", (xs[-1], ys[-1]),
                    textcoords="offset points", xytext=(6, -12), fontsize=8)

        # 13.56 MHz marker
        if freqs_MHz[0] <= 13.56 <= freqs_MHz[-1]:
            idx = int(np.argmin(np.abs(np.array(freqs_MHz) - 13.56)))
            ax.plot(xs[idx], ys[idx], marker='s', color='#E53935',
                    markersize=8, zorder=6)
            ax.annotate("13.56MHz", (xs[idx], ys[idx]),
                        textcoords="offset points", xytext=(6, 6),
                        fontsize=8, color='#E53935')

        self.draw()

    def _on_click(self, event):
        if event.xdata is None or not self._result_xs:
            return
        # Convert to display coords
        ax = self.ax
        data_pts = np.array(list(zip(self._result_xs, self._result_ys)))
        click_display = ax.transData.transform([[event.xdata, event.ydata]])[0]
        pts_display = ax.transData.transform(data_pts)
        dists = np.hypot(pts_display[:, 0] - click_display[0],
                         pts_display[:, 1] - click_display[1])
        idx = int(np.argmin(dists))
        if dists[idx] > 20:
            return

        # Remove old markers
        if self._click_marker:
            for artist in self._click_marker:
                artist.remove()
        if self._annot:
            self._annot.remove()
            self._annot = None

        freq = self._result_freqs[idx]
        Z = self._result_Z[idx]
        x, y = self._result_xs[idx], self._result_ys[idx]
        m, = ax.plot(x, y, 'o', color='#FF6F00', markersize=10, zorder=7)
        self._click_marker = [m]
        sign = '+' if Z.imag >= 0 else '-'
        label = f"f = {freq:.2f}MHz\nZ = {Z.real:.2f} {sign} j{abs(Z.imag):.2f} Ω"
        self._annot = ax.annotate(label, (x, y),
                                  textcoords="offset points", xytext=(8, 8),
                                  fontsize=8, color='#FF6F00',
                                  bbox=dict(boxstyle='round,pad=0.3',
                                           fc='white', alpha=0.8))
        self.draw()


# ---------------------------------------------------------------------------
# MNA Calculation Worker
# ---------------------------------------------------------------------------

class CalcWorker(QThread):
    progress = pyqtSignal(int)
    result = pyqtSignal(list, list)
    error = pyqtSignal(str)

    def __init__(self, f_start_mhz, f_end_mhz, components, wires, canvas_height):
        super().__init__()
        self.f_start = f_start_mhz
        self.f_end = f_end_mhz
        self.components = components
        self.wires = wires
        self.canvas_height = canvas_height

    def run(self):
        try:
            from scipy.linalg import solve as scipy_solve
        except ImportError:
            scipy_solve = np.linalg.solve

        try:
            node_map, comp_nodes = self._build_graph()
            n_nodes = max(node_map.values()) + 1  # includes node 0 (GND)
            if n_nodes < 2:
                self.error.emit('no_component')
                return

            freqs = np.arange(self.f_start, self.f_end + 0.005, 0.01)
            results = []
            n = len(freqs)

            for i, f_mhz in enumerate(freqs):
                Z = self._solve_mna(f_mhz * 1e6, node_map, comp_nodes, scipy_solve)
                results.append(Z)
                if i % 50 == 0:
                    self.progress.emit(int(i / n * 100))

            self.progress.emit(100)
            self.result.emit(freqs.tolist(), results)

        except Exception as e:
            self.error.emit(str(e))

    def _build_graph(self):
        """Union-Find to assign node IDs."""
        # parent dict
        parent = {}

        def find(x):
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # Keys: ('PORT', 'plus'), ('PORT', 'gnd'), (comp_id, pin)
        gnd_key = ('PORT', 'gnd')
        plus_key = ('PORT', 'plus')

        for w in self.wires:
            a = (w.start_comp_id, w.start_pin)
            b = (w.end_comp_id, w.end_pin)
            union(a, b)

        # Gather all pin keys
        all_keys = set()
        all_keys.add(gnd_key)
        all_keys.add(plus_key)
        for c in self.components:
            pins = ['left', 'right'] if c.rotation == 0 else ['top', 'bottom']
            for p in pins:
                all_keys.add((c.id, p))
        for w in self.wires:
            all_keys.add((w.start_comp_id, w.start_pin))
            all_keys.add((w.end_comp_id, w.end_pin))

        # Assign node IDs: GND=0, PLUS=1, rest 2+
        root_gnd = find(gnd_key)
        root_plus = find(plus_key)

        node_map = {}
        counter = [2]

        def get_node(key):
            r = find(key)
            if r == root_gnd:
                return 0
            if r == root_plus:
                return 1
            if r not in node_map:
                node_map[r] = counter[0]
                counter[0] += 1
            return node_map[r]

        full_node_map = {k: get_node(k) for k in all_keys}

        # Build comp_nodes
        comp_nodes = {}
        for c in self.components:
            pins = ['left', 'right'] if c.rotation == 0 else ['top', 'bottom']
            na = full_node_map.get((c.id, pins[0]), 0)
            nb = full_node_map.get((c.id, pins[1]), 0)
            comp_nodes[c.id] = (na, nb)

        return full_node_map, comp_nodes

    def _solve_mna(self, freq_hz, node_map, comp_nodes, scipy_solve):
        import math
        omega = 2 * math.pi * freq_hz

        # Determine matrix size (max node index, GND=0 excluded from matrix)
        max_node = 0
        for c in self.components:
            na, nb = comp_nodes[c.id]
            max_node = max(max_node, na, nb)

        N = max_node  # nodes 1..N mapped to indices 0..N-1
        if N < 1:
            return complex('inf')

        G = np.zeros((N, N), dtype=complex)

        for c in self.components:
            na, nb = comp_nodes[c.id]
            if c.value is None:
                continue
            v = c.value
            if c.type == 'R':
                Y = 1.0 / v
            elif c.type == 'L':
                L_H = v * 1e-9
                if omega == 0:
                    return complex('inf')
                Y = 1.0 / (1j * omega * L_H)
            elif c.type == 'C':
                C_F = v * 1e-12
                Y = 1j * omega * C_F
            else:
                continue

            # MNA stamp: G[a][a]+=Y, G[b][b]+=Y, G[a][b]-=Y, G[b][a]-=Y
            # Skip row/col for GND (node 0)
            if na > 0:
                G[na-1, na-1] += Y
            if nb > 0:
                G[nb-1, nb-1] += Y
            if na > 0 and nb > 0:
                G[na-1, nb-1] -= Y
                G[nb-1, na-1] -= Y

        I = np.zeros(N, dtype=complex)
        I[0] = 1.0  # inject 1A into node 1 (PORT plus), index 0

        try:
            V = scipy_solve(G, I)
            return complex(V[0])  # Z = V[node1] / 1A
        except Exception:
            return complex('inf')


# ---------------------------------------------------------------------------
# Advanced Window (윈도우2 - Impedance Analysis)
# ---------------------------------------------------------------------------

class AdvancedWindow(QDialog):
    """주파수 vs Magnitude / Phase 그래프 팝업 (비모달)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Advanced - Impedance Analysis')
        self.resize(800, 600)
        self.setMinimumSize(600, 400)
        # 모달 아님: 메인 윈도우와 동시 조작 가능
        self.setWindowFlags(self.windowFlags() | Qt.Window)

        self._freqs: Optional[np.ndarray] = None   # MHz 단위 주파수 배열
        self._Z_arr: Optional[np.ndarray] = None   # 복소 임피던스 배열
        self._cursor_freq: Optional[float] = None   # 현재 커서 주파수

        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # ── 툴바 ──────────────────────────────────────────────────────────
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel('Scale:'))
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(['Linear', 'Logarithmic'])
        self.scale_combo.setFixedWidth(110)
        self.scale_combo.currentTextChanged.connect(self._on_scale_changed)
        toolbar.addWidget(self.scale_combo)

        toolbar.addSpacing(20)
        toolbar.addWidget(QLabel('Mag Unit:'))
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(['Ω', 'dB'])
        self.unit_combo.setFixedWidth(70)
        self.unit_combo.currentTextChanged.connect(self._on_mag_unit_changed)
        toolbar.addWidget(self.unit_combo)
        toolbar.addStretch()
        root.addLayout(toolbar)

        # ── 그래프3: Magnitude ────────────────────────────────────────────
        from matplotlib.figure import Figure
        self._fig3, self._ax3 = plt.subplots(figsize=(7, 3))
        self._canvas3 = FigureCanvasQTAgg(self._fig3)
        self._canvas3.mpl_connect('button_press_event', self._on_graph_click)
        root.addWidget(self._canvas3, stretch=1)

        # ── 그래프4: Phase ────────────────────────────────────────────────
        self._fig4, self._ax4 = plt.subplots(figsize=(7, 3))
        self._canvas4 = FigureCanvasQTAgg(self._fig4)
        self._canvas4.mpl_connect('button_press_event', self._on_graph_click)
        root.addWidget(self._canvas4, stretch=1)

        # ── 상태바 ────────────────────────────────────────────────────────
        self._status_label = QLabel('Cursor: —')
        self._status_label.setStyleSheet('font-size: 10pt; padding: 2px 4px;')
        root.addWidget(self._status_label)

        # 초기 빈 그래프 표시
        self._show_no_data()

    # ------------------------------------------------------------------
    # 데이터 업데이트 (Cal 버튼 클릭 시 메인윈도우에서 호출)
    # ------------------------------------------------------------------

    def update_plots(self, frequencies: list, impedances: list):
        """Calculate 결과를 그래프에 반영."""
        self._freqs = np.array(frequencies)       # MHz
        self._Z_arr = np.array(impedances)         # complex
        self._cursor_freq = None
        self._redraw_all()

    def _show_no_data(self):
        """데이터 없을 때 안내 메시지 표시."""
        for ax, canvas in [(self._ax3, self._canvas3),
                           (self._ax4, self._canvas4)]:
            ax.clear()
            ax.text(0.5, 0.5, 'No data — Press Calculate',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12, color='#9E9E9E')
            ax.set_xticks([])
            ax.set_yticks([])
            canvas.draw()

    # ------------------------------------------------------------------
    # 그래프 그리기
    # ------------------------------------------------------------------

    def _redraw_all(self):
        if self._freqs is None or self._Z_arr is None:
            self._show_no_data()
            return
        self._redraw_mag()
        self._redraw_phase()

    def _redraw_mag(self):
        """그래프3: Magnitude 재렌더링."""
        ax = self._ax3
        ax.clear()
        freqs = self._freqs
        unit = self.unit_combo.currentText()
        mag = np.abs(self._Z_arr)
        if unit == 'dB':
            y_data = 20 * np.log10(np.where(mag > 0, mag, 1e-30))
            ylabel = 'Magnitude (dB)'
        else:
            y_data = mag
            ylabel = 'Magnitude (Ω)'

        scale = self.scale_combo.currentText()
        ax.plot(freqs, y_data, color='#1E88E5', linewidth=1.5)
        ax.set_xscale('log' if scale == 'Logarithmic' else 'linear')
        ax.set_xlabel('Frequency (MHz)', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, color='#E0E0E0', linestyle=':')
        ax.set_facecolor('white')

        # 커서 표시
        if self._cursor_freq is not None:
            self._draw_cursor_on_ax(ax, self._cursor_freq, freqs, y_data,
                                    unit if unit == 'dB' else 'Ω', '#FF9800')
        self._fig3.tight_layout()
        self._canvas3.draw()

    def _redraw_phase(self):
        """그래프4: Phase 재렌더링."""
        ax = self._ax4
        ax.clear()
        freqs = self._freqs
        phase = np.angle(self._Z_arr, deg=True)

        scale = self.scale_combo.currentText()
        ax.plot(freqs, phase, color='#E53935', linewidth=1.5)
        ax.set_xscale('log' if scale == 'Logarithmic' else 'linear')
        ax.set_xlabel('Frequency (MHz)', fontsize=9)
        ax.set_ylabel('Phase (°)', fontsize=9)
        ax.set_ylim(-180, 180)
        ax.set_yticks(range(-180, 181, 45))
        ax.grid(True, color='#E0E0E0', linestyle=':')
        ax.set_facecolor('white')

        # 커서 표시
        if self._cursor_freq is not None:
            self._draw_cursor_on_ax(ax, self._cursor_freq, freqs, phase,
                                    '°', '#FF9800')
        self._fig4.tight_layout()
        self._canvas4.draw()

    def _draw_cursor_on_ax(self, ax, freq, freqs, y_data, unit_str, color):
        """수직 커서라인 및 교차점 마커, 값 라벨 그리기."""
        ax.axvline(freq, color=color, linewidth=1, zorder=10)
        # 가장 가까운 데이터 포인트 인덱스
        idx = int(np.argmin(np.abs(freqs - freq)))
        yval = y_data[idx]
        ax.plot(freqs[idx], yval, 'o', color=color, markersize=4, zorder=11)
        sign = '+' if yval >= 0 else ''
        ax.annotate(f'{sign}{yval:.2f}{unit_str}',
                    (freqs[idx], yval),
                    textcoords='offset points', xytext=(6, 4),
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))

    # ------------------------------------------------------------------
    # 이벤트 핸들러
    # ------------------------------------------------------------------

    def _on_graph_click(self, event):
        """그래프 클릭 → 커서라인 동기화."""
        if event.xdata is None or self._freqs is None:
            return
        self._cursor_freq = event.xdata
        self._update_cursor()

    def _update_cursor(self):
        """커서 위치에 따라 두 그래프와 상태바 업데이트."""
        if self._freqs is None or self._cursor_freq is None:
            return
        freq = self._cursor_freq
        idx = int(np.argmin(np.abs(self._freqs - freq)))
        Z = self._Z_arr[idx]
        mag = abs(Z)
        phase = np.angle(Z, deg=True)
        unit = self.unit_combo.currentText()
        if unit == 'dB':
            mag_str = f'{20 * np.log10(max(mag, 1e-30)):.2f}dB'
        else:
            mag_str = f'{mag:.2f}Ω'
        sign = '+' if phase >= 0 else ''
        self._status_label.setText(
            f'Cursor: f={self._freqs[idx]:.2f}MHz  '
            f'|Z|={mag_str}  ∠Z={sign}{phase:.1f}°'
        )
        # 두 그래프 재렌더링 (커서 포함)
        self._redraw_mag()
        self._redraw_phase()

    def _on_scale_changed(self, _):
        """X축 스케일 전환 → 두 그래프 동시 업데이트."""
        self._redraw_all()

    def _on_mag_unit_changed(self, _):
        """Magnitude 단위 전환 → 그래프3만 업데이트."""
        self._redraw_mag()
        # 상태바도 단위 반영
        if self._cursor_freq is not None:
            self._update_cursor()


# ---------------------------------------------------------------------------
# Smith Chart Widget (캔버스 + Advanced 버튼 컨테이너)
# ---------------------------------------------------------------------------

class SmithChartWidget(QWidget):
    """SmithChartCanvas를 감싸고 우측 상단에 Advanced 버튼을 배치하는 컨테이너."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.canvas = SmithChartCanvas()
        layout.addWidget(self.canvas)

        # Advanced 버튼 (오버레이, 부모 위젯 기준 절대 위치)
        self.adv_btn = QPushButton('Advanced', self)
        self.adv_btn.setFixedSize(80, 28)
        self.adv_btn.setStyleSheet(
            'QPushButton { background-color: #4A90D9; color: white; '
            'font-size: 10pt; border-radius: 4px; }'
            'QPushButton:hover { background-color: #357ABD; }'
        )
        self._position_adv_btn()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_adv_btn()

    def _position_adv_btn(self):
        """Advanced 버튼을 우측 상단(margin 5px)에 위치."""
        self.adv_btn.move(self.width() - 80 - 5, 5)


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('RF Circuit Design - Smith Chart Analyzer')
        # 기본 크기 1800×900, 최소 1200×600, 자유 리사이즈
        self.resize(1800, 900)
        self.setMinimumSize(1200, 600)
        self._worker: Optional[CalcWorker] = None
        self._last_freqs: list = []
        self._last_Z: list = []
        self._advanced_win: Optional[AdvancedWindow] = None
        self._splitter_ratio: float = 0.6   # 기본 60:40 비율
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(4, 4, 4, 4)
        root_layout.setSpacing(4)

        # Top toolbar
        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)

        for label in ('R', 'L', 'C'):
            btn = QPushButton(label)
            btn.setFixedSize(40, 30)
            btn.clicked.connect(lambda _, t=label: self._set_placement(t))
            toolbar.addWidget(btn)

        toolbar.addSpacing(20)
        toolbar.addWidget(QLabel('시작(MHz):'))
        self.freq_start = QLineEdit('1')
        self.freq_start.setFixedWidth(70)
        toolbar.addWidget(self.freq_start)

        toolbar.addWidget(QLabel('끝(MHz):'))
        self.freq_end = QLineEdit('100')
        self.freq_end.setFixedWidth(70)
        toolbar.addWidget(self.freq_end)

        self.cal_btn = QPushButton('Cal')
        self.cal_btn.setFixedSize(50, 30)
        self.cal_btn.clicked.connect(self.on_cal_clicked)
        toolbar.addWidget(self.cal_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(150)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)
        toolbar.addWidget(self.progress_bar)

        toolbar.addStretch()
        root_layout.addLayout(toolbar)

        # Splitter: 좌=회로 편집(60%), 우=스미스차트(40%)
        self.splitter = QSplitter(Qt.Horizontal)

        self.circuit_canvas = CircuitCanvas()
        self.circuit_canvas.setMinimumWidth(480)

        # SmithChartWidget: 캔버스 + Advanced 버튼 컨테이너
        self.smith_widget = SmithChartWidget()
        self.smith_widget.setMinimumWidth(320)
        self.smith_canvas = self.smith_widget.canvas  # 기존 참조 유지

        self.splitter.addWidget(self.circuit_canvas)
        self.splitter.addWidget(self.smith_widget)
        # 초기 60:40 비율 설정
        total = 1800
        self.splitter.setSizes([int(total * 0.6), int(total * 0.4)])
        # 리사이즈 시 비율 유지
        self.splitter.splitterMoved.connect(self._on_splitter_moved)

        # Advanced 버튼 → AdvancedWindow 열기
        self.smith_widget.adv_btn.clicked.connect(self._open_advanced)

        root_layout.addWidget(self.splitter, stretch=1)

    def resizeEvent(self, event):
        """윈도우 리사이즈 시 현재 스플리터 비율 유지."""
        super().resizeEvent(event)
        total = self.splitter.width()
        if total > 0:
            self.splitter.setSizes([
                int(total * self._splitter_ratio),
                total - int(total * self._splitter_ratio)
            ])

    def _on_splitter_moved(self, pos: int, index: int):
        """사용자가 스플리터 핸들을 드래그한 경우 새 비율을 저장."""
        sizes = self.splitter.sizes()
        total = sum(sizes)
        if total > 0:
            self._splitter_ratio = sizes[0] / total

    def _open_advanced(self):
        """Advanced 버튼 클릭: AdvancedWindow 열기 (중복 방지)."""
        if self._advanced_win is None or not self._advanced_win.isVisible():
            self._advanced_win = AdvancedWindow(self)
            # 마지막 계산 결과가 있으면 즉시 표시
            if self._last_freqs:
                self._advanced_win.update_plots(self._last_freqs, self._last_Z)
            self._advanced_win.show()
        else:
            self._advanced_win.activateWindow()
            self._advanced_win.raise_()

    def _set_placement(self, comp_type: str):
        self.circuit_canvas.placement_mode = comp_type
        self.circuit_canvas.setFocus()

    def on_cal_clicked(self):
        err = self.validate_circuit()
        if err:
            # Highlight errors then show message
            self.circuit_canvas.update()
            QMessageBox.warning(self, '오류', ERRORS[err])
            return

        try:
            f_start = float(self.freq_start.text())
            f_end = float(self.freq_end.text())
        except ValueError:
            QMessageBox.warning(self, '오류', ERRORS['freq_empty'])
            return

        self.cal_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self._worker = CalcWorker(
            f_start, f_end,
            self.circuit_canvas.components,
            self.circuit_canvas.wires,
            self.circuit_canvas.height()
        )
        self._worker.progress.connect(self.progress_bar.setValue)
        self._worker.result.connect(self.on_calc_result)
        self._worker.error.connect(self._on_calc_error)
        self._worker.start()

    def on_calc_result(self, freqs: list, Z_list: list):
        self._last_freqs = freqs
        self._last_Z = Z_list
        self.progress_bar.setVisible(False)
        self.cal_btn.setEnabled(True)
        # 스미스 차트 업데이트
        self.smith_canvas.plot_results(freqs, Z_list)
        # AdvancedWindow가 열려있으면 함께 업데이트
        if self._advanced_win and self._advanced_win.isVisible():
            self._advanced_win.update_plots(freqs, Z_list)

    def _on_calc_error(self, msg: str):
        self.progress_bar.setVisible(False)
        self.cal_btn.setEnabled(True)
        QMessageBox.critical(self, '계산 오류', msg)

    def validate_circuit(self) -> Optional[str]:
        # Freq validation
        try:
            f_start = float(self.freq_start.text())
            f_end = float(self.freq_end.text())
        except ValueError:
            return 'freq_empty'
        if f_start >= f_end:
            return 'freq_order'

        comps = self.circuit_canvas.components
        wires = self.circuit_canvas.wires
        h = self.circuit_canvas.height()

        if not comps:
            return 'no_component'

        # Check no_value
        no_val = [c for c in comps if c.value is None]
        if no_val:
            for c in no_val:
                c.error_highlight = True
            return 'no_value'

        # Build adjacency for BFS
        adj: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}

        def add_edge(a, b):
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)

        # Each component connects its two pins
        for c in comps:
            pins = ['left', 'right'] if c.rotation == 0 else ['top', 'bottom']
            add_edge((c.id, pins[0]), (c.id, pins[1]))

        for w in wires:
            a = (w.start_comp_id, w.start_pin)
            b = (w.end_comp_id, w.end_pin)
            add_edge(a, b)

        # BFS from PORT plus
        start = ('PORT', 'plus')
        visited = set()
        q = deque([start])
        visited.add(start)
        while q:
            cur = q.popleft()
            for nb in adj.get(cur, []):
                if nb not in visited:
                    visited.add(nb)
                    q.append(nb)

        # Check PORT connected
        port_connected = any(c.id in [n[0] for n in visited] for c in comps)
        if not port_connected:
            return 'no_component'

        # Check GND reachable
        if ('PORT', 'gnd') not in visited:
            # Mark all components not in path
            for c in comps:
                c.error_highlight = True
            return 'isolated'

        # Check isolated components
        isolated = []
        for c in comps:
            pins = ['left', 'right'] if c.rotation == 0 else ['top', 'bottom']
            if (c.id, pins[0]) not in visited and (c.id, pins[1]) not in visited:
                isolated.append(c)

        if isolated:
            for c in isolated:
                c.error_highlight = True
            return 'isolated'

        return None


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
