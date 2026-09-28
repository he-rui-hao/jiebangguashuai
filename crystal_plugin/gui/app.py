"""晶体可视化插件 GUI（PySide6 + matplotlib Qt 后端）。

布局：左栏公共输入（材料/晶格常数/超胞/方向），右侧功能页：
  单晶2D（含单层切片 + 测量工具）、多方向对比、3D视图（含晶面标注）、
  SAED衍射、XRD、极射投影、双晶OR、孪晶。

约定（接口文档 §3.8）：GUI 只做输入校验与四指数→三指数转换，
所有晶体学数学都在 core/ 内核完成。
"""
from __future__ import annotations

import json
import os

import numpy as np

import matplotlib
matplotlib.use("QtAgg")  # 必须先于 core 模块导入，保住交互后端

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog,
    QFormLayout, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPushButton, QScrollArea, QSpinBox, QSplitter, QTabWidget, QVBoxLayout,
    QWidget,
)

from core.diffraction import (draw_saed, draw_xrd, simulate_diffraction,
                              simulate_powder_xrd)
from core.measure import angle, distance, nearest_index
from core.orientation import (four_to_three_dir, four_to_three_plane,
                              rotation_from_uvw)
from core.projector import (DEFAULT_COLORS, DEFAULT_RADII, _draw_proj,
                            draw_multi, project)
from core.stereographic import draw_stereographic, stereographic_poles
from crystio.database import list_materials, load_material
from bicrystal.or_builder import OR_PRESETS, build_bicrystal, rotation_from_or_name
from twin.twin_builder import build_twin

GRAIN_COLORS = ["#4c72b0", "#dd8452"]   # 基体/相1、孪晶/相2

# ----------------------------------------------------------------------
# 界面设置（"设置"菜单可调，持久化到 gui/settings.json）
# ----------------------------------------------------------------------
_SETTINGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "settings.json")
DEFAULT_SETTINGS = {
    "ui_font": 12,        # Qt 界面字号 (pt)
    "mpl_font": 13,       # 图中字号 (pt)
    "atom_scale": 1.0,    # 原子大小倍数
    "rot_speed": 0.15,    # 3D 旋转速度（度/像素）
    "zoom_step": 1.2,     # 滚轮缩放步长（>1）
    "dim_alpha": 0.15,    # 旋转时背景坐标系透明度
}


def load_settings() -> dict:
    try:
        with open(_SETTINGS_PATH, encoding="utf-8") as f:
            return {**DEFAULT_SETTINGS, **json.load(f)}
    except (OSError, json.JSONDecodeError):
        return dict(DEFAULT_SETTINGS)


def save_settings(settings: dict) -> None:
    try:
        with open(_SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
    except OSError:
        pass


def apply_mpl_font(pt: float) -> None:
    """matplotlib 全局字体：中文/希腊字母走微软雅黑，标题加粗，字号可调。

    这也修复了双晶/孪晶图注"相1/基体"等中文显示为方块的问题
    （默认 DejaVu Sans 无 CJK 字形）。
    """
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": pt,
        "axes.titlesize": pt + 3,
        "axes.titleweight": "bold",
        "axes.labelsize": pt,
        "axes.labelweight": "bold",
        "xtick.labelsize": pt - 1,
        "ytick.labelsize": pt - 1,
        "legend.fontsize": pt - 1,
    })


def qt_stylesheet(pt: float) -> str:
    return (f"QWidget {{ font-family: 'Microsoft YaHei'; font-size: {pt}pt; }} "
            "QPushButton { font-weight: bold; padding: 4px 10px; } "
            "QTabBar::tab { font-weight: bold; padding: 6px 14px; } "
            "QMenuBar { font-weight: bold; }")


# 希腊字母英文拼写 → 符号（材料数据键保持 ASCII 不变，仅显示层转换）
GREEK_WORDS = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε",
    "zeta": "ζ", "eta": "η", "theta": "θ", "kappa": "κ", "lambda": "λ",
    "mu": "μ", "nu": "ν", "rho": "ρ", "sigma": "σ", "tau": "τ", "omega": "ω",
}


def disp_name(name: str) -> str:
    """显示用材料名：gamma-Fe → γ-Fe、alpha-Fe → α-Fe。"""
    for word, sym in GREEK_WORDS.items():
        name = name.replace(word + "-", sym + "-")
    return name


def parse_indices(text: str, n: int = 3) -> tuple[int, ...]:
    """把 "1 0 0" 或 "1,0,0" 解析为 n 个整数。"""
    parts = text.replace(",", " ").split()
    if len(parts) != n:
        raise ValueError(f"需要 {n} 个整数指数，实际输入 {len(parts)} 个")
    return tuple(int(p) for p in parts)


class MplCanvas(FigureCanvasQTAgg):
    """单 Figure 画布，所有功能页复用。"""

    def __init__(self) -> None:
        self.fig = Figure(figsize=(6, 6))
        super().__init__(self.fig)

    def save_with_dialog(self, parent: QWidget) -> None:
        path, _ = QFileDialog.getSaveFileName(
            parent, "导出图片", "figure.png",
            "PNG 位图 (*.png);;SVG 矢量图 (*.svg);;PDF 矢量图 (*.pdf)")
        if path:
            self.fig.savefig(path, dpi=200, facecolor=self.fig.get_facecolor(),
                             bbox_inches="tight")


class Canvas3D(MplCanvas):
    """3D 画布：左键拖动旋转（速度可调）、滚轮缩放；
    旋转过程中背景坐标系（面板/网格/刻度数字）半透明化，松开后恢复。"""

    def __init__(self, settings: dict) -> None:
        super().__init__()
        self.settings = settings
        self.ax3d = None
        self._drag: tuple | None = None      # (x0, y0, elev0, azim0)
        self._saved_alpha: list | None = None
        self.mpl_connect("button_press_event", self._on_press)
        self.mpl_connect("motion_notify_event", self._on_move)
        self.mpl_connect("button_release_event", self._on_release)
        self.mpl_connect("scroll_event", self._on_scroll)

    def new_3d_axes(self):
        """清空画布并新建已接管交互的 3D axes。"""
        self.fig.clear()
        ax = self.fig.add_subplot(111, projection="3d")
        ax.disable_mouse_rotation()      # 关掉 mpl 原生交互（左键旋转/右键缩放）
        self.ax3d = ax
        self._drag = None
        self._saved_alpha = [
            (axis.pane.get_alpha(), axis._axinfo.get("grid", {}).get("alpha"))
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis)
        ]
        return ax

    # ---- 背景透明化 ----
    def _set_dim(self, on: bool) -> None:
        ax = self.ax3d
        if ax is None or self._saved_alpha is None:
            return
        dim = float(self.settings.get("dim_alpha", 0.15))
        for i, axis in enumerate((ax.xaxis, ax.yaxis, ax.zaxis)):
            pane_a, grid_a = (dim, dim) if on else self._saved_alpha[i]
            axis.pane.set_alpha(pane_a)
            try:
                if grid_a is None:
                    axis._axinfo["grid"].pop("alpha", None)
                else:
                    axis._axinfo["grid"]["alpha"] = grid_a
            except (KeyError, AttributeError):
                pass
            for t in axis.get_ticklabels():
                t.set_alpha(dim if on else 1.0)
            axis.label.set_alpha(dim if on else 1.0)
        self.draw_idle()

    # ---- 交互事件 ----
    def _on_press(self, event) -> None:
        if event.inaxes is self.ax3d and event.button == 1:
            self._drag = (event.x, event.y, self.ax3d.elev, self.ax3d.azim)
            self._set_dim(True)

    def _on_move(self, event) -> None:
        if self._drag is None or self.ax3d is None:
            return
        x0, y0, elev0, azim0 = self._drag
        speed = float(self.settings.get("rot_speed", 0.15))
        self.ax3d.view_init(elev=elev0 - (event.y - y0) * speed,
                            azim=azim0 - (event.x - x0) * speed)
        self.draw_idle()

    def _on_release(self, _event) -> None:
        if self._drag is not None:
            self._drag = None
            self._set_dim(False)

    def zoom(self, factor: float) -> None:
        """滚轮缩放：围绕数据中心等比缩放数据范围（factor>1 放大）。

        用 get_w_lims/set_lim3d 公开 API 实现（与 mpl 原生 zoom 同思路），
        不依赖 set_focal_length（本机 mpl 3.10.7 无此公开方法）。
        """
        ax = self.ax3d
        if ax is None:
            return
        s = 1.0 / factor                      # 放大 = 数据范围缩小
        x0, x1, y0, y1, z0, z1 = ax.get_w_lims()
        cx, cy, cz = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2
        ax.set_xlim3d(cx + (x0 - cx) * s, cx + (x1 - cx) * s)
        ax.set_ylim3d(cy + (y0 - cy) * s, cy + (y1 - cy) * s)
        ax.set_zlim3d(cz + (z0 - cz) * s, cz + (z1 - cz) * s)
        self.draw_idle()

    def _on_scroll(self, event) -> None:
        if event.inaxes is self.ax3d:
            step = float(self.settings.get("zoom_step", 1.2))
            self.zoom(step if event.button == "up" else 1.0 / step)


def _draw_atoms_3d(ax, coords: np.ndarray, labels: list[str],
                   colors: dict[str, str] | None = None,
                   size_scale: float = 1.0) -> None:
    """matplotlib 3D 原子球（交交由 Canvas3D 接管）。"""
    colors = colors or {}
    coords = np.asarray(coords)
    for lab in dict.fromkeys(labels):
        idx = [i for i, l in enumerate(labels) if l == lab]
        pts = coords[idx]
        elem = lab.split()[0]
        c = colors.get(elem, DEFAULT_COLORS.get(elem, "#888888"))
        s = (DEFAULT_RADII.get(elem, 1.30) * 14) ** 1.6 * size_scale
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=s, c=c,
                   edgecolors="#333333", linewidths=0.3, label=elem, depthshade=True)
    span = np.ptp(coords, axis=0).max() / 2 + 2
    mid = coords.mean(axis=0)
    ax.set_xlim(mid[0] - span, mid[0] + span)
    ax.set_ylim(mid[1] - span, mid[1] + span)
    ax.set_zlim(mid[2] - span, mid[2] + span)
    ax.set_box_aspect((1, 1, 1))
    ax.legend(loc="upper right")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("晶体可视化插件 — 任意方向原子排布特征生成")
        self.resize(1280, 780)

        self.settings = load_settings()
        self.apply_settings(self.settings, redraw=False)

        self.crystal = None            # 当前 Crystal
        self._proj_cache: dict = {}    # 单晶2D页投影缓存（供测量）
        self._last_gen = None          # 最近一次成功生成（设置变更后自动重绘）

        splitter = QSplitter()
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_tabs())
        splitter.setSizes([310, 950])
        self.setCentralWidget(splitter)
        self._build_menu()
        self.statusBar().showMessage("就绪")
        self._on_material_changed()

    # ------------------------------------------------------------------
    # 设置菜单
    # ------------------------------------------------------------------
    def _build_menu(self) -> None:
        m = self.menuBar().addMenu("设置(&S)")
        act = m.addAction("界面设置…")
        act.triggered.connect(self._open_settings)

    def _open_settings(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle("界面设置")
        form = QFormLayout(dlg)
        s = self.settings
        ui_font = QSpinBox(minimum=9, maximum=20, value=int(s["ui_font"]))
        ui_font.setSuffix(" pt")
        mpl_font = QSpinBox(minimum=8, maximum=24, value=int(s["mpl_font"]))
        mpl_font.setSuffix(" pt")
        atom = QDoubleSpinBox(minimum=0.3, maximum=2.5, singleStep=0.1,
                              value=float(s["atom_scale"]))
        rot = QDoubleSpinBox(minimum=0.05, maximum=0.60, singleStep=0.05,
                             decimals=2, value=float(s["rot_speed"]))
        rot.setSuffix(" °/px")
        zoom = QDoubleSpinBox(minimum=1.05, maximum=1.60, singleStep=0.05,
                              decimals=2, value=float(s["zoom_step"]))
        dim = QDoubleSpinBox(minimum=0.0, maximum=0.6, singleStep=0.05,
                             decimals=2, value=float(s["dim_alpha"]))
        form.addRow("界面字号", ui_font)
        form.addRow("图中字号", mpl_font)
        form.addRow("原子大小倍数", atom)
        form.addRow("3D 旋转速度", rot)
        form.addRow("滚轮缩放步长", zoom)
        form.addRow("旋转时背景透明度", dim)
        btns = QHBoxLayout()
        ok = QPushButton("应用")
        cancel = QPushButton("关闭")
        btns.addStretch(1)
        btns.addWidget(ok)
        btns.addWidget(cancel)
        form.addRow(btns)
        cancel.clicked.connect(dlg.reject)

        def _apply() -> None:
            self.apply_settings({
                "ui_font": ui_font.value(), "mpl_font": mpl_font.value(),
                "atom_scale": atom.value(), "rot_speed": rot.value(),
                "zoom_step": zoom.value(), "dim_alpha": dim.value(),
            })

        ok.clicked.connect(_apply)
        dlg.exec()

    def apply_settings(self, settings: dict, redraw: bool = True) -> None:
        """应用界面设置并持久化；redraw=True 时自动重绘最近一次生成的图。"""
        self.settings.update(settings)
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(qt_stylesheet(self.settings["ui_font"]))
        apply_mpl_font(self.settings["mpl_font"])
        save_settings(self.settings)
        if redraw and self._last_gen is not None:
            self._run(self._last_gen)
        self.statusBar().showMessage("设置已应用")

    # ------------------------------------------------------------------
    # 左栏：公共输入
    # ------------------------------------------------------------------
    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        form = QFormLayout(panel)

        self.material_combo = QComboBox()
        for key in list_materials():
            self.material_combo.addItem(disp_name(key), userData=key)
        self.material_combo.currentTextChanged.connect(self._on_material_changed)
        form.addRow("材料", self.material_combo)

        self.a_spin = QDoubleSpinBox(minimum=0.1, maximum=20.0, decimals=4)
        self.a_spin.valueChanged.connect(self._reload_crystal)
        form.addRow("a (Å)", self.a_spin)

        self.c_spin = QDoubleSpinBox(minimum=0.1, maximum=20.0, decimals=4)
        self.c_spin.valueChanged.connect(self._reload_crystal)
        form.addRow("c (Å, HCP)", self.c_spin)

        sc_row = QHBoxLayout()
        self.sc_spins = []
        for _ in range(3):
            sp = QSpinBox(minimum=1, maximum=8, value=3)
            self.sc_spins.append(sp)
            sc_row.addWidget(sp)
        form.addRow("超胞 n1 n2 n3", sc_row)

        # 观察方向（三/四指数）
        self.hcp4_dir = QCheckBox("HCP 四指数 [uvtw]")
        dir_row = QHBoxLayout()
        self.dir_spins = []
        for i, default in enumerate((1, 1, 0, 0)):
            sp = QSpinBox(minimum=-9, maximum=9, value=default)
            sp.setVisible(i < 3)
            self.dir_spins.append(sp)
            dir_row.addWidget(sp)
        self.hcp4_dir.toggled.connect(
            lambda on: [s.setVisible(True) for s in self.dir_spins])
        self.hcp4_dir.toggled.connect(
            lambda on: self.dir_spins[2].setValue(-(self.dir_spins[0].value()
                                                    + self.dir_spins[1].value())))
        form.addRow(self.hcp4_dir, dir_row)

        self.inplane_auto = QCheckBox("面内晶向自动")
        self.inplane_auto.setChecked(True)
        ip_row = QHBoxLayout()
        self.ip_spins = []
        for _ in range(3):
            sp = QSpinBox(minimum=-9, maximum=9)
            sp.setEnabled(False)
            self.ip_spins.append(sp)
            ip_row.addWidget(sp)
        self.inplane_auto.toggled.connect(
            lambda auto: [s.setEnabled(not auto) for s in self.ip_spins])
        form.addRow(self.inplane_auto, ip_row)

        note = QLabel("指数用负号表示，如 -1。\n改材料/晶格常数后点各页【生成】。")
        note.setWordWrap(True)
        form.addRow(note)

        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        return scroll

    def _on_material_changed(self) -> None:
        key = self.material_combo.currentData()
        params = load_material(key).meta["lattice_params"]
        self.a_spin.blockSignals(True)
        self.c_spin.blockSignals(True)
        self.a_spin.setValue(params["a"])
        has_c = params.get("c") is not None
        self.c_spin.setEnabled(has_c)
        if has_c:
            self.c_spin.setValue(params["c"])
        self.a_spin.blockSignals(False)
        self.c_spin.blockSignals(False)
        self._reload_crystal()

    def _reload_crystal(self) -> None:
        xtal = load_material(self.material_combo.currentData())
        p = xtal.meta["lattice_params"]
        a0, c0 = p["a"], p.get("c")
        ratio = self.a_spin.value() / a0
        L = xtal.lattice_vectors
        L[:2] *= ratio                       # 底面两行（立方 a1a2 / HCP a1a2 都含 a）
        if c0 is not None:
            L[2] *= self.c_spin.value() / c0
            p["c"] = self.c_spin.value()
        else:
            L[2] *= ratio
        p["a"] = self.a_spin.value()
        self.crystal = xtal
        self.statusBar().showMessage(f"已加载 {disp_name(self.crystal.name)}")

    def _supercell(self) -> tuple[int, int, int]:
        return tuple(sp.value() for sp in self.sc_spins)

    def _uvw(self) -> tuple[int, int, int]:
        vals = [sp.value() for sp in self.dir_spins]
        if self.hcp4_dir.isChecked():
            return four_to_three_dir(tuple(vals))
        return tuple(vals[:3])

    def _in_plane(self) -> tuple[int, int, int] | None:
        if self.inplane_auto.isChecked():
            return None
        return tuple(sp.value() for sp in self.ip_spins)

    # ------------------------------------------------------------------
    # 右侧功能页
    # ------------------------------------------------------------------
    def _build_tabs(self) -> QTabWidget:
        self.tabs = QTabWidget()
        self.tabs.addTab(self._tab_single2d(), "单晶2D")
        self.tabs.addTab(self._tab_multi(), "多方向对比")
        self.tabs.addTab(self._tab_3d(), "3D视图")
        self.tabs.addTab(self._tab_saed(), "SAED衍射")
        self.tabs.addTab(self._tab_xrd(), "XRD")
        self.tabs.addTab(self._tab_stereo(), "极射投影")
        self.tabs.addTab(self._tab_bicrystal(), "双晶OR")
        self.tabs.addTab(self._tab_twin(), "孪晶")
        return self.tabs

    def _page(self) -> tuple[QWidget, QVBoxLayout, MplCanvas]:
        page = QWidget()
        lay = QVBoxLayout(page)
        canvas = MplCanvas()
        lay.addWidget(canvas, stretch=1)
        return page, lay, canvas

    def _page3d(self) -> tuple[QWidget, QVBoxLayout, Canvas3D]:
        """3D 页：滚轮缩放、减速旋转、旋转时背景半透明。"""
        page = QWidget()
        lay = QVBoxLayout(page)
        canvas = Canvas3D(self.settings)
        lay.addWidget(canvas, stretch=1)
        return page, lay, canvas

    def _run(self, fn) -> None:
        """统一异常处理：内核 ValueError → 弹窗（接口文档 §5 协作规则 3）。"""
        try:
            fn()
        except (ValueError, KeyError) as e:
            QMessageBox.warning(self, "输入错误", str(e))

    # ---------- 页 1：单晶 2D ----------
    def _tab_single2d(self) -> QWidget:
        page, lay, self.cv2d = self._page()
        row = QHBoxLayout()
        btn = QPushButton("生成投影")
        btn.clicked.connect(lambda: self._run(self._gen_single2d))
        row.addWidget(btn)

        self.slice_chk = QCheckBox("单层切片")
        self.zmin = QDoubleSpinBox(minimum=-100, maximum=100, decimals=2, value=-0.5)
        self.zmax = QDoubleSpinBox(minimum=-100, maximum=100, decimals=2, value=0.5)
        self.zmin.setPrefix("z≥")
        self.zmax.setPrefix(" z≤")
        row.addWidget(self.slice_chk)
        row.addWidget(self.zmin)
        row.addWidget(self.zmax)

        self.measure_btn = QPushButton("测量模式")
        self.measure_btn.setCheckable(True)
        self.measure_btn.toggled.connect(self._toggle_measure)
        row.addWidget(self.measure_btn)

        save = QPushButton("导出图片")
        save.clicked.connect(lambda: self.cv2d.save_with_dialog(self))
        row.addWidget(save)
        row.addStretch(1)
        lay.addLayout(row)

        self._measure_clicks: list[int] = []
        self.cv2d.mpl_connect("button_press_event", self._on_measure_click)
        return page

    def _gen_single2d(self) -> None:
        R = rotation_from_uvw(self._uvw(), self.crystal.lattice_vectors,
                              self._in_plane())
        x, y, z, sp = project(self.crystal, R, self._supercell())
        zr = (self.zmin.value(), self.zmax.value()) if self.slice_chk.isChecked() else None
        self.cv2d.fig.clear()
        ax = self.cv2d.fig.add_subplot(111)
        _draw_proj(ax, x, y, z, sp, f"{disp_name(self.crystal.name)}  投影",
                   atom_scale=0.35 * self.settings["atom_scale"], z_range=zr)
        self.cv2d.draw_idle()
        self._proj_cache = {"x": x, "y": y, "z": z, "species": sp}
        self._measure_clicks.clear()
        self._last_gen = self._gen_single2d

    def _toggle_measure(self, on: bool) -> None:
        self._measure_clicks.clear()
        self.statusBar().showMessage(
            "测量模式：点击 2 个原子得距离，点第 3 个得键角" if on else "就绪")

    def _on_measure_click(self, event) -> None:
        if not self.measure_btn.isChecked() or event.inaxes is None:
            return
        if not self._proj_cache:
            return
        x, y, z = self._proj_cache["x"], self._proj_cache["y"], self._proj_cache["z"]
        i = nearest_index(x, y, event.xdata, event.ydata)
        self._measure_clicks.append(i)
        ax = event.inaxes
        ax.plot(x[i], y[i], "rx", markersize=12, markeredgewidth=2, zorder=99)
        pts = [np.array([x[j], y[j], z[j]]) for j in self._measure_clicks]
        if len(pts) == 2:
            d = distance(pts[0], pts[1])
            self.statusBar().showMessage(f"距离 = {d:.4f} Å")
            ax.annotate(f"{d:.3f} Å", ((x[self._measure_clicks[0]] + x[i]) / 2,
                                       (y[self._measure_clicks[0]] + y[i]) / 2),
                        color="red", fontsize=11, zorder=99)
        elif len(pts) == 3:
            ang = angle(pts[0], pts[1], pts[2])
            self.statusBar().showMessage(f"键角 = {ang:.3f}°（顶点在第 2 个点）")
            ax.annotate(f"{ang:.2f}°", (x[self._measure_clicks[1]],
                                        y[self._measure_clicks[1]]),
                        color="red", fontsize=11, zorder=99)
            self._measure_clicks.clear()
        self.cv2d.draw_idle()

    # ---------- 页 2：多方向对比 ----------
    def _tab_multi(self) -> QWidget:
        page, lay, self.cv_multi = self._page()
        row = QHBoxLayout()
        row.addWidget(QLabel("4 个方向（空格分隔）:"))
        self.view_edits = []
        for default in ("1 0 0", "1 1 0", "1 1 1", "1 1 2"):
            e = QLineEdit(default)
            e.setMaximumWidth(70)
            self.view_edits.append(e)
            row.addWidget(e)
        btn = QPushButton("生成对比图")
        btn.clicked.connect(lambda: self._run(self._gen_multi))
        row.addWidget(btn)
        save = QPushButton("导出图片")
        save.clicked.connect(lambda: self.cv_multi.save_with_dialog(self))
        row.addWidget(save)
        row.addStretch(1)
        lay.addLayout(row)
        return page

    def _gen_multi(self) -> None:
        views = [(parse_indices(e.text()), None) for e in self.view_edits]
        self.cv_multi.fig.clear()
        draw_multi(self.cv_multi.fig, self.crystal, views, self._supercell(),
                   atom_scale=0.30 * self.settings["atom_scale"])
        self.cv_multi.draw_idle()
        self._last_gen = self._gen_multi

    # ---------- 页 3：3D 视图 ----------
    def _tab_3d(self) -> QWidget:
        page, lay, self.cv3d = self._page3d()
        row = QHBoxLayout()
        self.plane_chk = QCheckBox("标注晶面 (hkl):")
        row.addWidget(self.plane_chk)
        self.plane_spins = []
        for i, default in enumerate((1, 1, 1)):
            sp = QSpinBox(minimum=-9, maximum=9, value=default)
            self.plane_spins.append(sp)
            row.addWidget(sp)
        btn = QPushButton("生成 3D")
        btn.clicked.connect(lambda: self._run(self._gen_3d))
        row.addWidget(btn)
        save = QPushButton("导出图片")
        save.clicked.connect(lambda: self.cv3d.save_with_dialog(self))
        row.addWidget(save)
        row.addWidget(QLabel("左键拖动旋转 · 滚轮缩放"))
        row.addStretch(1)
        lay.addLayout(row)
        return page

    def _gen_3d(self) -> None:
        frac, sp = self.crystal.supercell(*self._supercell())
        coords = frac @ self.crystal.lattice_vectors
        coords -= coords.mean(axis=0)
        ax = self.cv3d.new_3d_axes()
        _draw_atoms_3d(ax, coords, sp, size_scale=self.settings["atom_scale"])
        if self.plane_chk.isChecked():
            hkl = tuple(s.value() for s in self.plane_spins)
            g = np.linalg.inv(self.crystal.lattice_vectors) @ np.array(hkl, dtype=float)
            n = g / np.linalg.norm(g)
            ref = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
            e1 = np.cross(n, ref)
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(n, e1)
            half = np.linalg.norm(self.crystal.lattice_vectors, axis=1).mean() * 1.2
            corners = np.array([half * (e1 + e2), half * (e1 - e2),
                                -half * (e1 + e2), -half * (e1 - e2)])
            ax.plot_trisurf(corners[:, 0], corners[:, 1], corners[:, 2],
                            triangles=[[0, 1, 2], [0, 2, 3]],
                            color="#ff8800", alpha=0.25)
            ax.quiver(0, 0, 0, *n * half, color="#ff8800", arrow_length_ratio=0.15)
        ax.set_title(f"{disp_name(self.crystal.name)}  3D")
        self.cv3d.draw_idle()
        self._last_gen = self._gen_3d

    # ---------- 页 4：SAED ----------
    def _tab_saed(self) -> QWidget:
        page, lay, self.cv_saed = self._page()
        row = QHBoxLayout()
        row.addWidget(QLabel("max_index"))
        self.maxidx = QSpinBox(minimum=1, maximum=6, value=3)
        row.addWidget(self.maxidx)
        btn = QPushButton("生成衍射花样")
        btn.clicked.connect(lambda: self._run(self._gen_saed))
        row.addWidget(btn)
        save = QPushButton("导出图片")
        save.clicked.connect(lambda: self.cv_saed.save_with_dialog(self))
        row.addWidget(save)
        row.addStretch(1)
        lay.addLayout(row)
        return page

    def _gen_saed(self) -> None:
        pat = simulate_diffraction(self.crystal, self._uvw(), self._in_plane(),
                                   self.maxidx.value())
        self.cv_saed.fig.clear()
        self.cv_saed.fig.patch.set_facecolor("#111111")
        ax = self.cv_saed.fig.add_subplot(111)
        draw_saed(ax, pat, f"{disp_name(self.crystal.name)}  SAED")
        self.cv_saed.draw_idle()
        self._last_gen = self._gen_saed

    # ---------- 页 5：XRD ----------
    def _tab_xrd(self) -> QWidget:
        page, lay, self.cv_xrd = self._page()
        row = QHBoxLayout()
        row.addWidget(QLabel("波长 λ (Å)"))
        self.wl = QDoubleSpinBox(minimum=0.1, maximum=5.0, decimals=4, value=1.5406)
        row.addWidget(self.wl)
        row.addWidget(QLabel("2θ 范围"))
        self.tth_min = QDoubleSpinBox(minimum=1, maximum=170, value=10)
        self.tth_max = QDoubleSpinBox(minimum=10, maximum=179, value=120)
        row.addWidget(self.tth_min)
        row.addWidget(self.tth_max)
        btn = QPushButton("生成 XRD")
        btn.clicked.connect(lambda: self._run(self._gen_xrd))
        row.addWidget(btn)
        save = QPushButton("导出图片")
        save.clicked.connect(lambda: self.cv_xrd.save_with_dialog(self))
        row.addWidget(save)
        row.addStretch(1)
        lay.addLayout(row)
        return page

    def _gen_xrd(self) -> None:
        pat = simulate_powder_xrd(self.crystal, self.wl.value(),
                                  (self.tth_min.value(), self.tth_max.value()))
        self.cv_xrd.fig.clear()
        ax = self.cv_xrd.fig.add_subplot(111)
        draw_xrd(ax, pat,
                 f"{disp_name(self.crystal.name)}  XRD  (λ={self.wl.value()} Å)")
        self.cv_xrd.draw_idle()
        self._last_gen = self._gen_xrd

    # ---------- 页 6：极射投影 ----------
    def _tab_stereo(self) -> QWidget:
        page, lay, self.cv_st = self._page()
        row = QHBoxLayout()
        row.addWidget(QLabel("极点 (hkl)，逗号分隔，HCP 可写 4 指数"))
        self.pole_edit = QLineEdit("001, 100, 010, 110, 111, 112")
        row.addWidget(self.pole_edit, stretch=1)
        btn = QPushButton("生成极图")
        btn.clicked.connect(lambda: self._run(self._gen_stereo))
        row.addWidget(btn)
        save = QPushButton("导出图片")
        save.clicked.connect(lambda: self.cv_st.save_with_dialog(self))
        row.addWidget(save)
        lay.addLayout(row)
        return page

    def _gen_stereo(self) -> None:
        import re
        hkls = []
        for item in self.pole_edit.text().split(","):
            item = item.strip()
            tokens = item.split()
            if len(tokens) == 1 and re.fullmatch(r"-?\d{3,4}", item):
                # 紧凑写法 "001" / "111"：按单个数字拆分（仅支持个位数指数）
                idx = tuple(int(c) for c in item.lstrip("-"))
            else:
                idx = tuple(int(t) for t in tokens)
            if len(idx) == 4:
                idx = four_to_three_plane(idx)
            elif len(idx) != 3:
                raise ValueError(f"极点 {item!r} 不是 3 或 4 个指数")
            hkls.append(idx)
        poles = stereographic_poles(self.crystal, hkls, self._uvw(), self._in_plane())
        self.cv_st.fig.clear()
        ax = self.cv_st.fig.add_subplot(111)
        draw_stereographic(ax, poles,
                           f"{disp_name(self.crystal.name)}  极射投影")
        self.cv_st.draw_idle()
        self._last_gen = self._gen_stereo

    # ---------- 页 7：双晶 OR ----------
    def _tab_bicrystal(self) -> QWidget:
        page, lay, self.cv_bi = self._page3d()
        row = QHBoxLayout()
        keys = list(list_materials().keys())
        self.bi1 = QComboBox()
        self.bi2 = QComboBox()
        for key in keys:
            self.bi1.addItem(disp_name(key), userData=key)
            self.bi2.addItem(disp_name(key), userData=key)
        i1 = self.bi1.findData("gamma-Fe")
        i2 = self.bi2.findData("alpha-Fe")
        self.bi1.setCurrentIndex(max(i1, 0))
        self.bi2.setCurrentIndex(max(i2, 0))
        self.or_combo = QComboBox()
        self.or_combo.addItems(list(OR_PRESETS.keys()))
        row.addWidget(QLabel("相1"))
        row.addWidget(self.bi1)
        row.addWidget(QLabel("相2"))
        row.addWidget(self.bi2)
        row.addWidget(QLabel("取向关系"))
        row.addWidget(self.or_combo)
        row.addWidget(QLabel("界面 (hkl)"))
        self.bi_hkl = QLineEdit("1 1 0")
        self.bi_hkl.setMaximumWidth(70)
        row.addWidget(self.bi_hkl)
        btn = QPushButton("生成双晶")
        btn.clicked.connect(lambda: self._run(self._gen_bicrystal))
        row.addWidget(btn)
        save = QPushButton("导出图片")
        save.clicked.connect(lambda: self.cv_bi.save_with_dialog(self))
        row.addWidget(save)
        row.addWidget(QLabel("左键拖动旋转 · 滚轮缩放"))
        row.addStretch(1)
        lay.addLayout(row)
        self.bi_info = QLabel("（未生成）")
        lay.addWidget(self.bi_info)
        return page

    def _gen_bicrystal(self) -> None:
        p1 = load_material(self.bi1.currentData())
        p2 = load_material(self.bi2.currentData())
        R = rotation_from_or_name(p1, p2, self.or_combo.currentText())
        interface = parse_indices(self.bi_hkl.text())
        res = build_bicrystal(p1, p2, R, interface, supercell=(4, 4, 4))
        ax = self.cv_bi.new_3d_axes()
        for g, name in ((0, "相1"), (1, "相2")):
            pts = res.coords[res.grains == g]
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                       s=40 * self.settings["atom_scale"], c=GRAIN_COLORS[g],
                       edgecolors="#333333", linewidths=0.2, label=name, depthshade=True)
        span = np.ptp(res.coords, axis=0).max() / 2 + 2
        mid = res.coords.mean(axis=0)
        ax.set_xlim(mid[0] - span, mid[0] + span)
        ax.set_ylim(mid[1] - span, mid[1] + span)
        ax.set_zlim(mid[2] - span, mid[2] + span)
        ax.set_box_aspect((1, 1, 1))
        ax.legend()
        ax.set_title(f"{disp_name(p1.name)} / {disp_name(p2.name)}  "
                     f"{self.or_combo.currentText()} 双晶")
        self.cv_bi.draw_idle()
        self._last_gen = self._gen_bicrystal
        self.bi_info.setText(
            f"原子数 {res.n_atoms_before} → {len(res.coords)}（去重删 {res.n_removed_overlap}），"
            f"界面间距 gap={res.gap:.3f} Å")

    # ---------- 页 8：孪晶 ----------
    def _tab_twin(self) -> QWidget:
        page, lay, self.cv_tw = self._page3d()
        row = QHBoxLayout()
        row.addWidget(QLabel("孪生面 K1"))
        self.tw4 = QCheckBox("四指数")
        self.tw_spins = []
        for i, default in enumerate((1, 1, 1, 0)):
            sp = QSpinBox(minimum=-9, maximum=9, value=default)
            sp.setVisible(i < 3)
            self.tw_spins.append(sp)
            row.addWidget(sp)
        self.tw4.toggled.connect(
            lambda on: [s.setVisible(True) for s in self.tw_spins])
        row.addWidget(self.tw4)
        btn = QPushButton("生成孪晶")
        btn.clicked.connect(lambda: self._run(self._gen_twin))
        row.addWidget(btn)
        save = QPushButton("导出图片")
        save.clicked.connect(lambda: self.cv_tw.save_with_dialog(self))
        row.addWidget(save)
        row.addWidget(QLabel("左键拖动旋转 · 滚轮缩放"))
        row.addStretch(1)
        lay.addLayout(row)
        self.tw_info = QLabel("（未生成）")
        lay.addWidget(self.tw_info)
        return page

    def _gen_twin(self) -> None:
        vals = tuple(s.value() for s in self.tw_spins)
        plane = four_to_three_plane(vals) if self.tw4.isChecked() else vals[:3]
        res = build_twin(self.crystal, plane, supercell=(4, 4, 4))
        ax = self.cv_tw.new_3d_axes()
        for g, name in ((0, "基体"), (1, "孪晶")):
            pts = res.coords[res.grains == g]
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                       s=40 * self.settings["atom_scale"], c=GRAIN_COLORS[g],
                       edgecolors="#333333", linewidths=0.2, label=name, depthshade=True)
        span = np.ptp(res.coords, axis=0).max() / 2 + 2
        mid = res.coords.mean(axis=0)
        ax.set_xlim(mid[0] - span, mid[0] + span)
        ax.set_ylim(mid[1] - span, mid[1] + span)
        ax.set_zlim(mid[2] - span, mid[2] + span)
        ax.set_box_aspect((1, 1, 1))
        ax.legend()
        ax.set_title(f"{disp_name(self.crystal.name)}  "
                     f"({plane[0]}{plane[1]}{plane[2]}) 孪晶")
        self.cv_tw.draw_idle()
        self._last_gen = self._gen_twin
        self.tw_info.setText(
            f"原子数 {res.n_atoms_before} → {len(res.coords)}（界面去重删 {res.n_removed_overlap}）")
