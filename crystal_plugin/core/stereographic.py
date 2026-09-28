"""极射赤面投影（stereographic projection）：把晶面极点投到 2D 圆面。

科研动机：极射投影是晶体取向分析的通用语言——织构分析、孪生要素几何、
取向关系核对都在极图上进行。把任意 (hkl) 极点按观察方向 [uvw] 投影，
可直接对照 EBSD 极图。

原理：
  - 极点 = 晶面卡氏法线 n ∝ inv(L)·(hkl)（非立方晶系安全）
  - 旋转到观察坐标系后，从南极 (0,0,-1) 向赤道面投影：
        (px, py) = (dx, dy) / (1 + dz)
  - 下半球极点（dz<0）翻到上半球，空心标记（对跖点等价）
  - 极射投影保角：球面上两极点夹角 = 投影后沿大圆量的角度
"""
from __future__ import annotations

import matplotlib

if matplotlib.get_backend().lower() not in ("qtagg", "qt5agg", "qt6agg"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .orientation import rotation_from_uvw
from .structure import Crystal


def stereographic_poles(
    crystal: Crystal,
    hkl_list: list[tuple[int, int, int]],
    uvw: tuple[int, int, int] = (0, 0, 1),
    in_plane: tuple[int, int, int] | None = None,
) -> dict:
    """把若干 (hkl) 极点按观察方向 [uvw] 做极射赤面投影。

    返回 dict：
      px, py   : 投影坐标（投影圆半径 1）
      upper    : 布尔数组，True=上半球（实心点），False=下半球（空心点）
      hkls     : 输入的 (h,k,l) 列表
      R        : 使用的旋转矩阵
    """
    R = rotation_from_uvw(uvw, crystal.lattice_vectors, in_plane)
    invL = np.linalg.inv(crystal.lattice_vectors)

    px, py, upper = [], [], []
    for hkl in hkl_list:
        g = invL @ np.array(hkl, dtype=float)   # 倒易矢量 = 面法线方向
        n = g / np.linalg.norm(g)
        d = R @ n                                # 转到观察坐标系
        if d[2] >= 0:
            px.append(d[0] / (1 + d[2]))
            py.append(d[1] / (1 + d[2]))
            upper.append(True)
        else:                                    # 下半球翻到对跖点
            px.append(-d[0] / (1 - d[2]))
            py.append(-d[1] / (1 - d[2]))
            upper.append(False)

    return {
        "px": np.asarray(px),
        "py": np.asarray(py),
        "upper": np.asarray(upper, dtype=bool),
        "hkls": list(hkl_list),
        "R": R,
    }


def draw_stereographic(ax, poles: dict, title: str = "") -> None:
    """在给定 Axes 上绘制极射赤面投影（render_stereographic / GUI 共用）。"""
    # 投影圆 + 十字丝 + 同心圆（30°/60°）参考
    circle = plt.Circle((0, 0), 1.0, fill=False, color="#333333", linewidth=1.2)
    ax.add_patch(circle)
    ax.axhline(0, color="#bbbbbb", linewidth=0.6)
    ax.axvline(0, color="#bbbbbb", linewidth=0.6)
    for deg in (30, 60):
        r = np.tan(np.radians(deg / 2))
        ax.add_patch(plt.Circle((0, 0), r, fill=False, color="#dddddd",
                                linewidth=0.5, linestyle="--"))

    for x, y, up, hkl in zip(poles["px"], poles["py"], poles["upper"], poles["hkls"]):
        if up:
            ax.plot(x, y, "o", color="#c0392b", markersize=7)
        else:
            ax.plot(x, y, "o", markerfacecolor="none", markeredgecolor="#c0392b",
                    markersize=7, markeredgewidth=1.5)
        ax.annotate(f"({hkl[0]}{hkl[1]}{hkl[2]})", (x, y),
                    textcoords="offset points", xytext=(7, 7), fontsize=9,
                    color="#333333")

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)


def render_stereographic(
    poles: dict,
    title: str = "",
    out_path: str = "stereographic.png",
) -> str:
    """极射投影渲染。out_path 后缀决定格式：.png 位图，.svg/.pdf 矢量图。"""
    fig, ax = plt.subplots(figsize=(7, 7))
    draw_stereographic(ax, poles, title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    return out_path
