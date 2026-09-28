"""电子衍射斑点模拟（SAED 风格）：与实空间投影互补的倒空间视图。

科研动机：材料科研人员（尤其透射电镜方向）看原子排布时，最关心的问题
是"这个方向上的衍射花样长什么样"——实空间投影 + 衍射斑点并排展示，
取向关系、消光规律一目了然。

原理（运动学近似）：
  - 倒易矢量 G = inv(L)·(hkl)（单位 1/Å，不含 2π）
  - 晶带定律：h·u + k·v + l·w = 0 的 (hkl) 落在衍射图上（精确整数判据）
  - 结构因子 F(hkl) = Σ_j exp(2πi·(h x_j + k y_j + l z_j))，强度 I = |F|²
  - 系统消光（如 FCC 奇偶混杂指数 I=0）由结构因子自然给出，无需查表
"""
from __future__ import annotations

import matplotlib

if matplotlib.get_backend().lower() not in ("qtagg", "qt5agg", "qt6agg"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .orientation import rotation_from_uvw
from .structure import Crystal


def simulate_diffraction(
    crystal: Crystal,
    uvw: tuple[int, int, int],
    in_plane: tuple[int, int, int] | None = None,
    max_index: int = 3,
    intensity_tol: float = 1e-6,
) -> dict:
    """模拟 [uvw] 晶带轴的衍射斑点。

    返回 dict：
      gx, gy      : 斑点在图面上的坐标（1/Å，正交于观察方向）
      intensity   : 归一化强度（最大为 1）
      hkls        : 每个斑点的 (h,k,l)
      d_spacings  : 对应晶面间距（Å）
      R           : 使用的旋转矩阵
    """
    R = rotation_from_uvw(uvw, crystal.lattice_vectors, in_plane)
    invL = np.linalg.inv(crystal.lattice_vectors)
    basis = crystal.basis
    u, v, w = uvw

    spots, intens, hkls, ds = [], [], [], []
    for h in range(-max_index, max_index + 1):
        for k in range(-max_index, max_index + 1):
            for l in range(-max_index, max_index + 1):
                if (h, k, l) == (0, 0, 0):
                    continue
                if h * u + k * v + l * w != 0:      # 晶带定律（精确整数判据）
                    continue
                phases = 2j * np.pi * (basis @ np.array([h, k, l], dtype=float))
                F = np.exp(phases).sum()
                I = float(abs(F) ** 2)
                if I < intensity_tol:               # 系统消光
                    continue
                G = invL @ np.array([h, k, l], dtype=float)
                g_rot = R @ G
                spots.append(g_rot[:2])
                intens.append(I)
                hkls.append((h, k, l))
                ds.append(1.0 / np.linalg.norm(G))

    intens = np.asarray(intens)
    return {
        "gx": np.asarray(spots)[:, 0] if spots else np.array([]),
        "gy": np.asarray(spots)[:, 1] if spots else np.array([]),
        "intensity": intens / intens.max() if len(intens) else intens,
        "hkls": hkls,
        "d_spacings": np.asarray(ds),
        "R": R,
    }


def draw_saed(
    ax,
    pattern: dict,
    title: str = "",
    label_min_intensity: float = 0.15,
) -> None:
    """在给定 Axes 上绘制 SAED 花样（render_diffraction / GUI 共用）。"""
    gx, gy, I = pattern["gx"], pattern["gy"], pattern["intensity"]
    ax.set_facecolor("black")

    ax.scatter(gx, gy, s=20 + 700 * I, c="white", edgecolors="none", alpha=0.95)
    ax.scatter([0], [0], s=90, c="#ff5555", marker="o", zorder=5)   # 透射斑
    ax.annotate("(000)", (0, 0), textcoords="offset points", xytext=(8, -14),
                color="#ff5555", fontsize=10)

    for (x, y, inten, hkl) in zip(gx, gy, I, pattern["hkls"]):
        if inten >= label_min_intensity:
            ax.annotate(f"{hkl}", (x, y), textcoords="offset points",
                        xytext=(7, 7), color="#aaaaaa", fontsize=8)

    lim = max(np.abs(gx).max(), np.abs(gy).max()) * 1.18
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("g$_x$ (Å$^{-1}$)", color="white")
    ax.set_ylabel("g$_y$ (Å$^{-1}$)", color="white")
    ax.tick_params(colors="#666666")
    for spine in ax.spines.values():
        spine.set_color("#444444")
    ax.set_title(title, color="white", fontsize=12)


def render_diffraction(
    pattern: dict,
    title: str = "",
    out_path: str | None = None,
    label_min_intensity: float = 0.15,
) -> str | None:
    """SAED 风格渲染：黑底白斑。out_path 后缀决定格式：.png 位图，.svg/.pdf 矢量图。"""
    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
    draw_saed(ax, pattern, title, label_min_intensity)
    fig.patch.set_facecolor("#111111")
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(),
                    bbox_inches="tight")
        plt.close(fig)
        return out_path
    return None


# ----------------------------------------------------------------------
# 粉末 XRD 模拟
# ----------------------------------------------------------------------

def simulate_powder_xrd(
    crystal: Crystal,
    wavelength: float = 1.5406,          # Cu Kα，Å
    two_theta_range: tuple[float, float] = (10.0, 120.0),
    max_index: int = 4,
    lp_factor: bool = True,
    intensity_tol: float = 1e-6,
) -> dict:
    """粉末 X 射线衍射（运动学近似，全取向平均 → 只需 d 值 + 强度）。

    - 布拉格定律：λ = 2d·sinθ
    - 同一 d 值的等效晶面自动累加强度 = 多重性因子
    - lp_factor：乘洛伦兹-偏振因子 1/(sin²θ·cosθ)，使相对强度接近实测卡片

    返回 dict：
      two_theta : 峰位（度）
      intensity : 归一化强度（最大 100）
      d_spacings: 每个峰的 d（Å）
      hkls      : 每个峰的代表 (h,k,l)（同一 d 取第一个）
      wavelength: 使用的波长（Å）
    """
    invL = np.linalg.inv(crystal.lattice_vectors)
    basis = crystal.basis

    buckets: dict[float, list] = {}   # d → [I_sum, hkl]
    for h in range(-max_index, max_index + 1):
        for k in range(-max_index, max_index + 1):
            for l in range(-max_index, max_index + 1):
                if (h, k, l) == (0, 0, 0):
                    continue
                phases = 2j * np.pi * (basis @ np.array([h, k, l], dtype=float))
                I = float(abs(np.exp(phases).sum()) ** 2)
                if I < intensity_tol:                  # 系统消光
                    continue
                G = invL @ np.array([h, k, l], dtype=float)
                g = np.linalg.norm(G)
                d = 1.0 / g
                key = round(d, 4)                      # 合并等效晶面（多重性）
                if key in buckets:
                    buckets[key][0] += I
                else:
                    buckets[key] = [I, (h, k, l)]

    tth_min, tth_max = two_theta_range
    tt, inten, ds, hkls = [], [], [], []
    for d, (I, hkl) in buckets.items():
        sin_t = wavelength / (2 * d)
        if sin_t >= 1.0:
            continue
        theta = np.degrees(np.arcsin(sin_t))
        two_theta = 2 * theta
        if not (tth_min <= two_theta <= tth_max):
            continue
        if lp_factor:
            st, ct = np.sin(np.radians(theta)), np.cos(np.radians(theta))
            I = I / (st * st * ct)
        tt.append(two_theta)
        inten.append(I)
        ds.append(d)
        hkls.append(hkl)

    order = np.argsort(tt)
    inten = np.asarray(inten)[order]
    return {
        "two_theta": np.asarray(tt)[order],
        "intensity": inten / inten.max() * 100.0 if len(inten) else inten,
        "d_spacings": np.asarray(ds)[order],
        "hkls": [hkls[i] for i in order],
        "wavelength": wavelength,
    }


def draw_xrd(ax, pattern: dict, title: str = "", label_top: int = 8) -> None:
    """在给定 Axes 上绘制 XRD 图谱（棒状图，最强 label_top 个峰标注 (hkl)）。"""
    tt, I = pattern["two_theta"], pattern["intensity"]
    ax.vlines(tt, 0, I, color="#1f5fb4", linewidth=1.2)
    ax.set_xlabel("2θ (°)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.set_ylim(0, 118)
    ax.set_title(title)
    # 标注最强峰
    top = np.argsort(I)[::-1][:label_top]
    for i in top:
        hkl = pattern["hkls"][i]
        ax.annotate(f"({hkl[0]}{hkl[1]}{hkl[2]})", (tt[i], I[i]),
                    textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=8, rotation=90, color="#333333")


def render_xrd(
    pattern: dict,
    title: str = "",
    out_path: str = "xrd.png",
    label_top: int = 8,
) -> str:
    """XRD 图谱渲染。out_path 后缀决定格式：.png 位图，.svg/.pdf 矢量图。"""
    fig, ax = plt.subplots(figsize=(9, 5))
    draw_xrd(ax, pattern, title, label_top)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    return out_path
