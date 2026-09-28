"""投影器：旋转后的结构 → 2D 原子排布图。

对应接口文档 §3.3。正交投影 + 深度分层着色 + 等比例原子圆。
"""
from __future__ import annotations

import matplotlib

if matplotlib.get_backend().lower() not in ("qtagg", "qt5agg", "qt6agg"):
    matplotlib.use("Agg")  # 无界面环境渲染；GUI（Qt 后端）下保留交互后端

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from .structure import Crystal

# 常用元素默认配色（CPK/JMol 风格，可被用户输入覆盖）
DEFAULT_COLORS: dict[str, str] = {
    # 金属
    "Fe": "#b85450", "Cu": "#c87533", "Al": "#9db8d2", "Ni": "#7da87b",
    "Mg": "#5fb85f", "Ti": "#7a9cc6", "Zr": "#94e0e0", "Co": "#f090a0",
    "W": "#2194d6", "Mo": "#54b5b5", "Nb": "#73c2c9", "Ta": "#4da6ff",
    "V": "#a6a6ab", "Cr": "#8a99c7", "Zn": "#7d80b0", "Be": "#c2ff00",
    "Cd": "#ffd98f", "Ag": "#c0c0c0", "Au": "#ffd123", "Pb": "#575961",
    "Pt": "#d0d0e0", "Pd": "#7fc4c9", "Po": "#ab5c00", "Sn": "#668080",
    "Ga": "#c28f8f", "In": "#a67573", "Sb": "#9e63b5", "Cs": "#57178f",
    # 非金属/类金属
    "Si": "#f0c8a0", "Ge": "#668f8f", "C": "#404040",
    "Na": "#ab5cf2", "Cl": "#1ff01f", "O": "#ff0d0d", "S": "#ffff30",
    "As": "#bd80e3",
}

# 默认显示半径（Å，金属/共价半径量级，仅影响视觉，可被用户输入覆盖）
DEFAULT_RADII: dict[str, float] = {
    "Fe": 1.26, "Cu": 1.28, "Al": 1.43, "Ni": 1.24, "Mg": 1.60, "Ti": 1.45,
    "Zr": 1.60, "Co": 1.25, "W": 1.37, "Mo": 1.39, "Nb": 1.46, "Ta": 1.46,
    "V": 1.31, "Cr": 1.28, "Zn": 1.34, "Be": 1.12, "Cd": 1.49, "Ag": 1.45,
    "Au": 1.44, "Pb": 1.75, "Pt": 1.39, "Pd": 1.38, "Po": 1.68, "Sn": 1.41,
    "Ga": 1.35, "In": 1.67, "Sb": 1.45, "Cs": 2.44, "Si": 1.18, "Ge": 1.22,
    "C": 0.77, "Na": 1.66, "Cl": 1.75, "O": 1.40, "S": 1.84, "As": 1.25,
}


def project(
    crystal: Crystal,
    R: np.ndarray,
    supercell: tuple[int, int, int] = (3, 3, 3),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """超胞展开 → 居中 → 旋转 → 正交投影。

    返回 (x, y, z_depth, species_list)，z_depth 越大离观察者越近。
    """
    frac, species = crystal.supercell(*supercell)
    cart = frac @ crystal.lattice_vectors
    cart -= cart.mean(axis=0)  # 居中，使投影图以原点为中心
    rot = cart @ R.T
    return rot[:, 0], rot[:, 1], rot[:, 2], species


def _fallback_color(element: str) -> str:
    """未收录元素按名称哈希给稳定颜色。"""
    h = abs(hash(element))
    return f"#{(h & 0xFF):02x}{((h >> 8) & 0xFF):02x}{((h >> 16) & 0xFF):02x}"


def _draw_proj(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    species: list[str],
    title: str,
    colors: dict[str, str] | None = None,
    atom_scale: float = 0.35,
    z_range: tuple[float, float] | None = None,
) -> None:
    """在给定 Axes 上绘制 2D 原子排布（render / render_multi / GUI 共用）。

    - 按深度 z 分层（四舍五入到 0.5 Å），层间亮度渐变体现前后遮挡
    - 后层原子先画、前层后画，形成正确遮挡
    - atom_scale 为原子圆半径相对真实原子半径的缩放（VESTA 风格默认偏小）
    - z_range：只画深度在 [zmin, zmax]（Å）内的原子——单层切片功能
    """
    colors = colors or {}
    if z_range is not None:
        keep = (z >= z_range[0]) & (z <= z_range[1])
        x, y, z = x[keep], y[keep], z[keep]
        species = [s for s, k in zip(species, keep) if k]
    order = np.argsort(z)  # 深度升序：远处先画
    xs, ys, zs = x[order], y[order], z[order]
    sp = [species[i] for i in order]

    layers = np.round(zs / 0.5) * 0.5
    unique_layers = np.unique(layers)
    n_layers = len(unique_layers)
    layer_shade = {lv: 0.55 + 0.45 * i / max(n_layers - 1, 1)
                   for i, lv in enumerate(unique_layers)}  # 后层暗、前层亮

    span = max(np.ptp(xs), np.ptp(ys)) * 0.62 + 2.0
    for xi, yi, lvi, spi in zip(xs, ys, layers, sp):
        base = np.array(
            matplotlib.colors.to_rgb(colors.get(spi, DEFAULT_COLORS.get(spi, _fallback_color(spi))))
        )
        rgb = base * layer_shade[lvi]
        r = DEFAULT_RADII.get(spi, 1.30) * atom_scale
        ax.add_patch(Circle((xi, yi), r, facecolor=rgb, edgecolor="#222222",
                            linewidth=0.4, zorder=lvi))

    ax.set_xlim(-span, span)
    ax.set_ylim(-span, span)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=14)
    for spine in ax.spines.values():
        spine.set_visible(False)


def render(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    species: list[str],
    title: str,
    out_path: str,
    colors: dict[str, str] | None = None,
    atom_scale: float = 0.35,
    z_range: tuple[float, float] | None = None,
) -> None:
    """绘制 2D 原子排布图并保存。out_path 后缀决定格式：.png 位图，.svg/.pdf 矢量图。"""
    fig, ax = plt.subplots(figsize=(7, 7))
    _draw_proj(ax, x, y, z, species, title, colors, atom_scale, z_range)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)


def draw_multi(
    fig,
    crystal: Crystal,
    views: list[tuple[tuple[int, int, int], tuple[int, int, int] | None]],
    supercell: tuple[int, int, int] = (3, 3, 3),
    colors: dict[str, str] | None = None,
    atom_scale: float = 0.30,
    z_range: tuple[float, float] | None = None,
) -> None:
    """在给定 Figure 上绘制多方向联排对比（2×2，最多 4 个方向）。

    views : [((u,v,w), in_plane 或 None), ...]，每个子图标题自动标注 [uvw]
    """
    from .orientation import rotation_from_uvw

    views = views[:4]
    axes = fig.subplots(2, 2).ravel()
    for ax, (uvw, ip) in zip(axes, views):
        R = rotation_from_uvw(uvw, crystal.lattice_vectors, ip)
        x, y, z, sp = project(crystal, R, supercell)
        title = f"{crystal.name}  [{uvw[0]} {uvw[1]} {uvw[2]}]"
        _draw_proj(ax, x, y, z, sp, title, colors, atom_scale, z_range)
        ax.title.set_fontsize(11)
    for ax in axes[len(views):]:
        ax.axis("off")
    fig.tight_layout()


def render_multi(
    crystal: Crystal,
    views: list[tuple[tuple[int, int, int], tuple[int, int, int] | None]],
    supercell: tuple[int, int, int] = (3, 3, 3),
    out_path: str = "multi.png",
    colors: dict[str, str] | None = None,
    atom_scale: float = 0.30,
    z_range: tuple[float, float] | None = None,
) -> str:
    """多方向联排对比图（2×2）。out_path 后缀决定格式：.png 位图，.svg/.pdf 矢量图。"""
    fig = plt.figure(figsize=(12, 12))
    draw_multi(fig, crystal, views, supercell, colors, atom_scale, z_range)
    fig.savefig(out_path, dpi=180, facecolor="white")
    plt.close(fig)
    return out_path
