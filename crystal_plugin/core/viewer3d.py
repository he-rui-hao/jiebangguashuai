"""3D 交互可视化模块：单晶原子排布的三维可旋转视图。

技术路线：plotly（本地已装，零依赖增量）
  - 每个原子 = Scatter3d 标记点，坐标为固定卡氏坐标（数据不动、相机动，
    旋转拖动永不"错位"）
  - aspectmode='data' 锁定三轴等比例，保证晶格不被拉伸变形
  - 输出自包含 HTML：双击即可在任意浏览器打开，无需网络、无需 Python，
    直接满足答辩现场离线演示需求

桌面插件阶段（E 同学）可将本模块替换为 pyvistaqt 嵌入 PySide6，
数据生成部分（build_scene_data）原样复用。
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .projector import DEFAULT_COLORS, DEFAULT_RADII, _fallback_color
from .structure import Crystal

# 原子在 3D 视图中的显示半径 = 真实半径 × SCALE_3D（VESTA 风格偏小，便于看清层间）
SCALE_3D = 0.32


def build_scene_data(
    crystal: Crystal,
    supercell: tuple[int, int, int] = (3, 3, 3),
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """超胞展开并居中，返回 (cart_coords, species_list, cell_edges)。

    cell_edges 为超胞 12 条棱的端点对，形状 (12, 2, 3)，用于画晶胞框。
    所有坐标均为卡氏坐标（Å），居中于原点。
    """
    nx, ny, nz = supercell
    frac, species = crystal.supercell(nx, ny, nz)
    cart = frac @ crystal.lattice_vectors
    cart -= cart.mean(axis=0)

    # 超胞的 8 个顶点（相对居中坐标）
    L = crystal.lattice_vectors
    corners_frac = np.array(
        [[i, j, k] for i in (0, nx) for j in (0, ny) for k in (0, nz)],
        dtype=np.float64,
    )
    corners = corners_frac @ L
    corners -= corners.mean(axis=0)
    # 12 条棱：顶点索引对（二进制只差一位的顶点对）
    edges = []
    for a in range(8):
        for b in range(a + 1, 8):
            if bin(a ^ b).count("1") == 1:
                edges.append((corners[a], corners[b]))
    return cart, species, np.array(edges)


def _apply_layout(fig: go.Figure, title: str) -> None:
    """统一 3D 场景外观：三轴等比例（锁定晶格不变形）、隐藏坐标轴。"""
    fig.update_layout(
        title=dict(text=title, x=0.5),
        scene=dict(
            aspectmode="data",
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            bgcolor="white",
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=40, b=0),
    )


def build_figure_from_arrays(
    coords: np.ndarray,
    labels: list[str],
    title: str,
    colors: dict[str, str] | None = None,
    radii: dict[str, float] | None = None,
) -> go.Figure:
    """任意带标签点集的 3D 渲染（孪晶/双晶按晶粒着色用）。

    labels 形如 "Cu (基体)"；元素名取空格前部分用于查默认半径/颜色。
    colors / radii 以完整 label 为键，缺省回退到元素默认值。
    """
    coords = np.asarray(coords, dtype=np.float64)
    coords = coords - coords.mean(axis=0)
    colors = colors or {}
    radii = radii or {}
    span = np.abs(coords).max()

    fig = go.Figure()
    for lab in dict.fromkeys(labels):  # 保持出现顺序
        idx = [i for i, s in enumerate(labels) if s == lab]
        pts = coords[idx]
        element = lab.split(" ")[0].split("(")[0]
        r = radii.get(lab, DEFAULT_RADII.get(element, 1.30)) * SCALE_3D
        marker_d = r * 120.0 / max(span, 1.0)
        color = colors.get(lab, DEFAULT_COLORS.get(element, _fallback_color(element)))
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers", name=lab,
            marker=dict(size=marker_d, color=color,
                        line=dict(color="#222222", width=0.8), opacity=1.0),
            hovertemplate=f"{lab} (%{{x:.2f}}, %{{y:.2f}}, %{{z:.2f}}) Å<extra></extra>",
        ))
    _apply_layout(fig, title)
    return fig


def build_figure(
    crystal: Crystal,
    supercell: tuple[int, int, int] = (3, 3, 3),
    title: str | None = None,
    colors: dict[str, str] | None = None,
) -> go.Figure:
    """构建 plotly 3D 图（可交互旋转/缩放/平移）。"""
    colors = colors or {}
    cart, species, edges = build_scene_data(crystal, supercell)

    fig = go.Figure()

    # 晶胞框（12 条棱）
    for e in edges:
        fig.add_trace(go.Scatter3d(
            x=[e[0, 0], e[1, 0]], y=[e[0, 1], e[1, 1]], z=[e[0, 2], e[1, 2]],
            mode="lines", line=dict(color="#888888", width=2),
            showlegend=False, hoverinfo="skip",
        ))

    # 晶格矢量箭头（a1 红, a2 绿, a3 蓝），从超胞中心出发
    L = crystal.lattice_vectors
    for vec, c, name in zip(L, ["#d62728", "#2ca02c", "#1f77b4"], ["a1", "a2", "a3"]):
        fig.add_trace(go.Cone(
            x=[vec[0]], y=[vec[1]], z=[vec[2]],
            u=[vec[0]], v=[vec[1]], w=[vec[2]],
            sizemode="absolute", sizeref=1.2, anchor="tip",
            colorscale=[[0, c], [1, c]], showscale=False, name=name,
        ))

    # 原子：按元素分组各一条 trace（便于图例开关）
    elements = sorted(set(species))
    span = np.abs(cart).max()
    for el in elements:
        idx = [i for i, s in enumerate(species) if s == el]
        pts = cart[idx]
        r = DEFAULT_RADII.get(el, 1.30) * SCALE_3D
        # 标记直径（像素）与数据尺度（Å）按比例换算，保证"球"的视觉大小合理
        marker_d = r * 120.0 / max(span, 1.0)
        color = colors.get(el, DEFAULT_COLORS.get(el, _fallback_color(el)))
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers", name=el,
            marker=dict(
                size=marker_d, color=color,
                line=dict(color="#222222", width=0.8),
                opacity=1.0,
            ),
            hovertemplate=f"{el} (%{{x:.2f}}, %{{y:.2f}}, %{{z:.2f}}) Å<extra></extra>",
        ))

    _apply_layout(fig, title or f"{crystal.name} 3D view")
    return fig


def add_hkl_plane(
    fig: go.Figure,
    crystal: Crystal,
    hkl: tuple[int, int, int],
    half_size: float | None = None,
    color: str = "#ff8800",
    opacity: float = 0.25,
) -> go.Figure:
    """在 3D 场景中叠加指定 (hkl) 晶面的半透明面片 + 法线箭头。

    科研用途：旋转观察时直观回答"我现在看的是哪个晶面"，
    讲解晶面/晶向关系、孪晶面位置时尤其有用。面片过场景中心。
    """
    from twin.twin_builder import plane_normal_cart

    n = plane_normal_cart(hkl, crystal.lattice_vectors)
    # 面内两个正交基
    ref = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0])
    e1 = np.cross(n, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)

    if half_size is None:
        half_size = float(np.linalg.norm(crystal.lattice_vectors, axis=1).mean()) * 1.2
    corners = np.array([
        half_size * (e1 + e2), half_size * (e1 - e2),
        -half_size * (e1 + e2), -half_size * (e1 - e2),
    ])
    fig.add_trace(go.Mesh3d(
        x=corners[:, 0], y=corners[:, 1], z=corners[:, 2],
        i=[0, 0], j=[1, 2], k=[2, 3],
        color=color, opacity=opacity, name=f"({hkl[0]}{hkl[1]}{hkl[2]}) 面",
        hoverinfo="name",
    ))
    fig.add_trace(go.Cone(
        x=[n[0] * half_size], y=[n[1] * half_size], z=[n[2] * half_size],
        u=[n[0]], v=[n[1]], w=[n[2]],
        sizemode="absolute", sizeref=1.2, anchor="tip",
        colorscale=[[0, color], [1, color]], showscale=False,
        name=f"[{hkl[0]}{hkl[1]}{hkl[2]}] 法线",
    ))
    return fig


def save_html(fig: go.Figure, out_path: str) -> None:
    """保存为自包含 HTML（内嵌 plotly.js，双击离线打开）。"""
    fig.write_html(out_path, include_plotlyjs=True, full_html=True)


def show(fig: go.Figure) -> None:
    """在默认浏览器中打开交互视图。"""
    fig.show()
