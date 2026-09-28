"""演示脚本：K-S 取向关系双晶（FCC γ-Fe / BCC α-Fe）生成与验证。

验证点：
  [1] R 正交归一（RᵀR=I 且 det=1，非镜像）
  [2] OR 对齐精度：R 把相2 的面法线/晶向精确转到相1 方向（角度 < 1e-6°）
  [3] 界面拼接后无异常重叠（去重后最小间距报告）
  [4] 晶粒/界面统计

输出：
  output/KS_bicrystal_edge.png —— 沿 [1̄01]γ 侧视 2D 投影（界面水平）
  output/KS_bicrystal_3d.html  —— 3D 交互视图（两相双色）
"""
from __future__ import annotations

import os

import numpy as np

from bicrystal.or_builder import OR_PRESETS, build_bicrystal, rotation_from_or_name
from core.orientation import rotation_from_uvw
from core.projector import render
from core.structure import bcc, fcc
from core.viewer3d import build_figure_from_arrays, save_html
from twin.twin_builder import plane_normal_cart

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT_DIR, exist_ok=True)

COLOR_P1 = "#b85450"   # γ-Fe：砖红
COLOR_P2 = "#4a90d9"   # α-Fe：钢蓝


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    c = np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def verify(gamma, alpha, R, bi) -> None:
    # [1] 正交性
    err = np.abs(R.T @ R - np.eye(3)).max()
    det = np.linalg.det(R)
    print(f"  [1] |RᵀR−I| = {err:.2e}, det(R) = {det:.6f}"
          f"  {'PASS' if err < 1e-9 and abs(det - 1) < 1e-9 else 'FAIL'}")

    # [2] OR 对齐精度
    hkl1, uvw1, hkl2, uvw2, *_ = OR_PRESETS["K-S"]
    g1 = plane_normal_cart(hkl1, gamma.lattice_vectors)
    t1 = np.array(uvw1, float) @ gamma.lattice_vectors
    g2 = plane_normal_cart(hkl2, alpha.lattice_vectors)
    t2 = np.array(uvw2, float) @ alpha.lattice_vectors
    a_g = angle_deg(R @ g2, g1)
    a_t = angle_deg(R @ t2, t1)
    print(f"  [2] 面法线夹角 {a_g:.2e}°，晶向夹角 {a_t:.2e}°（应≈0）"
          f"  {'PASS' if max(a_g, a_t) < 1e-6 else 'FAIL'}")

    # [3] 最小间距
    d = np.linalg.norm(bi.coords[:, None, :] - bi.coords[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    print(f"  [3] 去重后最小间距 {d.min():.4f} Å（α-Fe 最近邻 2.482 Å，"
          f"界面错配属正常非共格）  {'PASS' if d.min() > 1.2 else 'FAIL'}")

    # [4] 统计
    print(f"  [4] 总原子 {len(bi.coords)}（γ {int((bi.grains == 0).sum())} + "
          f"α {int((bi.grains == 1).sum())}），界面 gap = {bi.gap:.3f} Å，"
          f"去重删 {bi.n_removed_overlap}")


if __name__ == "__main__":
    gamma = fcc("γ-Fe (FCC)", "Fe", a=3.591)
    alpha = bcc("α-Fe (BCC)", "Fe", a=2.866)
    print("=== K-S 双晶: (111)γ∥(011)α, [1̄01]γ∥[1̄1̄1]α ===")

    R = rotation_from_or_name(gamma, alpha, "K-S")
    bi = build_bicrystal(gamma, alpha, R, interface_hkl=(1, 1, 1),
                         supercell=(5, 5, 5))
    verify(gamma, alpha, R, bi)

    # ---- 2D 侧视投影：沿 [1̄01]γ（界面面内）看，界面法线 [111]γ 竖直 ----
    # x'=[12̄1]γ（界面面内另一方向）→ y' = z'×x' 恰好 ∥ [111]γ（界面法线，竖直）
    Rv = rotation_from_uvw((-1, 0, 1), in_plane=(1, -2, 1))
    centered = bi.coords - bi.coords.mean(axis=0)
    rot = centered @ Rv.T
    labels = [f"Fe ({'γ-FCC' if g == 0 else 'α-BCC'})" for g in bi.grains]
    edge_png = os.path.join(OUT_DIR, "KS_bicrystal_edge.png")
    render(rot[:, 0], rot[:, 1], rot[:, 2], labels,
           title="K-S bicrystal γ-Fe/α-Fe — view along [1-01]γ",
           out_path=edge_png,
           colors={"Fe (γ-FCC)": COLOR_P1, "Fe (α-BCC)": COLOR_P2})
    print(f"  [5] 2D 侧视图: {edge_png}")

    # ---- 3D 交互视图（相机侧视界面）----
    fig = build_figure_from_arrays(
        bi.coords, labels,
        title="K-S bicrystal γ-Fe/α-Fe — 拖拽旋转查看界面",
        colors={"Fe (γ-FCC)": COLOR_P1, "Fe (α-BCC)": COLOR_P2},
        radii={"Fe (γ-FCC)": 1.26, "Fe (α-BCC)": 1.26},
    )
    fig.update_layout(scene_camera=dict(eye=dict(x=1.6, y=-1.6, z=0.6)))
    html_path = os.path.join(OUT_DIR, "KS_bicrystal_3d.html")
    save_html(fig, html_path)
    print(f"  [6] 3D 交互视图: {html_path}")
    try:
        png3d = os.path.join(OUT_DIR, "KS_bicrystal_3d.png")
        fig.write_image(png3d, width=900, height=900, scale=2)
        print(f"  [7] 3D 验证截图: {png3d}")
    except Exception as e:
        print(f"  [7] 跳过 3D 截图（{e}）")
