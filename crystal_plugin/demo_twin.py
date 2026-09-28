"""演示脚本：FCC Cu {111} 孪晶生成与验证。

验证点：
  [1] 镜像矩阵 det(M) = -1
  [2] 合并后最小原子间距 ≥ 0.9×最近邻距（无重叠原子）
  [3] 镜像对称性：孪晶侧原子镜像回去能在基体侧找到对应原子
  [4] 界面层原子只属于基体侧一次（不多不少）

输出：
  output/FCC_Cu_twin_edge.png  —— 沿 [1-10] 侧视 2D 投影（孪晶面直立，镜像关系最直观）
  output/FCC_Cu_twin_3d.html   —— 3D 交互视图（基体/孪晶双色）
"""
from __future__ import annotations

import os

import numpy as np

from core.orientation import rotation_from_uvw
from core.projector import render
from core.structure import fcc
from core.viewer3d import build_figure_from_arrays, save_html
from twin.twin_builder import build_twin

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT_DIR, exist_ok=True)

COLOR_MATRIX = "#c87533"   # 基体：铜本色
COLOR_TWIN = "#4a90d9"     # 孪晶：钢蓝


def verify(twin, cu) -> None:
    # [1] det(M) = -1
    det = np.linalg.det(twin.mirror)
    print(f"  [1] det(镜像矩阵) = {det:.6f}  {'PASS' if abs(det + 1) < 1e-9 else 'FAIL'}")

    # [2] 合并后最小间距
    d = np.linalg.norm(twin.coords[:, None, :] - twin.coords[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    nn = cu.meta["lattice_params"]["a"] / np.sqrt(2)
    dmin = d.min()
    print(f"  [2] 合并后最小间距 {dmin:.4f} Å（最近邻 {nn:.4f} Å）"
          f"  {'PASS' if dmin > 0.9 * nn else 'FAIL'}")

    # [3] 镜像对称性：孪晶侧原子镜像回基体侧，应在基体侧找到对应原子
    n, c0 = twin.plane_normal, twin.plane_offset
    d_signed = twin.coords @ n - c0
    twin_side = twin.coords[(twin.grains == 1) & (d_signed > 0.2)]
    back = twin_side - 2.0 * ((twin_side @ n - c0)[:, None]) * n[None, :]
    matrix_side = twin.coords[twin.grains == 0]
    dist = np.linalg.norm(back[:, None, :] - matrix_side[None, :, :], axis=-1)
    match = (dist.min(axis=1) < 0.05).mean()
    print(f"  [3] 镜像对称匹配率 {match * 100:.1f}%（孪晶侧→基体侧）"
          f"  {'PASS' if match > 0.98 else 'FAIL'}")

    # [4] 界面层原子统计
    on_plane = np.abs(d_signed) < 0.1
    n_plane = int(on_plane.sum())
    n_plane_g1 = int((on_plane & (twin.grains == 1)).sum())
    ok = n_plane > 0 and n_plane_g1 == 0
    print(f"  [4] 界面层原子 {n_plane} 个，其中孪晶侧 {n_plane_g1} 个（应为 0）"
          f"  {'PASS' if ok else 'FAIL'}")
    print(f"  [5] 总原子 {len(twin.coords)}（构建 {twin.n_atoms_before}，去重删 {twin.n_removed_overlap}）")


if __name__ == "__main__":
    cu = fcc("FCC Cu", "Cu", a=3.615)
    print("=== FCC Cu (111) 孪晶 ===")
    twin = build_twin(cu, twin_plane=(1, 1, 1), supercell=(5, 5, 5))
    verify(twin, cu)

    # ---- 2D 侧视投影：沿 [1-10] 看，孪晶面 (111) 直立 ----
    R = rotation_from_uvw((1, -1, 0), in_plane=(1, 1, -2))
    centered = twin.coords - twin.coords.mean(axis=0)
    rot = centered @ R.T
    labels = [f"Cu ({'基体' if g == 0 else '孪晶'})" for g in twin.grains]
    edge_png = os.path.join(OUT_DIR, "FCC_Cu_twin_edge.png")
    render(rot[:, 0], rot[:, 1], rot[:, 2], labels,
           title="FCC Cu (111) twin — view along [1-10]",
           out_path=edge_png,
           colors={"Cu (基体)": COLOR_MATRIX, "Cu (孪晶)": COLOR_TWIN})
    print(f"  [6] 2D 侧视图: {edge_png}")

    # ---- 3D 交互视图 ----
    fig = build_figure_from_arrays(
        twin.coords, labels,
        title="FCC Cu (111) twin — 拖拽旋转查看镜像界面",
        colors={"Cu (基体)": COLOR_MATRIX, "Cu (孪晶)": COLOR_TWIN},
    )
    # 默认相机设为侧视孪晶面（眼位 ⊥ 面法线 (111)），两晶粒左右分明
    fig.update_layout(scene_camera=dict(eye=dict(x=1.6, y=-1.6, z=0.6)))
    html_path = os.path.join(OUT_DIR, "FCC_Cu_twin_3d.html")
    save_html(fig, html_path)
    print(f"  [7] 3D 交互视图: {html_path}")
    try:
        png3d = os.path.join(OUT_DIR, "FCC_Cu_twin_3d.png")
        fig.write_image(png3d, width=900, height=900, scale=2)
        print(f"  [8] 3D 验证截图: {png3d}")
    except Exception as e:
        print(f"  [8] 跳过 3D 截图（{e}）")
