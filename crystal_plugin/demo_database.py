"""演示脚本：晶体结构数据库读取 + 多结构类型验证。

验证点：
  [1] 全部材料可加载，基元/元素列表自洽
  [2] 各结构类型的最近邻距与理论公式一致（Po: a；金刚石: √3a/4；
      NaCl/CsCl/闪锌矿/L1₂ 均有解析答案）
  [3] Si [110] 投影（金刚石结构的"通道"特征）+ NaCl [001] 投影
  [4] Si 3D 视图叠加 (111) 晶面标注
"""
from __future__ import annotations

import os

import numpy as np

from core.orientation import rotation_from_uvw
from core.projector import project, render
from core.viewer3d import add_hkl_plane, build_figure, save_html
from crystio.database import list_materials, load_material

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT_DIR, exist_ok=True)


def nn_distance(crystal) -> float:
    frac, _ = crystal.supercell(2, 2, 2)
    cart = frac @ crystal.lattice_vectors
    d = np.linalg.norm(cart[:, None, :] - cart[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return float(d.min())


# 各结构类型的最近邻距理论公式（a 的函数）
NN_CASES = {
    "Po":   lambda a, c: a,                      # 简单立方
    "Si":   lambda a, c: np.sqrt(3) / 4 * a,     # 金刚石
    "Ge":   lambda a, c: np.sqrt(3) / 4 * a,
    "NaCl": lambda a, c: a / 2,                  # 岩盐 A-B
    "MgO":  lambda a, c: a / 2,
    "CsCl": lambda a, c: np.sqrt(3) / 2 * a,     # A-B 体对角一半
    "NiAl": lambda a, c: np.sqrt(3) / 2 * a,
    "ZnS":  lambda a, c: np.sqrt(3) / 4 * a,     # 闪锌矿 A-B
    "GaAs": lambda a, c: np.sqrt(3) / 4 * a,
    "Cu3Au": lambda a, c: a / np.sqrt(2),        # L1₂ 面心-顶点
    "Ni3Al": lambda a, c: a / np.sqrt(2),
}

if __name__ == "__main__":
    print("=== 数据库总览 ===")
    catalog = list_materials()
    print(f"  材料总数: {len(catalog)}")
    for k, v in catalog.items():
        print(f"    {k:10s} {v}")

    print("\n=== [1] 全量加载自洽性 ===")
    ok = True
    for key in catalog:
        x = load_material(key)
        if not (len(x.species) == len(x.basis) and x.basis.shape[1] == 3
                and np.all(x.basis >= 0) and np.all(x.basis < 1)):
            print(f"    {key}: FAIL（基元/元素列表不自洽）")
            ok = False
    print(f"    {len(catalog)} 种材料全部自洽  {'PASS' if ok else 'FAIL'}")

    print("\n=== [2] 最近邻距 vs 理论公式 ===")
    all_pass = True
    for key, formula in NN_CASES.items():
        x = load_material(key)
        a = x.meta["lattice_params"]["a"]
        c = x.meta["lattice_params"].get("c")
        got, want = nn_distance(x), formula(a, c)
        passed = abs(got - want) < 1e-3
        all_pass &= passed
        print(f"    {key:6s} 计算 {got:.4f} Å / 理论 {want:.4f} Å"
              f"  {'PASS' if passed else 'FAIL'}")
    print(f"    汇总  {'PASS' if all_pass else 'FAIL'}")

    print("\n=== [3] 2D 投影 ===")
    si = load_material("Si")
    R = rotation_from_uvw((1, -1, 0), si.lattice_vectors)
    x, y, z, sp = project(si, R, (3, 3, 3))
    p1 = os.path.join(OUT_DIR, "Si_diamond_110.png")
    render(x, y, z, sp, "Si (diamond) [1-10] - open channels", p1)
    print(f"    Si [110]: {p1}")

    nacl = load_material("NaCl")
    R2 = rotation_from_uvw((0, 0, 1), nacl.lattice_vectors)
    x2, y2, z2, sp2 = project(nacl, R2, (3, 3, 3))
    p2 = os.path.join(OUT_DIR, "NaCl_001.png")
    render(x2, y2, z2, sp2, "NaCl [001] (purple=Na, green=Cl)", p2)
    print(f"    NaCl [001]: {p2}")

    print("\n=== [4] 3D 晶面标注 ===")
    fig = build_figure(si, supercell=(2, 2, 2), title="Si 金刚石 + (111) 晶面标注")
    add_hkl_plane(fig, si, (1, 1, 1))
    p3 = os.path.join(OUT_DIR, "Si_3d_plane.html")
    save_html(fig, p3)
    print(f"    {p3}")
    try:
        p4 = os.path.join(OUT_DIR, "Si_3d_plane.png")
        fig.write_image(p4, width=900, height=900, scale=2)
        print(f"    {p4}")
    except Exception as e:
        print(f"    跳过截图（{e}）")
