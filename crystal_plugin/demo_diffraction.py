"""演示脚本：电子衍射斑点模拟（SAED）。

验证点（FCC Cu [110] 晶带轴，经典教科书案例）：
  [1] 允许斑点存在且强：(1̄11)、(2̄20)、(002)
  [2] 系统消光自动出现：(001)、(1̄10) 不出现在结果中（FCC 奇偶混杂消光）
  [3] 晶面间距正确：d(111) = a/√3
输出：output/FCC_Cu_SAED_110.png
"""
from __future__ import annotations

import os

import numpy as np

from core.diffraction import render_diffraction, simulate_diffraction
from crystio.database import load_material

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT_DIR, exist_ok=True)

if __name__ == "__main__":
    cu = load_material("Cu")
    a = cu.meta["lattice_params"]["a"]
    print("=== FCC Cu [110] 晶带轴衍射 ===")
    pat = simulate_diffraction(cu, (1, 1, 0), in_plane=(0, 0, 1), max_index=3)
    print(f"  斑点总数（不含透射斑）: {len(pat['hkls'])}")

    hkl_set = set(pat["hkls"])
    inten_map = dict(zip(pat["hkls"], pat["intensity"]))

    present = [(-1, 1, 1), (1, -1, 1), (2, -2, 0), (0, 0, 2)]
    extinct = [(0, 0, 1), (1, -1, 0), (1, 0, 0), (2, -1, 0)]
    ok1 = all(h in hkl_set and inten_map[h] > 0.9 for h in present)
    ok2 = all(h not in hkl_set for h in extinct)
    print(f"  [1] 允许斑点 {present} 全部存在且强度高  {'PASS' if ok1 else 'FAIL'}")
    print(f"  [2] 消光斑点 {extinct} 全部缺席（FCC 系统消光自动复现）"
          f"  {'PASS' if ok2 else 'FAIL'}")

    d_map = dict(zip(pat["hkls"], pat["d_spacings"]))
    d111 = d_map[(-1, 1, 1)]
    want = a / np.sqrt(3)
    print(f"  [3] d(111) = {d111:.4f} Å / 理论 {want:.4f} Å"
          f"  {'PASS' if abs(d111 - want) < 1e-6 else 'FAIL'}")

    out = os.path.join(OUT_DIR, "FCC_Cu_SAED_110.png")
    render_diffraction(pat, title="FCC Cu — [110] zone axis SAED", out_path=out)
    print(f"  [4] 衍射图: {out}")
