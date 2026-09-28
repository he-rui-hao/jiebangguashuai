"""科研增强功能验证：XRD / 极射投影 / 多方向联排 / 测量 / HCP 四指数转换。

验证依据：
[1] XRD（Cu Kα, λ=1.5406 Å）：FCC Cu 标准卡片峰位
    (111) 43.30° / (200) 50.43° / (220) 74.13°；(100) 系统消光不得出现
[2] 极射投影：[001] 轴下 (100) 极点在投影圆上（r=1）；
    (111) 与 (001) 夹角 54.7356° → r = tan(θ/2) = 0.51764（保角性的直接体现）
[3] 多方向联排：4 方向文件正常生成
[4] 测量：FCC 角-面心距 a/√2；直角四面体键角 109.47°
[5] 四指数转换：[21̄1̄0]→[300]≡[100]，[101̄0]→[210]，(101̄0)→(100)；非法输入抛错
"""
from __future__ import annotations

import os

import numpy as np

from core.diffraction import render_xrd, simulate_powder_xrd
from core.measure import angle, distance, nearest_index
from core.orientation import four_to_three_dir, four_to_three_plane
from core.projector import render_multi
from core.stereographic import render_stereographic, stereographic_poles
from crystio.database import load_material

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

results = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"  [{name}] {'PASS' if ok else 'FAIL'}  {detail}")


cu = load_material("Cu")

# ---- [1] XRD ----
print("=== [1] 粉末 XRD（FCC Cu, Cu Kα）===")
xrd = simulate_powder_xrd(cu, wavelength=1.5406, two_theta_range=(10, 120))
expected = {43.30: (1, 1, 1), 50.43: (2, 0, 0), 74.13: (2, 2, 0)}
all_ok = True
for exp_tt, exp_hkl in expected.items():
    i = int(np.argmin(np.abs(xrd["two_theta"] - exp_tt)))
    got = xrd["two_theta"][i]
    ok = abs(got - exp_tt) < 0.15
    all_ok &= ok
    print(f"    峰 {exp_hkl}: 理论 {exp_tt}° / 计算 {got:.2f}°  {'OK' if ok else 'X'}")
# (100) 消光：43.3° 以下不得有峰
check("XRD 峰位", all_ok)
check("(100) 系统消光", (xrd["two_theta"] > 43.0).all(),
      f"最低峰 {xrd['two_theta'].min():.2f}°")
p = os.path.join(OUT_DIR, "FCC_Cu_XRD.png")
render_xrd(xrd, "FCC Cu powder XRD (Cu Kα)", p)
print(f"    图: {p}")

# ---- [2] 极射投影 ----
print("=== [2] 极射赤面投影（[001] 轴）===")
poles = stereographic_poles(cu, [(0, 0, 1), (1, 0, 0), (1, 1, 1), (1, 1, 0),
                                 (1, 1, -1), (0, 1, 0)], uvw=(0, 0, 1))
r = np.hypot(poles["px"], poles["py"])
# (001) 在圆心
check("(001) 在圆心", r[0] < 1e-12, f"r={r[0]:.2e}")
# (100) 在圆上
check("(100) 在投影圆上", abs(r[1] - 1.0) < 1e-12, f"r={r[1]:.6f}")
# (111): r = tan(54.7356°/2)
theta = np.degrees(np.arccos(1 / np.sqrt(3)))
r_expect = np.tan(np.radians(theta / 2))
check("(111) 半径保角", abs(r[2] - r_expect) < 1e-10,
      f"r={r[2]:.6f} / 理论 {r_expect:.6f}")
# (11-1) 下半球 → 空心标记
check("(11-1) 下半球标记", not poles["upper"][4])
p2 = os.path.join(OUT_DIR, "FCC_Cu_stereo_001.png")
render_stereographic(poles, "FCC Cu stereographic, [001] axis", p2)
print(f"    图: {p2}")

# ---- [3] 多方向联排 ----
print("=== [3] 多方向 2×2 联排 ===")
views = [((1, 0, 0), None), ((1, 1, 0), None), ((1, 1, 1), None), ((1, 1, 2), None)]
p3 = os.path.join(OUT_DIR, "FCC_Cu_multi.png")
render_multi(cu, views, (3, 3, 3), p3)
check("联排图生成", os.path.exists(p3) and os.path.getsize(p3) > 10000,
      f"{os.path.getsize(p3)} bytes")

# ---- [4] 测量 ----
print("=== [4] 测量工具 ===")
a = cu.meta["lattice_params"]["a"]
d = distance([0, 0, 0], [a / 2, a / 2, 0])
check("FCC 角-面心距", abs(d - a / np.sqrt(2)) < 1e-10, f"{d:.4f} Å / 理论 {a/np.sqrt(2):.4f} Å")
# 四面体键角：中心 (1/4,1/4,1/4) 与两顶点
ang = angle([0, 0, 0], [0.25, 0.25, 0.25], [0.5, 0.5, 0])
check("四面体键角", abs(ang - 109.471) < 0.01, f"{ang:.3f}° / 理论 109.471°")
idx = nearest_index(np.array([0.0, 1.0, 5.0]), np.array([0.0, 1.0, 5.0]), 0.9, 1.1)
check("最近原子拾取", idx == 1, f"index={idx}")

# ---- [5] HCP 四指数转换 ----
print("=== [5] HCP 四指数 → 三指数 ===")
check("[21̄1̄0]→[300]", four_to_three_dir((2, -1, -1, 0)) == (3, 0, 0))
check("[101̄0]→[210]", four_to_three_dir((1, 0, -1, 0)) == (2, 1, 0))
check("(101̄0)→(100)", four_to_three_plane((1, 0, -1, 0)) == (1, 0, 0))
try:
    four_to_three_dir((1, 1, 0, 0))
    check("非法四指数抛错", False, "未抛错")
except ValueError:
    check("非法四指数抛错", True)

print("\n=== 汇总 ===")
n_pass = sum(1 for _, ok, _ in results if ok)
print(f"{n_pass}/{len(results)} 通过")
if n_pass != len(results):
    raise SystemExit(1)
