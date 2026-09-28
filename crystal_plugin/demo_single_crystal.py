"""演示脚本：单晶任意观察方向原子排布图生成与验证。

验证用参数（教科书标准晶格常数，单位 Å）：
    FCC Cu : a = 3.615
    BCC Fe : a = 2.866 (α-Fe)
    HCP Mg : a = 3.209, c = 5.211

运行：python demo_single_crystal.py
输出：output/ 目录下 4 张投影图 + 终端自动验证报告
"""
from __future__ import annotations

import os

import numpy as np

from core.orientation import rotation_from_uvw
from core.projector import project, render
from core.structure import bcc, fcc, hcp

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT_DIR, exist_ok=True)

# 最近邻距离理论值（Å），用于自动验证
NN_THEORY = {
    "FCC": lambda a: a / np.sqrt(2.0),
    "BCC": lambda a: a * np.sqrt(3.0) / 2.0,
    "HCP": lambda a, c: min(a, np.sqrt(a**2 / 3.0 + c**2 / 4.0)),
}


def check_rotation(R: np.ndarray, uvw, lattice_vectors=None) -> list[str]:
    """验证旋转矩阵的数学正确性，返回验证日志。"""
    logs = []
    # 1) 正交性：R·Rᵀ = I
    ortho_err = np.abs(R @ R.T - np.eye(3)).max()
    logs.append(f"  [1] 正交性 |R·Rᵀ-I|max = {ortho_err:.2e}  {'PASS' if ortho_err < 1e-12 else 'FAIL'}")
    # 2) 右手系：det(R) = +1
    det = np.linalg.det(R)
    logs.append(f"  [2] det(R) = {det:.12f}  {'PASS' if abs(det - 1) < 1e-12 else 'FAIL'}")
    # 3) 观察方向被转到 z 轴：R·d = (0, 0, |d|)
    d = np.asarray(uvw, dtype=np.float64)
    if lattice_vectors is not None:
        d = d @ lattice_vectors
    d /= np.linalg.norm(d)
    rotated = R @ d
    ok = abs(rotated[0]) < 1e-12 and abs(rotated[1]) < 1e-12 and rotated[2] > 0.999999
    logs.append(f"  [3] R·[{uvw[0]}{uvw[1]}{uvw[2]}] → ({rotated[0]:.2e}, {rotated[1]:.2e}, {rotated[2]:.6f})  {'PASS' if ok else 'FAIL'}")
    return logs


def check_nn(crystal, supercell=(3, 3, 3)) -> str:
    """验证超胞最近邻距离与理论值一致。"""
    frac, _ = crystal.supercell(*supercell)
    cart = frac @ crystal.lattice_vectors
    diffs = cart[:, None, :] - cart[None, :, :]
    d = np.linalg.norm(diffs, axis=-1)
    np.fill_diagonal(d, np.inf)
    nn = d.min()
    lt = crystal.meta["lattice_type"]
    p = crystal.meta["lattice_params"]
    theory = NN_THEORY[lt](**p) if lt == "HCP" else NN_THEORY[lt](p["a"])
    ok = abs(nn - theory) < 1e-6
    return f"  [4] 最近邻距离 计算 {nn:.4f} Å / 理论 {theory:.4f} Å  {'PASS' if ok else 'FAIL'}"


def run_case(crystal, uvw, supercell, in_plane=None, lattice_vectors=None):
    label = f"{crystal.name} [{uvw[0]}{uvw[1]}{uvw[2]}]"
    print(f"\n=== {label} ===")
    R = rotation_from_uvw(uvw, lattice_vectors=lattice_vectors, in_plane=in_plane)
    for line in check_rotation(R, uvw, lattice_vectors):
        print(line)
    print(check_nn(crystal))
    x, y, z, sp = project(crystal, R, supercell=supercell)
    n_layers = len(np.unique(np.round(z / 0.5)))
    print(f"  [5] 投影原子数 {len(x)}，深度层数 {n_layers}")
    out = os.path.join(OUT_DIR, f"{crystal.name.replace(' ', '_')}_{uvw[0]}{uvw[1]}{uvw[2]}.png")
    render(x, y, z, sp, title=label, out_path=out)
    print(f"  [6] 已保存: {out}")


if __name__ == "__main__":
    # 案例1：FCC Cu 沿 [110] 观察（经典案例，矩形对称花样）
    cu = fcc("FCC Cu", "Cu", a=3.615)
    run_case(cu, uvw=(1, 1, 0), supercell=(4, 4, 4))

    # 案例2：FCC Cu 沿 [111] 观察（应呈六重对称密排面投影）
    run_case(cu, uvw=(1, 1, 1), supercell=(4, 4, 4))

    # 案例3：BCC α-Fe 沿 [111] 观察
    fe = bcc("BCC Fe", "Fe", a=2.866)
    run_case(fe, uvw=(1, 1, 1), supercell=(4, 4, 4))

    # 案例4：HCP Mg 沿 [0001] 观察（六重对称，非立方晶系需传晶格矢量）
    mg = hcp("HCP Mg", "Mg", a=3.209, c=5.211)
    run_case(mg, uvw=(0, 0, 1), supercell=(3, 3, 3),
             lattice_vectors=mg.lattice_vectors)

    print("\n全部案例执行完毕。")
