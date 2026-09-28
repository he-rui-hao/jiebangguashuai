"""取向引擎：任意观察方向 [uvw] → 旋转矩阵。

对应接口文档 §3.2（本文件实现 rotation_from_uvw）。
约定：R 的行向量为新坐标系 x'、y'、z' 在旧坐标系中的表达，z' 恒为观察方向。
"""
from __future__ import annotations

import numpy as np


def _frac_dir_to_cart(direction: np.ndarray, lattice_vectors: np.ndarray | None) -> np.ndarray:
    """把分数（晶向指数）方向变到卡氏空间。立方晶系可传 None。"""
    if lattice_vectors is None:
        return direction.astype(np.float64)
    return direction @ lattice_vectors


def _pick_in_plane_index(
    uvw: tuple[int, int, int],
    lattice_vectors: np.ndarray | None = None,
) -> tuple[int, int, int]:
    """自动选取一个与 [uvw] 垂直的低指数晶向。

    立方晶系用整数点积即可；非立方晶系必须用度规张量 G = L·Lᵀ 判定
    （如 HCP [21̄1̄0]→三指数 (3,0,0)，其垂直晶向为 (1,2,0) 而非 (0,-3,0)）。
    """
    u, v, w = uvw
    if lattice_vectors is None:
        for cand in ((v, -u, 0), (w, 0, -u), (0, w, -v)):
            if cand != (0, 0, 0):
                return cand
        raise ValueError(f"无法为观察方向 [{u}{v}{w}] 自动选取面内晶向")
    d = np.asarray(uvw, dtype=np.float64) @ (lattice_vectors @ lattice_vectors.T)
    best: tuple[int, tuple[int, int, int]] | None = None
    for h in range(-3, 4):
        for k in range(-3, 4):
            for l in range(-3, 4):
                if (h, k, l) == (0, 0, 0):
                    continue
                if abs(float(d @ np.array([h, k, l]))) > 1e-10:
                    continue
                key = h * h + k * k + l * l
                if best is None or key < best[0]:
                    best = (key, (h, k, l))
    if best is None:
        raise ValueError(f"无法为观察方向 [{u}{v}{w}] 自动选取面内晶向")
    return best[1]


def rotation_from_uvw(
    uvw: tuple[int, int, int],
    lattice_vectors: np.ndarray | None = None,
    in_plane: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """构造旋转矩阵 R：z' 沿观察方向 [uvw]，x' 沿面内晶向 in_plane。

    参数
    ----
    uvw : 观察方向三指数，如 (1, 1, 0)
    lattice_vectors : 非立方晶系必传，用于分数方向→卡氏方向；立方晶系传 None
    in_plane : 面内 x' 晶向三指数，需与 uvw 垂直；None 时自动选取

    返回
    ----
    R : (3,3) 正交矩阵，det(R) = +1；使用方式 cart_new = cart_old @ R.T
    """
    uvw_arr = np.asarray(uvw, dtype=np.float64)
    if np.allclose(uvw_arr, 0):
        raise ValueError("观察方向 [uvw] 不能为零向量")

    ez = _frac_dir_to_cart(uvw_arr, lattice_vectors)
    ez /= np.linalg.norm(ez)

    if in_plane is None:
        in_plane = _pick_in_plane_index(uvw, lattice_vectors)
    in_plane_arr = np.asarray(in_plane, dtype=np.float64)

    # 垂直性检查（在分数空间用度规张量 G = L·Lᵀ 计算点积）
    if lattice_vectors is None:
        dot = float(uvw_arr @ in_plane_arr)
    else:
        G = lattice_vectors @ lattice_vectors.T
        dot = float(uvw_arr @ G @ in_plane_arr)
    if abs(dot) > 1e-10:
        raise ValueError(f"面内晶向 {in_plane} 与观察方向 {uvw} 不垂直")

    ex = _frac_dir_to_cart(in_plane_arr, lattice_vectors)
    ex /= np.linalg.norm(ex)

    ey = np.cross(ez, ex)
    ey /= np.linalg.norm(ey)
    # 重新正交化，消除数值误差
    ex = np.cross(ey, ez)
    ex /= np.linalg.norm(ex)

    R = np.vstack([ex, ey, ez])
    if np.linalg.det(R) < 0:
        R[1] = -R[1]  # 保证右手系
    return R


# ----------------------------------------------------------------------
# HCP 四指数 ↔ 三指数转换（GUI 层调用，内核只处理三指数）
# ----------------------------------------------------------------------

def four_to_three_dir(uvtw: tuple[int, int, int, int]) -> tuple[int, int, int]:
    """四指数晶向 [uvtw] → 三指数 [U V W]：U=2u+v, V=u+2v, W=w。

    等价于 U=u−t, V=v−t（利用 t=−(u+v)）。结果可能未约化（如 [21̄1̄0]→[300]），
    不影响任何方向计算；显示时可自行除以最大公约数。
    """
    u, v, t, w = uvtw
    if u + v + t != 0:
        raise ValueError(f"四指数晶向 [{u} {v} {t} {w}] 不满足 u+v+t=0")
    return (2 * u + v, u + 2 * v, w)


def four_to_three_plane(hkil: tuple[int, int, int, int]) -> tuple[int, int, int]:
    """四指数晶面 (hkil) → 三指数 (h k l)：直接去掉 i（需 h+k+i=0）。"""
    h, k, i, l = hkil
    if h + k + i != 0:
        raise ValueError(f"四指数晶面 ({h} {k} {i} {l}) 不满足 h+k+i=0")
    return (h, k, l)
