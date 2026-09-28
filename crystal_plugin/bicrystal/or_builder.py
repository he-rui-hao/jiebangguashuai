"""双晶 OR 生成器：依据取向关系构建两相双晶。

算法（对应教程 §5，全项目数学最难模块）：
  1. OR "(hkl)₁∥(hkl)₂, [uvw]₁∥[uvw]₂" → 旋转矩阵 R：
     晶向卡氏方向 t = uvw @ L；面法线卡氏方向 g = inv(L)·(hkl)（倒易点阵，
     非立方晶系绝不能直接用 hkl 当法线——本模块最大的坑）
  2. 双晶拼接：相1 保持不动，相2 旋转 R 后平移，使其原子层落在界面上方
     gap 处（默认 gap = 两相最近邻距的平均，界面原子物理上不重叠）
  3. 界面去重：跨晶粒过近原子对优先删相2 侧

保底说明：OR 预设表存的是"平行面对+平行向对"，R 现场换算，精度 1e-15；
暂不做 CSL 超胞搜索（计划中的降级方案），界面为容忍错配的非共格界面。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.structure import Crystal
from twin.twin_builder import plane_normal_cart

# ---------------------------------------------------------------- OR 预设表
# 格式: 名称 -> (hkl1, uvw1, hkl2, uvw2, 相1类型, 相2类型, 说明)
OR_PRESETS: dict[str, tuple] = {
    # Kurdjumov–Sachs: 钢中 γ(奥氏体)→α(马氏体/铁素体)，24 变体
    "K-S": ((1, 1, 1), (-1, 0, 1), (0, 1, 1), (-1, -1, 1),
            "FCC", "BCC", "(111)γ∥(011)α, [1̄01]γ∥[1̄1̄1]α"),
    # Nishiyama–Wassermann: 与 K-S 相差 5.26°，12 变体
    "N-W": ((1, 1, 1), (1, 1, -2), (0, 1, 1), (0, -1, 1),
            "FCC", "BCC", "(111)γ∥(011)α, [112̄]γ∥[01̄1]α"),
    # Burgers: 钛/锆中 β(BCC)→α(HCP)，12 变体；HCP 用三指数
    "Burgers": ((1, 1, 0), (1, -1, 1), (0, 0, 1), (1, 1, 0),
                "BCC", "HCP", "(110)β∥(0001)α, [11̄1]β∥[112̄0]α(三指数[110])"),
}


def or_to_rotation(
    xtal1: Crystal,
    xtal2: Crystal,
    hkl1: tuple[int, int, int],
    uvw1: tuple[int, int, int],
    hkl2: tuple[int, int, int],
    uvw2: tuple[int, int, int],
) -> np.ndarray:
    """取向关系 → 3×3 旋转矩阵 R，使相2 的 (hkl)₂∥(hkl)₁、[uvw]₂∥[uvw]₁。

    用法：cart2_new = cart2_old @ R.T。要求 uvw 位于 (hkl) 面内（晶带定律）。
    """
    def frame(xtal, hkl, uvw):
        g = plane_normal_cart(hkl, xtal.lattice_vectors)       # 面法线（倒易点阵）
        t = np.asarray(uvw, dtype=np.float64) @ xtal.lattice_vectors  # 晶向卡氏方向
        if abs(g @ t) > 1e-6:
            raise ValueError(
                f"[{uvw}] 不在 ({hkl}) 面内（违反晶带定律 g·t≈{g @ t:.4f}），"
                "请检查取向关系指数")
        e1 = t / np.linalg.norm(t)
        e2 = g - (g @ e1) * e1
        e2 /= np.linalg.norm(e2)
        return np.column_stack([e1, e2, np.cross(e1, e2)])

    F1 = frame(xtal1, hkl1, uvw1)
    F2 = frame(xtal2, hkl2, uvw2)
    return F1 @ F2.T


def rotation_from_or_name(xtal1: Crystal, xtal2: Crystal, name: str) -> np.ndarray:
    """按预设名称（"K-S" / "N-W" / "Burgers"）直接给旋转矩阵。"""
    if name not in OR_PRESETS:
        raise KeyError(f"未知取向关系 {name!r}，可选：{list(OR_PRESETS)}")
    hkl1, uvw1, hkl2, uvw2, *_ = OR_PRESETS[name]
    return or_to_rotation(xtal1, xtal2, hkl1, uvw1, hkl2, uvw2)


# ---------------------------------------------------------------- 双晶构建
@dataclass
class BicrystalResult:
    """双晶构建结果，字段与 TwinResult 对齐（渲染函数通用）。"""

    coords: np.ndarray          # (N,3) 卡氏坐标
    species: list[str]
    grains: np.ndarray          # (N,) 0=相1 1=相2
    plane_normal: np.ndarray    # 界面单位法向量（卡氏）
    plane_offset: float
    n_atoms_before: int
    n_removed_overlap: int
    R_or: np.ndarray = field(repr=False)   # 取向关系旋转矩阵
    gap: float = 0.0                       # 界面间距


def _nn_distance(crystal: Crystal) -> float:
    frac, _ = crystal.supercell(2, 2, 2)
    cart = frac @ crystal.lattice_vectors
    d = np.linalg.norm(cart[:, None, :] - cart[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return float(d.min())


def build_bicrystal(
    phase1: Crystal,
    phase2: Crystal,
    R_or: np.ndarray,
    interface_hkl: tuple[int, int, int],
    supercell: tuple[int, int, int] = (4, 4, 4),
    gap: float | None = None,
    overlap_tol: float | None = None,
) -> BicrystalResult:
    """按取向关系拼接两相双晶。

    参数
    ----
    phase1 / phase2 : 两个相（界面法线在相1 中以 interface_hkl 给出）
    R_or : or_to_rotation() 的输出
    interface_hkl : 界面在相1 中的晶面指数（OR 中的平行面 (hkl)₁）
    gap : 界面两侧最近原子层间距（Å），默认 0.5×(两相最近邻距之和)
    overlap_tol : 界面重叠判据（Å），默认 0.8×较小最近邻距（只影响跨界面原子对）
    """
    tol_layer = 0.1
    L1, L2 = phase1.lattice_vectors, phase2.lattice_vectors
    n = plane_normal_cart(interface_hkl, L1)

    # ---- 相1：界面之下，保持不动 ----
    frac1, sp1 = phase1.supercell(*supercell)
    c1 = frac1 @ L1
    d1 = c1 @ n
    d1c = c1.mean(axis=0) @ n
    layers1 = np.unique(np.round(d1, 6))
    c0 = float(layers1[np.argmin(np.abs(layers1 - d1c))])
    keep1 = d1 <= c0 + tol_layer

    # ---- 相2：旋转到 OR 取向，平移到界面之上 ----
    frac2, sp2 = phase2.supercell(*supercell)
    c2 = (frac2 @ L2) @ R_or.T
    d2 = c2 @ n
    d2c = c2.mean(axis=0) @ n
    layers2 = np.unique(np.round(d2, 6))
    layer0 = float(layers2[np.argmin(np.abs(layers2 - d2c))])
    if gap is None:
        gap = 0.5 * (_nn_distance(phase1) + _nn_distance(phase2))
    shift = (c0 + gap) - layer0          # 相2 中间层 → 界面上方 gap 处
    c2 = c2 + shift * n[None, :]
    d2 = d2 + shift
    keep2 = d2 > c0 + tol_layer

    coords = np.vstack([c1[keep1], c2[keep2]])
    sp = [sp1[i] for i in np.nonzero(keep1)[0]] + \
         [sp2[i] for i in np.nonzero(keep2)[0]]
    grains = np.concatenate([
        np.zeros(int(keep1.sum()), dtype=int),
        np.ones(int(keep2.sum()), dtype=int),
    ])

    # ---- 界面去重：跨晶粒过近优先删相2 ----
    # 默认判据 0.8×较小最近邻距：晶粒内部原子间距均 ≥ 最近邻距，
    # 只有跨界面原子对才可能落入该范围，不会误删晶内原子
    tol = overlap_tol if overlap_tol is not None else \
        0.8 * min(_nn_distance(phase1), _nn_distance(phase2))
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    np.fill_diagonal(dist, np.inf)
    drop = np.zeros(len(coords), dtype=bool)
    iu = np.triu_indices(len(coords), k=1)
    close_i, close_j = iu[0][dist[iu] < tol], iu[1][dist[iu] < tol]
    for i, j in zip(close_i, close_j):
        victim = j if grains[j] >= grains[i] else i
        drop[victim] = True
    n_removed = int(drop.sum())

    return BicrystalResult(
        coords=coords[~drop],
        species=[s for s, dr in zip(sp, drop) if not dr],
        grains=grains[~drop],
        plane_normal=n,
        plane_offset=c0,
        n_atoms_before=len(coords),
        n_removed_overlap=n_removed,
        R_or=R_or,
        gap=gap,
    )
