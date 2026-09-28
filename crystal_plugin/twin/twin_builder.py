"""孪晶生成器：镜像法（与 Atomsk 的 mirror+merge 流程一致）。

算法（对应教程 §6.4）：
  1. 求孪晶面 K1 的卡氏法向量 n（非立方晶系用倒易点阵：n = inv(L)·(hkl)）
  2. 超胞展开，选取过原子层且最接近体心的 K1 面作为分界面
  3. 一侧（基体）保持不动，另一侧做镜像：p' = p − 2(d−c0)·n
  4. 合并后去除界面重叠原子（距离 < 0.5×最近邻距判重叠）

已知限制（按计划留待后续）：HCP 非中心对称孪晶面需改用 180° 旋转法单独验证。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.structure import Crystal


@dataclass
class TwinResult:
    """孪晶构建结果。coords 为卡氏坐标（Å），grains 标记每个原子所属晶粒。"""

    coords: np.ndarray          # (N,3) 卡氏坐标
    species: list[str]          # 长度 N 的元素符号
    grains: np.ndarray          # (N,) 0=基体 1=孪晶
    plane_normal: np.ndarray    # 孪晶面单位法向量（卡氏）
    plane_offset: float         # 孪晶面位置 c0（p·n = c0）
    n_atoms_before: int
    n_removed_overlap: int
    mirror: np.ndarray = field(repr=False)  # 3×3 镜像矩阵，det=-1


def plane_normal_cart(hkl: tuple[int, int, int], lattice_vectors: np.ndarray) -> np.ndarray:
    """晶面 (hkl) 的卡氏单位法向量。

    面法线是倒易矢量 G = h·b1+k·b2+l·b3；倒易基矢（不含 2π）为 inv(L) 的列，
    故 n ∝ inv(L) @ hkl。立方晶系退化为 hkl 本身。
    """
    n = np.linalg.inv(lattice_vectors) @ np.asarray(hkl, dtype=np.float64)
    norm = np.linalg.norm(n)
    if norm < 1e-12:
        raise ValueError(f"晶面 {hkl} 法向量长度为零")
    return n / norm


def mirror_matrix(n: np.ndarray) -> np.ndarray:
    """关于法向 n 的平面的镜像矩阵 M = I − 2·n·nᵀ（det = −1）。"""
    n = np.asarray(n, dtype=np.float64)
    return np.eye(3) - 2.0 * np.outer(n, n)


def _nn_distance(crystal: Crystal) -> float:
    """用 2×2×2 超胞成对距离求最近邻距。"""
    frac, _ = crystal.supercell(2, 2, 2)
    cart = frac @ crystal.lattice_vectors
    d = np.linalg.norm(cart[:, None, :] - cart[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return float(d.min())


def build_twin(
    crystal: Crystal,
    twin_plane: tuple[int, int, int],
    supercell: tuple[int, int, int] = (4, 4, 4),
    overlap_tol: float | None = None,
) -> TwinResult:
    """按孪晶面 K1 生成孪晶（镜像法）。

    参数
    ----
    crystal : 基体单晶
    twin_plane : 孪晶面 K1 三指数，如 FCC 的 (1,1,1)
    supercell : 超胞倍数（镜像前的基体尺寸）
    overlap_tol : 界面重叠判据（Å），默认 0.5×最近邻距

    返回
    ----
    TwinResult，grains=0 为基体、1 为孪晶
    """
    L = crystal.lattice_vectors
    n = plane_normal_cart(twin_plane, L)
    M = mirror_matrix(n)
    tol_layer = 0.1  # 判定"原子在孪晶面上"的容差（Å）

    frac, species = crystal.supercell(*supercell)
    cart = frac @ L
    d = cart @ n

    # 选过原子层且最接近体心的面作为孪晶面（保证界面原子精确重合）
    d_center = cart.mean(axis=0) @ n
    layers = np.unique(np.round(d, 6))
    c0 = float(layers[np.argmin(np.abs(layers - d_center))])

    keep = d <= c0 + tol_layer          # 基体侧（含界面层），保持不动
    source = d < c0 - tol_layer         # 基体侧（不含界面层），镜像到对面形成孪晶
    mirrored = cart[source] - 2.0 * ((d[source] - c0)[:, None]) * n[None, :]

    coords = np.vstack([cart[keep], mirrored])
    sp = [species[i] for i in np.nonzero(keep)[0]] + \
         [species[i] for i in np.nonzero(source)[0]]
    grains = np.concatenate([
        np.zeros(int(keep.sum()), dtype=int),
        np.ones(int(source.sum()), dtype=int),
    ])

    # 界面去重：距离过近的原子对，优先删孪晶侧
    tol = overlap_tol if overlap_tol is not None else 0.5 * _nn_distance(crystal)
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    np.fill_diagonal(dist, np.inf)
    drop = np.zeros(len(coords), dtype=bool)
    iu = np.triu_indices(len(coords), k=1)
    close_i, close_j = iu[0][dist[iu] < tol], iu[1][dist[iu] < tol]
    for i, j in zip(close_i, close_j):
        victim = j if grains[j] >= grains[i] else i  # 同晶粒删序号大的，跨晶粒删孪晶侧
        drop[victim] = True
    n_removed = int(drop.sum())

    return TwinResult(
        coords=coords[~drop],
        species=[s for s, dr in zip(sp, drop) if not dr],
        grains=grains[~drop],
        plane_normal=n,
        plane_offset=c0,
        n_atoms_before=len(coords),
        n_removed_overlap=n_removed,
        mirror=M,
    )
