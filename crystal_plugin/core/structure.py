"""晶体结构内部表示与标准结构工厂函数。

对应接口文档 §2、§3.1。单位一律为埃（Å）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Crystal:
    """全组统一的内部结构对象（见 docs/接口文档.md §2）。"""

    name: str
    lattice_vectors: np.ndarray  # (3,3) 行向量 a1,a2,a3，单位 Å
    basis: np.ndarray            # (N,3) 分数坐标
    species: list[str]
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.lattice_vectors = np.asarray(self.lattice_vectors, dtype=np.float64)
        self.basis = np.asarray(self.basis, dtype=np.float64)
        if self.lattice_vectors.shape != (3, 3):
            raise ValueError("晶格矢量必须为 3×3 矩阵")
        if self.basis.ndim != 2 or self.basis.shape[1] != 3:
            raise ValueError("基元坐标必须为 N×3 数组")
        if len(self.species) != self.basis.shape[0]:
            raise ValueError("species 长度必须与基元原子数一致")

    def cartesian(self, frac: np.ndarray) -> np.ndarray:
        """分数坐标 → 卡氏坐标。"""
        return frac @ self.lattice_vectors

    def supercell(self, nx: int, ny: int, nz: int):
        """超胞展开，返回 (frac_coords, species_list)。"""
        frac_list, sp_list = [], []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    shift = np.array([i, j, k], dtype=np.float64)
                    frac_list.append(self.basis + shift)
                    sp_list.extend(self.species)
        frac = np.vstack(frac_list)
        # 归一到超胞的分数坐标（相对原单胞基矢的扩大坐标，直接乘原基矢即可）
        return frac, sp_list


def fcc(name: str, element: str, a: float) -> Crystal:
    """面心立方工厂函数，a 为晶格常数（Å）。"""
    basis = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ])
    return Crystal(
        name=name,
        lattice_vectors=np.eye(3) * a,
        basis=basis,
        species=[element] * 4,
        meta={"lattice_type": "FCC", "lattice_params": {"a": a}},
    )


def bcc(name: str, element: str, a: float) -> Crystal:
    """体心立方工厂函数，a 为晶格常数（Å）。"""
    basis = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
    ])
    return Crystal(
        name=name,
        lattice_vectors=np.eye(3) * a,
        basis=basis,
        species=[element] * 2,
        meta={"lattice_type": "BCC", "lattice_params": {"a": a}},
    )


def hcp(name: str, element: str, a: float, c: float) -> Crystal:
    """密排六方工厂函数（三指数惯用胞），a、c 为晶格常数（Å）。"""
    lattice = np.array([
        [a, 0.0, 0.0],
        [-a / 2.0, a * np.sqrt(3.0) / 2.0, 0.0],
        [0.0, 0.0, c],
    ])
    basis = np.array([
        [1.0 / 3.0, 2.0 / 3.0, 0.25],
        [2.0 / 3.0, 1.0 / 3.0, 0.75],
    ])
    return Crystal(
        name=name,
        lattice_vectors=lattice,
        basis=basis,
        species=[element] * 2,
        meta={"lattice_type": "HCP", "lattice_params": {"a": a, "c": c}},
    )
