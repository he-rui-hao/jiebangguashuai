"""测量工具：原子间距与键角计算 + 最近原子拾取。

科研动机：科研人员在观察原子排布时经常需要量化——"这两个原子间距多少"、
"这个键角是不是理想值"。GUI 里点击原子即可得到读数。
"""
from __future__ import annotations

import numpy as np


def distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """两点欧氏距离（Å）。"""
    return float(np.linalg.norm(np.asarray(p1, dtype=float) - np.asarray(p2, dtype=float)))


def angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """键角 ∠(p1-p2-p3)，顶点在 p2，返回角度（度）。"""
    v1 = np.asarray(p1, dtype=float) - np.asarray(p2, dtype=float)
    v2 = np.asarray(p3, dtype=float) - np.asarray(p2, dtype=float)
    cos = float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def nearest_index(xs: np.ndarray, ys: np.ndarray, cx: float, cy: float) -> int:
    """返回离点击位置 (cx, cy) 最近的原子下标（供 GUI 拾取）。"""
    d2 = (np.asarray(xs) - cx) ** 2 + (np.asarray(ys) - cy) ** 2
    return int(np.argmin(d2))
