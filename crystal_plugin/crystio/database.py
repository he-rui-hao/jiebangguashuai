"""本地晶体结构数据库读取层。

数据库文件：data/materials.json（纯 JSON，零依赖，满足"自动读取本地
晶体结构数据库"要求）。按 lattice_type 由生成器展开基元坐标——坐标可算
的不入库，库里只存不可推导的晶格常数，数据量小且不易出错。
"""
from __future__ import annotations

import json
import os

import numpy as np

from core.structure import Crystal

_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "materials.json")


# ------------------------------------------------------------ 基元生成器
def _cubic(a: float) -> np.ndarray:
    return np.eye(3) * a


def _hcp_lattice(a: float, c: float) -> np.ndarray:
    return np.array([[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, c]])


_FCC_POS = [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]


def _gen(lattice_type: str, a: float, c: float | None, elements: list[str]):
    """返回 (lattice_vectors, basis, species)。elements[0]=A 位，[1]=B 位。"""
    e0 = elements[0]
    e1 = elements[1] if len(elements) > 1 else elements[0]

    if lattice_type == "sc":
        return _cubic(a), np.array([[0, 0, 0]]), [e0]
    if lattice_type == "fcc":
        return _cubic(a), np.array(_FCC_POS), [e0] * 4
    if lattice_type == "bcc":
        return _cubic(a), np.array([[0, 0, 0], [0.5, 0.5, 0.5]]), [e0] * 2
    if lattice_type == "hcp":
        if c is None:
            raise ValueError("HCP 需要晶格常数 c")
        return _hcp_lattice(a, c), np.array([[0, 0, 0], [1 / 3, 2 / 3, 0.5]]), [e0] * 2
    if lattice_type == "diamond":
        off = [0.25, 0.25, 0.25]
        basis = [p for p in _FCC_POS] + [[(p[0] + off[0]) % 1, (p[1] + off[1]) % 1,
                                          (p[2] + off[2]) % 1] for p in _FCC_POS]
        return _cubic(a), np.array(basis), [e0] * 8
    if lattice_type == "nacl":
        basis_a = _FCC_POS
        basis_b = [[(p[0] + 0.5) % 1, p[1], p[2]] for p in _FCC_POS]
        return _cubic(a), np.array(basis_a + basis_b), [e0] * 4 + [e1] * 4
    if lattice_type == "cscl":
        return _cubic(a), np.array([[0, 0, 0], [0.5, 0.5, 0.5]]), [e0, e1]
    if lattice_type == "zincblende":
        basis_a = _FCC_POS
        basis_b = [[(p[0] + 0.25) % 1, (p[1] + 0.25) % 1, (p[2] + 0.25) % 1]
                   for p in _FCC_POS]
        return _cubic(a), np.array(basis_a + basis_b), [e0] * 4 + [e1] * 4
    if lattice_type == "l12":
        # A 位顶点，B 位面心（如 Cu3Au：Au 顶点，Cu 面心）
        basis = [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]
        return _cubic(a), np.array(basis), [e0, e1, e1, e1]
    raise KeyError(f"未支持的晶格类型 {lattice_type!r}")


# ------------------------------------------------------------ 对外接口
def _load_db(path: str | None = None) -> dict:
    with open(path or _DB_PATH, encoding="utf-8") as f:
        return json.load(f)


def list_materials(path: str | None = None) -> dict[str, str]:
    """返回 {键名: 摘要}，供 GUI 下拉框填充。"""
    db = _load_db(path)
    out = {}
    for key, m in db["materials"].items():
        lt = m["lattice_type"]
        lat = f"a={m['a']}" + (f", c={m['c']}" if m.get("c") else "")
        out[key] = f"{m.get('note', '')} [{lt}] {lat} Å"
    return out


def load_material(key: str, path: str | None = None) -> Crystal:
    """按键名读取材料并展开为 Crystal 对象。

    >>> cu = load_material("Cu")
    >>> cu.lattice_vectors[0, 0]
    3.6149
    """
    db = _load_db(path)
    if key not in db["materials"]:
        raise KeyError(f"数据库中无材料 {key!r}，可选：{sorted(db['materials'])}")
    m = db["materials"][key]
    L, basis, species = _gen(m["lattice_type"], m["a"], m.get("c"), m["elements"])
    return Crystal(
        name=key,
        lattice_vectors=L,
        basis=basis,
        species=species,
        meta={
            "lattice_type": m["lattice_type"],
            "lattice_params": {"a": m["a"], "c": m.get("c")},
            "elements": m["elements"],
            "note": m.get("note", ""),
            "source": "materials.json（本地数据库）",
        },
    )
