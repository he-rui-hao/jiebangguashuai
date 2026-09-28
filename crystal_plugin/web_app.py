"""晶体可视化插件 Web 版（Streamlit + Plotly）。

与桌面版（gui/app.py, PySide6）共用 core/ 内核，约定不变：
Web 层只做输入校验与四指数→三指数转换，所有晶体学数学都在 core/ 完成。

本地运行：streamlit run web_app.py
部署：Streamlit Community Cloud（入口文件填 web_app.py）。
"""
from __future__ import annotations

import io

import numpy as np
import streamlit as st

import matplotlib
matplotlib.use("Agg")  # 服务器无显示环境，必须 Agg
from matplotlib.figure import Figure
import plotly.graph_objects as go

from core.diffraction import (draw_saed, draw_xrd, simulate_diffraction,
                              simulate_powder_xrd)
from core.orientation import (four_to_three_dir, four_to_three_plane,
                              rotation_from_uvw)
from core.projector import (DEFAULT_COLORS, DEFAULT_RADII, _draw_proj,
                            draw_multi, project)
from core.stereographic import draw_stereographic, stereographic_poles
from crystio.database import list_materials, load_material
from bicrystal.or_builder import (OR_PRESETS, build_bicrystal,
                                  rotation_from_or_name)
from twin.twin_builder import build_twin

# ----------------------------------------------------------------------
# 全局：页面配置 / 字体 / 希腊字母显示名
# ----------------------------------------------------------------------
st.set_page_config(page_title="晶体可视化插件", layout="wide",
                   initial_sidebar_state="expanded")

# 云端 Linux 容器无微软雅黑：经 packages.txt 安装 fonts-noto-cjk 后回退到 Noto
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                        "WenQuanYi Zen Hei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
})

GRAIN_COLORS = ["#4c72b0", "#dd8452"]   # 基体/相1、孪晶/相2（与桌面版一致）

# 希腊字母英文拼写 → 符号（与 gui/app.py 的 disp_name 保持一致）
GREEK_WORDS = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε",
    "zeta": "ζ", "eta": "η", "theta": "θ", "kappa": "κ", "lambda": "λ",
    "mu": "μ", "nu": "ν", "rho": "ρ", "sigma": "σ", "tau": "τ", "omega": "ω",
}


def disp_name(name: str) -> str:
    """显示用材料名：gamma-Fe → γ-Fe、alpha-Fe → α-Fe。数据键保持 ASCII。"""
    for word, sym in GREEK_WORDS.items():
        name = name.replace(word + "-", sym + "-")
    return name


def parse_indices(text: str, n: int = 3) -> tuple[int, ...]:
    parts = text.replace(",", " ").split()
    if len(parts) != n:
        raise ValueError(f"需要 {n} 个整数指数，实际输入 {len(parts)} 个")
    return tuple(int(p) for p in parts)


# ----------------------------------------------------------------------
# 绘图辅助
# ----------------------------------------------------------------------
def mpl_download(fig: Figure, stem: str) -> None:
    """matplotlib 图 PNG 下载按钮。"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    st.download_button("下载 PNG", buf.getvalue(), f"{stem}.png", "image/png")


def plotly_atoms(coords: np.ndarray, labels: list[str],
                 colors: dict[str, str] | None = None,
                 size_scale: float = 1.0, title: str = "") -> go.Figure:
    """Plotly 3D 原子球（浏览器内拖动旋转 / 滚轮缩放）。"""
    coords = np.asarray(coords)
    colors = colors or {}
    fig = go.Figure()
    for lab in dict.fromkeys(labels):
        idx = [i for i, l in enumerate(labels) if l == lab]
        pts = coords[idx]
        elem = str(lab).split()[0]
        c = colors.get(elem, DEFAULT_COLORS.get(elem, "#888888"))
        size = max(DEFAULT_RADII.get(elem, 1.30) * 4.0 * size_scale, 2.0)
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers", name=str(lab),
            marker=dict(size=size, color=c,
                        line=dict(width=0.5, color="#333333"))))
    axis = dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#dddddd",
                showbackground=True, zerolinecolor="#bbbbbb")
    fig.update_layout(
        title=dict(text=title), height=640,
        scene=dict(aspectmode="data", xaxis=axis, yaxis=axis, zaxis=axis),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(itemsizing="constant"))
    return fig


def plotly_grains(coords: np.ndarray, grains: np.ndarray, names: tuple[str, str],
                  size_scale: float = 1.0, title: str = "") -> go.Figure:
    """双晶/孪晶 3D：按晶粒分两组着色。"""
    fig = go.Figure()
    for g, name in ((0, names[0]), (1, names[1])):
        pts = coords[grains == g]
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers", name=name,
            marker=dict(size=4.5 * size_scale, color=GRAIN_COLORS[g],
                        line=dict(width=0.3, color="#333333"))))
    axis = dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#dddddd",
                showbackground=True, zerolinecolor="#bbbbbb")
    fig.update_layout(
        title=dict(text=title), height=640,
        scene=dict(aspectmode="data", xaxis=axis, yaxis=axis, zaxis=axis),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(itemsizing="constant"))
    return fig


def to_xyz(coords: np.ndarray, species: list[str], comment: str) -> str:
    lines = [str(len(species)), comment]
    for s, (x, y, z) in zip(species, coords):
        lines.append(f"{s:2s} {x:14.6f} {y:14.6f} {z:14.6f}")
    return "\n".join(lines)


# 双晶/孪晶构建 O(N²) 去重较慢，按参数缓存
@st.cache_data(show_spinner=False)
def _cached_bicrystal(key1: str, key2: str, or_name: str,
                      interface: tuple[int, int, int], n: int):
    p1, p2 = load_material(key1), load_material(key2)
    R = rotation_from_or_name(p1, p2, or_name)
    res = build_bicrystal(p1, p2, R, interface, supercell=(n, n, n))
    return res.coords, res.species, res.grains, res.n_atoms_before, \
        res.n_removed_overlap, res.gap


@st.cache_data(show_spinner=False)
def _cached_twin(key: str, plane: tuple[int, int, int], n: int):
    res = build_twin(load_material(key), plane, supercell=(n, n, n))
    return res.coords, res.species, res.grains, res.n_atoms_before, \
        res.n_removed_overlap


# ----------------------------------------------------------------------
# 侧边栏：公共输入
# ----------------------------------------------------------------------
st.sidebar.header("公共输入")

MATERIAL_KEYS = list(list_materials().keys())
mat_key = st.sidebar.selectbox(
    "材料", MATERIAL_KEYS, format_func=disp_name,
    index=MATERIAL_KEYS.index("Cu") if "Cu" in MATERIAL_KEYS else 0)

_meta = load_material(mat_key).meta["lattice_params"]
_a0, _c0 = _meta["a"], _meta.get("c")
# 材料切换后重置晶格常数输入框
if st.session_state.get("_mat") != mat_key:
    st.session_state["_mat"] = mat_key
    st.session_state["a_in"] = float(_a0)
    if _c0 is not None:
        st.session_state["c_in"] = float(_c0)

a_val = st.sidebar.number_input("a (Å)", 0.1, 20.0, step=0.001,
                                format="%.4f", key="a_in")
if _c0 is not None:
    if "c_in" not in st.session_state:      # 从立方切到 HCP 的兜底
        st.session_state["c_in"] = float(_c0)
    c_val = st.sidebar.number_input("c (Å, HCP)", 0.1, 20.0, step=0.001,
                                    format="%.4f", key="c_in")
else:
    c_val = None

n_sc = st.sidebar.slider("超胞 n×n×n", 1, 6, 3)

st.sidebar.subheader("观察方向")
hcp4 = st.sidebar.checkbox("HCP 四指数 [uvtw]", value=False,
                           disabled=(_c0 is None))
if hcp4:
    dcols = st.sidebar.columns(4)
    du = dcols[0].number_input("u", -9, 9, 1, key="d0")
    dv = dcols[1].number_input("v", -9, 9, 1, key="d1")
    dcols[2].number_input("t=-(u+v)", value=-(du + dv), disabled=True, key="d2")
    dw = dcols[3].number_input("w", -9, 9, 0, key="d3")
    uvw = four_to_three_dir((du, dv, -(du + dv), dw))
else:
    dcols = st.sidebar.columns(3)
    uvw = tuple(dcols[i].number_input(l, -9, 9, d, key=f"d{i}")
                for i, (l, d) in enumerate((("u", 1), ("v", 1), ("w", 0))))

inplane_mode = st.sidebar.radio("面内晶向", ("自动", "手动"), horizontal=True)
in_plane: tuple[int, int, int] | None = None
if inplane_mode == "手动":
    pcols = st.sidebar.columns(3)
    in_plane = tuple(pcols[i].number_input(l, -9, 9, 0, key=f"p{i}")
                     for i, l in enumerate(("x", "y", "z")))
    if in_plane == (0, 0, 0):
        st.sidebar.warning("面内晶向不能为 (0 0 0)，已回退为自动")
        in_plane = None

with st.sidebar.expander("显示设置"):
    atom_scale = st.slider("原子大小倍数", 0.3, 2.5, 1.0, 0.1)
    font_pt = st.slider("图中字号", 8, 20, 12)
matplotlib.rcParams.update({
    "font.size": font_pt, "axes.titlesize": font_pt + 2,
    "axes.labelsize": font_pt,
    "xtick.labelsize": font_pt - 1, "ytick.labelsize": font_pt - 1,
    "legend.fontsize": font_pt - 1,
})

# 当前 Crystal（按 a/c 微调）
crystal = load_material(mat_key)
_p = crystal.meta["lattice_params"]
_ratio = a_val / _a0
crystal.lattice_vectors[:2] *= _ratio
if c_val is not None:
    crystal.lattice_vectors[2] *= c_val / _c0
    _p["c"] = c_val
else:
    crystal.lattice_vectors[2] *= _ratio
_p["a"] = a_val

# ----------------------------------------------------------------------
# 主区
# ----------------------------------------------------------------------
st.title("晶体任意方向原子排布特征生成")
st.caption("单晶投影 · 多方向对比 · 3D 视图 · SAED · XRD · 极射投影 · 双晶 OR · 孪晶"
           f" ｜ 当前材料：**{disp_name(mat_key)}**　观察方向：**[{uvw[0]} {uvw[1]} {uvw[2]}]**")

(tab_2d, tab_multi, tab_3d, tab_saed, tab_xrd,
 tab_stereo, tab_bi, tab_twin) = st.tabs(
    ["单晶2D", "多方向对比", "3D视图", "SAED衍射", "XRD",
     "极射投影", "双晶OR", "孪晶"])


def show_error(e: Exception) -> None:
    st.error(f"输入错误：{e}")


# ---------- 页 1：单晶 2D ----------
with tab_2d:
    c1, c2, c3 = st.columns([1, 1, 3])
    slice_on = c1.checkbox("单层切片", value=False)
    zmin = c2.number_input("z ≥", -100.0, 100.0, -0.5, 0.1)
    zmax = c2.number_input("z ≤", -100.0, 100.0, 0.5, 0.1)
    try:
        R = rotation_from_uvw(uvw, crystal.lattice_vectors, in_plane)
        x, y, z, sp = project(crystal, R, (n_sc, n_sc, n_sc))
        fig = Figure(figsize=(7, 7))
        ax = fig.add_subplot(111)
        _draw_proj(ax, x, y, z, sp,
                   f"{disp_name(crystal.name)}  [{uvw[0]} {uvw[1]} {uvw[2]}] 投影",
                   atom_scale=0.35 * atom_scale,
                   z_range=(zmin, zmax) if slice_on else None)
        st.pyplot(fig, clear_figure=True)
        mpl_download(fig, f"{mat_key}_{uvw[0]}{uvw[1]}{uvw[2]}_proj")
    except (ValueError, KeyError) as e:
        show_error(e)

# ---------- 页 2：多方向对比 ----------
with tab_multi:
    vcols = st.columns(4)
    vtexts = [vcols[i].text_input(f"方向 {i + 1}", v, key=f"v{i}")
              for i, v in enumerate(("1 0 0", "1 1 0", "1 1 1", "1 1 2"))]
    try:
        views = [(parse_indices(t), None) for t in vtexts]
        fig = Figure(figsize=(10, 10))
        draw_multi(fig, crystal, views, (n_sc, n_sc, n_sc),
                   atom_scale=0.30 * atom_scale)
        st.pyplot(fig, clear_figure=True)
        mpl_download(fig, f"{mat_key}_multi")
    except (ValueError, KeyError) as e:
        show_error(e)

# ---------- 页 3：3D 视图 ----------
with tab_3d:
    plane_on = st.checkbox("标注晶面 (hkl)", value=False)
    hkl3 = (1, 1, 1)
    if plane_on:
        hcols = st.columns(3)
        hkl3 = tuple(hcols[i].number_input(l, -9, 9, d, key=f"h{i}")
                     for i, (l, d) in enumerate((("h", 1), ("k", 1), ("l", 1))))
    frac, sp3 = crystal.supercell(n_sc, n_sc, n_sc)
    coords3 = frac @ crystal.lattice_vectors
    coords3 -= coords3.mean(axis=0)
    fig3 = plotly_atoms(coords3, sp3, size_scale=atom_scale,
                        title=f"{disp_name(crystal.name)}  3D（拖动旋转 · 滚轮缩放）")
    if plane_on:
        g = np.linalg.inv(crystal.lattice_vectors) @ np.array(hkl3, dtype=float)
        n_vec = g / np.linalg.norm(g)
        ref = np.array([1.0, 0, 0]) if abs(n_vec[0]) < 0.9 else np.array([0, 1.0, 0])
        e1 = np.cross(n_vec, ref)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n_vec, e1)
        half = np.linalg.norm(crystal.lattice_vectors, axis=1).mean() * 1.2
        corners = np.array([half * (e1 + e2), half * (e1 - e2),
                            -half * (e1 + e2), -half * (e1 - e2)])
        fig3.add_trace(go.Mesh3d(
            x=corners[:, 0], y=corners[:, 1], z=corners[:, 2],
            i=[0, 0], j=[1, 2], k=[2, 3],
            color="#ff8800", opacity=0.25, name=f"({hkl3[0]}{hkl3[1]}{hkl3[2]})"))
        tip = n_vec * half
        fig3.add_trace(go.Scatter3d(
            x=[0, tip[0]], y=[0, tip[1]], z=[0, tip[2]], mode="lines",
            line=dict(color="#ff8800", width=5), name="法线"))
    st.plotly_chart(fig3, width="stretch")

# ---------- 页 4：SAED ----------
with tab_saed:
    maxidx = st.slider("max_index", 1, 6, 3)
    try:
        pat = simulate_diffraction(crystal, uvw, in_plane, maxidx)
        fig = Figure(figsize=(7, 7))
        fig.patch.set_facecolor("#111111")
        ax = fig.add_subplot(111)
        draw_saed(ax, pat, f"{disp_name(crystal.name)}  SAED")
        st.pyplot(fig, clear_figure=True)
        mpl_download(fig, f"{mat_key}_{uvw[0]}{uvw[1]}{uvw[2]}_saed")
    except (ValueError, KeyError) as e:
        show_error(e)

# ---------- 页 5：XRD ----------
with tab_xrd:
    c1, c2, c3 = st.columns(3)
    wl = c1.number_input("波长 λ (Å)", 0.1, 5.0, 1.5406, 0.0001, format="%.4f")
    tth_min = c2.number_input("2θ 起始", 1.0, 170.0, 10.0, 1.0)
    tth_max = c3.number_input("2θ 终止", 10.0, 179.0, 120.0, 1.0)
    try:
        pat = simulate_powder_xrd(crystal, wl, (tth_min, tth_max))
        fig = Figure(figsize=(10, 4.5))
        ax = fig.add_subplot(111)
        draw_xrd(ax, pat, f"{disp_name(crystal.name)}  XRD  (λ={wl} Å)")
        st.pyplot(fig, clear_figure=True)
        mpl_download(fig, f"{mat_key}_xrd")
    except (ValueError, KeyError) as e:
        show_error(e)

# ---------- 页 6：极射投影 ----------
with tab_stereo:
    poles_text = st.text_input("极点 (hkl)，逗号分隔，HCP 可写 4 指数",
                               "001, 100, 010, 110, 111, 112")
    try:
        import re
        hkls = []
        for item in poles_text.split(","):
            item = item.strip()
            if not item:
                continue
            tokens = item.split()
            if len(tokens) == 1 and re.fullmatch(r"-?\d{3,4}", item):
                idx = tuple(int(ch) for ch in item.lstrip("-"))
            else:
                idx = tuple(int(t) for t in tokens)
            if len(idx) == 4:
                idx = four_to_three_plane(idx)
            elif len(idx) != 3:
                raise ValueError(f"极点 {item!r} 不是 3 或 4 个指数")
            hkls.append(idx)
        poles = stereographic_poles(crystal, hkls, uvw, in_plane)
        fig = Figure(figsize=(7, 7))
        ax = fig.add_subplot(111)
        draw_stereographic(ax, poles, f"{disp_name(crystal.name)}  极射投影")
        st.pyplot(fig, clear_figure=True)
        mpl_download(fig, f"{mat_key}_stereo")
    except (ValueError, KeyError) as e:
        show_error(e)

# ---------- 页 7：双晶 OR ----------
with tab_bi:
    c1, c2, c3, c4 = st.columns([1, 1, 1.4, 1])
    k1 = c1.selectbox("相1", MATERIAL_KEYS, format_func=disp_name,
                      index=MATERIAL_KEYS.index("gamma-Fe")
                      if "gamma-Fe" in MATERIAL_KEYS else 0)
    k2 = c2.selectbox("相2", MATERIAL_KEYS, format_func=disp_name,
                      index=MATERIAL_KEYS.index("alpha-Fe")
                      if "alpha-Fe" in MATERIAL_KEYS else 0)
    or_name = c3.selectbox("取向关系", list(OR_PRESETS.keys()))
    bi_hkl = c4.text_input("界面 (hkl)", "1 1 0")
    bi_n = st.slider("双晶超胞", 3, 6, 4, key="bi_n")
    try:
        interface = parse_indices(bi_hkl)
        coords, species, grains, n_before, n_removed, gap = _cached_bicrystal(
            k1, k2, or_name, interface, bi_n)
        fig_bi = plotly_grains(
            coords, grains, ("相1", "相2"), size_scale=atom_scale,
            title=f"{disp_name(k1)} / {disp_name(k2)}  {or_name} 双晶"
                  "（拖动旋转 · 滚轮缩放）")
        st.plotly_chart(fig_bi, width="stretch")
        st.info(f"原子数 {n_before} → {len(coords)}（去重删 {n_removed}），"
                f"界面间距 gap = {gap:.3f} Å")
        st.download_button(
            "下载 XYZ 结构文件",
            to_xyz(coords, species,
                   f"{k1}/{k2} {or_name} bicrystal, interface={interface}"),
            f"{k1}_{k2}_bicrystal.xyz", "chemical/x-xyz")
    except (ValueError, KeyError) as e:
        show_error(e)

# ---------- 页 8：孪晶 ----------
with tab_twin:
    c1, c2 = st.columns([2, 1])
    tw4 = c1.checkbox("四指数 (hkil)", value=False, disabled=(_c0 is None),
                      key="tw4")
    if tw4:
        tcols = c1.columns(4)
        th = tcols[0].number_input("h", -9, 9, 1, key="t0")
        tk = tcols[1].number_input("k", -9, 9, 1, key="t1")
        tcols[2].number_input("i=-(h+k)", value=-(th + tk), disabled=True,
                              key="t2")
        tl = tcols[3].number_input("l", -9, 9, 0, key="t3")
    else:
        tcols = c1.columns(3)
        th = tcols[0].number_input("h", -9, 9, 1, key="t0")
        tk = tcols[1].number_input("k", -9, 9, 1, key="t1")
        tl = tcols[2].number_input("l", -9, 9, 1, key="t3")
    tw_n = c2.slider("孪晶超胞", 3, 6, 4, key="tw_n")
    try:
        if tw4:
            plane = four_to_three_plane((th, tk, -(th + tk), tl))
        else:
            plane = (th, tk, tl)
        coords, species, grains, n_before, n_removed = _cached_twin(
            mat_key, plane, tw_n)
        fig_tw = plotly_grains(
            coords, grains, ("基体", "孪晶"), size_scale=atom_scale,
            title=f"{disp_name(crystal.name)}  ({plane[0]}{plane[1]}{plane[2]}) 孪晶"
                  "（拖动旋转 · 滚轮缩放）")
        st.plotly_chart(fig_tw, width="stretch")
        st.info(f"原子数 {n_before} → {len(coords)}（界面去重删 {n_removed}）")
        st.download_button(
            "下载 XYZ 结构文件",
            to_xyz(coords, species,
                   f"{mat_key} twin K1={plane}"),
            f"{mat_key}_twin_{plane[0]}{plane[1]}{plane[2]}.xyz",
            "chemical/x-xyz")
    except (ValueError, KeyError) as e:
        show_error(e)

st.divider()
st.caption("内核与桌面版完全一致（core/ 取向引擎 · 投影 · 衍射 · 双晶/孪晶构建）。"
           "测量工具（键长/键角）请使用桌面版。")
