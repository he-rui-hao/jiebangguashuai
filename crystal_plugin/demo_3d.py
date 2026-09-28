"""演示脚本：单晶原子排布 3D 交互可视化。

运行：python demo_3d.py
输出：
  output/FCC_Cu_3d.html  —— 自包含交互网页，双击浏览器打开，
                           鼠标左键旋转 / 滚轮缩放 / 右键平移
  output/FCC_Cu_3d.png   —— 静态截图（需 kaleido），用于自动验证
"""
from __future__ import annotations

import os

from core.structure import fcc
from core.viewer3d import build_figure, save_html

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT_DIR, exist_ok=True)

if __name__ == "__main__":
    cu = fcc("FCC Cu", "Cu", a=3.615)
    fig = build_figure(cu, supercell=(3, 3, 3), title="FCC Cu (a=3.615 Å) — 拖拽旋转查看任意方向")

    html_path = os.path.join(OUT_DIR, "FCC_Cu_3d.html")
    save_html(fig, html_path)
    print(f"[1] 交互 HTML 已保存: {html_path}")

    # 静态截图用于自动验证（kaleido 可选依赖）
    try:
        png_path = os.path.join(OUT_DIR, "FCC_Cu_3d.png")
        fig.write_image(png_path, width=900, height=900, scale=2)
        print(f"[2] 验证截图已保存: {png_path}")
    except Exception as e:
        print(f"[2] 跳过静态截图（未安装 kaleido: {e}）")

    print(f"[3] trace 数 = {len(fig.data)}（12 棱 + 3 轴箭头 + 1 元素组 = 16 为正常）")
