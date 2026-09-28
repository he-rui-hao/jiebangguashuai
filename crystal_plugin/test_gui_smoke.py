"""GUI 离屏冒烟测试：不弹窗，实例化主窗口并依次触发各功能页生成。

用法：python test_gui_smoke.py（自动设置 QT_QPA_PLATFORM=offscreen）
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from gui.app import MainWindow


def main() -> None:
    app = QApplication([])
    win = MainWindow()

    checks = []

    def check(name, fn):
        try:
            fn()
            checks.append((name, True, ""))
            print(f"  [{name}] PASS")
        except Exception as e:  # noqa: BLE001 - 冒烟测试要抓所有异常
            checks.append((name, False, str(e)))
            print(f"  [{name}] FAIL  {type(e).__name__}: {e}")

    print("=== GUI 冒烟测试（离屏） ===")
    check("单晶2D 投影", win._gen_single2d)
    check("多方向对比", win._gen_multi)
    check("3D 视图", win._gen_3d)
    win.plane_chk.setChecked(True)
    check("3D 晶面标注", win._gen_3d)

    def _wheel_zoom():
        # 滚轮缩放回归：数据范围必须真的变化（防止 API 不存在被静默吞掉）
        ax = win.cv3d.ax3d
        assert ax is not None
        before = ax.get_w_lims()
        win.cv3d.zoom(win.settings["zoom_step"])       # 放大
        mid = ax.get_w_lims()
        win.cv3d.zoom(1.0 / win.settings["zoom_step"])  # 缩回
        after = ax.get_w_lims()
        assert mid[1] - mid[0] < before[1] - before[0], "放大未改变数据范围"
        assert abs(after[1] - after[0] - (before[1] - before[0])) < 1e-9
    check("3D 滚轮缩放", _wheel_zoom)
    check("SAED 衍射", win._gen_saed)
    check("XRD", win._gen_xrd)
    check("极射投影", win._gen_stereo)
    check("双晶 OR", win._gen_bicrystal)
    check("孪晶", win._gen_twin)

    # HCP 材料 + 四指数路径
    win.material_combo.setCurrentText("Mg")
    win.hcp4_dir.setChecked(True)
    win.dir_spins[0].setValue(2)
    win.dir_spins[1].setValue(-1)
    win.dir_spins[2].setValue(-1)
    win.dir_spins[3].setValue(0)
    check("HCP 四指数 [21̄1̄0] 投影", win._gen_single2d)
    check("HCP XRD", win._gen_xrd)

    # 测量工具：直接调用核心函数
    from core.measure import angle, distance
    d = distance([0, 0, 0], [1, 0, 0])
    a = angle([1, 0, 0], [0, 0, 0], [0, 1, 0])
    checks.append(("测量函数", abs(d - 1.0) < 1e-12 and abs(a - 90.0) < 1e-9, ""))
    print(f"  [测量函数] {'PASS' if checks[-1][1] else 'FAIL'}")

    n = sum(1 for _, ok, _ in checks if ok)
    print(f"\n{n}/{len(checks)} 通过")
    win.close()
    if n != len(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
