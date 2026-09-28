"""GUI 启动入口：python run_gui.py"""
import sys

from PySide6.QtWidgets import QApplication

from gui.app import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
