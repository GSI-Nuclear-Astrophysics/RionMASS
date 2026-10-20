"""GUI entry point (``rionmass gui`` / ``rionmass-gui`` console script)."""

from __future__ import annotations

import sys

_HELP_TEXT = """usage: rionmass-gui [-h]

Launch the RionMASS desktop GUI. Takes no arguments once running.

options:
  -h, --help  show this message and exit
"""


def main(argv: list[str] | None = None) -> int:
    # Handle help before importing Qt or starting the event loop.
    args = sys.argv[1:] if argv is None else argv
    if any(a in ("-h", "--help") for a in args):
        print(_HELP_TEXT)
        return 0

    from PySide6.QtWidgets import QApplication

    from .main_window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
