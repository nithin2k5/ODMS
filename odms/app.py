"""Application entry point."""
from __future__ import annotations

import logging
import sys
import tkinter as tk


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if sys.platform.startswith("win"):
        try:  # crisp rendering on high-DPI displays
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass
    from .ui.main_window import MainWindow

    root = tk.Tk()
    app = MainWindow(root)
    args = sys.argv[1:] if argv is None else argv
    if args:
        from pathlib import Path
        root.after(200, lambda: app.load_image(Path(args[0])))
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
