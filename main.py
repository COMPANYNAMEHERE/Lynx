"""Entry point that gracefully reports fatal startup errors."""

from __future__ import annotations

import tkinter as tk


def fatal(msg: str) -> None:
    """Show an error message with an Exit button and stop."""
    root = tk.Tk()
    root.title("Lynx Error")
    tk.Label(root, text=msg, justify="left", wraplength=400).pack(padx=20, pady=10)
    tk.Button(root, text="Exit", command=root.destroy).pack(pady=(0, 10))
    root.mainloop()


def safe_main() -> None:
    try:
        from lynx.gui import main as gui_main
    except Exception as e:  # pragma: no cover - GUI import failed
        fatal(f"Failed to start GUI: {e}")
        return

    try:
        gui_main()
    except Exception as e:  # pragma: no cover - runtime failure
        fatal(str(e))


if __name__ == "__main__":
    safe_main()
