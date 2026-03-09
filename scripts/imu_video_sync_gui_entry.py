from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Optional


def _resolve_error_log_path() -> Path:
    if getattr(sys, "frozen", False):
        exe_path = Path(sys.executable)
        return exe_path.with_name("IMUVideoSync-gui-error.txt")
    return Path.cwd() / "IMUVideoSync-gui-error.txt"


def _write_gui_error_log(details: str) -> Optional[Path]:
    try:
        path = _resolve_error_log_path()
        path.write_text(details, encoding="utf-8")
        return path
    except Exception:
        return None


def _show_error_dialog_windows(message: str) -> None:
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, message, "IMUVideoSync", 0x10)
    except Exception:
        return None


def _show_error_dialog_macos(message: str) -> None:
    try:
        import subprocess

        safe = message.replace('"', '\\"')
        subprocess.run(["osascript", "-e", f'display dialog "{safe}" with title "IMUVideoSync"'])
    except Exception:
        return None


def _report_gui_launch_error(exc: Exception) -> None:
    details = traceback.format_exc()
    log_path = _write_gui_error_log(details)
    base = "IMUVideoSync failed to launch the GUI."
    if log_path:
        msg = f"{base}\n\nSee {log_path.name} for details."
    else:
        msg = f"{base}\n\n{exc}"

    if sys.platform.startswith("win"):
        _show_error_dialog_windows(msg)
    elif sys.platform == "darwin":
        _show_error_dialog_macos(msg)
    else:
        try:
            sys.stderr.write(msg + "\n")
        except Exception:
            return None


if __name__ == "__main__":
    try:
        from imu_video_sync.gui import main as gui_main

        gui_main()
    except Exception as exc:
        _report_gui_launch_error(exc)
        sys.exit(1)
