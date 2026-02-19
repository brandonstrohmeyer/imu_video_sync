from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Optional


def _parent_process_name() -> str:
    if sys.platform.startswith("win"):
        return _parent_process_name_windows() or ""
    if sys.platform == "darwin":
        return _parent_process_name_macos() or ""
    return ""


def _parent_process_name_windows() -> Optional[str]:
    names = _windows_ancestor_names(limit=1)
    return names[0] if names else None

def _windows_ancestor_names(limit: int = 6) -> list[str]:
    try:
        import ctypes
        import ctypes.wintypes as wintypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

        class PROCESS_BASIC_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("Reserved1", ctypes.c_void_p),
                ("PebBaseAddress", ctypes.c_void_p),
                ("Reserved2", ctypes.c_void_p * 2),
                ("UniqueProcessId", ctypes.c_void_p),
                ("InheritedFromUniqueProcessId", ctypes.c_void_p),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        ntdll = ctypes.WinDLL("ntdll", use_last_error=True)

        def _get_parent_pid(pid: int) -> Optional[int]:
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not handle:
                return None
            try:
                info = PROCESS_BASIC_INFORMATION()
                size = wintypes.ULONG()
                status = ntdll.NtQueryInformationProcess(
                    handle,
                    0,
                    ctypes.byref(info),
                    ctypes.sizeof(info),
                    ctypes.byref(size),
                )
                if status != 0:
                    return None
                parent = ctypes.cast(info.InheritedFromUniqueProcessId, ctypes.c_void_p).value
                return int(parent) if parent else None
            finally:
                kernel32.CloseHandle(handle)

        def _get_process_name(pid: int) -> Optional[str]:
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not handle:
                return None
            try:
                size = wintypes.DWORD(260)
                buf = ctypes.create_unicode_buffer(size.value)
                if not kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
                    return None
                return os.path.basename(buf.value) if buf.value else None
            finally:
                kernel32.CloseHandle(handle)

        names: list[str] = []
        pid = os.getpid()
        seen: set[int] = set()
        for _ in range(limit):
            parent_pid = _get_parent_pid(pid)
            if not parent_pid or parent_pid in seen:
                break
            seen.add(parent_pid)
            name = _get_process_name(parent_pid)
            if name:
                names.append(name)
            pid = parent_pid
        return names
    except Exception:
        return []


def _windows_launched_from_shell() -> bool:
    shell_names = {
        "cmd.exe",
        "powershell.exe",
        "pwsh.exe",
        "wt.exe",
        "windowsterminal.exe",
        "bash.exe",
    }
    for name in _windows_ancestor_names():
        if name.lower().strip() in shell_names:
            return True
    return False


def _parent_process_name_macos() -> Optional[str]:
    import subprocess

    ppid = os.getppid()
    try:
        out = subprocess.check_output(
            ["ps", "-p", str(ppid), "-o", "comm="], text=True
        )
        return out.strip() or None
    except Exception:
        return None


def _launched_from_shell(parent_name: str) -> bool:
    parent = parent_name.lower().strip()
    shell_names = {
        "cmd.exe",
        "powershell.exe",
        "pwsh.exe",
        "wt.exe",
        "windowsterminal.exe",
        "bash",
        "zsh",
        "fish",
        "sh",
        "tcsh",
        "ksh",
    }
    return parent in shell_names


def _should_launch_gui(argv: list[str]) -> bool:
    if len(argv) > 1:
        return False
    if sys.platform not in ("win32", "darwin"):
        return False
    if sys.platform == "win32" and _windows_launched_from_shell():
        return False
    parent = _parent_process_name()
    if parent and _launched_from_shell(parent):
        return False
    return True


def _hide_console_window_windows() -> None:
    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes

        hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        if hwnd:
            ctypes.windll.user32.ShowWindow(hwnd, 0)
    except Exception:
        return None


def _show_console_window_windows() -> None:
    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes

        hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        if hwnd:
            ctypes.windll.user32.ShowWindow(hwnd, 5)
    except Exception:
        return None


def _ensure_console_windows() -> None:
    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        ATTACH_PARENT_PROCESS = -1
        attached = kernel32.AttachConsole(ATTACH_PARENT_PROCESS)
        if not attached:
            if not kernel32.AllocConsole():
                return
        sys.stdout = open("CONOUT$", "w", encoding="utf-8", errors="replace", buffering=1)
        sys.stderr = open("CONOUT$", "w", encoding="utf-8", errors="replace", buffering=1)
        sys.stdin = open("CONIN$", "r", encoding="utf-8", errors="replace")
    except Exception:
        return None


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

    try:
        if sys.stdin and sys.stdin.isatty():
            input("Press Enter to close...")
    except Exception:
        return None


if __name__ == "__main__":
    if sys.platform.startswith("win") and len(sys.argv) <= 1:
        _hide_console_window_windows()
    if _should_launch_gui(sys.argv):
        if sys.platform.startswith("win"):
            _hide_console_window_windows()
        try:
            from imu_video_sync.gui import main as gui_main

            gui_main()
        except Exception as exc:
            _report_gui_launch_error(exc)
            sys.exit(1)
    else:
        if sys.platform.startswith("win"):
            _ensure_console_windows()
        from imu_video_sync.cli import main as cli_main

        cli_main()
