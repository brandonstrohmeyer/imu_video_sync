from __future__ import annotations

import sys
from typing import Optional


def detect_system_theme() -> Optional[str]:
    if sys.platform.startswith("win"):
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize",
            )
            value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
            winreg.CloseKey(key)
            if value == 0:
                return "dark"
            if value == 1:
                return "light"
        except Exception:
            return None
        return None

    if sys.platform == "darwin":
        try:
            import subprocess

            result = subprocess.run(
                ["defaults", "read", "-g", "AppleInterfaceStyle"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0 and "Dark" in result.stdout:
                return "dark"
            return "light"
        except Exception:
            return None

    return None


def choose_ttkbootstrap_theme(default: str = "lumen") -> str:
    theme = detect_system_theme()
    if theme == "dark":
        return "darkly"
    if theme == "light":
        return "lumen"
    return default
