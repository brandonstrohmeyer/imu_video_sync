# -*- mode: python ; coding: utf-8 -*-
import os
import sys

from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_dynamic_libs
from PyInstaller.utils.hooks import collect_submodules

datas = [('assets\\\\icon\\\\IMUVideoSync.ico', 'assets\\\\icon'), ('assets\\\\icon\\\\IMUVideoSync.png', 'assets\\\\icon')]
binaries = []
hiddenimports = []
datas += collect_data_files('telemetry_parser')
binaries += collect_dynamic_libs('telemetry_parser')
hiddenimports += collect_submodules('telemetry_parser')


a = Analysis(
    ['scripts\\imu_video_sync_entry.py'],
    pathex=['src'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['scipy'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

icon_path = os.path.join("assets", "icon", "IMUVideoSync.ico")
if sys.platform == "darwin":
    icns_candidates = [
        os.path.join("build", "IMUVideoSync.icns"),
        os.path.join("assets", "icon", "IMUVideoSync.icns"),
    ]
    for candidate in icns_candidates:
        if os.path.exists(candidate):
            icon_path = candidate
            break

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='IMUVideoSync',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=[icon_path],
)
