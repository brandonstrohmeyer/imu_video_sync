Place all required Windows x64 wheels in this folder before building.

Minimum wheels required:
- telemetry_parser-<ver>-cp312-cp312-win_amd64.whl
- numpy-<ver>-cp312-cp312-win_amd64.whl
- pandas-<ver>-cp312-cp312-win_amd64.whl
- scipy-<ver>-cp312-cp312-win_amd64.whl
- pyinstaller-<ver>-py3-none-any.whl
- pyinstaller_hooks_contrib-<ver>-py3-none-any.whl
- pytest-<ver>-py3-none-any.whl

Also include any dependency wheels required by the above packages (pip will not access the network).

Then run:
  .\scripts\build_exe.ps1
