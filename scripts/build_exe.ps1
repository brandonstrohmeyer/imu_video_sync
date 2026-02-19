param(
    [string]$PythonExe = ".venv312\\Scripts\\python.exe",
    [switch]$Offline,
    [string]$WheelDir = "vendor\\wheels",
    [string]$Requirements = "requirements-build.txt",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

if (-not (Test-Path $PythonExe)) {
    throw "Python 3.12 venv not found at $PythonExe. Create .venv312 or pass -PythonExe."
}
if ($Offline) {
    if (-not (Test-Path $WheelDir)) {
        throw "Wheelhouse not found: $WheelDir"
    }
    if (-not (Test-Path $Requirements)) {
        throw "Requirements file not found: $Requirements"
    }
}

$venvRoot = Split-Path -Parent (Split-Path -Parent $PythonExe)
$venvPip = Join-Path $venvRoot "Scripts\\pip.exe"
$venvPyInstaller = Join-Path $venvRoot "Scripts\\pyinstaller.exe"
$pytestReport = Join-Path "build" "pytest-report.xml"

if (-not (Test-Path $venvPip)) {
    throw "pip not found in venv at $venvPip"
}

Write-Host "Installing build dependencies..."
& $venvPip install --upgrade pip
if ($Offline) {
    Write-Host "Offline mode: installing from wheelhouse."
    & $venvPip install --no-index --find-links $WheelDir -r $Requirements
} else {
    Write-Host "Online mode: installing from PyPI/git."
    & $venvPip install -r requirements.txt
    & $venvPip install -r requirements-build.txt
}

if (-not (Test-Path $venvPyInstaller)) {
    throw "pyinstaller not found after install at $venvPyInstaller"
}

Write-Host "Running tests..."
New-Item -ItemType Directory -Force (Split-Path $pytestReport) | Out-Null
& $PythonExe -m pytest --junitxml $pytestReport
if ($LASTEXITCODE -ne 0) {
    throw "Pytest failed with exit code $LASTEXITCODE"
}
if (-not (Test-Path $pytestReport)) {
    throw "Pytest report missing at $pytestReport"
}
[xml]$pytestXml = Get-Content $pytestReport
$skippedNodes = $pytestXml.SelectNodes("//skipped")
$skippedCount = 0
if ($skippedNodes) {
    $skippedCount = $skippedNodes.Count
}
if ($skippedCount -gt 0) {
    throw "Pytest reported $skippedCount skipped test(s). Failing build."
}

Write-Host "Building IMUVideoSync.exe..."
$pyinstallerArgs = @(
    "--noconfirm",
    "--onefile",
    "--noconsole",
    "--exclude-module", "scipy",
    "--icon", "assets\\icon\\IMUVideoSync.ico",
    "--name", "IMUVideoSync",
    "--paths", "src",
    "--add-data", "assets\\icon\\IMUVideoSync.ico;assets\\icon",
    "--add-data", "assets\\icon\\IMUVideoSync.png;assets\\icon",
    "--collect-binaries", "telemetry_parser",
    "--collect-data", "telemetry_parser",
    "--collect-submodules", "telemetry_parser",
    "scripts\\imu_video_sync_entry.py"
)
if ($Clean) {
    $pyinstallerArgs = @("--clean") + $pyinstallerArgs
}

if (Test-Path "dist\\IMUVideoSync") {
    Write-Host "Removing stale dist\\IMUVideoSync folder..."
    Remove-Item -Recurse -Force "dist\\IMUVideoSync"
}
& $venvPyInstaller @pyinstallerArgs

Write-Host "Done. Output: dist\\IMUVideoSync.exe"
