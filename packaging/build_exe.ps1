<#
.SYNOPSIS
    Freezes the DeepTune desktop app (desktop/app.py) into a standalone
    Windows build with PyInstaller: packaging\dist\DeepTune\DeepTune.exe
    plus its dependencies, needing no Python install on the machine that
    runs it.

.DESCRIPTION
    Run from anywhere; paths are resolved relative to this script. Uses the
    project's own .venv (the one requirements.txt was installed into) so the
    frozen build matches exactly what's importable in the checkout -- not a
    system Python, which may have different (or missing) packages.

    This only produces the frozen app folder. For the single Setup.exe most
    users should actually download, run build_installer.ps1 instead (it
    calls this script and then Inno Setup).

.EXAMPLE
    packaging\build_exe.ps1
#>
[CmdletBinding()]
param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Spec = Join-Path $PSScriptRoot "deeptune.spec"
$DistPath = Join-Path $PSScriptRoot "dist"
$WorkPath = Join-Path $PSScriptRoot "build"

if (-not (Test-Path $Python)) {
    throw "No .venv found at $Python. Create it and install requirements.txt first:`n" +
          "  python -m venv .venv`n  .venv\Scripts\python.exe -m pip install -r requirements.txt"
}

# PyInstaller (like many native tools) writes its own normal, successful-run
# logging to stderr. PowerShell 5.1 wraps every stderr line from a native
# command into a terminating NativeCommandError whenever that stream is
# being redirected or captured (piped, sent to a file, or captured by an
# automation harness) -- with $ErrorActionPreference = "Stop" in effect,
# that aborts the script on the very first log line even though the tool
# itself hasn't failed at all. Native calls below run under "Continue" and
# check $LASTEXITCODE themselves instead, which is what actually reflects
# success/failure for a non-PowerShell process.
$originalErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
    & $Python -m PyInstaller --version *> $null
} finally {
    $ErrorActionPreference = $originalErrorActionPreference
}
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing PyInstaller and its community hook package into .venv..." -ForegroundColor Cyan
    & $Python -m pip install pyinstaller pyinstaller-hooks-contrib
    if ($LASTEXITCODE -ne 0) { throw "pip install pyinstaller failed." }
}

if ($Clean -and (Test-Path $DistPath)) {
    Write-Host "Removing previous build output: $DistPath" -ForegroundColor DarkGray
    Remove-Item -Recurse -Force $DistPath
}
if ($Clean -and (Test-Path $WorkPath)) {
    Remove-Item -Recurse -Force $WorkPath
}

Write-Host "Building DeepTune.exe with PyInstaller (this bundles torch/transformers/tabpfn/etc. -- expect several minutes and a multi-GB output folder)..." -ForegroundColor Cyan
$originalErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
    & $Python -m PyInstaller $Spec --distpath $DistPath --workpath $WorkPath --noconfirm
} finally {
    $ErrorActionPreference = $originalErrorActionPreference
}
if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed (exit code $LASTEXITCODE)." }

$ExePath = Join-Path $DistPath "DeepTune\DeepTune.exe"
if (-not (Test-Path $ExePath)) { throw "Build reported success but $ExePath is missing." }

$SizeGB = [math]::Round(((Get-ChildItem (Join-Path $DistPath "DeepTune") -Recurse | Measure-Object -Property Length -Sum).Sum / 1GB), 2)
Write-Host ""
Write-Host "Built: $ExePath  ($SizeGB GB folder)" -ForegroundColor Green
Write-Host "Run it directly to test, or run build_installer.ps1 to produce a single Setup.exe for distribution." -ForegroundColor Green
