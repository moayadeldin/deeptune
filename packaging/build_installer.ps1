<#
.SYNOPSIS
    Builds the single DeepTune-Setup-<version>.exe installer that end users
    download: freezes the app with PyInstaller (build_exe.ps1), then wraps
    it with Inno Setup (installer.iss).

.DESCRIPTION
    Requires Inno Setup 6 (ISCC.exe) installed -- https://jrsoftware.org/isinfo.php.
    The version number is read from the repo's VERSION file (e.g. "v1.1.0"
    -> installer version "1.1.0"); override with -Version if needed.

.EXAMPLE
    packaging\build_installer.ps1

.EXAMPLE
    packaging\build_installer.ps1 -Version 1.2.0 -SkipBuild
    Re-packages the existing packaging\dist\DeepTune build (skips
    re-running PyInstaller) under an explicit version number.
#>
[CmdletBinding()]
param(
    [string]$Version,
    [switch]$SkipBuild,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot

if (-not $Version) {
    $VersionFile = Join-Path $RepoRoot "VERSION"
    if (Test-Path $VersionFile) {
        $Version = (Get-Content $VersionFile -Raw).Trim().TrimStart("v")
    } else {
        $Version = "0.0.0-dev"
    }
}
Write-Host "Packaging DeepTune $Version" -ForegroundColor Cyan

$IsccCandidates = @(
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "${env:ProgramFiles}\Inno Setup 6\ISCC.exe"
)
$Iscc = $IsccCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $Iscc) {
    $OnPath = Get-Command iscc.exe -ErrorAction SilentlyContinue
    if ($OnPath) { $Iscc = $OnPath.Source }
}
if (-not $Iscc) {
    throw "Inno Setup 6 (ISCC.exe) not found. Install it from https://jrsoftware.org/isinfo.php, " +
          "then re-run this script."
}

if (-not $SkipBuild) {
    $BuildArgs = @{}
    if ($Clean) { $BuildArgs["Clean"] = $true }
    & (Join-Path $PSScriptRoot "build_exe.ps1") @BuildArgs
}

$ExePath = Join-Path $PSScriptRoot "dist\DeepTune\DeepTune.exe"
if (-not (Test-Path $ExePath)) {
    throw "$ExePath not found. Run without -SkipBuild first, or run build_exe.ps1."
}

Write-Host "Running Inno Setup..." -ForegroundColor Cyan
# See build_exe.ps1's matching comment: native tools writing normal output
# to stderr get wrapped into terminating errors by PowerShell 5.1 whenever
# that stream is captured/redirected, under $ErrorActionPreference = "Stop".
$originalErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
    & $Iscc (Join-Path $PSScriptRoot "installer.iss") "/DMyAppVersion=$Version"
} finally {
    $ErrorActionPreference = $originalErrorActionPreference
}
if ($LASTEXITCODE -ne 0) { throw "Inno Setup compile failed (exit code $LASTEXITCODE)." }

$SetupExe = Join-Path $PSScriptRoot "installer-dist\DeepTune-Setup-$Version.exe"
if (Test-Path $SetupExe) {
    $SizeMB = [math]::Round(((Get-Item $SetupExe).Length / 1MB), 1)
    Write-Host ""
    Write-Host "Installer ready: $SetupExe  ($SizeMB MB)" -ForegroundColor Green
    Write-Host "This is the one file to share -- it installs DeepTune for the current user, no admin rights or separate Python install needed." -ForegroundColor Green
} else {
    Write-Warning "Inno Setup reported success but $SetupExe was not found; check its OutputDir setting in installer.iss."
}
